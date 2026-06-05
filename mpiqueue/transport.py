"""
Transport — the main entry point for the MPI queue.

Each rank creates a Transport, starts it, and uses put()/get() directly.
A SocketGateway can be attached via start_gateway() so local external
processes can join through a Unix socket.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, Optional

from mpi4py import MPI

from .membership import MembershipTable, NodeInfo
from .message import Message

if TYPE_CHECKING:
    from .gateway import SocketGateway

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

log = logging.getLogger(__name__)

# Reserved system tags — keep high to avoid collision with user tags.
TAG_PEER_JOINED   = 0x7FFB   # 32763
TAG_PEER_LEFT     = 0x7FFC   # 32764
TAG_RELAY         = 0x7FFD   # 32765 — envelope: dest is a socket client
TAG_BUFFER_HEADER = 0x7FFE   # 32766 — envelope before a zero-copy buffer send
TAG_SHUTDOWN      = 0x7FFF   # 32767


@dataclasses.dataclass
class _BufMeta:
    """Pickle envelope sent immediately before a zero-copy buffer transfer."""
    dtype: str
    shape: tuple
    tag: int      # original user tag (used for the buffer Isend/Recv)
    src: int
    dest: int
    msg_id: str


def _is_buffer_payload(obj: Any) -> bool:
    """Return True if *obj* supports the buffer protocol (e.g. numpy array)."""
    if _NUMPY and isinstance(obj, np.ndarray):
        return True
    if isinstance(obj, (bytearray, memoryview)):
        return True
    return False


class Transport:
    """
    Background MPI I/O thread with optional Unix socket gateway.

    Usage::

        from mpi4py import rc
        rc.thread_level = "multiple"   # required if start_gateway() is used
        from mpiqueue import Transport

        t = Transport()
        t.start()
        t.start_gateway()              # optional — opens /tmp/mpiqueue_rank{N}.sock

        t.put(dest=1, payload={"job": 42}, tag=1)
        msg = t.get(timeout=5.0)

        t.stop()

    Or as a context manager::

        with Transport() as t:
            t.start_gateway()
            t.put(dest=1, payload="hello", tag=1)
            msg = t.get()
    """

    def __init__(
        self,
        comm: MPI.Comm | None = None,
        poll_interval: float = 0.0001,   # set to 0 for busy-wait / minimum latency
        outbox_batch: int = 64,
    ) -> None:
        self.comm = comm or MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()
        self.size: int = self.comm.Get_size()

        # local_rank: rank within the shared-memory domain (same host).
        # Used as the default socket path discriminator.
        _local_comm = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
        self.local_rank: int = _local_comm.Get_rank()

        self._poll_interval = poll_interval
        self._outbox_batch = outbox_batch

        self._outbox: Queue[Message] = Queue()
        self._inbox:  Queue[Message] = Queue()

        self.membership = MembershipTable(self.size)

        self._tag_handlers: dict[int, Callable[[Message], Any]] = {}

        self._running = False
        self._thread: threading.Thread | None = None
        self._pending: list[MPI.Request] = []

        self._gateway: SocketGateway | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "Transport":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"mpi-transport-rank{self.rank}",
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        if self._gateway:
            self._gateway.stop()
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def __enter__(self) -> "Transport":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Gateway
    # ------------------------------------------------------------------

    def start_gateway(self, socket_path: Optional[str] = None) -> "Transport":
        """
        Open a Unix domain socket on this rank so local external processes
        can connect via SocketClient.

        Default path: ``/tmp/mpiqueue_rank{local_rank}.sock``

        Requires ``rc.thread_level = "multiple"`` before importing mpi4py.
        """
        from .gateway import SocketGateway
        self._gateway = SocketGateway(self, socket_path=socket_path)
        self._gateway.start()
        return self

    @property
    def socket_path(self) -> Optional[str]:
        return self._gateway.socket_path if self._gateway else None

    # ------------------------------------------------------------------
    # Messaging API
    # ------------------------------------------------------------------

    def put(self, dest: int, payload: Any, tag: int = 0) -> Message:
        """Send *payload* to *dest* node_id with *tag*. Non-blocking."""
        msg = Message(payload=payload, dest=dest, tag=tag)
        self.send(msg)
        return msg

    def get(self, timeout: Optional[float] = None) -> Message:
        """Block until a message arrives (raises queue.Empty on timeout)."""
        return self._inbox.get(timeout=timeout)

    def get_nowait(self) -> Message:
        """Return a message immediately or raise queue.Empty."""
        return self._inbox.get_nowait()

    def broadcast(self, payload: Any, tag: int = 0, exclude_self: bool = True) -> None:
        """Fan out *payload* to all known nodes."""
        for info in self.membership:
            if exclude_self and info.node_id == self.rank:
                continue
            self.put(dest=info.node_id, payload=payload, tag=tag)

    # ------------------------------------------------------------------
    # Internal send (takes a Message directly — used by SocketGateway)
    # ------------------------------------------------------------------

    def send(self, msg: Message) -> None:
        msg.src = self.rank
        self._outbox.put(msg)

    def register_tag_handler(self, tag: int, handler: Callable[[Message], Any]) -> None:
        self._tag_handlers[tag] = handler

    # ------------------------------------------------------------------
    # Send dispatch
    # ------------------------------------------------------------------

    def _dispatch_send(self, msg: Message) -> None:
        info = self.membership.get(msg.dest)
        if info is None:
            log.warning("Rank %d: unknown dest node_id=%d — dropping", self.rank, msg.dest)
            return

        if info.is_static:
            if _is_buffer_payload(msg.payload):
                # Two-phase zero-copy path:
                #   1. Small pickle envelope (dtype/shape/meta).
                #   2. Raw buffer send — no pickle, RDMA-capable on InfiniBand.
                meta = _BufMeta(
                    dtype=str(msg.payload.dtype) if _NUMPY else "bytes",
                    shape=getattr(msg.payload, "shape", (len(msg.payload),)),
                    tag=msg.tag,
                    src=msg.src,
                    dest=msg.dest,
                    msg_id=msg.msg_id,
                )
                r1 = self.comm.isend(meta, dest=msg.dest, tag=TAG_BUFFER_HEADER)
                r2 = self.comm.Isend(msg.payload, dest=msg.dest, tag=msg.tag)
                self._pending.extend([r1, r2])
            else:
                # Fallback: pickle path for arbitrary Python objects.
                req = self.comm.isend(msg, dest=msg.dest, tag=msg.tag)
                self._pending.append(req)
        else:
            gateway = info.gateway_rank
            if gateway == self.rank:
                handler = self._tag_handlers.get(TAG_RELAY)
                if handler:
                    handler(msg)
                else:
                    log.warning("Rank %d: no relay handler for socket client %d", self.rank, msg.dest)
            else:
                relay = Message(payload=msg, dest=gateway, tag=TAG_RELAY)
                relay.src = self.rank
                self._pending.append(self.comm.isend(relay, dest=gateway, tag=TAG_RELAY))

    # ------------------------------------------------------------------
    # Receive dispatch
    # ------------------------------------------------------------------

    def _handle_incoming(self, msg: Message) -> None:
        tag = msg.tag

        if tag == TAG_PEER_JOINED:
            info: NodeInfo = msg.payload
            self.membership.register(info)
            log.info("Rank %d: peer joined — %r", self.rank, info)
            self._inbox.put(msg)
            return

        if tag == TAG_PEER_LEFT:
            info = msg.payload
            self.membership.remove(info.node_id)
            log.info("Rank %d: peer left — %r", self.rank, info)
            self._inbox.put(msg)
            return

        if tag == TAG_RELAY:
            handler = self._tag_handlers.get(TAG_RELAY)
            if handler:
                handler(msg.payload)
            else:
                log.warning("Rank %d: received relay but no handler registered", self.rank)
            return

        if tag in self._tag_handlers:
            self._tag_handlers[tag](msg)
            return

        self._inbox.put(msg)

    # ------------------------------------------------------------------
    # I/O thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            active = False

            if self._pending:
                self._pending = [r for r in self._pending if not r.Test()]

            for _ in range(self._outbox_batch):
                try:
                    msg = self._outbox.get_nowait()
                except Empty:
                    break
                self._dispatch_send(msg)
                active = True

            status = MPI.Status()
            while self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
                src = status.Get_source()
                tag = status.Get_tag()

                if tag == TAG_BUFFER_HEADER:
                    # Receive the small envelope, then the raw buffer.
                    meta: _BufMeta = self.comm.recv(source=src, tag=TAG_BUFFER_HEADER)
                    if _NUMPY:
                        buf = np.empty(meta.shape, dtype=meta.dtype)
                    else:
                        buf = bytearray(meta.shape[0])
                    # Blocking Recv: buffer send is already in-flight (issued
                    # immediately after the envelope on the sender side).
                    self.comm.Recv(buf, source=src, tag=meta.tag)
                    msg = Message(
                        payload=buf,
                        dest=meta.dest,
                        src=meta.src,
                        tag=meta.tag,
                        msg_id=meta.msg_id,
                    )
                else:
                    msg = self.comm.recv(source=src, tag=tag)

                self._handle_incoming(msg)
                active = True
                status = MPI.Status()

            if not active:
                time.sleep(self._poll_interval)

        if self._pending:
            MPI.Request.Waitall(self._pending)
            self._pending.clear()
