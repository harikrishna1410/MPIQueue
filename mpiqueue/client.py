"""
SocketClient — connects a local external process to the MPI queue via a
Unix domain socket (AF_UNIX).

The client has no knowledge of the cluster topology.  It receives only its
own ``node_id`` on connect.  All routing decisions live in the queue layer;
the client just names a destination node_id that it learned through the
application (e.g. a coordinator told it, or it replies to msg.src).

Usage::

    from mpiqueue.client import SocketClient

    with SocketClient("/tmp/mpiqueue_rank1.sock") as client:
        # node_id is the only identity the client knows about itself.
        print("my id:", client.node_id)

        # Destination node_ids come from the application, not from here.
        client.put(dest=coordinator_id, payload={"job": "hello"}, tag=1)

        msg = client.get(timeout=5.0)
        client.put(dest=msg.src, payload="ack", tag=2)   # reply by src

Wire protocol
-------------
Each frame:  [ 4-byte big-endian unsigned int length ][ N bytes pickle data ]
"""
from __future__ import annotations

import logging
import pickle
import queue
import socket
import struct
import threading
from typing import Any, Optional

from .message import Message

log = logging.getLogger(__name__)

_FRAME_HEADER = struct.Struct("!I")


def _send_frame(sock: socket.socket, obj: object) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_FRAME_HEADER.pack(len(data)) + data)


def _recv_frame(sock: socket.socket) -> object:
    header = _recv_exact(sock, _FRAME_HEADER.size)
    (length,) = _FRAME_HEADER.unpack(header)
    return pickle.loads(_recv_exact(sock, length))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionResetError("socket closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


class SocketClient:
    """
    Handle for an external process to participate in the MPI queue.

    The client is topology-blind: it only knows its own ``node_id``.
    Destination node_ids for ``put()`` come from the application
    (a coordinator message, a received ``msg.src``, etc.).

    Parameters
    ----------
    socket_path:
        Path to the Unix socket file of the local MPI rank's gateway.
        Default: ``/tmp/mpiqueue_rank0.sock``.
    connect_timeout:
        Seconds to wait while connecting (default 10 s).
    """

    def __init__(
        self,
        socket_path: str = "/tmp/mpiqueue_rank0.sock",
        connect_timeout: float = 10.0,
    ) -> None:
        self._socket_path = socket_path
        self._connect_timeout = connect_timeout

        self._sock: Optional[socket.socket] = None
        self._node_id: int = -1

        self._inbox: queue.Queue[Message] = queue.Queue()
        self._recv_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> "SocketClient":
        """Connect and receive node_id assignment from the gateway."""
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(self._connect_timeout)
        self._sock.connect(self._socket_path)
        self._sock.settimeout(None)

        welcome: Message = _recv_frame(self._sock)  # type: ignore[assignment]
        self._node_id = welcome.payload["node_id"]

        self._running = True
        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            daemon=True,
            name=f"uds-recv-{self._node_id}",
        )
        self._recv_thread.start()
        log.info("SocketClient connected: node_id=%d path=%s", self._node_id, self._socket_path)
        return self

    def disconnect(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._recv_thread:
            self._recv_thread.join(timeout=3.0)

    def __enter__(self) -> "SocketClient":
        return self.connect()

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @property
    def node_id(self) -> int:
        """This client's opaque identifier within the cluster."""
        return self._node_id

    def put(self, dest: int, payload: Any, tag: int = 0) -> Message:
        """
        Send *payload* to *dest* node_id.

        The queue layer handles all routing; this client has no knowledge of
        how *dest* is reached.
        """
        if self._sock is None:
            raise RuntimeError("Not connected — call connect() first")
        msg = Message(payload=payload, dest=dest, tag=tag, src=self._node_id)
        _send_frame(self._sock, msg)
        return msg

    def get(self, timeout: Optional[float] = None) -> Message:
        """Block until a message arrives (raises queue.Empty on timeout)."""
        return self._inbox.get(timeout=timeout)

    def get_nowait(self) -> Message:
        """Return a message immediately or raise queue.Empty."""
        return self._inbox.get_nowait()

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        try:
            while self._running:
                msg: Message = _recv_frame(self._sock)  # type: ignore[assignment]
                self._inbox.put(msg)
        except (ConnectionResetError, EOFError, OSError):
            if self._running:
                log.warning("SocketClient %d: connection lost", self._node_id)
        finally:
            self._running = False

    def __repr__(self) -> str:
        return f"SocketClient(node_id={self._node_id}, path={self._socket_path!r})"
