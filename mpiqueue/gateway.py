"""
SocketGateway — bridges local external processes into the MPI message mesh
using Unix domain sockets (AF_UNIX).

Unix sockets are file-based and local-only, which matches the design:
a client always connects to the MPI rank that is local to it.  They are
faster than TCP because they bypass the network stack entirely.

Each rank creates its own socket file, e.g. ``/tmp/mpiqueue_rank0.sock``.
Clients connect by path, so no port allocation is needed.

How it works
------------
1. Each MPI rank that calls ``start_gateway()`` creates a Unix socket file
   and starts an accept loop thread.
2. A local external process connects to that file via ``SocketClient``.
3. On every new connection the gateway:
     a. Assigns a unique node_id (>= COMM_WORLD size).
     b. Sends the client a welcome frame with its node_id + gateway rank.
     c. Fan-outs a PEER_JOINED message to all COMM_WORLD ranks so every
        MPI process updates its membership table.
     d. Starts a per-client receive thread (socket → MPI outbox).
4. Inbound relay envelopes (TAG_RELAY) from other MPI ranks are forwarded
   to the correct local socket client.

Wire protocol
-------------
Each frame:  [ 4-byte big-endian unsigned int length ][ N bytes pickle data ]
"""

from __future__ import annotations

import logging
import os
import pickle
import socket
import struct
import threading
from typing import Optional

from .membership import NodeInfo
from .message import Message
from .transport import TAG_PEER_JOINED, TAG_PEER_LEFT, TAG_RELAY, Transport

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


class _ClientSession:
    """Manages one connected Unix socket client."""

    def __init__(
        self,
        node_id: int,
        sock: socket.socket,
        gateway: "SocketGateway",
    ) -> None:
        self.node_id = node_id
        self.sock = sock
        self._gateway = gateway
        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            daemon=True,
            name=f"uds-client-{node_id}",
        )

    def start(self) -> None:
        self._recv_thread.start()

    def send(self, msg: Message) -> None:
        try:
            _send_frame(self.sock, msg)
        except OSError as exc:
            log.warning("Client %d send error: %s", self.node_id, exc)
            self._gateway._on_client_disconnect(self.node_id)

    def _recv_loop(self) -> None:
        try:
            while True:
                msg: Message = _recv_frame(self.sock)  # type: ignore[assignment]
                msg.src = self.node_id
                log.debug(
                    "UDS client %d → rank %d [tag=%d]",
                    self.node_id,
                    msg.dest,
                    msg.tag,
                )
                self._gateway._transport.send(msg)
        except (ConnectionResetError, EOFError, OSError):
            pass
        finally:
            self._gateway._on_client_disconnect(self.node_id)


class SocketGateway:
    """
    Unix domain socket gateway on a single MPI rank.

    Each rank creates its own socket file.  External processes connect to
    the file that belongs to their local rank — there is no central gateway.

    Parameters
    ----------
    transport:
        The ``Transport`` instance on this rank.
    socket_path:
        Path for the Unix socket file.  Defaults to
        ``/tmp/mpiqueue_rank{N}.sock``.
    backlog:
        Accept backlog (max pending connections).
    """

    def __init__(
        self,
        transport: Transport,
        socket_path: Optional[str] = None,
        backlog: int = 10,
    ) -> None:
        self._transport = transport
        self._socket_path = (
            socket_path or f"/tmp/mpiqueue_rank{transport.local_rank}.sock"
        )
        self._backlog = backlog

        self._clients: dict[int, _ClientSession] = {}
        self._lock = threading.Lock()

        self._server_sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._running = False

        # Intercept TAG_RELAY so the transport thread can forward inbound
        # relay envelopes to the correct local socket client.
        self._transport.register_tag_handler(TAG_RELAY, self._on_relay)

    @property
    def socket_path(self) -> str:
        return self._socket_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        # Remove stale socket file from a previous run.
        try:
            os.unlink(self._socket_path)
        except FileNotFoundError:
            pass

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self._socket_path)
        self._server_sock.listen(self._backlog)
        self._running = True

        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            daemon=True,
            name=f"uds-accept-rank{self._transport.rank}",
        )
        self._accept_thread.start()
        log.info(
            "SocketGateway rank %d listening on %s",
            self._transport.rank,
            self._socket_path,
        )

    def stop(self) -> None:
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
        if self._accept_thread:
            self._accept_thread.join(timeout=3.0)
        try:
            os.unlink(self._socket_path)
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Accept loop
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        while self._running:
            try:
                self._server_sock.settimeout(1.0)
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            info = self._transport.membership.add_dynamic(
                gateway_rank=self._transport.rank
            )
            session = _ClientSession(node_id=info.node_id, sock=conn, gateway=self)
            with self._lock:
                self._clients[info.node_id] = session

            # Welcome frame: only tell the client its assigned node_id.
            # Peer discovery and routing are the queue layer's responsibility.
            welcome = Message(
                payload={"node_id": info.node_id},
                dest=info.node_id,
                src=self._transport.rank,
                tag=0,
            )
            session.send(welcome)

            # Fan-out PEER_JOINED to the entire MPI cluster.
            self._broadcast_membership_event(info, TAG_PEER_JOINED)

            session.start()
            log.info(
                "Rank %d: client node_id=%d connected via %s",
                self._transport.rank,
                info.node_id,
                self._socket_path,
            )

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------

    def _on_client_disconnect(self, node_id: int) -> None:
        with self._lock:
            session = self._clients.pop(node_id, None)
        if session is None:
            return
        info = self._transport.membership.remove(node_id)
        if info:
            self._broadcast_membership_event(info, TAG_PEER_LEFT)
        log.info(
            "Rank %d: client node_id=%d disconnected",
            self._transport.rank,
            node_id,
        )

    # ------------------------------------------------------------------
    # Relay: MPI rank → local socket client
    # ------------------------------------------------------------------

    def _on_relay(self, msg: Message) -> None:
        """Called by the transport thread for inbound TAG_RELAY envelopes."""
        with self._lock:
            session = self._clients.get(msg.dest)
        if session:
            session.send(msg)
        else:
            log.warning(
                "Rank %d: relay for unknown client node_id=%d — dropping",
                self._transport.rank,
                msg.dest,
            )

    # ------------------------------------------------------------------
    # Membership fan-out
    # ------------------------------------------------------------------

    def _broadcast_membership_event(self, info: NodeInfo, tag: int) -> None:
        """Send a membership event point-to-point to all COMM_WORLD ranks."""
        notification = Message(payload=info, dest=-1, src=self._transport.rank, tag=tag)
        for rank in range(self._transport.size):
            if rank == self._transport.rank:
                self._transport.membership.register(info)
                self._transport._inbox.put(notification)
            else:
                self._transport.send(
                    Message(payload=info, dest=rank, src=self._transport.rank, tag=tag)
                )
