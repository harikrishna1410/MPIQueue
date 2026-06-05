"""
MPIQueue — dynamic peer-to-peer messaging over MPI with Unix socket joining.

Core idea: MPI processes form a static cluster.  External processes join
dynamically by connecting to a local rank's Unix socket.  Once joined,
every node addresses every other by node_id — the underlying MPI topology
is invisible to the application.

Quickstart (MPI side)::

    from mpi4py import rc
    rc.thread_level = "multiple"
    from mpiqueue import Transport

    with Transport() as t:
        t.start_gateway()                       # /tmp/mpiqueue_rank{N}.sock
        t.put(dest=1, payload={"job": 42}, tag=1)
        msg = t.get(timeout=5.0)

Quickstart (external client — no mpi4py needed)::

    from mpiqueue.client import SocketClient

    with SocketClient("/tmp/mpiqueue_rank1.sock") as c:
        print("my node_id:", c.node_id)
        c.put(dest=coordinator_id, payload={"job": "hello"}, tag=1)
        msg = c.get(timeout=5.0)
"""

from .client import SocketClient
from .gateway import SocketGateway
from .membership import MembershipTable, NodeInfo
from .message import Message
from .transport import TAG_PEER_JOINED, TAG_PEER_LEFT, TAG_SHUTDOWN, Transport

__all__ = [
    "Transport",
    "SocketClient",
    "SocketGateway",
    "Message",
    "MembershipTable",
    "NodeInfo",
    "TAG_PEER_JOINED",
    "TAG_PEER_LEFT",
    "TAG_SHUTDOWN",
]
