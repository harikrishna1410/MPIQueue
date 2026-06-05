# MPIQueue

Dynamic peer-to-peer messaging over MPI with Unix socket joining.

MPI processes form a static cluster. External processes join dynamically by connecting to a local rank's Unix domain socket. Once joined, every node addresses every other by `node_id` -- the underlying MPI topology is invisible to the application.

## Features

- **Background I/O thread** -- non-blocking `put()`/`get()` with automatic MPI send/recv in a dedicated thread
- **Dynamic membership** -- external processes connect at runtime via Unix sockets; no MPI relaunch needed
- **Zero-copy transfers** -- numpy arrays and buffer-protocol objects use MPI `Isend`/`Recv` (RDMA-capable on InfiniBand)
- **Topology-blind clients** -- `SocketClient` knows only its `node_id`; all routing is handled by the transport layer
- **Tag-based dispatch** -- register per-tag handlers or read from the inbox queue directly
- **Membership events** -- `TAG_PEER_JOINED` / `TAG_PEER_LEFT` notify all ranks when clients connect or disconnect

## Installation

```bash
pip install mpi4py
pip install -e .
```

Requires an MPI implementation (e.g. OpenMPI, MPICH).

## Quick Start

### MPI ranks communicating directly

```python
from mpi4py import rc
rc.thread_level = "funneled"
from mpiqueue import Transport

with Transport() as t:
    if t.rank == 0:
        msg = t.get(timeout=5.0)
        print(f"Received: {msg.payload}")
    else:
        t.put(dest=0, payload={"job": 42}, tag=1)
```

```bash
mpiexec -n 2 python my_script.py
```

### External process joining dynamically

**Server** (MPI side):

```python
from mpi4py import rc
rc.thread_level = "multiple"   # required for gateway
from mpiqueue import Transport

with Transport() as t:
    t.start_gateway()           # opens /tmp/mpiqueue_rank{N}.sock
    msg = t.get(timeout=30.0)
    print(f"From client: {msg.payload}")
```

**Client** (no mpi4py needed):

```python
from mpiqueue.client import SocketClient

with SocketClient("/tmp/mpiqueue_rank0.sock") as c:
    print(f"My node_id: {c.node_id}")
    c.put(dest=coordinator_id, payload={"job": "hello"}, tag=1)
    reply = c.get(timeout=5.0)
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        MPI Cluster                               │
│                                                                  │
│  ┌──────────┐    MPI isend/recv    ┌──────────┐                  │
│  │  Rank 0   │◄──────────────────►│  Rank 1   │                  │
│  │ Transport │                     │ Transport │                  │
│  │          │                     │          │                  │
│  │ Gateway  │                     │ Gateway  │                  │
│  └────┬─────┘                     └────┬─────┘                  │
│       │ Unix socket                    │ Unix socket             │
│       │                                │                         │
│  ┌────┴─────┐                     ┌────┴─────┐                  │
│  │ Client A │                     │ Client B │                  │
│  │ node_id=2│                     │ node_id=3│                  │
│  └──────────┘                     └──────────┘                  │
└──────────────────────────────────────────────────────────────────┘
```

- **Transport** -- one per MPI rank. Runs a background thread for MPI I/O. Exposes `put()`/`get()` and an optional `SocketGateway`.
- **SocketGateway** -- accepts Unix socket connections on a rank. Assigns `node_id`s, relays messages between socket clients and MPI ranks, and broadcasts membership events.
- **SocketClient** -- connects to a gateway socket. Topology-blind: it only knows its own `node_id`. Destinations come from the application (e.g. a coordinator message or `msg.src`).
- **MembershipTable** -- thread-safe registry of all nodes (static MPI ranks + dynamic socket clients). Updated automatically via `PEER_JOINED`/`PEER_LEFT` events.
- **Message** -- dataclass envelope with `payload`, `dest`, `src`, `tag`, `msg_id`, and `timestamp`.

## Examples

All examples are in the `examples/` directory.

| Example | Description | Command |
|---------|-------------|---------|
| `basic.py` | Producer-consumer: ranks 1..N send tasks to rank 0 | `mpiexec -n 4 python examples/basic.py` |
| `bidirectional.py` | Ring ping-pong: every rank sends and receives | `mpiexec -n 4 python examples/bidirectional.py` |
| `router_example.py` | Tag-based dispatch with a handler dict | `mpiexec -n 3 python examples/router_example.py` |
| `dynamic_server.py` | Gateway server waiting for socket clients | `mpiexec -n 3 python examples/dynamic_server.py` |
| `dynamic_client.py` | Socket client that joins and sends tasks | `python examples/dynamic_client.py [rank]` |

For the dynamic example, start `dynamic_server.py` first, then run `dynamic_client.py` in a separate terminal.

## API Reference

### Transport

```python
Transport(comm=None, poll_interval=0.0001, outbox_batch=64)
```

| Method | Description |
|--------|-------------|
| `start()` | Start the background I/O thread |
| `stop()` | Stop the I/O thread and gateway |
| `start_gateway(socket_path=None)` | Open a Unix socket for external clients |
| `put(dest, payload, tag=0)` | Send a message (non-blocking) |
| `get(timeout=None)` | Receive next message (blocks, raises `queue.Empty` on timeout) |
| `get_nowait()` | Receive immediately or raise `queue.Empty` |
| `broadcast(payload, tag=0)` | Send to all known nodes |
| `register_tag_handler(tag, handler)` | Register a callback for a specific tag |

### SocketClient

```python
SocketClient(socket_path="/tmp/mpiqueue_rank0.sock", connect_timeout=10.0)
```

| Method | Description |
|--------|-------------|
| `connect()` | Connect and receive `node_id` assignment |
| `disconnect()` | Close the connection |
| `put(dest, payload, tag=0)` | Send a message to a node |
| `get(timeout=None)` | Receive next message |
| `node_id` | This client's assigned identifier |

Both `Transport` and `SocketClient` support context managers (`with` statements).

### Reserved Tags

| Constant | Value | Purpose |
|----------|-------|---------|
| `TAG_PEER_JOINED` | `0x7FFB` | A new client connected |
| `TAG_PEER_LEFT` | `0x7FFC` | A client disconnected |
| `TAG_SHUTDOWN` | `0x7FFF` | Shutdown signal |

User-defined tags should stay below `0x7FFB` to avoid collisions.

## Thread Level Requirements

- **MPI ranks only** (no gateway): `rc.thread_level = "funneled"` is sufficient
- **With gateway**: `rc.thread_level = "multiple"` is required (set before importing `mpi4py`)

## License

MIT
