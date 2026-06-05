"""
Dynamic server — every rank opens a Unix socket gateway for local clients.

Run with:
    mpiexec -n 3 python examples/dynamic_server.py

Each rank creates /tmp/mpiqueue_rank{local_rank}.sock.
Run dynamic_client.py in a separate terminal to join dynamically.
"""
from mpi4py import rc

rc.thread_level = "multiple"

import time  # noqa: E402

from mpiqueue import Transport, TAG_PEER_JOINED  # noqa: E402

TAG_TASK = 1
TAG_RESULT = 2


def main() -> None:
    with Transport() as t:
        t.start_gateway()
        print(f"[rank {t.rank}] Gateway ready at {t.socket_path}")

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                msg = t.get(timeout=1.0)
            except Exception:
                continue

            if msg.tag == TAG_PEER_JOINED:
                info = msg.payload
                print(f"[rank {t.rank}] *** PEER_JOINED node_id={info.node_id} at rank {info.gateway_rank} ***")
                # Tell the new client this rank's node_id so it can address us.
                if info.gateway_rank == t.rank:
                    t.put(dest=info.node_id, payload={"coordinator": t.rank}, tag=0)

            elif msg.tag == TAG_TASK:
                print(f"[rank {t.rank}] Task from node {msg.src}: {msg.payload}")
                t.put(dest=msg.src, payload={"by": t.rank, "echo": msg.payload}, tag=TAG_RESULT)


if __name__ == "__main__":
    main()
