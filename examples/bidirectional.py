"""
Bidirectional — every rank is both producer and consumer simultaneously.

Run with:
    mpiexec -n 4 python examples/bidirectional.py

Each rank pings its right neighbour (ring) and replies with a pong.
Send and receive run in parallel threads on the same rank.
"""
import threading

from mpi4py import rc

rc.thread_level = "funneled"

from mpiqueue import Transport  # noqa: E402

TAG_PING = 1
TAG_PONG = 2


def consumer_thread(t: Transport, received: list) -> None:
    waiting = {"ping": True, "pong": True}
    while any(waiting.values()):
        msg = t.get(timeout=10)
        if msg.tag == TAG_PING:
            print(f"[rank {t.rank}] PING from {msg.src}: {msg.payload}")
            t.put(dest=msg.src, payload=f"pong from {t.rank}", tag=TAG_PONG)
            waiting["ping"] = False
        elif msg.tag == TAG_PONG:
            print(f"[rank {t.rank}] PONG from {msg.src}: {msg.payload}")
            received.append(msg.payload)
            waiting["pong"] = False


def main() -> None:
    with Transport() as t:
        if t.size < 2:
            raise SystemExit("Need at least 2 ranks.")

        received: list = []
        thread = threading.Thread(target=consumer_thread, args=(t, received), daemon=True)
        thread.start()

        next_rank = (t.rank + 1) % t.size
        t.put(dest=next_rank, payload=f"ping from {t.rank}", tag=TAG_PING)

        thread.join(timeout=15)
        print(f"[rank {t.rank}] Done. received={received}")


if __name__ == "__main__":
    main()
