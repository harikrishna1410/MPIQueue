"""
Basic producer → consumer example.

Run with:
    mpiexec -n 4 python examples/basic.py

Rank 0: consumer.
Ranks 1..N-1: producers — each sends 3 tasks then a done signal.
"""
from mpi4py import rc

rc.thread_level = "funneled"

from mpiqueue import Transport  # noqa: E402

TAG_TASK = 1
TAG_DONE = 2
CONSUMER_RANK = 0


def main() -> None:
    with Transport() as t:
        if t.size < 2:
            raise SystemExit("Need at least 2 ranks.")

        if t.rank == CONSUMER_RANK:
            done_count = 0
            num_producers = t.size - 1
            print(f"[rank {t.rank}] Consumer ready.")
            while done_count < num_producers:
                msg = t.get(timeout=30)
                if msg.tag == TAG_DONE:
                    done_count += 1
                elif msg.tag == TAG_TASK:
                    print(f"[rank {t.rank}] Task from {msg.src}: {msg.payload}")
        else:
            for i in range(3):
                t.put(dest=CONSUMER_RANK, payload={"producer": t.rank, "i": i}, tag=TAG_TASK)
            t.put(dest=CONSUMER_RANK, payload=None, tag=TAG_DONE)
            print(f"[rank {t.rank}] Producer done.")


if __name__ == "__main__":
    main()
