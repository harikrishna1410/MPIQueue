"""
Tag-based dispatch using a simple handler dict — no Router class needed.

Run with:
    mpiexec -n 3 python examples/router_example.py

Rank 0: sends two types of tasks then shuts workers down.
Ranks 1, 2: dispatch by tag using a plain dict of handlers.
"""
import time

from mpi4py import rc

rc.thread_level = "funneled"

from mpiqueue import Transport, TAG_SHUTDOWN  # noqa: E402

TAG_COMPUTE = 1
TAG_REPORT = 2


def main() -> None:
    with Transport() as t:
        if t.size < 2:
            raise SystemExit("Need at least 2 ranks.")

        if t.rank == 0:
            workers = [r for r in range(t.size) if r != 0]
            for i, dest in enumerate(workers):
                t.put(dest=dest, payload={"value": (i + 1) * 10}, tag=TAG_COMPUTE)
            for dest in workers:
                t.put(dest=dest, payload={"report": "status"}, tag=TAG_REPORT)
            time.sleep(0.5)
            for dest in workers:
                t.put(dest=dest, payload=None, tag=TAG_SHUTDOWN)
            return

        # Worker: dispatch by tag.
        handlers = {
            TAG_COMPUTE: lambda msg: print(f"[rank {t.rank}] compute: {msg.payload['value'] ** 2}"),
            TAG_REPORT:  lambda msg: print(f"[rank {t.rank}] report from {msg.src}: {msg.payload}"),
        }

        while True:
            msg = t.get(timeout=10)
            if msg.tag == TAG_SHUTDOWN:
                break
            handler = handlers.get(msg.tag)
            if handler:
                handler(msg)


if __name__ == "__main__":
    main()
