"""
Dynamic client — connects to its local MPI rank via Unix domain socket.

Run AFTER starting dynamic_server.py:
    python examples/dynamic_client.py [rank]

The client is topology-blind:
  - It connects by socket path only.
  - It receives its own node_id and nothing else about the cluster.
  - The first message from the server tells it where to send work
    (a coordinator node_id).  All subsequent addressing is by node_id
    learned from the application — never from internal cluster topology.
"""
import sys
import time

from mpiqueue.client import SocketClient

TAG_ANNOUNCE = 0   # server sends coordinator node_id on client join
TAG_TASK = 1
TAG_RESULT = 2


def main() -> None:
    target_rank = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sock_path = f"/tmp/mpiqueue_rank{target_rank}.sock"

    print(f"Connecting via {sock_path} ...")

    with SocketClient(socket_path=sock_path) as client:
        print(f"Connected. My node_id={client.node_id}")

        # The client waits for the server to announce where to send work.
        # It does not enumerate the cluster — it just waits to be told.
        announce = client.get(timeout=10.0)
        coordinator_id = announce.payload["coordinator"]
        print(f"Coordinator is node_id={coordinator_id}\n")

        # Send tasks addressed by node_id from the application, not by rank.
        for i in range(3):
            client.put(
                dest=coordinator_id,
                payload={"task_id": i, "data": f"job-{i}"},
                tag=TAG_TASK,
            )
            print(f"Sent task {i} to node {coordinator_id}")

        # Collect replies — reply to whoever sent them (msg.src).
        print("\nWaiting for replies ...")
        for _ in range(3):
            try:
                msg = client.get(timeout=10.0)
                print(f"Reply from node {msg.src}: {msg.payload}")
            except Exception:
                print("Timed out.")
                break

        time.sleep(0.3)
        print("\nClient done.")


if __name__ == "__main__":
    main()
