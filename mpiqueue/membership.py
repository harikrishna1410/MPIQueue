"""
Membership — tracks all known nodes (static + dynamically joined).

Static nodes  : node_id == MPI rank, part of the original COMM_WORLD.
Dynamic nodes : node_id >= initial_size, connected later via MPI ports.
                They are reachable through a gateway rank in the static cluster.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Iterator, Optional


@dataclass
class NodeInfo:
    node_id: int
    is_dynamic: bool
    gateway_rank: Optional[int] = None   # None for static nodes
    meta: dict = field(default_factory=dict)  # application metadata

    @property
    def is_static(self) -> bool:
        return not self.is_dynamic

    def __repr__(self) -> str:
        kind = "dynamic" if self.is_dynamic else "static"
        gw = f", gateway={self.gateway_rank}" if self.is_dynamic else ""
        return f"NodeInfo(id={self.node_id}, {kind}{gw})"


class MembershipTable:
    """
    Thread-safe registry of all known nodes.

    Initialised with the static COMM_WORLD ranks; dynamic nodes are added
    as they join via :meth:`add`.
    """

    def __init__(self, comm_size: int) -> None:
        self._lock = RLock()
        self._nodes: dict[int, NodeInfo] = {
            r: NodeInfo(node_id=r, is_dynamic=False)
            for r in range(comm_size)
        }
        self._next_dynamic_id = comm_size

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_dynamic(self, gateway_rank: int, meta: dict | None = None) -> NodeInfo:
        """Assign a new node_id to a dynamically joined node."""
        with self._lock:
            node_id = self._next_dynamic_id
            self._next_dynamic_id += 1
            info = NodeInfo(
                node_id=node_id,
                is_dynamic=True,
                gateway_rank=gateway_rank,
                meta=meta or {},
            )
            self._nodes[node_id] = info
            return info

    def register(self, info: NodeInfo) -> None:
        """Register a node (used by non-gateway ranks receiving a JOIN event)."""
        with self._lock:
            self._nodes[info.node_id] = info
            if info.node_id >= self._next_dynamic_id:
                self._next_dynamic_id = info.node_id + 1

    def remove(self, node_id: int) -> Optional[NodeInfo]:
        with self._lock:
            return self._nodes.pop(node_id, None)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, node_id: int) -> Optional[NodeInfo]:
        with self._lock:
            return self._nodes.get(node_id)

    def all(self) -> list[NodeInfo]:
        with self._lock:
            return list(self._nodes.values())

    def static_ranks(self) -> list[int]:
        with self._lock:
            return [n.node_id for n in self._nodes.values() if n.is_static]

    def dynamic_nodes(self) -> list[NodeInfo]:
        with self._lock:
            return [n for n in self._nodes.values() if n.is_dynamic]

    def __len__(self) -> int:
        with self._lock:
            return len(self._nodes)

    def __iter__(self) -> Iterator[NodeInfo]:
        with self._lock:
            return iter(list(self._nodes.values()))

    def __repr__(self) -> str:
        with self._lock:
            return f"MembershipTable({list(self._nodes.values())})"
