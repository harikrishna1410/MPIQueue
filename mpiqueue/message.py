"""
Message dataclass — the unit of exchange between ranks.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A point-to-point message envelope."""

    payload: Any                # arbitrary application data
    dest: int                   # target MPI rank
    tag: int = 0                # user-defined routing tag (0 = default)
    src: int = -1               # filled in by transport on send
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.monotonic)

    def reply(self, payload: Any, tag: int | None = None) -> "Message":
        """Create a reply message addressed back to the sender."""
        return Message(
            payload=payload,
            dest=self.src,
            tag=tag if tag is not None else self.tag,
        )

    def __repr__(self) -> str:
        return (
            f"Message(src={self.src}, dest={self.dest}, "
            f"tag={self.tag}, id={self.msg_id[:8]}, payload={self.payload!r})"
        )
