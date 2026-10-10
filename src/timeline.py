"""Timestamp conventions and shared-timeline <-> source mapping.

Convention (documented in README):

* The *shared timeline* is measured in seconds. Its zero is Player A's
  recording start.
* Each player has an ``offset``: ``source_time = shared_time + offset``.
  Player A's offset is 0 by default. If Player B started recording 5 s after
  Player A, B's file at t=0 corresponds to shared t=5, so ``offset_b = -5``.
* The usable shared range is the intersection of both recordings, so both video
  and the (single) audio source always have footage.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import EDLError

PLAYERS = ("player_a", "player_b")


@dataclass(frozen=True)
class SyncMapping:
    offset_a: float
    offset_b: float
    duration_a: float
    duration_b: float

    @classmethod
    def from_b_offset(cls, offset_b: float, duration_a: float, duration_b: float) -> "SyncMapping":
        return cls(0.0, offset_b, duration_a, duration_b)

    def offset(self, player: str) -> float:
        _check_player(player)
        return self.offset_a if player == "player_a" else self.offset_b

    def duration(self, player: str) -> float:
        _check_player(player)
        return self.duration_a if player == "player_a" else self.duration_b

    def shared_to_source(self, player: str, t: float) -> float:
        return t + self.offset(player)

    def source_to_shared(self, player: str, t: float) -> float:
        return t - self.offset(player)

    def player_range(self, player: str) -> tuple[float, float]:
        """Shared-timeline interval covered by ``player``'s recording."""
        off = self.offset(player)
        return -off, self.duration(player) - off

    def common_range(self) -> tuple[float, float]:
        """Shared-timeline interval covered by both recordings."""
        a, b = self.player_range("player_a"), self.player_range("player_b")
        start, end = max(a[0], b[0]), min(a[1], b[1])
        if end <= start:
            raise EDLError("The recordings do not overlap on the shared timeline (check the offset)")
        return start, end


def _check_player(player: str) -> None:
    if player not in PLAYERS:
        raise EDLError(f"Unknown player {player!r}; expected one of {PLAYERS}")
