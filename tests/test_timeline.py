import pytest

from src.errors import EDLError
from src.timeline import SyncMapping


def test_offset_convention():
    # B started 5 s after A -> B's t=0 is shared t=5 -> offset_b = -5
    m = SyncMapping.from_b_offset(-5.0, 100.0, 60.0)
    assert m.shared_to_source("player_a", 10) == 10
    assert m.shared_to_source("player_b", 10) == 5
    assert m.source_to_shared("player_b", 0) == 5
    assert m.player_range("player_b") == (5.0, 65.0)


def test_different_durations_common_range():
    m = SyncMapping.from_b_offset(-5.0, 100.0, 60.0)
    assert m.common_range() == (5.0, 65.0)
    m = SyncMapping.from_b_offset(3.0, 100.0, 200.0)  # B started before A
    assert m.common_range() == (0.0, 100.0)


def test_no_overlap_and_bad_player():
    with pytest.raises(EDLError):
        SyncMapping.from_b_offset(-500.0, 100.0, 60.0).common_range()
    with pytest.raises(EDLError):
        SyncMapping.from_b_offset(0, 1, 1).offset("player_c")
