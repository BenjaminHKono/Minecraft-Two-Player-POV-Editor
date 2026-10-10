import json

import pytest

from src.config import Config, config_from_dict, load_config
from src.errors import ConfigError


def test_defaults():
    c = load_config(None)
    assert c.output.resolution == "1920x1080" and c.output.fps == 60
    assert c.editing.minimum_shot_seconds == 2.0
    assert c.editing.speaker_switch_threshold_seconds == 1.0
    assert c.synchronization.manual_offset_seconds is None


def test_partial_override():
    c = config_from_dict({"editing": {"minimum_shot_seconds": 4}, "output": {"fps": 30}})
    assert c.editing.minimum_shot_seconds == 4 and c.output.fps == 30
    assert c.output.resolution == "1920x1080"


@pytest.mark.parametrize("data", [
    {"output": {"resolution": "big"}},
    {"output": {"resolution": "1921x1080"}},
    {"output": {"fps": 0}},
    {"editing": {"minimum_shot_seconds": -1}},
    {"editing": {"bogus": 1}},
    {"nonsense": {}},
    {"silence": {"short_pause_seconds": 5, "long_pause_seconds": 1}},
    {"speakers": {"mapping": {"SPEAKER_00": "player_c"}}},
    {"speakers": {"diarizer": "segments_file"}},
    {"synchronization": {"method": "magic"}},
])
def test_invalid(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


def test_load_file_errors(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(ConfigError):
        load_config(bad)
    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"output": {"fps": 30}}))
    assert load_config(good).output.fps == 30
    assert isinstance(Config().to_dict(), dict)
