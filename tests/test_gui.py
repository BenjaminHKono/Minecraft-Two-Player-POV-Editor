import pytest

from src import pipeline
from src.gui import parse_offset, run_project


@pytest.mark.parametrize(("value", "expected"), [("", None), ("  ", None), ("-2.5", -2.5), ("0", 0.0)])
def test_parse_offset(value, expected):
    assert parse_offset(value) == expected


@pytest.mark.parametrize("value", ["abc", "nan", "inf", "-inf"])
def test_parse_offset_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_offset(value)


def test_run_project_uses_shared_pipeline(monkeypatch):
    project = object()
    expected = ("output/final.mp4", object(), object())
    callback = lambda _: None
    calls = {}

    def load_project(settings, overrides):
        calls["load"] = (settings, overrides)
        return project

    def run_all(actual_project, manual_offset, progress):
        calls["run"] = (actual_project, manual_offset, progress)
        return expected

    monkeypatch.setattr(pipeline, "load_project", load_project)
    monkeypatch.setattr(pipeline, "run_all", run_all)

    result = run_project("a.mp4", "b.mp4", "output/final.mp4", "settings.json", -1.25, callback)

    assert result == expected
    assert calls["load"] == (
        "settings.json",
        {"player_a": "a.mp4", "player_b": "b.mp4", "output": "output/final.mp4"},
    )
    assert calls["run"] == (project, -1.25, callback)
