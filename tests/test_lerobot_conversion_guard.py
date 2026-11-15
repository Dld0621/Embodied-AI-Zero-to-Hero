"""Conversion control flow, not official LeRobot/Parquet compatibility tests."""

import builtins
import sys
from types import ModuleType

import numpy as np
import pytest

from examples.robot_foundation_models.common import to_lerobot
from examples.robot_foundation_models.common.canonical_dataset import EpisodeBuilder


@pytest.fixture
def episode():
    builder = EpisodeBuilder(
        task="push", robot_type="teaching", control_frequency=20,
        action_type="ee_delta",
    )
    builder.add_step(
        observation={"images": {"front": np.zeros((2, 2, 3), dtype=np.uint8)},
                     "state": np.zeros(2, dtype=np.float32)},
        action=np.zeros(2, dtype=np.float32), language="push", reward=0, success=False,
    )
    return builder.to_episode()


@pytest.mark.parametrize("mock", [False, True])
def test_existing_output_is_preserved_without_calling_either_writer(monkeypatch, tmp_path, episode, mock):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "previous-result.txt"
    sentinel.write_text("keep my prior run")

    def unexpected(*args):
        pytest.fail("No converter should touch an existing output.")

    monkeypatch.setattr(to_lerobot, "_convert_with_lerobot", unexpected)
    monkeypatch.setattr(to_lerobot, "_convert_with_pyarrow", unexpected)
    with pytest.raises(FileExistsError, match="does not delete or overwrite"):
        to_lerobot.convert_to_lerobot([episode], output, mock=mock)
    assert sentinel.read_text() == "keep my prior run"
    assert list(output.iterdir()) == [sentinel]


def test_missing_official_dependency_never_writes_a_mock_fallback(monkeypatch, tmp_path, episode):
    output = tmp_path / "new-official-run"

    def missing(*args):
        raise ImportError("mocked missing official dependency")

    def forbidden(*args):
        pytest.fail("Official conversion must never switch to mock serialization.")

    monkeypatch.setattr(to_lerobot, "_convert_with_lerobot", missing)
    monkeypatch.setattr(to_lerobot, "_convert_with_pyarrow", forbidden)
    with pytest.raises(ImportError, match="No mock fallback") as error:
        to_lerobot.convert_to_lerobot([episode], output)
    assert "mocked missing" in str(error.value.__cause__)
    assert not output.exists()


def test_explicit_mock_selects_only_the_teaching_writer(monkeypatch, tmp_path, episode):
    calls = []

    def teaching_writer(*args):
        calls.append(args)

    def official(*args):
        pytest.fail("Explicit mock mode must not import or invoke official LeRobot.")

    monkeypatch.setattr(to_lerobot, "_convert_with_lerobot", official)
    monkeypatch.setattr(to_lerobot, "_convert_with_pyarrow", teaching_writer)
    output = tmp_path / "new-teaching-run"
    to_lerobot.convert_to_lerobot([episode], output, dataset_name="lesson", mock=True)
    assert calls == [([episode], output, "lesson", 20)]


def test_empty_inputs_fail_before_writing_or_importing_optional_dependencies(monkeypatch, tmp_path, episode):
    def unexpected(*args):
        pytest.fail("Empty input must be rejected before choosing a writer.")

    monkeypatch.setattr(to_lerobot, "_convert_with_lerobot", unexpected)
    monkeypatch.setattr(to_lerobot, "_convert_with_pyarrow", unexpected)
    empty_episode = EpisodeBuilder(
        task="push", robot_type="teaching", control_frequency=20,
        action_type="ee_delta",
    ).to_episode()
    for episodes in ([], [empty_episode], [episode, empty_episode]):
        with pytest.raises(ValueError, match="nonempty episode"):
            to_lerobot.convert_to_lerobot(episodes, tmp_path / "never-written")
    assert not (tmp_path / "never-written").exists()


def test_missing_mock_dependencies_raise_instead_of_returning_success(monkeypatch, tmp_path, episode, capsys):
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("mocked missing pandas")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    output = tmp_path / "no-parquet"
    with pytest.raises(ImportError, match="requires pandas and pyarrow"):
        to_lerobot.convert_to_lerobot([episode], output, mock=True)
    assert not output.exists()
    assert "written" not in capsys.readouterr().out


@pytest.mark.parametrize("fail_frame", [False, True])
@pytest.mark.parametrize("fail_finalize", [False, True])
def test_official_writer_finalizes_before_success_and_on_frame_failure(monkeypatch, tmp_path, episode, capsys, fail_frame, fail_finalize):
    # A fake API verifies calls and cleanup, not real v3 schema compatibility.
    events = []

    class Dataset:
        @classmethod
        def create(cls, **kwargs):
            events.append("create")
            assert not kwargs["root"].exists()
            return cls()

        def add_frame(self, frame):
            events.append("frame")
            assert frame["task"] == "push"
            if fail_frame:
                raise ValueError("mocked frame rejection")

        def save_episode(self):
            events.append("save")

        def finalize(self):
            events.append("finalize")
            if fail_finalize:
                raise RuntimeError("mocked finalize rejection")

    for name in ("lerobot", "lerobot.datasets", "lerobot.datasets.lerobot_dataset"):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["lerobot.datasets.lerobot_dataset"].LeRobotDataset = Dataset
    output = tmp_path / "official-new"
    if fail_frame:
        with pytest.raises(ValueError, match="mocked frame rejection") as error:
            to_lerobot.convert_to_lerobot([episode], output)
        assert events == ["create", "frame", "finalize"]
        assert "Dataset written" not in capsys.readouterr().out
        if fail_finalize and hasattr(error.value, "add_note"):
            assert any("mocked finalize rejection" in note for note in error.value.__notes__)
    elif fail_finalize:
        with pytest.raises(RuntimeError, match="mocked finalize rejection"):
            to_lerobot.convert_to_lerobot([episode], output)
        assert events == ["create", "frame", "save", "finalize"]
        assert "Dataset written" not in capsys.readouterr().out
    else:
        to_lerobot.convert_to_lerobot([episode], output)
        assert events == ["create", "frame", "save", "finalize"]
        assert "Dataset written" in capsys.readouterr().out


def test_fractional_or_invalid_fps_is_not_silently_truncated(tmp_path, episode):
    for fps in (20.5, 0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive integer fps"):
            to_lerobot.convert_to_lerobot([episode], tmp_path / "new", fps=fps)
    assert not (tmp_path / "new").exists()
