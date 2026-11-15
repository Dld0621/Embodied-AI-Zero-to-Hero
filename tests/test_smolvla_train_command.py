"""Versioned command and local-input contracts; no model/GPU/network required."""

import sys

import pytest

from examples.robot_foundation_models.smolvla import finetune


def _flags(command):
    return dict(argument.split("=", 1) for argument in command[1:])


def test_local_command_uses_dataset_root_and_policy_optimizer_preset(tmp_path):
    root = tmp_path / "dataset with spaces"
    command = finetune.build_train_command(
        "lerobot-train", str(root), False, str(tmp_path / "new-run"),
        {"learning_rate": 2e-5, "steps": 25, "batch_size": 2},
    )
    flags = _flags(command)
    assert flags["--dataset.root"] == str(root)
    assert flags["--dataset.root"] != str(root.parent)
    assert flags["--dataset.repo_id"] == "local/dataset-with-spaces"
    assert flags["--policy.optimizer_lr"] == "2e-05"
    assert "--lr" not in flags and "--optimizer.lr" not in flags
    assert flags["--policy.push_to_hub"] == "false"
    assert flags["--wandb.enable"] == "false"
    assert flags["--steps"] == "25" and flags["--batch_size"] == "2"


def test_hub_command_requires_explicit_selection_and_has_no_local_root(tmp_path):
    flags = _flags(finetune.build_train_command(
        "lerobot-train", "owner/dataset", True, str(tmp_path / "new-run"),
        {"use_wandb": True},
    ))
    assert flags["--dataset.repo_id"] == "owner/dataset"
    assert "--dataset.root" not in flags
    assert flags["--policy.push_to_hub"] == "false"
    assert flags["--wandb.enable"] == "true"
    for invalid in ("/abs/path", "./dataset", "../dataset", "a/b/c", "a/", "a/b c"):
        with pytest.raises(ValueError):
            finetune.validate_hub_dataset_id(invalid)


def test_missing_local_paths_fail_before_optional_imports_or_training(tmp_path):
    # Both strings contain '/', which formerly made nonexistent local paths
    # silently become Hub identifiers. Neither call should reach LeRobot.
    for local in (tmp_path / "missing", "datasets/not-present"):
        with pytest.raises(FileNotFoundError, match="Local dataset directory"):
            finetune.real_train(local, tmp_path / "new-run", {})
    assert not (tmp_path / "new-run").exists()


def test_local_metadata_preflight_rejects_legacy_placeholder_and_existing_output(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset_info.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="meta/info.json"):
        finetune.validate_real_training_inputs(dataset, tmp_path / "new-run")
    (dataset / "meta").mkdir()
    (dataset / "meta" / "info.json").write_text("{}")
    assert finetune.validate_real_training_inputs(dataset, tmp_path / "new-run") == str(dataset)
    # Presence is only a preflight condition; the trainer validates v3 schema.
    output = tmp_path / "existing-run"
    output.mkdir()
    with pytest.raises(FileExistsError, match="resume"):
        finetune.validate_real_training_inputs(dataset, output)


@pytest.mark.parametrize("hub", [False, True])
def test_main_routes_sources_without_precreating_real_output(monkeypatch, tmp_path, hub):
    output = tmp_path / "parent" / "new-run"
    reference = "owner/dataset" if hub else str(tmp_path / "local-dataset")
    flag = "--dataset_repo_id" if hub else "--dataset_dir"
    monkeypatch.setattr(sys, "argv", ["finetune.py", flag, reference, "--output_dir", str(output)])
    monkeypatch.setattr(finetune, "load_config", lambda _: {})
    calls = []

    def record(dataset_reference, output_dir, config, *, is_hf_repo=False):
        assert not output_dir.exists(), "The official trainer must create its own new output."
        calls.append((dataset_reference, output_dir, is_hf_repo))

    monkeypatch.setattr(finetune, "real_train", record)
    finetune.main()
    assert calls == [(reference, output, hub)]
    assert not output.exists()


def test_mock_rejects_hub_source_without_starting_training(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["finetune.py", "--mock", "--dataset_repo_id", "owner/dataset"])
    with pytest.raises(SystemExit) as error:
        finetune.main()
    assert error.value.code == 2
