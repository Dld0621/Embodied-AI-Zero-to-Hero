# Release Checklist

## Evidence

- [ ] `python scripts/audit_repository.py` passes.
- [ ] `python scripts/check_markdown_links.py` passes.
- [ ] `python scripts/run_pipeline.py --validate` passes.
- [ ] `python -m pytest tests/ -q` passes on supported Python versions.
- [ ] Every changed metric matches `results/benchmarks/benchmark_v2.json` or a newly committed raw artifact.
- [ ] Failed, missing, and hardware-dependent results remain explicitly labelled.

## Reproducibility

- [ ] Commands, seeds, dependency lock, hardware, dataset version, and evaluation episode count are recorded.
- [ ] Generated checkpoints are not accidentally committed.
- [ ] A clean environment can run the documented Quick Start.
- [ ] The Docker image builds, or the release notes explain why it is not applicable.

## Documentation and governance

- [ ] English and Chinese landing pages remain aligned.
- [ ] Local Markdown links and MkDocs strict build pass.
- [ ] `CITATION.cff`, `CHANGELOG.md`, `SECURITY.md`, and third-party notices reflect the release.
- [ ] GitHub description, Topics, release notes, and version tag match the repository state.

## Hardware, when claimed

- [ ] Robot model, serial/configuration, firmware, calibration, payload, site, operator, and safety devices are recorded.
- [ ] HIL, shadow-mode, guarded-rollout, intervention, and rollback evidence is attached.
- [ ] The claim is scoped to the tested hardware and task.
