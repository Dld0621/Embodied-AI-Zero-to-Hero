# Embodied AI · Zero to Hero

An evidence-aware learning and engineering stack for embodied intelligence.

## Choose an entry point

| Goal | Start here | Completion signal |
|---|---|---|
| Learn the prerequisites | [English foundations overview](foundations/README_EN.md) or [中文路线图](foundations/00-roadmap.md) | Complete the lesson checks and run the linked example |
| Build one system | [Pipeline catalog](pipelines/README.md) or [中文总览](pipelines/README_CN.md) | Produce the declared artifacts and metrics |
| Compare methods | [Benchmark report](benchmark_report.md) | Reproduce a result with the recorded protocol and seeds |
| Prepare deployment | [Sim-to-Real guide](19-sim-to-real-guide.md) | Pass simulation, replay, HIL, shadow-mode, and safety gates |

## Evidence before claims

DoF uses five levels of evidence: import, smoke execution, deterministic test, benchmark, and hardware validation. A lower level never implies a higher one. Read the [validation policy](VALIDATION.md) before interpreting a result.

## Repository commands

```bash
python scripts/run_pipeline.py --validate
python scripts/run_pipeline.py --list
python scripts/audit_repository.py
python -m pytest tests/ -q
```

The root [English README](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/README.md) and [中文 README](https://github.com/Dld0621/Embodied-AI-Zero-to-Hero/blob/master/README_CN.md) remain the product landing pages. This site is the deeper documentation layer.
