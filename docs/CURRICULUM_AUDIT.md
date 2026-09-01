# Curriculum Audit and the 100-Point Quality Contract

**English · [简体中文](CURRICULUM_AUDIT_CN.md)**

This audit scores the repository's **curriculum design, learner workflow, and locally verifiable quality contracts** before and after this upgrade. The reviewed baseline is `master` commit `42a093a8dea450a80a4e01177bd0febfd68a5118` on 2026-09-02. It does not score state-of-the-art models, hardware performance, or an individual learner.

## Initial score: 85/100

The starting repository was strong: a 45-node knowledge graph, 14 foundation lessons, 11 Pipelines, seven research routes, explicit evidence levels, bilingual core navigation, and extensive tests. Its main gap was the learner loop, not content volume: there was no single first-hour entry, evidence diagnostic, formal progress record, common review standard, staged capstone, or expert graduation contract.

| Criterion | Before | After | Upgrade |
|---|---:|---:|---|
| Beginner onboarding and diagnosis | 7 | 10 | First-hour route, evidence diagnostic, background entries |
| Dependency-ordered curriculum | 10 | 10 | Preserve all 45 nodes and bind them to 12 modules |
| Embodied AI conceptual coverage | 10 | 10 | Preserve the mathematics-to-deployment chain |
| Runnable practice and engineering progression | 9 | 10 | Bind 11 Pipelines to module artifacts and gates |
| Mastery assessment and promotion | 6 | 10 | 100-point module rubric, critical failures, review workflow |
| Portfolio and capstones | 5 | 10 | C0/C1/C2, evidence templates, independent review form |
| Research literacy and experiment design | 10 | 10 | Preserve matched baselines, ablations, uncertainty, negative results |
| Systems, safety, and transfer boundaries | 10 | 10 | Preserve separation of simulation evidence and hardware authorization |
| Evidence integrity and reproducibility | 10 | 10 | Extend evidence contracts to learner progress and graduation |
| Navigation, role accessibility, bilingual core | 8 | 10 | Add bilingual learner operating manual and background-aware entry |
| **Total** | **85** | **100** | **All 10 contracts implemented and checked** |

The machine-readable source is [`curriculum/quality_rubric.json`](../curriculum/quality_rubric.json). Automated checks enforce ten criteria, totals, evidence files, and complete 45-node coverage so the score cannot silently drift from implementation.

## What “100” means precisely

A score of 100 means every criterion in this repository rubric has:

1. an explicit contract;
2. a locatable implementation;
3. a local structural check;
4. a stated evidence boundary.

It does **not** mean the curriculum never needs updating or that reading it guarantees expertise. Papers, software, and hardware change. Learners still submit raw artifacts and pass independent human review.

## Upgrade evidence

- [Start here](start-here.md): first hour, background entries, plan generation, and learner loop.
- [`curriculum/manifest.json`](../curriculum/manifest.json): L0–L5, M00–M11, goals, and three capstones.
- [`scripts/run_curriculum.py`](../scripts/run_curriculum.py): diagnosis, planning, progress initialization, evidence-record audit, and contract validation.
- [Assessment](assessment.md): 100-point module rubric, critical failures, and review workflow.
- [Capstones](capstone.md): engineering, learning-system, and independent-research graduation projects.
- [Learner templates](../learner/README.md): experiment card, failure report, and independent review form.
- [`tests/test_curriculum_journey.py`](../tests/test_curriculum_journey.py): contract, CLI, and progress-evidence regression checks.

## Honest remaining boundaries

- Repository smoke tests do not replace large-scale benchmarks or hardware results.
- C2 requires independent review, but software cannot prove reviewer independence or scientific correctness.
- Detailed foundation chapters are primarily Chinese; bilingual core coverage is not paragraph-for-paragraph translation parity.
- Mutable software versions and frontier papers must be rechecked against current primary sources.

The 100 is therefore a transparent, testable **repository-quality contract**, not a marketing claim of a perfect course.
