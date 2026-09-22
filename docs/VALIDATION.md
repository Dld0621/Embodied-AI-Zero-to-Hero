# Validation and Claim Policy / 验证与声明规范

> **逐点图解 / Concept close-ups：**[硬件证据与运行安全](knowledge-atlas/deploy-hardware-gates/index.md)。每个小点配原理、算例、图、自测；这是中文细解，保留英文术语。

DoF does not equate code execution with research success. Every public claim must identify its evidence level, source artifact, and known boundary.

## Evidence ladder

| Level | Required evidence | Allowed claim | Disallowed inference |
|---|---|---|---|
| Import | Module imports and schemas load | Interface is syntactically available | Algorithm works |
| Smoke | Minimum command completes | Execution path is connected | Useful task performance |
| Deterministic test | Fixed assertion passes across supported environments | Contract or regression is verified | Generalization |
| Benchmark | Frozen protocol, artifact, seeds, and metrics are recorded | Result holds for that protocol | Transfer to another task or robot |
| Hardware | Robot, calibration, operator, safety gates, and incidents are recorded | Result holds for that physical setup | Certification or universal safety |

### Smoke qualifiers / Smoke 限定词

- **synthetic-smoke** means a fixed, generated fixture exercises the pipeline and its metrics. It does not add a new evidence level or imply performance on captured sensor data.
- **synthetic-contact-dynamics** is a synthetic smoke in which the simulator resolves hand-object contact. It can support claims about that fixture's contact, lift, and retention checks, but not policy quality, transfer to a production hand, or hardware performance.
- **interface-tested** means schemas, adapters, or planners connect without requiring production weights or hardware. It does not imply the underlying model is useful.
- Every smoke artifact must state both what it supports and what it cannot support; those boundaries travel with the metric JSON.

`synthetic-smoke` 表示固定的合成夹具能够执行管线与指标记录；它不是新的证据等级，也不能推出真实传感器数据上的性能。`synthetic-contact-dynamics` 表示仿真器确实求解了手—物体接触，但只能支持该夹具内的接触、抬升与保持检查，不能证明策略质量、生产级手型迁移或真机性能。所有 smoke 产物都必须同时记录“可支持的结论”和“不能支持的结论”。

## Sources of truth

| Claim family | Canonical source |
|---|---|
| Pipeline status, commands, artifacts, metrics | [`pipelines/manifest.json`](../pipelines/manifest.json) |
| Knowledge-node prerequisites, stages, documents, and learner evidence | [`knowledge/manifest.json`](../knowledge/manifest.json) |
| Research-route deliverables, metrics, and promotion gates | [`learning_paths/manifest.json`](../learning_paths/manifest.json) |
| PushCube numerical results | [`results/benchmarks/benchmark_v2.json`](../results/benchmarks/benchmark_v2.json) |
| Dependency compatibility | Lock files and successful CI jobs |
| Development-host teaching defaults | [`tools/robotdev/stack_matrix.json`](../tools/robotdev/stack_matrix.json) plus its dated upstream links |
| Foundation concepts | [Primary-source registry](SOURCES.md) |
| Scientific wording and evidence labels | [Claim review and accuracy gate](CLAIM_REVIEW.md) |
| Third-party terms | Bundled upstream license files and [third-party notices](../THIRD_PARTY_NOTICES.md) |
| Hardware readiness | A hardware-specific report; none is currently claimed locally |

Narrative documents summarize these sources and must not override them.

## Accuracy audit

Run before every release or major documentation change:

```bash
python scripts/check_markdown_links.py
python scripts/check_markdown_format.py
python scripts/check_claims.py
python scripts/run_knowledge_map.py --validate
python scripts/run_pipeline.py --validate
python scripts/run_learning_path.py --validate
python scripts/audit_repository.py
python -m pytest tests/ -q
```

The automated audit verifies repository-local links, GitHub-compatible math delimiters, required governance files, knowledge-graph structure, Pipeline and research-route contracts, benchmark consistency, bilingual entry points, source pointers, and declared artifacts. Graph validation proves that identifiers, prerequisites, stages, documents, and coverage are internally consistent; it does not prove learner mastery. The audit also cannot prove the truth of an unexecuted physical experiment or every semantic interpretation of an external source.

### Formula display / 公式显示

Encoding checks alone cannot establish readable mathematics. Run the build and inspect its actual formula output:

```bash
python -m mkdocs build --strict --clean
python scripts/check_site_math.py
```

The build uses committed, pinned-renderer SVG and semantic MathML, and fails on missing or invalid formulas. The independent HTML audit runs automatically at the end of every normal MkDocs build, including the existing Python-only Pages CI. It requires real formula output on both laboratory pages, rejects visible raw math fallback (including the table of contents) and error nodes, and checks that no runtime math CDN is reintroduced. These are structural checks, **not visual acceptance**.

Browser acceptance additionally runs `node scripts/test_learning_lab_browser.cjs` against a locally served build, with Playwright installed. It checks formulas with external resources blocked and JavaScript disabled, repeated bilingual navigation, and 320/360/1440 px layouts. Preserve screenshots and the actual result; do not claim an unexecuted browser check passed. GitHub's Markdown preview is a separate renderer and still needs source-preview verification after changes to mathematical notation.

乱码验收分为三层：文件编码与美元价格检查、生成网页中的静态公式检查、浏览器实际显示检查。前两层通过不代表第三层已经通过；交互按钮可用也不能证明公式已经渲染。

## Benchmark rules

- Never replace `null` or “not evaluated” with zero.
- Keep training data, model scale, compute, seeds, evaluation episodes, and horizon visible.
- Use closed-loop task performance as the primary policy metric; offline loss is supporting evidence.
- Report failed and negative results using the same protocol as successful results.
- Regenerate or manually update narrative tables only after changing the machine-readable source.

## Hardware rules

- Local simulation does not authorize robot motion.
- A real deployment requires calibration, limit checks, watchdog, emergency stop, low-risk commissioning, rollback, and an identified operator.
- “Available model” means an asset exists; “adapter” means interface code exists; neither implies real-robot validation.

## Correction process

When a conflict is found, preserve the raw artifact, identify the incorrect derived statement, fix the narrative or generator, add a regression test, and record the correction in `CHANGELOG.md`.
