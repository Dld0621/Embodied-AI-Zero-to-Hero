# 知识细解作者契约 / Atomic lesson authoring contract

This layer expands the existing 45 knowledge nodes; it does not replace their
prerequisites, evidence requirements, original lessons, or curriculum gates.
Coverage is measured against `knowledge/manifest.json`, not all possible topics
in embodied AI. Chinese explanations retain essential English terminology.

Author `*.json` files in this directory as objects with `schema_version: 1` and
an `atoms` list. Every atom contains these fields:

- `id`: globally unique lowercase ASCII hyphenated identifier.
- `node_id`: an existing knowledge-node ID.
- `title`, `english`: one teachable concept, not a chapter-sized topic.
- `prerequisites`: short names of the concepts needed before this atom.
- `intuition`: why this concept is needed, in concrete beginner-friendly terms.
- `mechanism`: at least three causal explanation steps. Define new symbols,
  quantities, units, and assumptions before using them.
- `worked_example`: at least three steps with specific inputs, calculation or
  trace, and an interpretable result. Toy data must be labelled as such.
- `misconception`: `wrong`, `why`, and `right` strings.
- `check`: one specific `question` and a reasoned `answer` (not merely yes/no).
- `visual`: one of the diagram specifications below.
- `reading`: at least two observations explaining what to read in the figure
  and what the figure cannot establish.
- `sources`: original/official source objects containing `title` and `url`.

Use 4–10 genuinely distinct atoms per existing node, chosen by concept density.
Do not pad coverage with renamed duplicates. All explanations, calculations,
misconceptions, diagrams and answers must agree. Keep dollar currency out of
math, never use TeX in headings, and do not invent benchmark or hardware results.

## Diagram specifications

Every diagram also has a concise `title` and a `caption` explaining its point.
Labels may contain plain Unicode symbols but not raw TeX. Diagrams are teaching
examples, not model-performance evidence. Prefer geometric or numeric diagrams
for geometric/numeric concepts; flowcharts are for actual sequences/dependencies.

- `flow`: `nodes` (3–5 objects with `label` and `detail`); optional `edges`
  (objects with integer `from`, `to`, and `label`). If edges are omitted, connect
  consecutive nodes. Use short labels and explain feedback in text as needed.
- `compare`: `panels` (2–3 objects with `title` and `lines`, each 2–4 strings).
- `plot`: `x_label`, `y_label` with units; `series` (1–3 objects with `label`
  and 2–12 `[x, y]` numeric `points`). State that connecting lines are schematic
  if sparse samples are not a continuous model.
- `matrix`: rectangular numeric `values` (at most 6 by 6), matching `row_labels`
  and `column_labels`, and a `note` stating what cells and colors mean.
- `timeline`: `unit` and `events` (2–6 objects with `label`, numeric `start` and
  `end`); `note` stating the clock origin and simplifying assumptions.
- `vectors`: 2–4 `vectors` (objects with `label`, numeric `x` and `y`), `unit`,
  and `note`. Every arrow starts at the common origin; translated coordinate
  frames must be explained using a separate sequence or worked example.
- `bars`: equal-length `labels` and numeric `values` (2–6), `axis_label` with
  units, and a `note` identifying the teaching example or actual source.

Source assets and lesson pages are generated deterministically by
`scripts/build_knowledge_atlas.py`. Commit source JSON and generated outputs
together. Browser layout inspection and content review are separate from
schema coverage; neither a visited page nor a correct toy calculation certifies
learner mastery or authorizes physical robot motion.
