# Interactive laboratory validation / 交互实验室验证

The [English laboratory](learning-lab.md) and [中文实验室](learning-lab-cn.md) teach seven concepts through deterministic, local calculations. The sixth module checks symmetric two-finger vertical force balance after adding and clipping a manual force correction. The seventh maps normalized VLA actions into physical units and distinguishes action spacing, average command rate, replanning frequency and observation age. Exported JSON records the model assumptions, parameters and learner explanations for subsequent review.

## Three complementary checks / 三层检查

| Layer | What it checks | What it does not establish |
| --- | --- | --- |
| Numerical model tests | Known transforms, forward/inverse kinematics, control integration, action timing, confidence intervals, clipped two-finger force balance and VLA action contracts | Real-robot accuracy or policy performance |
| Static repository tests | Bilingual lesson assets, source references, local-only scripts and the proposed CI patch | Visual readability or complete accessibility |
| Chromium end-to-end tests | Parameter updates, reset, export, invalid inputs, deep links, keyboard tabs, instant navigation, light/dark layout and no-JavaScript fallback | Every browser, assistive technology, device or network configuration |

The browser runner rejects uncaught JavaScript errors and checks for horizontal overflow and clipped SVG labels at 320 px, 360 px and 1440 px. It blocks external HTTPS assets during interactive checks: the laboratory must calculate and draw without MathJax or a remote API. Static worked examples remain available without JavaScript, including when reading the Markdown on GitHub. Saved panel screenshots hide the unrelated sticky site header so it does not cover the top tabs after capture scrolling. Screenshots supplement these assertions; they do not replace a human review of explanations, contrast, labels and diagrams.

An additional 320 px coordinate-frame regression checks the rightmost attainable plotted point and its label: angle 45°, translation (1.5, 1.5) m, local point (1, −1) m. The displayed world point must round to (2.91, 1.50) m while its label remains inside the plot. Invalid-input checks also require the value readout to show the new input rather than a stale valid value.

For a local preview, the browser harness maps the production sitemap root to the supplied preview URL. Material's real instant-navigation router then recognizes the preview's port and path. The HTML, scripts, and production sitemap on disk are not rewritten. A same-document marker and the newly mounted language root verify that the language switch actually used instant navigation. Production routing still needs the separate post-deployment check below.

浏览器检查覆盖中文与英文的七个模块，检查参数变化、重置、不可达目标、空输入、导出文件、深链接、键盘切换以及即时导航。接触模块额外检查执行上限、零摩擦和导出数值。VLA 模块检查训练范围改变物理动作、K/H 无效配置清图、零延迟边界与导出。手机窄屏和桌面宽屏都检查浅色、深色模式。验收范围以实际运行记录为准。

## 2026-11-15 local verification

The seven-module runner passed **640 browser assertions** with Playwright 1.62.1 and installed headless Chrome 156.0.8078.76. Coverage includes both languages, 320/360/1440 px, both themes, parameter/reset/export checks, keyboard/deep links, instant language navigation, blocked external HTTPS assets and no-JavaScript companions. The VLA module also checks invalid K/H, zero latency, training-bound changes and physical/time values in exported JSON.

A separate temporary browser check passed **129 assertions** across eight updated VLA pages at 320/1440 px in both themes. The pages had no whole-page horizontal overflow or replacement characters, and every displayed equation fit without horizontal scrolling. The new course SVG had no text outside its viewBox. Narrow-screen formula and diagram screenshots, the Chinese dark mobile module and the English desktop module received visual inspection. Temporary screenshots are local QA artifacts rather than committed course content.

The final Python suite passed **344 tests**, with **17 PyTorch-dependent tests skipped**. The interaction, reading-shell and math-renderer suites passed **43 Node tests**; cache-specific checks were rerun after formula line breaks changed. Strict MkDocs checked **639 formula occurrences across 32 pages**, backed by **432 unique cached expressions**. Each laboratory companion contains **65 rendered expressions**.

Independent review checked the new lesson, paper provenance, numerical models, source/target interfaces and the bilingual integration. New source-backed material, deterministic teaching calculations, fixed command construction and conversion guards do not establish GPU training, official dataset-loader compatibility or real-robot performance. Current scoped review and remaining executable findings are recorded in the [VLA update review](reviews/vla-update-2026-11-15.md).

## 2026-11-14 historical local verification

The previous six-module runner passed **554 browser assertions** using Playwright 1.62.1 with installed headless Chrome 156.0.8078.76. It exercised both languages, 320/360/1440 px viewports, light/dark themes, parameter changes, reset, JSON exports, keyboard navigation, instant language switches, blocked remote resources and JavaScript-disabled reading.

The full Python suite passed **315 tests**, with **17 PyTorch-dependent tests skipped** because PyTorch was unavailable in this environment. Numerical interaction, reading-shell and math-renderer tests passed **38 Node tests**. The strict site build checked **624 formula occurrences on 27 pages**, backed by **419 unique cached expressions**; each laboratory page contains 63 rendered formulas.

The new contact module and chapter diagram received narrow/wide screenshot inspection. The new chapter also passed 320/1440 px layout checks with all four equations statically rendered. Physics review checked the default +0.338 N margin, the −0.162 N actuator-limit case, zero friction, negative corrections and the distinction between friction capacity and equilibrium force. This record covers local browser and teaching-model verification.

## Reproduce locally / 本地复现

From the repository root, use Node.js 22 and the documented Python environment:

```bash
python -m mkdocs build --strict --clean
node --test tests/interactive/lab-models.test.cjs
python -m pytest tests/test_learning_lab.py -q
npm install --no-save --package-lock=false playwright@1.62.1
npx playwright install chromium
python -m http.server 8765 --bind 127.0.0.1 --directory site
```

In a second terminal:

```bash
node scripts/test_learning_lab_browser.cjs http://127.0.0.1:8765
```

Set `LAB_QA_ARTIFACTS` to a local output directory to retain screenshots. The runner accepts a different base URL as its first argument, including a project-path prefix. It creates no learner progress records and sends no laboratory parameters to a server. Stop the local preview server when finished.

## CI status / 持续集成状态

**Full browser CI is not enabled.** The proposed browser job is preserved in [learning-lab-ci.patch](patches/learning-lab-ci.patch); it has not been applied. The 2026-11-15 update adds pandas/PyArrow only to the existing RFM serialization smoke job so it can perform an actual Parquet write, while missing dependencies still raise errors. The full browser results above are from the local runner, not a GitHub browser job.

The existing pytest CI job discovers `tests/test_learning_lab.py`. Its numerical-model test uses `EMBODIED_LAB_NODE` when explicitly configured, or an existing `node` executable on `PATH`; if neither is available, pytest reports an explicit skip. It installs no runtime or package and does not change credentials. To choose an already-installed runtime locally:

```bash
EMBODIED_LAB_NODE=/absolute/path/to/node python -m pytest tests/test_learning_lab.py -q
```

For a future browser-CI change, first inspect the separate proposal and verify it still applies to the current workflow:

```bash
git apply --check docs/patches/learning-lab-ci.patch
```

When implementing that separate CI change:

```bash
git apply docs/patches/learning-lab-ci.patch
```

Review the resulting workflow diff and use the repository's normal review/publishing process. Do not enable broader account permissions solely to bypass a rejected push.

## Before publishing / 发布前人工复核

- Read the two language versions side by side; verify that units, assumptions, default parameters and worked examples agree.
- Inspect each diagram at narrow and wide widths in both color schemes; do not rely only on an overflow assertion.
- Change one parameter at a time and explain why the result changes. Check the displayed assumptions against the lesson and primary sources.
- Try keyboard-only interaction and confirm that focus remains visible. A keyboard smoke test is not a full screen-reader audit.
- Verify the deployed page and its commit independently of a successful local build or pushed branch. A pull request is not an updated default branch.

If the separate browser-workflow patch is applied and published, the documentation workflow will repeat model and browser checks for changes to laboratory JavaScript, CSS, lesson Markdown, test files and that workflow. Until then, use the local browser runner and inspect the existing pytest results, including any explicit Node-runtime skip. Actual run status must be verified in GitHub Actions; an unexecuted or future run is not a passing run.

收尾复核：写入异常与 writer 收尾异常同时出现时，保留主要写入错误；正常写入后的收尾错误仍传播。新增两个失败组合后，转换保护定向测试共 11 项通过。上文 344 项为此前实际执行的完整测试回执，本次补强后未重新运行整套。

CI 序列化补充：在隔离 Python 3.11.16 环境安装 NumPy 1.26.4、pandas 2.2.3 和 PyArrow 19.0.1，实际运行转换烟测；独立用 pandas 和 PyArrow 均读回 3 条 episode 共 30 帧。产物明确标记教学格式和 `official_lerobot_compatible=false`，没有把这次写读称作官方 v3 loader 验证。
