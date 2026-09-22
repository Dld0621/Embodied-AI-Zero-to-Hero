# Interactive laboratory validation / 交互实验室验证

The [English laboratory](learning-lab.md) and [中文实验室](learning-lab-cn.md) teach five concepts through deterministic, local calculations. These are educational models, not robot control software, training results, or proof of learner expertise. Exported JSON is an unreviewed learning record; it does not update the evidence-gated curriculum progress file.

## Three complementary checks / 三层检查

| Layer | What it checks | What it does not establish |
| --- | --- | --- |
| Numerical model tests | Known transforms, forward/inverse kinematics, control integration, action timing and confidence-interval edge cases | Real-robot accuracy or policy performance |
| Static repository tests | Bilingual lesson assets, source references, local-only scripts and the proposed CI patch | Visual readability or complete accessibility |
| Chromium end-to-end tests | Parameter updates, reset, export, invalid inputs, deep links, keyboard tabs, instant navigation, light/dark layout and no-JavaScript fallback | Every browser, assistive technology, device or network configuration |

The browser runner rejects uncaught JavaScript errors and checks for horizontal overflow and clipped SVG labels at 320 px, 360 px and 1440 px. It blocks external HTTPS assets during interactive checks: the laboratory must calculate and draw without MathJax or a remote API. Static worked examples remain available without JavaScript, including when reading the Markdown on GitHub. Saved panel screenshots hide the unrelated sticky site header so it does not cover the top tabs after capture scrolling. Screenshots supplement these assertions; they do not replace a human review of explanations, contrast, labels and diagrams.

An additional 320 px coordinate-frame regression checks the rightmost attainable plotted point and its label: angle 45°, translation (1.5, 1.5) m, local point (1, −1) m. The displayed world point must round to (2.91, 1.50) m while its label remains inside the plot. Invalid-input checks also require the value readout to show the new input rather than a stale valid value.

For a local preview, the browser harness maps the production sitemap root to the supplied preview URL. Material's real instant-navigation router then recognizes the preview's port and path. The HTML, scripts, and production sitemap on disk are not rewritten. A same-document marker and the newly mounted language root verify that the language switch actually used instant navigation. Production routing still needs the separate post-deployment check below.

浏览器检查会覆盖中文与英文的五个模块，并检查参数变化、重置、不可达目标、空输入、导出文件、深链接、键盘切换以及即时导航。手机窄屏和桌面宽屏都检查浅色、深色模式。检查通过只表示所列场景通过，不等于获得无障碍认证，也不等于完成课程考核。

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

**Browser CI wiring is pending workflow-change permission.** The active GitHub workflow files are unchanged. The full proposed configuration is preserved in [learning-lab-ci.patch](patches/learning-lab-ci.patch), including path triggers, pinned Node/Playwright setup, numerical and browser checks, and screenshot artifacts. This patch is a reviewable proposal, not an active GitHub Actions job.

现有凭据不具备修改 GitHub workflow 的权限，因此本次不扩展账号权限。浏览器检查已在本地执行，自动浏览器 CI 配置暂存为补丁；不能把它描述为已在 GitHub 自动运行。

The existing pytest CI job discovers `tests/test_learning_lab.py`. Its numerical-model test uses `EMBODIED_LAB_NODE` when explicitly configured, or an existing `node` executable on `PATH`; if neither is available, pytest reports an explicit skip. It installs no runtime or package and does not change credentials. To choose an already-installed runtime locally:

```bash
EMBODIED_LAB_NODE=/absolute/path/to/node python -m pytest tests/test_learning_lab.py -q
```

Only an authorized maintainer with permission to change workflows should apply and publish the proposed CI configuration. From the repository root, first inspect the patch and verify it still applies:

```bash
git apply --check docs/patches/learning-lab-ci.patch
```

After explicit authorization to change the workflow:

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

Once an authorized maintainer publishes the proposed workflow patch, the documentation workflow will repeat model and browser checks for changes to laboratory JavaScript, CSS, lesson Markdown, test files and that workflow. Until then, use the local browser runner and inspect the existing pytest results, including any explicit Node-runtime skip. Actual run status must be verified in GitHub Actions; an unexecuted or future run is not a passing run.
