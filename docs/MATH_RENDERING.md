# 公式显示与维护 / Equation rendering and maintenance

## 阅读者：不需要安装公式插件

文档站的公式在发布前已经转换为 SVG 图形。普通公式的字符是内嵌矢量路径，无需从外部服务器下载 MathJax、网页字体或额外 JavaScript；断开外部 CDN、关闭 JavaScript 或使用站内即时导航时，公式内容仍然存在。

每个 SVG 同时带有屏幕阅读器可使用的 MathML 版本。页面只在视觉上隐藏 MathML，不用 `display: none` 或 `aria-hidden` 隐藏其语义内容。公式中的中文说明保留为 SVG 文本，使用设备已有的中文字体；因此仍需操作系统具备中文字符支持，但无需网站字体服务。

GitHub 文件预览继续使用原始 Markdown 中的标准数学语法，由 GitHub 自己排版。本站的静态渲染不修改 GitHub 的渲染器，也不声称能控制 GitHub 服务故障。长公式保留正常字号，通过公式区域横向滚动阅读。

## Maintainers: ordinary builds remain Python-only

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict --clean
python scripts/generate_math_cache.py --check
python scripts/check_site_math.py
```

The committed `generated/math-cache.json` contains the rendered formulas. A Python MkDocs hook replaces each generated Arithmatex wrapper with self-contained SVG and assistive MathML. Node.js is **not required** for a normal build or cache-coverage check. The Pages workflow does not need a new dependency-installation stage.

A missing, invalid or uncached equation stops the build with a page-specific error. It never silently falls back to displaying TeX source. `--check` additionally rejects unused cache entries left after deleting or changing formulas.

## 修改数学内容后 / After changing equations

```bash
# Maintainer workstation only; use Node.js 18 or newer.
npm ci --ignore-scripts
python scripts/generate_math_cache.py
node --test tests/math/math-render.test.cjs
python -m mkdocs build --strict --clean
python scripts/check_site_math.py
```

If Node.js is not on `PATH`, pass `--node /absolute/path/to/node` or set `EMBODIED_MATH_NODE`. Commit the Markdown changes and regenerated cache together. `package-lock.json` fixes the complete maintenance dependency tree; do not edit it by hand.

The generator performs two isolated temporary builds: one collects the actual expressions emitted by the Markdown parser; the other verifies a normal fail-closed build against the new cache. Neither temporary build overwrites `site/`, so generation cannot replace a currently served preview with unrendered equations.

The renderer is pinned to MathJax **3.2.2**, using only the `base`, `ams` and `boldsymbol` TeX packages. Unknown commands, unclosed groups, HTML injection extensions and external resources are rejected. SVG uses `fontCache: "none"`; each formula contains its own glyph paths without shared page IDs or external font files. Both SVG and MathML come from the same parsed expression. Repeated generation is deterministic.

The transitive `@xmldom/xmldom` dependency is explicitly overridden to **0.9.12**, replacing the deprecated 0.9.10 dependency selected by the upstream speech engine. It is not used by the selected lightweight HTML adaptor or shipped to readers. Run `npm audit` when updating the maintenance lockfile; the absence of current advisories is not a permanent security guarantee.

## What to test before publication

- Full-site output: every formula has SVG plus MathML; there is no unrendered delimiter, error node, external glyph reference or formula CDN script.
- Changed formulas: verify the mathematics and not only whether it parses. Rendering does not certify a derivation, units, or assumptions.
- Actual reading: inspect both languages, long equations, matrices, superscripts, fractions and Chinese labels at narrow and wide widths.
- Degraded network: formulas remain visible with external requests blocked and with JavaScript disabled.
- Navigation: changing pages must preserve formula display without rerunning a typesetter.

The Python cache tests run when documentation dependencies are installed; maintainer renderer tests additionally require `npm ci` and Node.js. Skipped tests are not visual verification. The [validation policy](VALIDATION.md) distinguishes automated checks from browser observations.

## Upstream implementation and licensing

The implementation follows the installed 3.2.2 source for `AbstractMathDocument.convert` and `MathItem`, serializes that same item's MathML tree, and uses the documented [SVG output options](https://docs.mathjax.org/en/v3.2/options/output/svg.html). See also the upstream [server-side examples](https://github.com/mathjax/MathJax-demos-node).

The generated glyph paths derive from MathJax's TeX SVG font data, copyright 2017–2022 The MathJax Consortium, under the [Apache License 2.0](licenses/MathJax-3.2.2-LICENSE.txt). The original license is included in the published site. See [repository third-party notices](../THIRD_PARTY_NOTICES.md) for the separation between original teaching content and upstream components.
