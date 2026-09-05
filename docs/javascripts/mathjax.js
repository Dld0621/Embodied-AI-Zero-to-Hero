window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
  },
  options: {
    ignoreHtmlClass: "[a-zA-Z]+_html",
    processHtmlClass: "arithmatex",
  },
};

// Initial typesetting belongs to MathJax startup. The local lab must also work
// when that optional CDN cannot load.
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    if (typeof window.MathJax.typesetPromise === "function") {
      window.MathJax.typesetClear();
      window.MathJax.typesetPromise().catch(() => {
        // Keep source formulas available if optional rendering fails.
      });
    }
  });
}
