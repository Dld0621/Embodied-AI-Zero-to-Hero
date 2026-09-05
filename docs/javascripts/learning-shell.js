/* Device-local reading preferences only. No analytics or assessed progress. */
(function (scope) {
  "use strict";
  const preferenceKey = "embodied-reading-preferences-v1";
  const bookmarkKey = "embodied-reading-bookmark-v1";
  const fonts = ["standard", "large", "larger"];
  const idPattern = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/;
  const normalize = value => String(value || "").normalize("NFKC").toLocaleLowerCase().trim();
  function matches(query, content) {
    return normalize(query).split(/\s+/).filter(Boolean).every(word => normalize(content).includes(word));
  }
  function preferences(value) {
    return {font: fonts.includes(value?.font) ? value.font : "standard", focus: value?.focus === true};
  }
  function bookmark(value) {
    if (!value || typeof value.id !== "string" || typeof value.chapter !== "string" ||
        value.id.length > 100 || value.chapter.length > 100 ||
        !idPattern.test(value.id) || !idPattern.test(value.chapter) ||
        typeof value.title !== "string" || !value.title.trim() || value.title.length > 200) return null;
    return {id: value.id, chapter: value.chapter, title: value.title};
  }
  const api = {matches, preferences, bookmark};
  if (typeof module === "object" && module.exports) module.exports = api;
  if (!scope.document) return;
  const document = scope.document;
  function read(key) { try { return JSON.parse(scope.localStorage.getItem(key)); } catch { return null; } }
  function write(key, value) { try { scope.localStorage.setItem(key, JSON.stringify(value)); return true; } catch { return false; } }
  let state = preferences(read(preferenceKey));
  function apply() {
    document.documentElement.dataset.studyFont = state.font;
    document.documentElement.dataset.studyFocus = String(state.focus);
  }
  apply();
  function init() {
    const toolbar = document.querySelector(".study-toolbar");
    if (toolbar && !toolbar.dataset.ready) {
      toolbar.dataset.ready = "true"; toolbar.hidden = false;
      const select = toolbar.querySelector("[data-study-font]");
      const focus = toolbar.querySelector("[data-study-focus]");
      const mark = toolbar.querySelector("[data-study-bookmark]");
      const status = toolbar.querySelector(".study-toolbar-status");
      select.value = state.font;
      focus.setAttribute("aria-pressed", String(state.focus));
      const savePreferences = () => {
        apply();
        if (!write(preferenceKey, state)) status.textContent = "设置仅在当前会话生效；浏览器未允许本地存储。";
      };
      select.addEventListener("change", () => { state.font = preferences({font: select.value}).font; savePreferences(); });
      focus.addEventListener("click", () => {
        state.focus = !state.focus; focus.setAttribute("aria-pressed", String(state.focus));
        savePreferences();
      });
      const lesson = document.querySelector(".study-lesson");
      if (lesson) {
        mark.hidden = false;
        mark.addEventListener("click", () => {
          const entry = bookmark({id: lesson.dataset.studyId, chapter: lesson.dataset.studyChapter,
            title: lesson.dataset.studyTitle});
          status.textContent = entry && write(bookmarkKey, entry)
            ? "已在本浏览器记住本节；回学习首页可继续，不计入课程通过。"
            : "未能保存阅读位置；请使用浏览器书签。";
        });
      }
    }
    const resume = document.querySelector(".study-resume");
    if (resume && !resume.dataset.ready) {
      resume.dataset.ready = "true";
      const saved = bookmark(read(bookmarkKey));
      if (saved) {
        const link = resume.querySelector("[data-study-resume]");
        const base = new URL(resume.dataset.atlasBase, scope.location.href);
        link.href = new URL(`${saved.chapter}/${saved.id}/`, base).href;
        link.textContent = saved.title;
        resume.hidden = false;
        resume.querySelector("button").addEventListener("click", () => {
          try { scope.localStorage.removeItem(bookmarkKey); resume.hidden = true; }
          catch { resume.querySelector("[role=status]").textContent = "无法清除本地记录，请在浏览器设置中操作。"; }
        });
      }
    }
    const filter = document.querySelector(".study-filter");
    if (filter && !filter.dataset.ready) {
      filter.dataset.ready = "true"; filter.hidden = false;
      const input = filter.querySelector("input"), status = filter.querySelector("[role=status]");
      const cards = Array.from(document.querySelectorAll(".study-catalog-item"));
      function update() {
        let count = 0;
        cards.forEach(card => { card.hidden = !matches(input.value, card.dataset.studyKeywords); if (!card.hidden) count++; });
        document.querySelectorAll(".study-domain").forEach(domain => {
          domain.hidden = !Array.from(domain.querySelectorAll(".study-catalog-item")).some(card => !card.hidden);
        });
        status.textContent = count ? `显示 ${count} / ${cards.length} 章；匹配章节标题、英文术语和小节名。` : "没有匹配章节，试试更短的关键词，或使用顶部全文搜索。";
      }
      input.addEventListener("input", update); update();
      document.querySelector(".study-domain-nav")?.addEventListener("click", event => {
        const link = event.target.closest?.("a");
        if (link && link.getAttribute("href")?.startsWith("#") && input.value) {
          // Restore hidden destinations before the browser follows the domain anchor.
          input.value = ""; update();
        }
      });
    }
  }
  if (typeof document$ !== "undefined") document$.subscribe(init);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, {once: true});
  else init();
})(typeof window !== "undefined" ? window : globalThis);
