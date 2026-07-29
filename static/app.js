(() => {
  const editorWrap = document.querySelector(".editor-wrap");
  const editor = document.getElementById("editor");
  const editorHighlight = document.getElementById("editorHighlight");
  const checkBtn = document.getElementById("checkBtn");
  const acceptAllBtn = document.getElementById("acceptAllBtn");
  const downloadBtn = document.getElementById("downloadBtn");
  const statusMsg = document.getElementById("statusMsg");
  const suggestionsList = document.getElementById("suggestionsList");
  const countPill = document.getElementById("countPill");
  const stripeBar = document.getElementById("stripeBar");
  const langTabs = document.querySelectorAll(".lang-tab");

  const PLACEHOLDERS = {
    yo: "Kọ ọ̀rọ̀ rẹ sí ibí…",
    ig: "Dee ihe ị chọrọ ide ebe a…",
    ha: "Rubuta abin da kake so a nan…",
  };

  let currentLang = "yo";
  let lastResults = [];
  let lastCheckedLength = 0;
  let scanDebounceTimer = null;

  langTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      langTabs.forEach((t) => {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
      });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
      currentLang = tab.dataset.lang;
      editor.placeholder = PLACEHOLDERS[currentLang] || "";
      resetSuggestions();
      scanForTypos();
    });
  });

  function resetSuggestions() {
    lastResults = [];
    lastCheckedLength = 0;
    suggestionsList.innerHTML =
      '<p class="empty-state">Write something, then press <strong>Check text</strong>. ' +
      "Flagged sentences will show up here with a suggested fix and why. " +
      "Finished sentences are also checked live in the background as you type.</p>";
    setCount(0);
    acceptAllBtn.disabled = true;
    downloadBtn.disabled = true;
  }

  function setCount(n) {
    countPill.textContent = String(n);
    countPill.classList.toggle("zero", n === 0);
  }

  function markSnippets(text, snippets, tag) {
    let out = text;
    for (const snip of snippets) {
      if (!snip) continue;
      const idx = out.indexOf(snip);
      if (idx === -1) continue;
      out =
        out.slice(0, idx) +
        `<${tag}>${escapeHtml(snip)}</${tag}>` +
        out.slice(idx + snip.length);
    }
    return out;
  }

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // ---- Live typo underlining (Tier 1 only, fast, debounced) ----

  function renderHighlight(text, flags) {
    // Build HTML mirroring the plain text exactly, wrapping flagged
    // character ranges in <mark>. Offsets must be applied back-to-front
    // so earlier insertions don't shift later offsets.
    const sorted = [...flags].sort((a, b) => b.start - a.start);
    let html = escapeHtml(text);

    // Since we escaped first, we need offsets computed against the raw
    // text, then re-inserted into the escaped string at equivalent
    // positions. Simplest reliable approach: build from the raw string
    // directly, escaping each segment as we go.
    let result = "";
    let cursor = text.length;
    const ascending = [...flags].sort((a, b) => a.start - b.start);
    let pieces = [];
    let pos = 0;
    for (const f of ascending) {
      if (f.start < pos) continue; // skip overlapping/out-of-order flags
      pieces.push(escapeHtml(text.slice(pos, f.start)));
      pieces.push(`<mark>${escapeHtml(text.slice(f.start, f.end))}</mark>`);
      pos = f.end;
    }
    pieces.push(escapeHtml(text.slice(pos)));
    editorHighlight.innerHTML = pieces.join("");
  }

  async function scanForTypos() {
    const text = editor.value;
    if (!text.trim()) {
      editorHighlight.innerHTML = "";
      return;
    }
    try {
      const res = await fetch("/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lang: currentLang, text }),
      });
      if (!res.ok) return;
      const data = await res.json();
      if (data.error) return;
      renderHighlight(text, data.flags || []);
    } catch {
      // fail quietly — underlining is a nice-to-have, not critical path
    }
  }

  function scheduleTypoScan() {
    clearTimeout(scanDebounceTimer);
    scanDebounceTimer = setTimeout(scanForTypos, 400);
  }

  editor.addEventListener("scroll", () => {
    editorHighlight.scrollTop = editor.scrollTop;
    editorHighlight.scrollLeft = editor.scrollLeft;
  });

  // -----------------------------------------------------------------

  function renderSuggestions() {
    const flagged = lastResults.filter((r) => r.escalated);
    setCount(flagged.filter((r) => r.has_error).length);

    if (flagged.length === 0) {
      suggestionsList.innerHTML =
        '<p class="empty-state">No issues flagged. Looking good.</p>';
      acceptAllBtn.disabled = true;
      downloadBtn.disabled = false;
      return;
    }

    suggestionsList.innerHTML = "";
    let anyFixable = false;

    flagged.forEach((r, i) => {
      const card = document.createElement("div");
      card.className = "suggestion-card" + (r.has_error ? "" : " clean");

      if (!r.has_error) {
        card.innerHTML = `
          <span class="tag">Reviewed</span>
          <p class="orig-sentence">${escapeHtml(r.sentence)}</p>
          <p class="explanation">Flagged for review, but no correction needed.</p>
        `;
        suggestionsList.appendChild(card);
        return;
      }

      anyFixable = true;
      const origSnippets = (r.edits || [])
        .filter((e) => e.op !== "insert")
        .map((e) => e.original);
      const suggSnippets = (r.edits || [])
        .filter((e) => e.op !== "delete")
        .map((e) => e.suggestion);

      const origHtml = markSnippets(escapeHtml(r.sentence), origSnippets.map(escapeHtml), "del");
      const corrHtml = markSnippets(
        escapeHtml(r.corrected_sentence || ""),
        suggSnippets.map(escapeHtml),
        "ins"
      );

      const driftWarning =
        typeof r.semantic_similarity === "number" && r.semantic_similarity < 0.75
          ? `<p class="explanation" style="color:var(--osun);font-weight:600;">
               ⚠ This suggestion may change the sentence's meaning, not just its wording — review closely.
             </p>`
          : "";

      card.innerHTML = `
        <span class="tag">${r.auto_applied ? "Auto-suggested" : "Needs review"}</span>
        <p class="orig-sentence">${origHtml}</p>
        <p class="corrected-sentence">${corrHtml}</p>
        <p class="explanation" data-explanation-idx="${i}">${escapeHtml(r.explanation || "")}</p>
        <div style="display:flex; gap:8px; margin-bottom:10px;">
          <button class="chip-btn translate-btn" data-idx="${i}">Translate to English</button>
          <button class="chip-btn revert-btn" data-idx="${i}" style="display:none;">Revert to original</button>
        </div>
        ${driftWarning}
        <div class="card-actions">
          <button class="chip-btn accept" data-idx="${i}">Accept</button>
          <button class="chip-btn dismiss" data-idx="${i}">Dismiss</button>
        </div>
      `;
      suggestionsList.appendChild(card);
    });

    acceptAllBtn.disabled = !anyFixable;
    downloadBtn.disabled = false;

    suggestionsList.querySelectorAll(".chip-btn.accept").forEach((btn) => {
      btn.addEventListener("click", () => acceptOne(flagged[Number(btn.dataset.idx)]));
    });
    suggestionsList.querySelectorAll(".chip-btn.dismiss").forEach((btn) => {
      btn.addEventListener("click", () => {
        btn.closest(".suggestion-card").style.opacity = "0.45";
        btn.closest(".suggestion-card").querySelectorAll("button").forEach((b) => (b.disabled = true));
      });
    });
    suggestionsList.querySelectorAll(".translate-btn").forEach((btn) => {
      btn.addEventListener("click", () => translateExplanation(btn, flagged[Number(btn.dataset.idx)]));
    });
    suggestionsList.querySelectorAll(".revert-btn").forEach((btn) => {
      btn.addEventListener("click", () => revertExplanation(btn, flagged[Number(btn.dataset.idx)]));
    });
  }

  async function translateExplanation(btn, result) {
    if (!result || !result.explanation) return;
    const card = btn.closest(".suggestion-card");
    const p = card.querySelector(`[data-explanation-idx="${btn.dataset.idx}"]`);
    const revertBtn = card.querySelector(".revert-btn");
    btn.disabled = true;
    btn.textContent = "Translating…";
    try {
      const res = await fetch("/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lang: currentLang, text: result.explanation }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      p.textContent = data.translated;
      btn.textContent = "Translated ✓";
      if (revertBtn) revertBtn.style.display = "inline-block";
    } catch (err) {
      btn.textContent = "Translate to English";
      btn.disabled = false;
      statusMsg.textContent = `Couldn't translate: ${err.message}`;
    }
  }

  function revertExplanation(revertBtn, result) {
    if (!result) return;
    const card = revertBtn.closest(".suggestion-card");
    const p = card.querySelector(`[data-explanation-idx="${revertBtn.dataset.idx}"]`);
    const translateBtn = card.querySelector(".translate-btn");
    p.textContent = result.explanation;
    revertBtn.style.display = "none";
    if (translateBtn) {
      translateBtn.textContent = "Translate to English";
      translateBtn.disabled = false;
    }
  }

  function acceptOne(result) {
    if (!result || !result.corrected_sentence) return;
    const idx = editor.value.indexOf(result.sentence);
    if (idx === -1) {
      statusMsg.textContent = "Couldn't locate that sentence in the editor (did you edit it manually?)";
      return;
    }
    editor.value =
      editor.value.slice(0, idx) +
      result.corrected_sentence +
      editor.value.slice(idx + result.sentence.length);
    statusMsg.textContent = "Applied. Re-check to refresh suggestions.";
    scanForTypos();
  }

  acceptAllBtn.addEventListener("click", () => {
    const rebuilt = lastResults
      .map((r) => (r.has_error && r.corrected_sentence ? r.corrected_sentence : r.sentence))
      .join(" ");
    editor.value = rebuilt;
    statusMsg.textContent = "All suggestions applied. Re-check to confirm everything's clean.";
    acceptAllBtn.disabled = true;
    scanForTypos();
  });

  downloadBtn.addEventListener("click", () => {
    const blob = new Blob([editor.value], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${currentLang}-corrected.txt`;
    a.click();
    URL.revokeObjectURL(url);
  });

  async function fetchReadability(text) {
    if (!text || !text.trim()) return null;
    try {
      const res = await fetch("/readability", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) return null;
      const data = await res.json();
      if (data.error) return null;
      return data;
    } catch {
      return null;
    }
  }

  function renderReadability(before, after) {
    const panel = document.getElementById("readabilityPanel");
    const beforeEl = document.getElementById("readabilityBefore");
    const afterEl = document.getElementById("readabilityAfter");

    if (!before) {
      panel.style.display = "none";
      return;
    }
    panel.style.display = "flex";
    beforeEl.innerHTML = `<span class="tag">Before</span>LIX ${before.lix_score} — ${before.label}`;

    if (after && after.lix_score !== before.lix_score) {
      afterEl.innerHTML = `<span class="tag">After corrections</span>LIX ${after.lix_score} — ${after.label}`;
      afterEl.style.display = "inline-block";
    } else {
      afterEl.style.display = "none";
    }
  }

  async function runCheck() {
    const text = editor.value.trim();
    if (!text) {
      statusMsg.textContent = "Nothing to check yet.";
      return;
    }
    checkBtn.disabled = true;
    stripeBar.classList.add("checking");
    statusMsg.textContent = "Checking… this can take a moment on CPU.";

    try {
      const res = await fetch("/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lang: currentLang, text }),
      });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      lastResults = data.sentences || [];
      lastCheckedLength = editor.value.length;
      renderSuggestions();

      const rebuilt = lastResults
        .map((r) => (r.has_error && r.corrected_sentence ? r.corrected_sentence : r.sentence))
        .join(" ");
      const [before, after] = await Promise.all([
        fetchReadability(text),
        fetchReadability(rebuilt),
      ]);
      renderReadability(before, after);

      statusMsg.textContent = `Done — ${lastResults.filter((r) => r.escalated).length} sentence(s) flagged.`;
    } catch (err) {
      statusMsg.textContent = `Couldn't reach the checker: ${err.message}. Is the app server running?`;
    } finally {
      checkBtn.disabled = false;
      stripeBar.classList.remove("checking");
    }
  }

  checkBtn.addEventListener("click", runCheck);
  editor.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runCheck();
  });

  // ---- Live/background grammar checking (per finished sentence) ----

  function getLastFinishedSentence(text) {
    const enders = [...text.matchAll(/[.!?](?=\s|$)/g)];
    if (enders.length === 0) return null;

    const last = enders[enders.length - 1];
    const endIdx = last.index + 1;
    if (endIdx <= lastCheckedLength) return null;

    const before = text.slice(0, endIdx - 1);
    const priorEnders = [...before.matchAll(/[.!?](?=\s|$)/g)];
    const startIdx = priorEnders.length > 0
      ? priorEnders[priorEnders.length - 1].index + 1
      : 0;

    const sentence = text.slice(startIdx, endIdx).trim();
    return sentence ? { sentence, endIdx } : null;
  }

  async function liveCheckLastSentence() {
    const text = editor.value;
    const found = getLastFinishedSentence(text);
    if (!found) return;

    lastCheckedLength = found.endIdx;

    try {
      const res = await fetch("/check_sentence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lang: currentLang, sentence: found.sentence }),
      });
      if (!res.ok) return;
      const result = await res.json();
      if (result.error) return;

      const i = lastResults.findIndex((r) => r.sentence === result.sentence);
      if (i >= 0) {
        lastResults[i] = result;
      } else {
        lastResults.push(result);
      }
      renderSuggestions();
      downloadBtn.disabled = false;
    } catch {
      // ignore silently in live mode
    }
  }

  editor.addEventListener("input", () => {
    if (/[.!?]\s*$/.test(editor.value)) {
      liveCheckLastSentence();
    }
    scheduleTypoScan();
  });
  // -------------------------------------------------------------------
})();