

https://github.com/user-attachments/assets/9736a2bc-40aa-49c2-9703-1fff40c905b3



# N-ATLaS Grammar Checker (Yoruba / Igbo / Hausa) — Fully Offline

A two-tier, fully offline grammar checker for Nigerian indigenous languages,
built on [N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) (Llama-3 8B
fine-tuned for Hausa, Igbo, Yoruba). No server process, no cloud API, no
internet connection required once installed — everything runs locally on
your machine, including the LLM itself, loaded in-process from a local
quantized GGUF file.

---

## Table of contents

1. [Architecture overview](#architecture-overview)
2. [Tier 1 — the rule/lexicon gate](#tier-1--the-rulelexicon-gate)
3. [Tier 2 — N-ATLaS correction](#tier-2--n-atlas-correction)
4. [Live typo underlining](#live-typo-underlining)
5. [Live background grammar checking](#live-background-grammar-checking)
6. [Deferred reveal (no mid-typing pop-ins)](#deferred-reveal-no-mid-typing-pop-ins)
7. [Translation with quote preservation](#translation-with-quote-preservation)
8. [Readability scoring (LIX)](#readability-scoring-lix)
9. [Setup](#setup)
10. [Usage](#usage)
11. [Configuration reference](#configuration-reference)
12. [Why semantic matching is disabled](#why-semantic-matching-is-disabled)
13. [Known limitations](#known-limitations)
14. [Troubleshooting](#troubleshooting)
15. [License notes](#license-notes)

---

## Architecture overview

```
 text in
    │
    ▼
 sentence splitter (per-language)
    │
    ▼
 ┌─────────────────────────────┐
 │ TIER 1 — Rule/Lexicon Gate   │   milliseconds, CPU only, no model weights
 │  - character-set validation  │
 │  - lexicon lookup            │
 │    (exact match required)    │
 │  - fuzzy "did you mean"      │
 │    (does NOT suppress flags) │
 └──────────────┬───────────────┘
                │ flagged sentences only
                ▼
 ┌─────────────────────────────┐
 │ TIER 2 — N-ATLaS (gated)     │   in-process, CPU, quantized GGUF (4-bit)
 │  - llama-cpp-python           │
 │  - structured JSON output     │
 │  - explains before correcting │
 │  - word-level diff vs input   │
 │  - per-language auto-apply    │
 │    threshold                  │
 │  - surface-level rephrase     │
 │    guard (word-diff ratio)    │
 └──────────────┬───────────────┘
                │
                ▼
     inline suggestions (accept/reject), never a silent rewrite
```

Two supporting systems sit alongside this pipeline:

- A **fast, Tier-1-only live scan** (`/scan`) drives real-time underlining
  in the editor as you type, completely separate from the (slower) Tier 2
  correction pipeline.
- A **readability scorer** (`/readability`) reports a LIX score for the
  text before and after corrections, so you can see whether the fixes
  actually made the writing easier to read.

---

## Tier 1 — the rule/lexicon gate

`natlas_grammar/lightweight_classifier.py`

Tier 1 exists because there is no labeled "grammar error" dataset at scale
for Yoruba, Igbo, or Hausa (unlike English, where something like
CoLA/JFLEG exists to train a classifier on). Instead, Tier 1 is fully
deterministic:

- **Character-set validation** (`language_rules.py`) — each language has a
  fixed, known set of valid letters. A token using a character outside
  that set is flagged as `invalid_character`.
- **Lexicon lookup** — every word is checked against a real wordlist
  (`resources/lexicons/{yo,ig,ha}.txt`, ~11k–18k words each). A word must
  be an **exact match** to be considered known; anything else — including
  words that are merely close to a real word — is flagged
  `out_of_vocabulary`.
- **Fuzzy "did you mean"** — when a word isn't an exact match, the gate
  also looks for the closest real word using Python's built-in `difflib`
  (no extra dependency, fully offline). This closest match is attached to
  the flag purely as a hint for the person reading the suggestion — it
  does **not** cause the flag to be suppressed. A near-miss typo still
  gets flagged; it just also comes with a "did you mean X" pointer.

### Bugs that were fixed here (and why the code looks the way it does)

- **Unicode normalization.** Diacritic letters like `ẹ`, `ọ`, `ị`, `ụ` can
  be stored as one single character (NFC) or as a base letter plus a
  separate combining mark (NFD) — visually identical, different raw bytes.
  Every word, from both the lexicon file and the live text, is now forced
  through `unicodedata.normalize("NFC", ...)` before comparison, or
  lookups would silently fail for a huge share of real words.
- **Apostrophe stripping.** The original punctuation-stripping logic
  removed `'` from every token — but in Hausa specifically, the apostrophe
  is a real letter (representing the hooked `ƴ` sound, e.g. `'yan`, `'ya`),
  not punctuation. Stripping it broke lookups for a large share of genuine
  Hausa vocabulary. The strip list no longer includes `'`.
- **The missing-diacritic heuristic was too noisy for live use.** A
  separate check (`flag_missing_diacritics` in `language_rules.py`) flags
  any word 3+ letters long that doesn't contain one of the language's
  special letters — but most ordinary, correctly-spelled words simply
  don't need one, so this flagged almost everything. It's still defined in
  `language_rules.py` but is **not called** from the live token-flagging
  path (`_flag_token`) for this reason.
- **Minimum lexicon size guard.** If a lexicon file has fewer than
  `min_lexicon_size` words (default `200`), OOV/spelling checks are
  disabled entirely for that language rather than producing near-universal
  false positives from too-sparse data. A `[warn]` is printed at startup if
  this triggers.

---

## Tier 2 — N-ATLaS correction

`natlas_grammar/llm_corrector.py`

Only sentences Tier 1 flags are sent here — this is the expensive,
precision layer. The model runs **in-process** (not as a separate server)
via `llama-cpp-python`, loading your local `models/n-atlas-q4_k_m.gguf`
directly into the same Python process.

- **Structured JSON output, never a silent rewrite.** The model is
  prompted to return `{"has_error": bool, "explanation": "...", "corrected": "..."}`.
  The orchestrator (`orchestrator.py`) computes a word-level diff between
  the original and corrected sentence, so the UI can show exactly what
  changed rather than swapping in an entirely different sentence.
- **Explain before correcting.** The JSON schema asks for `explanation`
  *before* `corrected`. Without this ordering, the model tends to lock in
  "corrected = original unchanged" before it has actually reasoned through
  what's wrong. Explaining first acts as a lightweight chain-of-thought the
  correction step can build on.
- **Explanations are written in the target language**, not English — most
  natural for a native-speaking writer reading their own correction.
- **Auto-apply is conservative and per-language.** A correction is only
  auto-applied if it passes *both* a confidence threshold
  (`auto_apply_threshold`) *and* a surface-level rephrase guard — if more
  than 60% of a sentence's words changed, it's treated as a rephrase, not
  a targeted fix, and is never auto-applied regardless of threshold.
  **Yoruba's threshold is `null` — nothing is ever auto-applied for
  Yoruba**, reflecting N-ATLaS's own reported weaker fluency there
  (2.71/5 vs Hausa 4.23/5 in the model card's human eval).

### A truncation bug worth knowing about

Longer sentences occasionally produced JSON responses that got cut off
mid-string before N-ATLaS finished generating (visible as no closing `}`
in the raw output). This silently failed to parse and fell back to
"no correction needed" — looking like a missed error rather than a
truncation. Fixed by:
- Raising `max_new_tokens` (now `1024` in `config.yaml`, up from an
  original `512`).
- Tightening the prompt's explanation-length rule to one short sentence
  under 20 words, reducing how much the model needs to generate per
  response.

If you still see truncation on unusually long sentences, raising
`max_new_tokens` further (e.g. `1536`) is the next lever to pull.

---

## Live typo underlining

`natlas_grammar/lightweight_classifier.py` (`scan()` method),
`app.py` (`/scan` route), `static/app.js`, `static/index.html`,
`static/style.css`

Browsers give no way to style individual words inside a plain
`<textarea>` while keeping it editable. This is solved with a standard
overlay technique:

- `static/index.html` wraps the editor in `.editor-wrap`, containing two
  stacked elements: a hidden `#editorHighlight` div showing the same text
  with flagged words wrapped in `<mark>` (styled with a wavy red
  underline), sitting *behind* the real `#editor` textarea — which has its
  background made transparent so the underline shows through.
- Both layers share identical font, size, line-height, and padding
  (`static/style.css`), so the underline aligns exactly under the real
  typed letters. A `scroll` listener keeps them in sync when the text
  scrolls.
- `LightweightGate.scan(text)` runs the same character/lexicon checks as
  Tier 1's sentence-level gate, but returns every individual flagged
  token's character offsets directly — no sentence-level aggregate
  decision needed, since this is purely about "which words to underline."
- The frontend calls `/scan` on a **400ms debounce** after typing pauses —
  fast enough to feel responsive, but not firing on every single
  keystroke.

This only reflects Tier 1 (character/lexicon) issues — genuine
grammar/word-order errors that only the LLM can catch still require the
sentence-completion trigger or the "Check text" button, since running the
LLM on every pause would be far too slow for live typing feedback.

---

## Live background grammar checking

`natlas_grammar/orchestrator.py` (`check_one()`), `app.py`
(`/check_sentence` route), `static/app.js`

As you finish typing a sentence — ending it with `.`, `!`, or `?` — that
single sentence is sent through the full Tier 1 → Tier 2 pipeline in the
background, without needing to click "Check text." This reuses the exact
same logic as the full-document check (`GrammarChecker.check_one()`, which
`check()` also calls per sentence), so there's no duplicated logic between
the "check everything" and "check one sentence live" paths.

---

## Deferred reveal (no mid-typing pop-ins)

`static/app.js`

Live background checking (above) still runs continuously, but its results
are collected into a hidden buffer (`pendingResults`) rather than being
rendered into the suggestions panel immediately. Nothing appears in the
UI until the editor has been idle for **2.5 seconds** (`REVEAL_IDLE_MS`),
at which point everything collected so far is revealed to the panel at
once. This avoids suggestion cards popping in one at a time while you're
still actively writing.

Clicking **Check text** bypasses this entirely — it always re-checks the
full document and renders results immediately, cancelling any pending
idle-reveal timer so there's no conflicting double-render.

Live typo underlining (above) is **not** subject to this delay — it's a
separate, much cheaper Tier-1-only path and continues to update
immediately on its own 400ms debounce.

---

## Translation with quote preservation

`natlas_grammar/llm_corrector.py` (`translate_to_english()`)

Since explanations are written in the target language by design, the UI
offers a **Translate to English** button per suggestion. A naive
translation prompt would translate the actual local-language words being
discussed too (e.g. `'nagodessosai'` → some English guess), making the
explanation useless for matching back to the original text. The
translation prompt explicitly instructs the model: translate the
surrounding explanatory prose, but leave anything inside quotation marks
exactly as written, untranslated. A concrete before/after example is
included directly in the prompt to anchor the pattern.

This depends on the model consistently quoting the words it discusses in
its original explanation (which the Tier 2 correction prompt already
encourages) — usually reliable on an 8B quantized model, but not
absolutely guaranteed by instruction-following alone.

---

## Readability scoring (LIX)

`natlas_grammar/readability.py`, `app.py` (`/readability` route),
`static/app.js`, `static/index.html`

Flesch-Kincaid-style formulas depend on syllable counting, which relies on
English-specific vowel-cluster heuristics that don't transfer cleanly to
Yoruba, Igbo, or Hausa (different vowel systems, diacritics interacting
with syllables, Igbo's agglutinative structure). Instead, this uses
**LIX** (Läsbarhetsindex), which needs only two language-agnostic inputs:
average words per sentence, and the percentage of "long words" (over 6
characters).

```
LIX = (words / sentences) + (long_words × 100 / words)
```

The resulting score is bucketed into five bands (Very easy → Very
difficult). These bands come from LIX's original Swedish calibration and
have **not** been independently re-validated for Yoruba/Igbo/Hausa
specifically — treat the label as a rough, internally consistent signal
for comparison rather than a certified reading-level claim. What it's
genuinely useful for: after running "Check text," the app scores both the
original text and what the text would look like with every Tier 2
suggestion applied, and shows both scores side by side — a concrete,
comparable "before vs. after" readability signal even before you click
Accept.

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

If `llama-cpp-python` fails to build from source:
```powershell
python -m pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Place your model at `models/n-atlas-q4_k_m.gguf` and confirm
`llm.model_path` in `config.yaml` points to it. If the file is missing,
`NAtlasCorrector` raises a clear error at startup and the app automatically
falls back to Tier-1-only mode (flagging and underlining still work;
corrections and translation are skipped) instead of crashing.

---

## Usage

**CLI:**
```powershell
python cli.py --lang yo --text "Mo je nse oko"
python cli.py --lang ha --file draft.txt --apply-all
python cli.py --lang ig --text "..." --tier1-only   # no model needed
```

**Web UI:**
```powershell
uvicorn app:app --reload
```
Open http://127.0.0.1:8000. The model loads once at startup and is shared
across all three languages, rather than loading the multi-gigabyte
quantized weights three separate times.

---

## Configuration reference

`config.yaml`

| Key | Meaning |
|---|---|
| `llm.model_path` | path to your local GGUF |
| `llm.n_ctx` | 8092 — matches N-ATLaS's max context |
| `llm.n_gpu_layers` | 0 — CPU-only setup |
| `llm.max_new_tokens` | 1024 — needs headroom or JSON responses truncate |
| `languages.<lang>.auto_apply_threshold` | confidence needed to auto-apply; `null` = never |
| `tier1.fuzzy_match_enabled` / `fuzzy_match_threshold` | offline "did you mean" matching (0–1 ratio, does not suppress flags) |
| `tier1.oov_ratio_flag_threshold` | how high a sentence's OOV ratio must be to force-flag it |
| `semantic.enabled` | `false` — see below |

---

## Why semantic matching is disabled

An optional LaBSE-based layer (`natlas_grammar/semantic.py`) exists in the
codebase for meaning-based (rather than spelling-based) matching and a
meaning-drift guard on corrections — but it requires a ~1.8GB download of
LaBSE from Hugging Face Hub on first use, incompatible with fully offline
use. It's kept in the codebase (not deleted) so it can be re-enabled later
with internet access, by setting `semantic.enabled: true` and installing
`sentence-transformers` — no code changes required, since both `cli.py`
and `app.py` already handle `semantic_matcher=None` as a fully supported
state.

Without it: Tier 1 lexicon matching uses exact match + offline fuzzy
matching (`difflib`); Tier 2's auto-apply guard uses the word-diff ratio
only.

---

## Known limitations

- 8,092 token context window — batch by paragraph, not whole documents.
- Yoruba fluency is notably weaker in N-ATLaS's own eval — always
  suggested, never auto-applied.
- No dialect handling.
- CPU-only inference — a few seconds per flagged sentence, including live
  background checks.
- Readability band labels aren't independently validated for these three
  languages (see [Readability scoring](#readability-scoring-lix)).
- Translation's quote-preservation depends on consistent model
  instruction-following, not a hard guarantee.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Model file not found` | Check `llm.model_path` matches your GGUF's actual path/name. |
| `WinError 193` on install | Use the prebuilt-wheel install command above. |
| Only "no correction needed", never a fix | Check terminal for truncated JSON — raise `max_new_tokens`, confirm model loaded (`[warn]` at startup means it didn't). |
| A new route returns `404 Not Found` | Confirm the route is actually saved in `app.py` (`Select-String -Path app.py -Pattern "/routename"`), then fully kill stale processes (`Get-Process python \| Stop-Process -Force`) before restarting — `--reload` doesn't always catch every change reliably. |
| `uvicorn` hangs after startup | Model is loading into RAM — can take 1–2 min on CPU. |
| `Fatal error in launcher` on `pip` | Use `python -m pip install ...` instead of `pip install ...`. |
| UI shows stale results | Hard refresh (Ctrl+Shift+R) — `app.js` gets cached. |
| Every word gets underlined/flagged | Check lexicon size (`min_lexicon_size` guard) and Unicode normalization — see [Tier 1 bugs](#tier-1--the-rulelexicon-gate). |

---

## License notes

N-ATLaS requires attribution to the Federal Ministry of Communications,
Innovation and Digital Economy / Awarri Technologies, has a 1000-active-user
cap before needing a commercial license, and derivatives must carry "Powered
by Awarri" if renamed. See the
[model card](https://huggingface.co/NCAIR1/N-ATLaS) for full terms.

---

## Acknowledgements

**Team**
- **Okpor Victor (Group Leader)**
- **Yusuf Sada**
- **Joshua Egberibo**
- **Matilda Obot**
- **David Adeyeni**
- **Aminat Bakare**
- **Asher Nzurum**
- **Umar Ahmad**
- **Lasisi Abdulmalik Bolaji**

**Facilitators**
- Victor Rizama
- Stephen Ayuba

**NCAIR** — for providing the N-ATLaS model that powers this project's
Tier 2 correction layer.
