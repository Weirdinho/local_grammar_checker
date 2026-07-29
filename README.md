
# N-ATLaS Grammar Checker (Yoruba / Igbo / Hausa) — Fully Offline

Two-tier grammar checker for Nigerian indigenous languages, built on
[N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) (Llama-3 8B fine-tuned for
Hausa, Igbo, Yoruba). Fully offline — no server process, no cloud API.

## Architecture

```
 text in
    │
    ▼
 sentence splitter (per-language)
    │
    ▼
 ┌─────────────────────────────┐
 │ TIER 1 — Rule/Lexicon Gate   │   < 20ms/sentence, CPU, no model weights
 │  - orthographic char check   │
 │  - diacritic completeness    │
 │  - wordlist / lexicon lookup │
 │    (exact string match)      │
 └──────────────┬───────────────┘
                │ flagged sentences only
                ▼
 ┌─────────────────────────────┐
 │ TIER 2 — N-ATLaS (gated)     │   in-process, CPU (quantized GGUF, 4-bit)
 │  - llama-cpp-python          │
 │  - structured JSON output    │
 │  - word-level diff vs input  │
 │  - per-language accept       │
 │    threshold                 │
 │  - surface-level rephrase    │
 │    guard (word-diff ratio)   │
 └──────────────┬───────────────┘
                │
                ▼
     inline suggestions (accept/reject), not silent rewrites
```

- **Tier 1**: character/diacritic validation + lexicon lookup. Exact match
  first; if a word isn't found, falls back to offline fuzzy spelling-match
  (`difflib`, no dependency) — close-enough words are treated as likely
  typos instead of hard errors, not flagged.
- **Tier 2**: runs your local `n-atlas-q4_k_m.gguf` in-process via
  `llama-cpp-python`. Returns structured JSON, explains before correcting
  (otherwise it tends to echo the input unchanged).
- **Live checking**: as you type, each sentence is checked automatically the
  moment you end it with `.`, `!`, or `?` — no need to press "Check text"
  for sentences already finished. The button still works for full re-checks.
- **Translation**: explanations are written in the target language by
  default. The "Translate to English" button translates the explanation's
  prose but leaves quoted words (e.g. `'nagodessosai'`) untouched, so the
  reader can match the explanation back to their actual text.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

If `llama-cpp-python` fails to build:
```powershell
python -m pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Place your model at `models/n-atlas-q4_k_m.gguf` and confirm `llm.model_path`
in `config.yaml` matches. If missing, falls back to Tier-1-only automatically.

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
Open http://127.0.0.1:8000. Model loads once at startup, shared across all
three languages.

## Config essentials (`config.yaml`)

| Key | Meaning |
|---|---|
| `llm.model_path` | path to your GGUF |
| `llm.max_new_tokens` | 768 — needs headroom or JSON responses truncate |
| `languages.<lang>.auto_apply_threshold` | confidence needed to auto-apply; `null` = never |
| `tier1.fuzzy_match_enabled` / `fuzzy_match_threshold` | offline spelling-distance fallback (0–1 ratio) |
| `semantic.enabled` | `false` — needs a ~1.8GB download, incompatible with offline use |

Without semantic matching: lexicon lookup uses exact match + fuzzy match;
Tier 2 auto-apply guard uses word-diff ratio only (>60% words changed =
never auto-applied); Yoruba never auto-applies regardless of threshold.

## Known limitations

- 8,092 token context — batch by paragraph, not full documents.
- Yoruba fluency is weaker in N-ATLaS's own eval (2.71/5 vs Hausa 4.23/5) —
  always suggested, never auto-applied.
- No dialect handling.
- CPU-only inference — a few seconds per flagged sentence, live checks
  included.
- Translation's quote-preservation depends on the model consistently
  quoting the words it discusses — usually reliable, not guaranteed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Model file not found` | Check `llm.model_path` matches your GGUF's actual path/name. |
| `WinError 193` on install | Use the prebuilt-wheel install command above. |
| Only "no correction needed", never a fix | Check terminal for truncated JSON — raise `max_new_tokens`, confirm model loaded (`[warn]` at startup means it didn't). |
| `uvicorn` hangs after startup | Model loading into RAM — 1–2 min on CPU. |
| `Fatal error in launcher` on `pip` | Use `python -m pip install ...`. |
| UI shows stale results | Hard refresh (Ctrl+Shift+R) — `app.js` gets cached. |

## License notes

N-ATLaS requires attribution to the Federal Ministry of Communications,
Innovation and Digital Economy / Awarri Technologies, has a 1000-active-user
cap before needing a commercial license, and derivatives must carry "Powered
by Awarri" if renamed. See the [model card](https://huggingface.co/NCAIR1/N-ATLaS).
