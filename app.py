from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cli import build_semantic_matcher, load_config
from natlas_grammar.lightweight_classifier import LightweightGate
from natlas_grammar.llm_corrector import NAtlasCorrector
from natlas_grammar.orchestrator import GrammarChecker

app = FastAPI(title="N-ATLaS Grammar Checker (offline)")

_config = load_config()
_semantic_matcher = build_semantic_matcher(_config)  # None, since semantic.enabled is false

# Load N-ATLaS ONCE and share it across all three languages, instead of
# loading the ~4-5GB quantized model three separate times.
_llm_cfg = _config["llm"]
try:
    _shared_corrector = NAtlasCorrector(
        model_path=_llm_cfg["model_path"],
        n_ctx=_llm_cfg.get("n_ctx", 8092),
        n_threads=_llm_cfg.get("n_threads"),
        n_gpu_layers=_llm_cfg.get("n_gpu_layers", 0),
        temperature=_llm_cfg["temperature"],
        repeat_penalty=_llm_cfg["repeat_penalty"],
        max_new_tokens=_llm_cfg["max_new_tokens"],
    )
except RuntimeError as e:
    print(f"[warn] {e}\nRunning Tier-1-only (no corrections) for all languages.")
    _shared_corrector = None

_checkers = {}
for _lang in _config["languages"]:
    _lang_cfg = _config["languages"][_lang]
    _tier1_cfg = _config["tier1"]
    _sem_cfg = _config.get("semantic", {})

    _gate = LightweightGate(
        lang_code=_lang,
        lexicon_path=_lang_cfg.get("lexicon_path"),
        oov_ratio_flag_threshold=_tier1_cfg["oov_ratio_flag_threshold"],
        kenlm_path=_lang_cfg.get("kenlm_path"),
        perplexity_zscore_flag_threshold=_tier1_cfg["perplexity_zscore_flag_threshold"],
        semantic_matcher=_semantic_matcher,
        semantic_oov_accept_similarity=_sem_cfg.get("oov_accept_similarity", 0.82),
        fuzzy_match_enabled=_tier1_cfg.get("fuzzy_match_enabled", True),
        fuzzy_match_threshold=_tier1_cfg.get("fuzzy_match_threshold", 0.82),
    )

    _checkers[_lang] = GrammarChecker(
        lang_code=_lang,
        gate=_gate,
        corrector=_shared_corrector,
        auto_apply_threshold=_lang_cfg.get("auto_apply_threshold"),
        semantic_matcher=_semantic_matcher,
        drift_guard_similarity=_sem_cfg.get("drift_guard_similarity") if _semantic_matcher else None,
    )

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


class CheckRequest(BaseModel):
    lang: str
    text: str


class CheckSentenceRequest(BaseModel):
    lang: str
    sentence: str
    context: str | None = None


class ScanRequest(BaseModel):
    lang: str
    text: str


class TranslateRequest(BaseModel):
    lang: str
    text: str


class ReadabilityRequest(BaseModel):
    text: str


@app.post("/translate")
def translate(req: TranslateRequest):
    checker = _checkers.get(req.lang)
    if checker is None:
        return {"error": f"unsupported language '{req.lang}'"}
    if checker.corrector is None:
        return {"error": "N-ATLaS model isn't loaded — check server logs for the load error."}
    try:
        translated = checker.corrector.translate_to_english(req.text, req.lang)
    except RuntimeError as e:
        return {"error": str(e)}
    return {"translated": translated}


@app.post("/check")
def check(req: CheckRequest):
    if req.lang not in _checkers:
        return {"error": f"unsupported language '{req.lang}'"}
    results = _checkers[req.lang].check(req.text)
    return {
        "sentences": [
            {
                "sentence": r.sentence,
                "escalated": r.escalated,
                "has_error": r.has_error,
                "corrected_sentence": r.corrected_sentence,
                "explanation": r.explanation,
                "auto_applied": r.auto_applied,
                "semantic_similarity": r.semantic_similarity,
                "edits": [
                    {"op": e.op, "original": e.original, "suggestion": e.suggestion}
                    for e in r.edits
                ],
            }
            for r in results
        ]
    }


@app.post("/check_sentence")
def check_sentence(req: CheckSentenceRequest):
    """
    Live/background checking — called right as the user finishes a
    sentence (on ., !, or ?), instead of re-checking the whole document.
    """
    checker = _checkers.get(req.lang)
    if checker is None:
        return {"error": f"unsupported language '{req.lang}'"}

    r = checker.check_one(req.sentence, context=req.context)
    return {
        "sentence": r.sentence,
        "escalated": r.escalated,
        "has_error": r.has_error,
        "corrected_sentence": r.corrected_sentence,
        "explanation": r.explanation,
        "auto_applied": r.auto_applied,
        "semantic_similarity": r.semantic_similarity,
        "edits": [
            {"op": e.op, "original": e.original, "suggestion": e.suggestion}
            for e in r.edits
        ],
    }


@app.post("/scan")
def scan(req: ScanRequest):
    """
    Fast Tier-1-only scan for live underlining — no LLM call. Meant to be
    called frequently (on a typing debounce) since it's rule/lexicon-based
    only, no model inference involved.
    """
    checker = _checkers.get(req.lang)
    if checker is None:
        return {"error": f"unsupported language '{req.lang}'"}

    tokens = checker.gate.scan(req.text)
    return {
        "flags": [
            {"start": t.start, "end": t.end, "token": t.token, "reasons": t.reasons}
            for t in tokens
        ]
    }


@app.post("/readability")
def readability(req: ReadabilityRequest):
    """
    LIX readability score, since that depends on
    English-specific syllable counting. LIX only needs words-per-sentence
    and a long-word ratio, both language-agnostic. Can be called on raw
    text either before or after corrections are applied, to compare.
    """
    from natlas_grammar.readability import score_text  # lazy import, tiny/cheap either way

    result = score_text(req.text)
    if result is None:
        return {"error": "no text to score"}
    return {
        "lix_score": result.lix_score,
        "label": result.label,
        "word_count": result.word_count,
        "sentence_count": result.sentence_count,
        "long_word_count": result.long_word_count,
        "avg_words_per_sentence": result.avg_words_per_sentence,
        "long_word_ratio": result.long_word_ratio,
    }


@app.get("/")
def index():
    return FileResponse(_static_dir / "index.html")