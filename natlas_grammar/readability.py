"""
Readability scoring — LIX (Läsbarhetsindex), not Flesch-Kincaid.

Flesch-Kincaid and similar English-tuned formulas depend on syllable
counting, which relies on English-specific vowel-cluster heuristics that
don't transfer cleanly to Yoruba, Igbo, or Hausa. LIX avoids syllable
counting entirely — it's based only on words-per-sentence and the
proportion of long words (over 6 characters), both straightforward to
compute in any space-delimited, punctuation-marked language.

This does NOT mean LIX's classic difficulty bands are precisely calibrated
for these languages — they were originally calibrated on Swedish/European
text. Treat the label as a rough, consistent signal for comparison (e.g.
before vs. after correction), not an authoritative reading-level claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"[^\s]+", re.UNICODE)

LONG_WORD_CHAR_THRESHOLD = 6


@dataclass
class ReadabilityResult:
    lix_score: float
    label: str
    word_count: int
    sentence_count: int
    long_word_count: int
    avg_words_per_sentence: float
    long_word_ratio: float


def _label_for_lix(score: float) -> str:
    if score < 25:
        return "Very easy"
    if score < 35:
        return "Easy"
    if score < 45:
        return "Medium"
    if score < 55:
        return "Difficult"
    return "Very difficult"


def score_text(text: str) -> ReadabilityResult | None:
    text = text.strip()
    if not text:
        return None

    words = [w.strip(".,!?;:\"()") for w in _WORD_RE.findall(text)]
    words = [w for w in words if re.search(r"\w", w, re.UNICODE)]
    word_count = len(words)
    if word_count == 0:
        return None

    sentence_count = len([s for s in _SENTENCE_RE.split(text) if s.strip()])
    sentence_count = max(sentence_count, 1)  # avoid div-by-zero on unterminated text

    long_word_count = sum(1 for w in words if len(w) > LONG_WORD_CHAR_THRESHOLD)

    avg_words_per_sentence = word_count / sentence_count
    long_word_ratio = long_word_count / word_count

    lix = avg_words_per_sentence + (long_word_count * 100 / word_count)

    return ReadabilityResult(
        lix_score=round(lix, 1),
        label=_label_for_lix(lix),
        word_count=word_count,
        sentence_count=sentence_count,
        long_word_count=long_word_count,
        avg_words_per_sentence=round(avg_words_per_sentence, 1),
        long_word_ratio=round(long_word_ratio, 3),
    )