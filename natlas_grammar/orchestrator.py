from __future__ import annotations

import re
from dataclasses import dataclass, field

from .diff_utils import Edit, edit_ratio, word_level_diff
from .lightweight_classifier import LightweightGate
from .llm_corrector import NAtlasCorrector

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

REPHRASE_GUARD_RATIO = 0.6


@dataclass
class SentenceResult:
    sentence: str
    escalated: bool
    has_error: bool = False
    edits: list[Edit] = field(default_factory=list)
    explanation: str = ""
    auto_applied: bool = False
    corrected_sentence: str | None = None
    gate_reasons: list[str] = field(default_factory=list)
    semantic_similarity: float | None = None
    verifier_checked: bool = False
    verifier_meaning_preserved: bool | None = None
    verifier_reason: str = ""


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


class GrammarChecker:
    def __init__(
        self,
        lang_code: str,
        gate: LightweightGate,
        corrector: NAtlasCorrector | None,
        auto_apply_threshold: float | None,
        semantic_matcher=None,
        drift_guard_similarity: float | None = None,
    ):
        self.lang_code = lang_code
        self.gate = gate
        self.corrector = corrector
        self.auto_apply_threshold = auto_apply_threshold
        self.semantic_matcher = semantic_matcher
        self.drift_guard_similarity = drift_guard_similarity

    def check_one(self, sentence: str, context: str | None = None) -> SentenceResult:
        """
        Check a single sentence in isolation. Used both by check() (batch,
        whole-document mode) and by the live/background per-sentence
        endpoint — same logic, one path, no duplication.
        """
        gate_result = self.gate.check_sentence(sentence)

        if not gate_result.flagged:
            return SentenceResult(sentence=sentence, escalated=False)

        reasons = sorted({reason for t in gate_result.tokens for reason in t.reasons})

        if self.corrector is None:
            return SentenceResult(
                sentence=sentence,
                escalated=True,
                explanation="flagged by Tier 1; Tier 2 not configured",
                gate_reasons=reasons,
            )

        correction = self.corrector.correct(sentence, self.lang_code, context=context)

        if not correction.has_error or correction.corrected.strip() == sentence.strip():
            return SentenceResult(sentence=sentence, escalated=True, has_error=False)

        ratio = edit_ratio(sentence, correction.corrected)
        edits = word_level_diff(sentence, correction.corrected)

        semantic_similarity = None
        drift_ok = True
        if self.semantic_matcher is not None:
            semantic_similarity = self.semantic_matcher.similarity(sentence, correction.corrected)
            if self.drift_guard_similarity is not None:
                drift_ok = semantic_similarity >= self.drift_guard_similarity

        auto_apply = (
            self.auto_apply_threshold is not None
            and ratio < REPHRASE_GUARD_RATIO
            and (1 - ratio) >= self.auto_apply_threshold
            and drift_ok
        )

        return SentenceResult(
            sentence=sentence,
            escalated=True,
            has_error=True,
            edits=edits,
            explanation=correction.explanation,
            auto_applied=auto_apply,
            corrected_sentence=correction.corrected,
            semantic_similarity=semantic_similarity,
        )

    def check(self, text: str) -> list[SentenceResult]:
        results: list[SentenceResult] = []
        sentences = split_sentences(text)

        for idx, sentence in enumerate(sentences):
            context = sentences[idx - 1] if idx > 0 else None
            results.append(self.check_one(sentence, context=context))

        return results