"""
Turn a (original, corrected) sentence pair into a list of minimal word-level
edits, so the UI can show "X -> Y" suggestions instead of swapping in a whole
new sentence. This matters more here than in an English tool, because N-ATLaS
is an 8B instruction model prone to some rephrasing drift, especially in
Yoruba where the model card reports weaker fluency.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass
class Edit:
    op: str  # "replace" | "insert" | "delete"
    original: str
    suggestion: str
    position: int  # word index in the original sentence


def word_level_diff(original: str, corrected: str) -> list[Edit]:
    orig_words = original.split()
    corr_words = corrected.split()

    sm = difflib.SequenceMatcher(a=orig_words, b=corr_words, autojunk=False)
    edits: list[Edit] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        original_span = " ".join(orig_words[i1:i2])
        corrected_span = " ".join(corr_words[j1:j2])
        if tag == "replace":
            edits.append(Edit("replace", original_span, corrected_span, i1))
        elif tag == "delete":
            edits.append(Edit("delete", original_span, "", i1))
        elif tag == "insert":
            edits.append(Edit("insert", "", corrected_span, i1))

    return edits


def edit_ratio(original: str, corrected: str) -> float:
    """Fraction of words touched — used to detect over-aggressive rewrites."""
    orig_words = original.split()
    if not orig_words:
        return 0.0
    edits = word_level_diff(original, corrected)
    touched = sum(
        max(len(e.original.split()), len(e.suggestion.split())) for e in edits
    )
    return min(touched / len(orig_words), 1.0)
