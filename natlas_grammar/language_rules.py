"""
Per-language orthography definitions.

These are deterministic, no-ML checks: valid grapheme sets and a couple of
cheap structural heuristics per language. They're the backbone of Tier 1
because diacritic/character-set problems are extremely common and fully
checkable without a model.

NOTE: these sets are deliberately conservative starting points, not
linguistically exhaustive. Extend them (especially the Yoruba tone-mark
logic) against a real corpus before relying on this in production.

NOTE on flag_missing_diacritics(): this heuristic is defined here but is
NOT currently used by LightweightGate's live scan/underline path — it
flags any word 3+ letters long that contains none of the language's
special diacritic letters, which in practice means most ordinary, correctly
spelled words (since not every word needs a special letter) get flagged.
Too noisy for live underlining. Kept here in case it's useful for a more
targeted future check (e.g. only applied to specific known word patterns),
but not wired into token flagging right now.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


def _strip_accents(s: str) -> str:
    """Return the base letters of a string with diacritics removed."""
    decomposed = unicodedata.normalize("NFD", s)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    name: str
    # base letters valid in the orthography (lowercase)
    base_letters: str
    # combining/underdot letters that carry meaning beyond the base letter
    special_letters: tuple[str, ...] = field(default_factory=tuple)
    # tone-mark combining characters this language uses (empty = not tonal in writing)
    tone_marks: tuple[str, ...] = field(default_factory=tuple)
    # digraphs that should be treated as single units when tokenizing
    digraphs: tuple[str, ...] = field(default_factory=tuple)

    def valid_char_pattern(self) -> re.Pattern:
        chars = re.escape(self.base_letters + "".join(self.special_letters))
        # allow combining tone marks (Unicode Mn) plus the language's own letters
        return re.compile(rf"^[{chars}\u0300-\u036f\s'\-]+$", re.IGNORECASE)

    def flag_invalid_chars(self, token: str) -> bool:
        """True if token contains characters outside this language's orthography."""
        if not token:
            return False
        return self.valid_char_pattern().match(token) is None

    def flag_missing_diacritics(self, token: str) -> bool:
        """
        Heuristic: if the *undiacritized* form of the token is a common word
        in this language's special-letter inventory but the token as typed
        carries none of the special letters/tone marks, it's a candidate for
        a missing-diacritic error. This is intentionally cheap and noisy —
        it's a Tier-1 flag, not a verdict. NOT currently called from
        LightweightGate's token-flagging path — see module docstring.
        """
        if not self.special_letters and not self.tone_marks:
            return False
        has_any_special = any(ch in token for ch in self.special_letters)
        has_tone_mark = any(
            unicodedata.combining(ch) for ch in unicodedata.normalize("NFD", token)
        )
        # only meaningful for tokens long enough to plausibly carry a diacritic
        return len(token) >= 3 and not has_any_special and not has_tone_mark


YORUBA = LanguageProfile(
    code="yo",
    name="Yoruba",
    base_letters="abdefghijklmnoprstuwy",
    special_letters=("ẹ", "ọ", "ṣ", "gb"),
    tone_marks=("\u0301", "\u0300"),  # acute (high), grave (low); mid = unmarked
    digraphs=("gb",),
)

IGBO = LanguageProfile(
    code="ig",
    name="Igbo",
    base_letters="abdefghijklmnoprstuvwyz",
    special_letters=("ị", "ọ", "ụ", "ṅ", "gb", "gh", "kp", "kw", "nw", "ny", "ch", "sh"),
    tone_marks=("\u0301", "\u0300"),
    digraphs=("gb", "gh", "kp", "kw", "nw", "ny", "ch", "sh"),
)

HAUSA = LanguageProfile(
    code="ha",
    name="Hausa",
    base_letters="abcdefghijklmnorstuwyz",
    special_letters=("ɓ", "ɗ", "ƙ", "ƴ", "'y"),
    tone_marks=(),  # tone/vowel length not standardly marked in boko orthography
    digraphs=("ts", "sh"),
)

LANGUAGES: dict[str, LanguageProfile] = {p.code: p for p in (YORUBA, IGBO, HAUSA)}


def get_profile(lang_code: str) -> LanguageProfile:
    try:
        return LANGUAGES[lang_code]
    except KeyError as e:
        raise ValueError(
            f"Unsupported language '{lang_code}'. Supported: {list(LANGUAGES)}"
        ) from e