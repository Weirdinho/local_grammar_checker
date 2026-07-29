from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .language_rules import LanguageProfile, get_profile

_WORD_RE = re.compile(r"[^\s]+", re.UNICODE)


def _normalize(s: str) -> str:
    """
    Force consistent Unicode form before any comparison. Yoruba/Igbo/Hausa
    diacritics (ẹ ọ ṣ, ị ọ ụ ṅ, ɓ ɗ ƙ ƴ) can be encoded either as a single
    composed character (NFC) or as a base letter + separate combining mark
    (NFD) — visually identical, but different raw bytes. Without this,
    lexicon lookups can silently fail depending on which form the lexicon
    file vs. the typed text happens to use.
    """
    return unicodedata.normalize("NFC", s)


@dataclass
class FlaggedToken:
    token: str
    start: int
    end: int
    reasons: list[str]
    closest_meaning: tuple[str, float] | None = None


@dataclass
class GateResult:
    sentence: str
    flagged: bool
    tokens: list[FlaggedToken] = field(default_factory=list)
    oov_ratio: float = 0.0


class LightweightGate:
    def __init__(
        self,
        lang_code: str,
        lexicon_path: str | Path | None = None,
        oov_ratio_flag_threshold: float = 0.2,
        kenlm_path: str | Path | None = None,
        perplexity_zscore_flag_threshold: float = 1.5,
        semantic_matcher=None,
        semantic_oov_accept_similarity: float = 0.82,
        fuzzy_match_enabled: bool = True,
        fuzzy_match_threshold: float = 0.82,
        min_lexicon_size: int = 200,
    ):
        self.profile: LanguageProfile = get_profile(lang_code)
        self.oov_ratio_flag_threshold = oov_ratio_flag_threshold
        self.perplexity_zscore_flag_threshold = perplexity_zscore_flag_threshold
        self.lexicon: set[str] = self._load_lexicon(lexicon_path)
        self._lexicon_words: list[str] = sorted(self.lexicon)
        self._lm = self._load_kenlm(kenlm_path)
        self.semantic_matcher = semantic_matcher
        self.semantic_oov_accept_similarity = semantic_oov_accept_similarity
        self.fuzzy_match_enabled = fuzzy_match_enabled
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.min_lexicon_size = min_lexicon_size

        # A lexicon this small would flag nearly every real word as unknown
        # rather than actually distinguishing typos from valid vocabulary.
        # Rather than produce noisy, near-universal false positives, skip
        # OOV checks entirely until a real wordlist is in place.
        self.lexicon_usable = len(self.lexicon) >= self.min_lexicon_size
        if not self.lexicon_usable and self.lexicon:
            print(
                f"[warn] {lang_code} lexicon has only {len(self.lexicon)} words "
                f"(< {self.min_lexicon_size}) — OOV checks disabled until a "
                "larger lexicon is provided. See README for sources."
            )

    @staticmethod
    def _load_lexicon(path: str | Path | None) -> set[str]:
        if not path:
            return set()
        p = Path(path)
        if not p.exists():
            return set()
        with p.open(encoding="utf-8") as f:
            return {_normalize(line.strip().lower()) for line in f if line.strip()}

    @staticmethod
    def _load_kenlm(path: str | Path | None):
        if not path or not Path(path).exists():
            return None
        try:
            import kenlm  # optional dependency
        except ImportError:
            return None
        return kenlm.Model(str(path))

    def _tokens(self, text: str) -> list[re.Match]:
        return list(_WORD_RE.finditer(text))

    def _closest_by_fuzzy_match(self, token: str) -> tuple[str, float] | None:
        if not self._lexicon_words:
            return None
        matches = difflib.get_close_matches(
            token, self._lexicon_words, n=1, cutoff=0.6
        )
        if not matches:
            return None
        best = matches[0]
        ratio = difflib.SequenceMatcher(None, token, best).ratio()
        return best, ratio

    def _flag_token(self, raw: str, start: int, end: int) -> tuple[FlaggedToken | None, bool]:
        """
        Runs per-token checks (char validity + lexicon lookup) on a single
        token. Any token not an EXACT match in the lexicon is flagged as
        out-of-vocabulary — including near-miss/typo words that a fuzzy or
        semantic match found something close for. The closest match is
        still looked up and attached as a "did you mean" hint, but it no
        longer silently suppresses the flag — a fuzzy-close word is still
        not a confirmed correct word, so it stays underlined.

        Shared by both check_sentence() (single sentence, plus aggregate
        OOV-ratio / perplexity decisions) and scan() (arbitrary raw text,
        just wants per-token flags for underlining, no aggregate decision
        needed).

        Returns (FlaggedToken or None, is_oov) — is_oov feeds into the
        caller's oov_ratio calculation.
        """
        # Do NOT strip apostrophes — they're a real letter in Hausa/Yoruba/
        # Igbo orthography (e.g. Hausa 'y / ƴ, elision marks), not just
        # punctuation.
        token = _normalize(raw.strip(".,!?;:\"()").lower())
        reasons: list[str] = []
        is_oov = False
        closest_meaning: tuple[str, float] | None = None

        if self.profile.flag_invalid_chars(token):
            reasons.append("invalid_character")

        if self.lexicon_usable and self.lexicon and token and token not in self.lexicon:
            if re.search(r"\w", token, re.UNICODE):
                is_oov = True
                reasons.append("out_of_vocabulary")

                # Look up a closest match purely as a "did you mean"
                # suggestion — this does NOT suppress the flag anymore.
                if self.semantic_matcher is not None:
                    closest_meaning = self.semantic_matcher.closest_lexicon_word(
                        token, self.profile.code, self._lexicon_words
                    )
                if closest_meaning is None and self.fuzzy_match_enabled:
                    closest_meaning = self._closest_by_fuzzy_match(token)

        if not reasons:
            return None, is_oov

        return (
            FlaggedToken(token=raw, start=start, end=end, reasons=reasons, closest_meaning=closest_meaning),
            is_oov,
        )

    def check_sentence(self, sentence: str) -> GateResult:
        matches = self._tokens(sentence)
        if not matches:
            return GateResult(sentence=sentence, flagged=False)

        flagged_tokens: list[FlaggedToken] = []
        oov_count = 0

        for m in matches:
            flagged, is_oov = self._flag_token(m.group(), m.start(), m.end())
            if is_oov:
                oov_count += 1
            if flagged is not None:
                flagged_tokens.append(flagged)

        word_tokens = [m for m in matches if re.search(r"\w", m.group(), re.UNICODE)]
        oov_ratio = (
            oov_count / len(word_tokens)
            if (word_tokens and self.lexicon_usable)
            else 0.0
        )

        flagged = bool(flagged_tokens) or self._perplexity_flag(sentence)
        if oov_ratio >= self.oov_ratio_flag_threshold:
            flagged = True

        return GateResult(sentence=sentence, flagged=flagged, tokens=flagged_tokens, oov_ratio=oov_ratio)

    def scan(self, text: str) -> list[FlaggedToken]:
        """
        Fast, Tier-1-only, no sentence-level aggregate decision — just
        returns every individual flagged token with its character offsets
        in the raw input text. Meant to be called frequently (e.g. on a
        typing debounce) to drive live underlining in the editor.
        """
        flagged_tokens: list[FlaggedToken] = []
        for m in self._tokens(text):
            flagged, _ = self._flag_token(m.group(), m.start(), m.end())
            if flagged is not None:
                flagged_tokens.append(flagged)
        return flagged_tokens

    def _perplexity_flag(self, sentence: str) -> bool:
        if self._lm is None:
            return False
        score = self._lm.score(sentence, bos=True, eos=True)
        n_words = max(len(sentence.split()), 1)
        avg_log_prob = score / n_words
        return avg_log_prob < -6.0