import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from natlas_grammar.diff_utils import edit_ratio, word_level_diff
from natlas_grammar.lightweight_classifier import LightweightGate
from natlas_grammar.orchestrator import split_sentences


class FakeSemanticMatcher:
    """Deterministic stand-in for SemanticMatcher — no model download needed.
    Give it a dict of {token: (closest_word, similarity)} to control output."""

    def __init__(self, canned: dict[str, tuple[str, float]]):
        self.canned = canned

    def closest_lexicon_word(self, token, lang_code, lexicon_words):
        return self.canned.get(token)


def test_split_sentences():
    text = "Mo lọ si ile-iwe. Ó ti pẹ́ jù! Ṣé o ń bọ̀?"
    sents = split_sentences(text)
    assert len(sents) == 3


def test_yoruba_lexicon_hit_not_flagged():
    gate = LightweightGate("yo", lexicon_path="resources/lexicons/yo.txt")
    result = gate.check_sentence("mo je oko")
    # 'je' and 'oko' are in the sample lexicon, 'mo' too
    assert result.oov_ratio == 0.0


def test_yoruba_oov_flags():
    gate = LightweightGate("yo", lexicon_path="resources/lexicons/yo.txt")
    result = gate.check_sentence("mo zzqxwv oko")
    assert result.flagged is True
    assert any("out_of_vocabulary" in t.reasons for t in result.tokens)


def test_hausa_invalid_char_flagged():
    gate = LightweightGate("ha", lexicon_path="resources/lexicons/ha.txt")
    # '@' is not part of Hausa orthography
    result = gate.check_sentence("gida@ ruwa")
    assert any("invalid_character" in t.reasons for t in result.tokens)


def test_word_level_diff_minimal():
    edits = word_level_diff("mo je oko", "mo jẹ oko")
    assert len(edits) == 1
    assert edits[0].op == "replace"
    assert edits[0].original == "je"
    assert edits[0].suggestion == "jẹ"


def test_edit_ratio_small_for_single_word_fix():
    ratio = edit_ratio("mo je nse oko ni owuro yi", "mo jẹ ń ṣe oko ní ọwúrọ̀ yìí")
    assert 0 < ratio <= 1.0


def test_semantic_match_above_threshold_is_not_flagged():
    # 'zzqxwv' isn't in the lexicon, but the fake matcher says it's a close
    # meaning-match to 'oko' (0.9 similarity) — above the accept threshold,
    # so it should NOT show up as a hard-flagged token.
    fake = FakeSemanticMatcher({"zzqxwv": ("oko", 0.9)})
    gate = LightweightGate(
        "yo",
        lexicon_path="resources/lexicons/yo.txt",
        semantic_matcher=fake,
        semantic_oov_accept_similarity=0.82,
    )
    result = gate.check_sentence("mo zzqxwv oko")
    hard_flagged = [t for t in result.tokens if "out_of_vocabulary" in t.reasons]
    assert hard_flagged == []


def test_semantic_match_below_threshold_still_flagged_with_suggestion():
    # similarity 0.3 is below the accept threshold — should still be flagged
    # as OOV, but now carrying a closest_meaning suggestion for the UI to show.
    fake = FakeSemanticMatcher({"zzqxwv": ("oko", 0.3)})
    gate = LightweightGate(
        "yo",
        lexicon_path="resources/lexicons/yo.txt",
        semantic_matcher=fake,
        semantic_oov_accept_similarity=0.82,
    )
    result = gate.check_sentence("mo zzqxwv oko")
    flagged = [t for t in result.tokens if t.token == "zzqxwv"]
    assert len(flagged) == 1
    assert "out_of_vocabulary" in flagged[0].reasons
    assert flagged[0].closest_meaning == ("oko", 0.3)


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
