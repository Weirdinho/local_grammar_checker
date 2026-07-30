"""
Tier 2 — N-ATLaS itself, run fully offline via llama-cpp-python, loading the
GGUF weights directly in-process. No server process, no network calls.

Only called for sentences the Tier-1 gate flagged.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from .language_rules import get_profile

_LANG_NAMES = {"yo": "Yoruba", "ig": "Igbo", "ha": "Hausa"}

_ALPHABET_NOTES = {
    "yo": (
        "Standard Yorùbá orthography uses only: a b d e ẹ f g gb h i j k l "
        "m n o ọ p r s ṣ t u w y (gb is a single digraph sound). It does "
        "NOT use c, q, v, x, or z. Tone marks (´ high, ` low, unmarked mid) "
        "and the underdots on ẹ/ọ/ṣ are meaningful — do not add or remove "
        "them unless that is the actual error."
    ),
    "ig": (
        "Standard Igbo (Ọnwụ) orthography uses: a b ch d e f g gb gh gw h "
        "i ị j k kp kw l m n ṅ nw ny o ọ p r s sh t u ụ v w y z (ch, gb, "
        "gh, gw, kp, kw, nw, ny, sh are single digraph sounds). It does NOT "
        "use standalone c, q, or x. The underdots on ị/ọ/ụ/ṅ are "
        "meaningful — do not add or remove them unless that is the actual "
        "error."
    ),
    "ha": (
        "Standard Hausa (boko) orthography uses: a b ɓ c d ɗ e f g h i j "
        "k ƙ l m n o r s sh t ts u w y ƴ z (sh and ts are single digraph "
        "sounds; ɓ, ɗ, ƙ, ƴ are distinct letters, not typos of b/d/k/y). It "
        "does NOT use p, q, v, or x — never introduce these. Tone and "
        "vowel length are not normally marked in writing."
    ),
}

_EXAMPLE_RESPONSE = (
    '{{"has_error": true, "explanation": "<short explanation of what was '
    'wrong, written in {lang}>", "corrected": "<corrected sentence>"}}'
)

_SYSTEM_TEMPLATE = (
    "You are a professional {lang} language editor. You correct grammar, "  
    "spelling, word choice, and diacritics/tone marks in {lang} text written "
    "by native speakers.\n\n"
    "ALPHABET: {alphabet_note}\n\n"
    "STRICT RULES:\n"
    "1. Make the SMALLEST edit that fixes an actual error. Never rephrase, "
    "expand, or rewrite a sentence that is already correct.\n"
    "2. NEVER change what the sentence means. Fixing grammar/spelling must "
    "preserve the original meaning exactly. If you cannot fix the error "
    "without changing the meaning, set has_error to false instead of "
    "guessing.\n"
    "3. If the input is short, a fragment, or ambiguous, do NOT reinterpret "
    "or 'translate' it into a different sentence you think makes more "
    "sense. Only touch clear, mechanical errors (spelling, diacritics, "
    "agreement, word order). If you are not confident what the writer "
    "meant, set has_error to false and leave it unchanged rather than "
    "substituting your own guess at their intent.\n"
    "4. Never introduce a letter that isn't part of {lang}'s alphabet as "
    "described above.\n"
    "5. First figure out the error and write a short natural explanation "
    "in {lang} — the same language as the sentence itself. THEN, based on "
    "that explanation, write the actual corrected sentence with the fix "
    "applied. The 'corrected' field must reflect the fix described in "
    "'explanation' — never leave 'corrected' identical to the original "
    "when has_error is true.\n"
    "6. Keep the explanation to ONE short sentence, under 20 words. Do not "
    "repeat words or restate the same point twice.\n\n"
    "Respond with ONLY a JSON object, no other text, in this exact field "
    "ORDER: " + _EXAMPLE_RESPONSE + ". If has_error is false, set "  # pyright: ignore[reportImplicitStringConcatenation]
    '"explanation" to an empty string and "corrected" to the original '
    "sentence unchanged."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Correction:
    original: str
    has_error: bool
    corrected: str
    explanation: str
    raw_response: str


class NAtlasCorrector:
    def __init__(
        self,
        model_path: str | Path = "models/n-atlas-q4_k_m.gguf",
        n_ctx: int = 8092,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        temperature: float = 0.1,
        repeat_penalty: float = 1.12,
        max_new_tokens: int = 1024,
        server_url: str | None = None,
        timeout_s: float | None = None,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python isn't installed. Run "  # pyright: ignore[reportImplicitStringConcatenation]
                "`pip install llama-cpp-python`, or use --tier1-only to run "
                "without corrections."
            ) from e

        model_path = Path(model_path)
        if not model_path.exists():
            raise RuntimeError(
                f"Model file not found at '{model_path}'. Point llm.model_path "  # pyright: ignore[reportImplicitStringConcatenation]
                "in config.yaml at your downloaded n-atlas-q4_k_m.gguf."
            )

        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

        try:
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                chat_format="llama-3",
                verbose=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model at '{model_path}': {e}"
            ) from e

    def _chat(self, system: str, user_content: str) -> str:
        with self._lock:
            resp = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.temperature,
                repeat_penalty=self.repeat_penalty,
                max_tokens=self.max_new_tokens,
            )
        return resp["choices"][0]["message"]["content"].strip()  # pyright: ignore[reportUnknownMemberType]

    def correct(self, sentence: str, lang_code: str, context: str | None = None) -> Correction:
        lang_name = _LANG_NAMES.get(lang_code, get_profile(lang_code).name)
        alphabet_note = _ALPHABET_NOTES.get(lang_code, "")
        system = _SYSTEM_TEMPLATE.format(lang=lang_name, alphabet_note=alphabet_note)

        user_content = sentence if not context else f"Context: {context}\nSentence to check: {sentence}"

        try:
            text = self._chat(system, user_content)
        except Exception as e:
            raise RuntimeError(f"Local inference failed: {e}") from e

        # ---- TEMPORARY DEBUG LINE — remove once confirmed stable ----
        print(f"\n[DEBUG] RAW MODEL OUTPUT for '{sentence}':\n{text}\n{'-'*60}")
        # ----------------------------------------------------------------

        return self._parse(sentence, text)

    def translate_to_english(self, text: str, lang_code: str) -> str:
        lang_name = _LANG_NAMES.get(lang_code, get_profile(lang_code).name)
        system = (
            f"You are a translator. Translate the following {lang_name} "
            "explanation into clear, natural English.\n\n"
            "IMPORTANT: Any word or phrase inside quotation marks (' ' or "
            f'" ") is an actual {lang_name} word being discussed — leave '
            "it EXACTLY as written, unquoted or unchanged, do NOT translate "
            "it. Only translate the surrounding explanatory sentence "
            "structure around those quoted words.\n\n"
            "Example: if the input is:\n"
            "  Kalmar 'nagodessosai' ba daidai ba ce. Ya kamata ya zama "
            "'nagode'.\n"
            "the correct output is:\n"
            "  The word 'nagodessosai' isn't correct. It should be "
            "'nagode'.\n"
            "— note 'nagodessosai' and 'nagode' stay untranslated exactly "
            "as given.\n\n"
            "Respond with ONLY the English translation — no notes, no "
            "extra commentary, no explanation of what you did."
        )
        try:
            return self._chat(system, text)
        except Exception as e:
            raise RuntimeError(f"Local translation failed: {e}") from e

    @staticmethod
    def _parse(original: str, raw_text: str) -> Correction:
        match = _JSON_RE.search(raw_text)
        if not match:
            return Correction(
                original=original,
                has_error=False,
                corrected=original,
                explanation="(model response not parseable — no correction applied)",
                raw_response=raw_text,
            )
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return Correction(
                original=original,
                has_error=False,
                corrected=original,
                explanation="(malformed JSON from model — no correction applied)",
                raw_response=raw_text,
            )

        return Correction(
            original=original,
            has_error=bool(data.get("has_error", False)),
            corrected=str(data.get("corrected", original)),
            explanation=str(data.get("explanation", "")),
            raw_response=raw_text,
        )