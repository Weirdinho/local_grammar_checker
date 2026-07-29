"""
Optional semantic layer, using a local LaBSE sentence-embedding model.

Two jobs:

1. "Closest meaning" lexicon lookup (Tier 1). When a token isn't found by
   exact string lookup, instead of just flagging it as unknown, find the
   lexicon word whose embedding is closest to it. If that's close enough,
   treat the token as a plausible variant rather than hard-flagging it —
   and either way, surface the closest word as a "did you mean" suggestion
   grounded in meaning rather than spelling.

2. Meaning-drift guard (Tier 2). After N-ATLaS proposes a correction,
   compare the embedding of the original sentence to the corrected one.
   The existing word-diff "rephrase guard" (see diff_utils.edit_ratio)
   only measures surface-level change; two sentences can share almost all
   the same words and still mean something different, or look very
   different on paper and mean the same thing. This catches the case the
   surface-level guard can't.

Why LaBSE specifically: most multilingual sentence-embedding models (e.g.
paraphrase-multilingual-MiniLM) only cover ~50 higher-resource languages
and don't include Yoruba, Igbo, or Hausa at all. LaBSE's training set
explicitly includes all three (ISO codes yo, ig, ha) alongside 106 other
languages. That said, they're a small slice of LaBSE's training data
compared to languages like French or Spanish, so treat its similarity
scores as a useful heuristic to catch egregious cases, not a precise or
authoritative judgment of meaning.

This is heavier than the rest of the pipeline: sentence-transformers pulls
in torch + transformers, and the model itself is a ~1.8GB download from
Hugging Face Hub on first use (cached locally afterward — fully offline
from then on). It's imported lazily so Tier-1-only / non-semantic runs
never need any of this installed. See README "Semantic matching (optional)"
for setup, including a Windows-specific note on installing a CPU-only torch
build to avoid the DLL issues heavier ML installs can trigger there.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class SemanticMatcher:
    MODEL_NAME = "sentence-transformers/LaBSE"

    def __init__(self, cache_dir: str | Path = "resources/embedding_cache"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers isn't installed. Run "
                "`pip install sentence-transformers` (see README 'Semantic "
                "matching (optional)' for a Windows-safe install order) or "
                "set semantic.enabled: false in config.yaml."
            ) from e

        try:
            self.model = SentenceTransformer(self.MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Could not load {self.MODEL_NAME}. On first run this "
                "downloads ~1.8GB from Hugging Face Hub, so make sure you "
                "have an internet connection and enough disk space. "
                f"(underlying error: {e})"
            ) from e

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns L2-normalized embeddings, so dot product == cosine similarity."""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def similarity(self, a: str, b: str) -> float:
        vecs = self.embed([a, b])
        return float(np.dot(vecs[0], vecs[1]))

    def _lexicon_embeddings(self, lang_code: str, words: list[str]) -> np.ndarray:
        """Cached on disk, keyed by a hash of the wordlist so it's rebuilt
        automatically if you swap in a bigger/different lexicon file."""
        key = hashlib.sha1("\n".join(words).encode("utf-8")).hexdigest()[:12]
        cache_path = self.cache_dir / f"{lang_code}_{key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        vecs = self.embed(words)
        np.save(cache_path, vecs)
        return vecs

    def closest_lexicon_word(
        self, token: str, lang_code: str, lexicon_words: list[str]
    ) -> tuple[str, float] | None:
        """Returns (closest_word, cosine_similarity), or None if the lexicon is empty."""
        if not lexicon_words:
            return None
        lex_vecs = self._lexicon_embeddings(lang_code, lexicon_words)
        tok_vec = self.embed([token])[0]
        sims = lex_vecs @ tok_vec
        best_idx = int(np.argmax(sims))
        return lexicon_words[best_idx], float(sims[best_idx])
