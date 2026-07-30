
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
