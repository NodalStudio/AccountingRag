"""Encodeur dense configurable. Défaut : multilingual-e5-small (384 dims, léger)."""
import os

_DEFAULT = "intfloat/multilingual-e5-small"


class Embedder:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # import paresseux (lourd)

        self.model_name = model_name or os.environ.get("ACCRAG_EMB_MODEL", _DEFAULT)
        self._model = SentenceTransformer(self.model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self._e5 = "e5" in self.model_name.lower()

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts] if self._e5 else texts
        return self._model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False).tolist()

    def encode_query(self, text: str) -> list[float]:
        prefixed = f"query: {text}" if self._e5 else text
        return self._model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
