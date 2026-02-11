"""
GloVe Embeddings Module

Provides support for loading and using the NRCan domain-specific GloVe
embeddings trained on geoscience corpora.

The preferred GloVe model was trained using the AdaGrad algorithm with:
- Minimum token frequency: 5
- Context window size: 15
- Number of iterations: 15
- Fixed weighting functions: x_max=10, alpha=0.75
- 300-dimensional vectors

References:
    - Lawley, C.J.M., et al. (2022). Geoscience language models and their
      intrinsic evaluation. Applied Computing and Geosciences, 14, 100084.
    - Raimondo, S., et al. (2022). Datasets to support geoscience language
      models. Geological Survey of Canada, Open File 8848.
    - Pennington, J., et al. (2014). GloVe: Global Vectors for Word
      Representation. EMNLP 2014.
    - NRCan/Geoscience_Language_Models: https://github.com/NRCan/Geoscience_Language_Models
"""

from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
import numpy as np


# Default GloVe configuration matching the NRCan geoscience model
GLOVE_CONFIG = {
    "vector_size": 300,
    "window_size": 15,
    "iterations": 15,
    "min_count": 5,
    "x_max": 10,
    "alpha": 0.75,
}


class GeoscienceGloVeEmbeddings:
    """
    Load and use NRCan domain-specific GloVe embeddings.

    The geoscience GloVe model provides 300-dimensional word vectors
    retrained on 27,000+ GEOSCAN documents plus additional provincial
    geological survey publications.

    The model can be loaded from:
    1. A GloVe-format text file (word vec1 vec2 ... vec300)
    2. A gensim KeyedVectors file (.kv or .bin)
    3. The NRCan CSV file (Geoscience_GloVe_Embedding_Minerals.csv)

    Reference:
        NRCan/Geoscience_Language_Models repository contains the GloVe
        training scripts (project_tools/embeddings/glove/) and the mineral
        embeddings CSV (Geoscience_GloVe_Embedding_Minerals.csv).
    """

    def __init__(self, vector_size: int = 300):
        """
        Initialize the GloVe embeddings handler.

        Args:
            vector_size: Dimension of the word vectors (300 for NRCan model)
        """
        self.vector_size = vector_size
        self.word_vectors: Optional[Dict[str, np.ndarray]] = None
        self._keyed_vectors = None

    def load_from_text_file(self, filepath: str) -> None:
        """
        Load GloVe vectors from a text file (standard GloVe format).

        Each line: word val1 val2 ... valN

        Args:
            filepath: Path to GloVe text file
        """
        self.word_vectors = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < self.vector_size + 1:
                    continue
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:self.vector_size + 1]])
                self.word_vectors[word] = vector

    def load_from_keyed_vectors(self, filepath: str) -> None:
        """
        Load embeddings from a gensim KeyedVectors file.

        This is the format used by the NRCan repository for storing
        and querying embeddings (functions/glove.py, functions/bert.py).

        Args:
            filepath: Path to gensim KeyedVectors file (.kv or .bin)

        Raises:
            ImportError: If gensim is not installed
        """
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError(
                "gensim is required for KeyedVectors loading. "
                "Install with: pip install gensim"
            )
        self._keyed_vectors = KeyedVectors.load(filepath)
        self.vector_size = self._keyed_vectors.vector_size
        # Also populate the dict for direct access
        self.word_vectors = {
            word: self._keyed_vectors[word]
            for word in self._keyed_vectors.key_to_index
        }

    def load_from_nrcan_csv(self, filepath: str) -> None:
        """
        Load mineral embeddings from the NRCan CSV file.

        The NRCan repository provides Geoscience_GloVe_Embedding_Minerals.csv
        with 300-d vectors for ~1893 mineral names.

        Args:
            filepath: Path to the NRCan mineral embeddings CSV
        """
        import pandas as pd
        df = pd.read_csv(filepath, index_col=0)
        self.word_vectors = {}
        self.vector_size = df.shape[1]
        for word in df.index:
            self.word_vectors[str(word).lower()] = df.loc[word].values.astype(np.float32)

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get the embedding vector for a single word.

        Args:
            word: Word to look up

        Returns:
            Vector array of shape (vector_size,) or None if not found
        """
        if self.word_vectors is None:
            raise RuntimeError("No embeddings loaded. Call a load_from_* method first.")
        return self.word_vectors.get(word.lower())

    def embed_text(
        self,
        text: str,
        aggregation: str = "mean"
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Compute a text embedding by aggregating word vectors.

        This follows the approach used in the NRCan repository
        (functions/utils.py: embed_text).

        Args:
            text: Input text string
            aggregation: How to combine word vectors - "mean" or "sum"

        Returns:
            Tuple of (embedding vector, list of words not in vocabulary)
        """
        if self.word_vectors is None:
            raise RuntimeError("No embeddings loaded. Call a load_from_* method first.")

        words = text.lower().split()
        vectors = []
        not_in_vocab = []

        for word in words:
            vec = self.word_vectors.get(word)
            if vec is not None:
                vectors.append(vec)
            else:
                not_in_vocab.append(word)

        if not vectors:
            return None, not_in_vocab

        vectors_array = np.stack(vectors)
        if aggregation == "mean":
            embedding = vectors_array.mean(axis=0)
        elif aggregation == "sum":
            embedding = vectors_array.sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}. Use 'mean' or 'sum'.")

        return embedding, not_in_vocab

    def find_similar(
        self,
        word_or_vector: Union[str, np.ndarray],
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar words to a given word or vector.

        Uses cosine similarity, matching the NRCan repository's
        find_similar_keyword_in_vocab function.

        Args:
            word_or_vector: Word string or embedding vector
            top_n: Number of similar words to return

        Returns:
            List of (word, similarity_score) tuples
        """
        if self.word_vectors is None:
            raise RuntimeError("No embeddings loaded. Call a load_from_* method first.")

        if isinstance(word_or_vector, str):
            query_vec = self.word_vectors.get(word_or_vector.lower())
            if query_vec is None:
                return []
        else:
            query_vec = word_or_vector

        # Compute cosine similarities
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-10:
            return []

        similarities = []
        for word, vec in self.word_vectors.items():
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                continue
            sim = np.dot(query_vec, vec) / (query_norm * vec_norm)
            similarities.append((word, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)

        # Skip the query word itself if present
        if isinstance(word_or_vector, str):
            similarities = [
                (w, s) for w, s in similarities
                if w != word_or_vector.lower()
            ]

        return similarities[:top_n]

    @property
    def vocab_size(self) -> int:
        """Number of words in the vocabulary."""
        if self.word_vectors is None:
            return 0
        return len(self.word_vectors)
