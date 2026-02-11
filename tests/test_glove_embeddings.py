"""
Tests for GloVe embeddings module.
"""

import pytest
import numpy as np
import os
import tempfile
from geoscience_transformers.glove_embeddings import GeoscienceGloVeEmbeddings, GLOVE_CONFIG


def test_glove_config():
    """Test that default GloVe config matches NRCan specs."""
    assert GLOVE_CONFIG["vector_size"] == 300
    assert GLOVE_CONFIG["window_size"] == 15
    assert GLOVE_CONFIG["iterations"] == 15
    assert GLOVE_CONFIG["min_count"] == 5
    assert GLOVE_CONFIG["x_max"] == 10
    assert GLOVE_CONFIG["alpha"] == 0.75


def test_glove_init():
    """Test GloVe embeddings initialization."""
    emb = GeoscienceGloVeEmbeddings(vector_size=300)
    assert emb.vector_size == 300
    assert emb.word_vectors is None
    assert emb.vocab_size == 0


def test_glove_load_from_text_file():
    """Test loading GloVe vectors from text file."""
    # Create a temporary GloVe file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        # Write 3 words with 10-dimensional vectors
        for word in ['lithium', 'copper', 'gold']:
            vec = ' '.join([f'{v:.4f}' for v in np.random.randn(10)])
            f.write(f'{word} {vec}\n')
        tmp_path = f.name

    try:
        emb = GeoscienceGloVeEmbeddings(vector_size=10)
        emb.load_from_text_file(tmp_path)

        assert emb.vocab_size == 3
        assert emb.get_word_vector('lithium') is not None
        assert emb.get_word_vector('copper') is not None
        assert emb.get_word_vector('gold') is not None
        assert emb.get_word_vector('unknown_word') is None

        # Check vector dimension
        vec = emb.get_word_vector('lithium')
        assert vec.shape == (10,)
    finally:
        os.unlink(tmp_path)


def test_glove_embed_text():
    """Test text embedding via word vector aggregation."""
    emb = GeoscienceGloVeEmbeddings(vector_size=5)
    emb.word_vectors = {
        'lithium': np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        'deposit': np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        'rich': np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    }

    # Test mean aggregation
    vec, oov = emb.embed_text("lithium deposit rich", aggregation="mean")
    assert vec is not None
    np.testing.assert_array_almost_equal(vec, [1/3, 1/3, 1/3, 0, 0])
    assert len(oov) == 0

    # Test with OOV words
    vec, oov = emb.embed_text("lithium unknown word", aggregation="mean")
    assert vec is not None
    assert 'unknown' in oov
    assert 'word' in oov

    # Test sum aggregation
    vec, oov = emb.embed_text("lithium deposit", aggregation="sum")
    np.testing.assert_array_almost_equal(vec, [1.0, 1.0, 0.0, 0.0, 0.0])


def test_glove_embed_text_all_oov():
    """Test text embedding when all words are out of vocabulary."""
    emb = GeoscienceGloVeEmbeddings(vector_size=5)
    emb.word_vectors = {'lithium': np.zeros(5)}

    vec, oov = emb.embed_text("unknown words only")
    assert vec is None
    assert len(oov) == 3


def test_glove_find_similar():
    """Test finding similar words."""
    emb = GeoscienceGloVeEmbeddings(vector_size=3)
    emb.word_vectors = {
        'lithium': np.array([1.0, 0.0, 0.0]),
        'copper': np.array([0.9, 0.1, 0.0]),
        'gold': np.array([0.0, 0.0, 1.0]),
    }

    similar = emb.find_similar('lithium', top_n=2)
    assert len(similar) == 2
    # Copper should be most similar to lithium
    assert similar[0][0] == 'copper'
    assert similar[0][1] > 0.9  # High similarity


def test_glove_find_similar_by_vector():
    """Test finding similar words by vector."""
    emb = GeoscienceGloVeEmbeddings(vector_size=3)
    emb.word_vectors = {
        'lithium': np.array([1.0, 0.0, 0.0]),
        'copper': np.array([0.9, 0.1, 0.0]),
    }

    query = np.array([1.0, 0.0, 0.0])
    similar = emb.find_similar(query, top_n=2)
    assert len(similar) == 2
    assert similar[0][0] == 'lithium'


def test_glove_load_not_initialized():
    """Test error when trying to use embeddings before loading."""
    emb = GeoscienceGloVeEmbeddings()
    with pytest.raises(RuntimeError, match="No embeddings loaded"):
        emb.get_word_vector('test')
    with pytest.raises(RuntimeError, match="No embeddings loaded"):
        emb.embed_text('test')


def test_glove_load_from_nrcan_csv():
    """Test loading from NRCan-style CSV file."""
    import pandas as pd

    # Create a temporary CSV with mineral embeddings
    data = {
        'dim_0': [0.1, 0.4, 0.7],
        'dim_1': [0.2, 0.5, 0.8],
        'dim_2': [0.3, 0.6, 0.9],
    }
    df = pd.DataFrame(data, index=['quartz', 'feldspar', 'mica'])

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f)
        tmp_path = f.name

    try:
        emb = GeoscienceGloVeEmbeddings()
        emb.load_from_nrcan_csv(tmp_path)

        assert emb.vocab_size == 3
        assert emb.vector_size == 3
        vec = emb.get_word_vector('quartz')
        assert vec is not None
        np.testing.assert_array_almost_equal(vec, [0.1, 0.2, 0.3])
    finally:
        os.unlink(tmp_path)
