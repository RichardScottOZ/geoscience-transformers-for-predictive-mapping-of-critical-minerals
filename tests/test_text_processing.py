"""
Tests for text processing module.
"""

import pytest
import numpy as np
from geoscience_transformers.text_processing import GeoscienceTextProcessor


def test_text_processor_initialization():
    """Test that text processor initializes correctly."""
    processor = GeoscienceTextProcessor(model_name="bert-base-uncased")
    assert processor.model_name == "bert-base-uncased"
    assert processor.max_length == 512
    assert processor.device in ["cuda", "cpu"]


def test_preprocess_text():
    """Test text preprocessing."""
    processor = GeoscienceTextProcessor()
    
    text = "The LITHIUM deposit at http://example.com contains 5% Li2O!"
    clean_text = processor.preprocess_text(text, lowercase=True)
    
    # Should be lowercase
    assert clean_text.islower()
    # Should not contain URLs
    assert "http" not in clean_text
    # Should not contain multiple spaces
    assert "  " not in clean_text


def test_extract_mineral_mentions():
    """Test mineral extraction."""
    processor = GeoscienceTextProcessor()
    
    text = "The deposit contains lithium, cobalt, and rare earth elements."
    minerals = processor.extract_mineral_mentions(text)
    
    assert "lithium" in minerals
    assert "cobalt" in minerals
    assert "rare earth" in minerals


def test_extract_features():
    """Test feature extraction."""
    processor = GeoscienceTextProcessor()
    
    texts = ["Lithium deposit", "Copper mineralization"]
    embeddings = processor.extract_features(texts)
    
    # Check shape
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0  # Should have embedding dimensions
    
    # Check type
    assert isinstance(embeddings, np.ndarray)


def test_process_document_batch():
    """Test batch document processing."""
    processor = GeoscienceTextProcessor()
    
    documents = [
        {"text": "Lithium deposit with high grades"},
        {"text": "Copper and gold mineralization"}
    ]
    
    result = processor.process_document_batch(documents)
    
    assert len(result) == len(documents)
    assert "clean_text" in result.columns
    assert "embedding" in result.columns
    assert "minerals_mentioned" in result.columns
