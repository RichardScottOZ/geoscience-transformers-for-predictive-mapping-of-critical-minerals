"""
Tests for model module.
"""

import pytest
import torch
from geoscience_transformers.models import (
    GeoscienceTransformer, 
    ProspectivityModel,
    SelfSupervisedWrapper
)


def test_geoscience_transformer():
    """Test GeoscienceTransformer model."""
    model = GeoscienceTransformer(
        input_dim=50,
        hidden_dim=128,
        num_layers=2,
        num_heads=4
    )
    
    # Test forward pass
    x = torch.randn(8, 50)  # batch_size=8, input_dim=50
    output = model(x)
    
    assert output.shape == (8, 1)  # Default output_dim=1


def test_prospectivity_model():
    """Test ProspectivityModel."""
    model = ProspectivityModel(
        geochemical_dim=30,
        geophysical_dim=10,
        text_embedding_dim=768,
        hidden_dim=128,
        num_layers=2
    )
    
    # Test with all modalities
    geochemical = torch.randn(4, 30)
    geophysical = torch.randn(4, 10)
    text = torch.randn(4, 768)
    
    output = model(geochemical=geochemical, geophysical=geophysical, text=text)
    
    assert output.shape == (4, 1)


def test_prospectivity_model_partial():
    """Test ProspectivityModel with partial inputs."""
    model = ProspectivityModel(
        geochemical_dim=30,
        geophysical_dim=10,
        text_embedding_dim=768,
        hidden_dim=128,
        use_text=False
    )
    
    # Test with only geochemical and geophysical
    geochemical = torch.randn(4, 30)
    geophysical = torch.randn(4, 10)
    
    output = model(geochemical=geochemical, geophysical=geophysical)
    
    assert output.shape == (4, 1)


def test_self_supervised_wrapper():
    """Test SelfSupervisedWrapper."""
    base_model = GeoscienceTransformer(
        input_dim=50,
        hidden_dim=128,
        num_layers=2
    )
    
    wrapper = SelfSupervisedWrapper(base_model, projection_dim=64)
    
    # Test contrastive loss
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)
    
    loss = wrapper.contrastive_loss(z1, z2)
    
    assert loss.item() >= 0  # Loss should be non-negative
    assert not torch.isnan(loss)
