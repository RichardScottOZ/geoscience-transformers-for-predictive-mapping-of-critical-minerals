"""
Tests for model module.
"""

import pytest
import torch
from geoscience_transformers.models import (
    GeoscienceTransformer, 
    ProspectivityModel,
    SelfSupervisedWrapper,
    TabTransformer,
    FTTransformer,
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


def test_tab_transformer():
    """Test TabTransformer for tabular data."""
    num_features = 20
    model = TabTransformer(
        num_features=num_features,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        output_dim=1
    )

    # Test forward pass
    x = torch.randn(8, num_features)
    output = model(x)

    assert output.shape == (8, 1)

    # Test get_embeddings (mean-pooled across columns)
    embeddings = model.get_embeddings(x)
    assert embeddings.shape == (8, 64)  # Mean-pooled to hidden_dim


def test_tab_transformer_column_attention():
    """Test that TabTransformer uses column-wise self-attention."""
    model = TabTransformer(
        num_features=5,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
    )

    x = torch.randn(4, 5)
    output = model(x)
    assert output.shape == (4, 1)

    # Verify column_type_embedding has correct shape
    assert model.column_type_embedding.shape == (5, 32)


def test_ft_transformer():
    """Test FT-Transformer with [CLS] token."""
    num_features = 15
    model = FTTransformer(
        num_features=num_features,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        output_dim=1
    )

    # Test forward pass
    x = torch.randn(8, num_features)
    output = model(x)

    assert output.shape == (8, 1)

    # Test get_embeddings returns [CLS] token representation
    embeddings = model.get_embeddings(x)
    assert embeddings.shape == (8, 64)  # CLS token dim == hidden_dim


def test_ft_transformer_cls_token():
    """Test FT-Transformer [CLS] token and position embeddings."""
    model = FTTransformer(
        num_features=10,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
    )

    # CLS token should be (1, 1, hidden_dim)
    assert model.cls_token.shape == (1, 1, 32)

    # Position embedding covers CLS + all features
    assert model.position_embedding.shape == (1, 11, 32)  # 10 features + 1 CLS


def test_tab_transformer_self_supervised():
    """Test TabTransformer with SelfSupervisedWrapper."""
    base_model = TabTransformer(
        num_features=10,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
    )

    wrapper = SelfSupervisedWrapper(base_model, projection_dim=16)

    # get_embeddings should work through wrapper
    x = torch.randn(4, 10)
    emb = wrapper.get_embeddings(x)
    assert emb.shape[0] == 4

    # Contrastive loss should work with TabTransformer embeddings
    z1 = wrapper.get_embeddings(torch.randn(4, 10))
    z2 = wrapper.get_embeddings(torch.randn(4, 10))
    loss = wrapper.contrastive_loss(z1, z2)
    assert not torch.isnan(loss)
