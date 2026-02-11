"""
Tests for training module - masked value prediction and pseudo-labeling.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from geoscience_transformers.models import (
    GeoscienceTransformer,
    TabTransformer,
    SelfSupervisedWrapper,
)
from geoscience_transformers.training import (
    GeoscienceDataset,
    SelfSupervisedTrainer,
    PseudoLabelTrainer,
)


def test_masked_input_creation():
    """Test that masked inputs are created correctly."""
    model = GeoscienceTransformer(input_dim=10, hidden_dim=32, num_layers=1)
    wrapper = SelfSupervisedWrapper(model)
    trainer = SelfSupervisedTrainer(wrapper, mask_ratio=0.3)

    features = torch.randn(8, 10)
    masked, mask, original = trainer.create_masked_input(features)

    # Shapes should match
    assert masked.shape == features.shape
    assert mask.shape == features.shape
    assert original.shape == features.shape

    # Masked positions should be zero
    assert torch.all(masked[mask] == 0.0)

    # Unmasked positions should be unchanged
    assert torch.allclose(masked[~mask], original[~mask])


def test_masked_value_prediction_loss():
    """Test masked value prediction loss computation."""
    model = GeoscienceTransformer(input_dim=10, hidden_dim=32, num_layers=1)
    wrapper = SelfSupervisedWrapper(model)
    trainer = SelfSupervisedTrainer(wrapper, mask_ratio=0.3)

    # Create dummy embeddings and features
    embeddings = torch.randn(4, 32)
    mask = torch.zeros(4, 10, dtype=torch.bool)
    mask[:, :3] = True  # Mask first 3 features
    original = torch.randn(4, 10)

    loss = trainer.masked_value_prediction_loss(embeddings, mask, original)

    assert not torch.isnan(loss)
    assert loss.item() >= 0


def test_self_supervised_trainer_with_mask_prediction():
    """Test that self-supervised training includes masked value prediction."""
    model = GeoscienceTransformer(input_dim=10, hidden_dim=32, num_layers=1)
    wrapper = SelfSupervisedWrapper(model, projection_dim=16)
    trainer = SelfSupervisedTrainer(wrapper, mask_ratio=0.15)

    # Create dataset
    data = np.random.randn(32, 10).astype(np.float32)
    dataset = GeoscienceDataset(data)
    loader = DataLoader(dataset, batch_size=8)

    # Train for 1 epoch
    loss = trainer.train_epoch(loader, epoch=0)
    assert loss > 0
    assert not np.isnan(loss)


def test_self_supervised_with_tab_transformer():
    """Test self-supervised training with TabTransformer."""
    model = TabTransformer(num_features=10, hidden_dim=32, num_layers=1, num_heads=4)
    wrapper = SelfSupervisedWrapper(model, projection_dim=16)
    trainer = SelfSupervisedTrainer(wrapper, mask_ratio=0.15)

    data = np.random.randn(16, 10).astype(np.float32)
    dataset = GeoscienceDataset(data)
    loader = DataLoader(dataset, batch_size=8)

    loss = trainer.train_epoch(loader, epoch=0)
    assert not np.isnan(loss)


def test_pseudo_label_trainer_init():
    """Test PseudoLabelTrainer initialization."""
    model = GeoscienceTransformer(input_dim=10, hidden_dim=32, num_layers=1)
    trainer = PseudoLabelTrainer(
        model,
        confidence_threshold=0.9,
    )
    assert trainer.confidence_threshold == 0.9


def test_pseudo_label_generation():
    """Test pseudo-label generation."""
    model = GeoscienceTransformer(input_dim=10, hidden_dim=32, num_layers=1)
    trainer = PseudoLabelTrainer(model, confidence_threshold=0.5)

    # Create unlabeled dataset
    data = np.random.randn(32, 10).astype(np.float32)
    dataset = GeoscienceDataset(data)
    loader = DataLoader(dataset, batch_size=16)

    features, pseudo_labels, mask = trainer.generate_pseudo_labels(loader)

    assert features.shape[0] == 32
    assert pseudo_labels.shape[0] == 32
    assert mask.shape[0] == 32
    # With threshold=0.5, all samples should be above threshold
    # (sigmoid output is always >= 0 or <= 1)


def test_pseudo_label_trainer_fit():
    """Test PseudoLabelTrainer training loop."""
    model = GeoscienceTransformer(input_dim=10, hidden_dim=32, num_layers=1)
    trainer = PseudoLabelTrainer(model, confidence_threshold=0.9)

    # Create labeled dataset
    labeled_features = np.random.randn(16, 10).astype(np.float32)
    labeled_labels = np.random.randint(0, 2, 16).astype(np.float32)
    labeled_dataset = GeoscienceDataset(labeled_features, labeled_labels)
    labeled_loader = DataLoader(labeled_dataset, batch_size=8)

    # Create unlabeled dataset
    unlabeled_features = np.random.randn(32, 10).astype(np.float32)
    unlabeled_dataset = GeoscienceDataset(unlabeled_features)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16)

    # Train for 1 pseudo-labeling round
    history = trainer.fit(
        labeled_loader,
        unlabeled_loader,
        num_epochs=1,
        num_pseudo_rounds=2,
    )

    assert "train_loss" in history
    assert "pseudo_count" in history
    assert len(history["train_loss"]) >= 1
    assert len(history["pseudo_count"]) == 2


def test_mask_ratio_parameter():
    """Test that mask_ratio parameter is used correctly."""
    model = GeoscienceTransformer(input_dim=100, hidden_dim=32, num_layers=1)
    wrapper = SelfSupervisedWrapper(model)

    # Test with different mask ratios
    trainer_low = SelfSupervisedTrainer(wrapper, mask_ratio=0.1)
    trainer_high = SelfSupervisedTrainer(wrapper, mask_ratio=0.5)

    features = torch.randn(1000, 100)

    _, mask_low, _ = trainer_low.create_masked_input(features)
    _, mask_high, _ = trainer_high.create_masked_input(features)

    # Higher mask ratio should mask more features on average
    assert mask_high.float().mean() > mask_low.float().mean()
