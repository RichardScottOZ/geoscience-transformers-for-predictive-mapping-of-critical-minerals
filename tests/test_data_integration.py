"""
Tests for data integration module.
"""

import pytest
import numpy as np
import pandas as pd
from geoscience_transformers.data_integration import MultimodalDataIntegrator


def test_integrator_initialization():
    """Test integrator initialization."""
    integrator = MultimodalDataIntegrator(spatial_resolution=1000.0)
    assert integrator.spatial_resolution == 1000.0
    assert integrator.normalize is True
    assert not integrator.is_fitted


def test_add_geochemical_features():
    """Test geochemical feature engineering."""
    integrator = MultimodalDataIntegrator()
    
    data = pd.DataFrame({
        'Cu': [10, 20, 30],
        'Au': [0.1, 0.2, 0.3],
        'Ag': [1, 2, 3]
    })
    
    result = integrator.add_geochemical_features(data, ['Cu', 'Au', 'Ag'])
    
    # Check ratios were added
    assert 'Cu_Au_ratio' in result.columns
    # Check log transforms were added
    assert 'Cu_log' in result.columns


def test_add_spatial_features():
    """Test spatial feature engineering."""
    integrator = MultimodalDataIntegrator()
    
    data = pd.DataFrame({
        'x': [0, 100, 200],
        'y': [0, 100, 200]
    })
    
    result = integrator.add_spatial_features(data)
    
    assert 'distance_from_center' in result.columns
    assert 'spatial_density' in result.columns


def test_prepare_features():
    """Test feature preparation."""
    integrator = MultimodalDataIntegrator()
    
    data = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [4, 5, 6]
    })
    
    X, feature_names = integrator.prepare_features(data, ['feat1', 'feat2'], fit=True)
    
    assert X.shape == (3, 2)
    assert integrator.is_fitted
    assert feature_names == ['feat1', 'feat2']


def test_integrate_text_features():
    """Test text feature integration."""
    integrator = MultimodalDataIntegrator()
    
    data = pd.DataFrame({'x': [1, 2, 3]})
    embeddings = np.random.rand(3, 10)
    
    result = integrator.integrate_text_features(data, embeddings, prefix='text')
    
    # Check embeddings were added
    assert 'text_0' in result.columns
    assert 'text_9' in result.columns
    assert len(result) == 3
