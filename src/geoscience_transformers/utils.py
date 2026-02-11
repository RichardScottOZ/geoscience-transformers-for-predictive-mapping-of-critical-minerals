"""
Utility functions for the geoscience transformers package.
"""

import numpy as np
from typing import List, Tuple, Optional
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Array of labels
        
    Returns:
        Tensor of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return torch.FloatTensor(weights)


def split_data(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Feature array
        labels: Optional labels array
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of split data
    """
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # First split: train and temp (val + test)
    if labels is not None:
        X_train, X_temp, y_train, y_temp = train_test_split(
            data, labels,
            test_size=(1 - train_ratio),
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: val and test
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=relative_test_ratio,
            random_state=random_state,
            stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train, X_temp = train_test_split(
            data,
            test_size=(1 - train_ratio),
            random_state=random_state
        )
        
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        X_val, X_test = train_test_split(
            X_temp,
            test_size=relative_test_ratio,
            random_state=random_state
        )
        
        return X_train, X_val, X_test


def save_config(config: dict, path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        path: Output file path
    """
    import yaml
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        path: Config file path
        
    Returns:
        Configuration dictionary
    """
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)
