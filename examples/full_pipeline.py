"""
Full Pipeline Example

This example demonstrates a complete prospectivity mapping pipeline
from data preparation to prediction and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_synthetic_data(n_samples=500):
    """Create synthetic geoscience data for demonstration."""
    print("Creating synthetic data...")
    
    # Spatial coordinates
    x = np.random.uniform(0, 10000, n_samples)
    y = np.random.uniform(0, 10000, n_samples)
    
    # Geochemical data (log-normal distributions)
    Cu = np.random.lognormal(0, 1, n_samples)
    Au = np.random.lognormal(-2, 1, n_samples)
    Ag = np.random.lognormal(-1, 1, n_samples)
    Pb = np.random.lognormal(0.5, 0.8, n_samples)
    Zn = np.random.lognormal(1, 0.9, n_samples)
    
    # Geophysical data
    magnetic = np.random.normal(50000, 5000, n_samples)
    gravity = np.random.normal(9.81, 0.05, n_samples)
    
    # Synthetic labels (deposit presence)
    # Higher values more likely to have deposits
    deposit_prob = (
        (Cu > np.percentile(Cu, 75)) & 
        (Au > np.percentile(Au, 75)) |
        (magnetic > np.percentile(magnetic, 70))
    ).astype(float)
    
    # Add some noise
    labels = (deposit_prob + np.random.uniform(0, 0.3, n_samples)) > 0.6
    
    data = pd.DataFrame({
        'x': x,
        'y': y,
        'Cu': Cu,
        'Au': Au,
        'Ag': Ag,
        'Pb': Pb,
        'Zn': Zn,
        'magnetic_intensity': magnetic,
        'gravity': gravity,
        'deposit_present': labels.astype(int)
    })
    
    print(f"  Created {n_samples} samples")
    print(f"  Deposit ratio: {labels.mean():.2%}")
    
    return data


def main():
    print("=" * 70)
    print("Geoscience Transformers - Full Pipeline Example")
    print("=" * 70)
    
    # 1. Data Preparation
    print("\n1. DATA PREPARATION")
    print("-" * 70)
    
    data = create_synthetic_data(n_samples=500)
    
    # Split data
    from geoscience_transformers.utils import split_data
    
    feature_cols = ['Cu', 'Au', 'Ag', 'Pb', 'Zn', 'magnetic_intensity', 'gravity']
    X = data[feature_cols].values
    y = data['deposit_present'].values
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    
    # 2. Feature Engineering
    print("\n2. FEATURE ENGINEERING")
    print("-" * 70)
    
    from geoscience_transformers import MultimodalDataIntegrator
    
    integrator = MultimodalDataIntegrator()
    
    # Add engineered features to training data
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df = integrator.add_geochemical_features(
        train_df, 
        element_columns=['Cu', 'Au', 'Ag', 'Pb', 'Zn']
    )
    
    # Prepare features
    all_feature_cols = [col for col in train_df.columns if col in feature_cols or '_' in col]
    X_train_proc, _ = integrator.prepare_features(train_df, all_feature_cols, fit=True)
    
    # Apply same transformations to validation and test
    val_df = pd.DataFrame(X_val, columns=feature_cols)
    val_df = integrator.add_geochemical_features(
        val_df,
        element_columns=['Cu', 'Au', 'Ag', 'Pb', 'Zn']
    )
    X_val_proc, _ = integrator.prepare_features(val_df, all_feature_cols, fit=False)
    
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df = integrator.add_geochemical_features(
        test_df,
        element_columns=['Cu', 'Au', 'Ag', 'Pb', 'Zn']
    )
    X_test_proc, _ = integrator.prepare_features(test_df, all_feature_cols, fit=False)
    
    print(f"  Engineered features: {X_train_proc.shape[1]}")
    
    # 3. Model Training
    print("\n3. MODEL TRAINING")
    print("-" * 70)
    
    from geoscience_transformers import GeoscienceTransformer
    from geoscience_transformers.training import SupervisedTrainer, GeoscienceDataset
    from torch.utils.data import DataLoader
    
    # Create model
    model = GeoscienceTransformer(
        input_dim=X_train_proc.shape[1],
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    train_dataset = GeoscienceDataset(X_train_proc, y_train)
    val_dataset = GeoscienceDataset(X_val_proc, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train
    trainer = SupervisedTrainer(model, learning_rate=1e-3)
    
    print("  Training model (5 epochs)...")
    history = trainer.fit(
        train_loader, 
        val_loader,
        num_epochs=5,
        checkpoint_path="/tmp/best_model.pt"
    )
    
    # 4. Evaluation
    print("\n4. EVALUATION")
    print("-" * 70)
    
    from geoscience_transformers import ProspectivityPredictor
    
    predictor = ProspectivityPredictor(model)
    
    # Make predictions
    test_preds = predictor.predict(X_test_proc)
    
    # Calculate metrics
    metrics = predictor.calculate_metrics(test_preds, y_test)
    
    print("  Test Set Metrics:")
    for metric, value in metrics.items():
        print(f"    {metric.upper():12s}: {value:.4f}")
    
    # 5. Visualization
    print("\n5. VISUALIZATION")
    print("-" * 70)
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy History')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/training_history.png', dpi=150, bbox_inches='tight')
    print("  Saved training history plot to /tmp/training_history.png")
    
    # Plot ROC curve
    fig = predictor.plot_roc_curve(test_preds, y_test, output_path='/tmp/roc_curve.png')
    print("  Saved ROC curve to /tmp/roc_curve.png")
    
    # 6. Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nOutput files:")
    print("  - /tmp/best_model.pt - Trained model checkpoint")
    print("  - /tmp/training_history.png - Training curves")
    print("  - /tmp/roc_curve.png - ROC curve")
    print("\nFinal test accuracy: {:.2%}".format(metrics['accuracy']))
    print("Final test AUC: {:.4f}".format(metrics['roc_auc']))
    

if __name__ == "__main__":
    # Set random seed for reproducibility
    from geoscience_transformers.utils import set_seed
    set_seed(42)
    
    main()
