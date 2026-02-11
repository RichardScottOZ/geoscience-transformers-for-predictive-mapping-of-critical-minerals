"""
Basic Usage Example

This script demonstrates the basic usage of the geoscience transformers package.
"""

import numpy as np
import pandas as pd
from geoscience_transformers import (
    GeoscienceTextProcessor,
    MultimodalDataIntegrator,
    GeoscienceTransformer,
    SelfSupervisedTrainer
)

def main():
    print("=" * 60)
    print("Geoscience Transformers - Basic Usage Example")
    print("=" * 60)
    
    # 1. Text Processing
    print("\n1. Processing Geoscience Text...")
    processor = GeoscienceTextProcessor(model_name="bert-base-uncased")
    
    sample_texts = [
        "The lithium deposit shows high-grade mineralization with associated cobalt.",
        "Copper and gold mineralization occurs in altered volcanic rocks.",
        "Rare earth elements are enriched in the carbonatite intrusion."
    ]
    
    for text in sample_texts:
        clean_text = processor.preprocess_text(text)
        minerals = processor.extract_mineral_mentions(text)
        print(f"  Text: {text[:50]}...")
        print(f"  Minerals found: {', '.join(minerals)}")
    
    # Extract embeddings
    embeddings = processor.extract_features(sample_texts)
    print(f"\n  Extracted embeddings shape: {embeddings.shape}")
    
    # 2. Data Integration
    print("\n2. Integrating Multimodal Data...")
    
    # Create sample data
    n_samples = 100
    sample_data = pd.DataFrame({
        'x': np.random.uniform(0, 1000, n_samples),
        'y': np.random.uniform(0, 1000, n_samples),
        'Cu': np.random.lognormal(0, 1, n_samples),
        'Au': np.random.lognormal(-2, 1, n_samples),
        'Ag': np.random.lognormal(-1, 1, n_samples),
        'magnetic_intensity': np.random.normal(50000, 1000, n_samples),
        'gravity': np.random.normal(9.81, 0.01, n_samples)
    })
    
    integrator = MultimodalDataIntegrator()
    
    # Add features
    sample_data = integrator.add_geochemical_features(
        sample_data, 
        element_columns=['Cu', 'Au', 'Ag']
    )
    sample_data = integrator.add_geophysical_features(
        sample_data,
        geophysical_columns=['magnetic_intensity', 'gravity']
    )
    sample_data = integrator.add_spatial_features(sample_data)
    
    print(f"  Total features created: {len(sample_data.columns)}")
    print(f"  Sample data shape: {sample_data.shape}")
    
    # 3. Model Creation
    print("\n3. Creating Transformer Model...")
    
    # Prepare features
    feature_columns = [col for col in sample_data.columns 
                      if col not in ['x', 'y']]
    X, _ = integrator.prepare_features(sample_data, feature_columns)
    
    # Create model
    model = GeoscienceTransformer(
        input_dim=X.shape[1],
        hidden_dim=128,
        num_layers=2,
        num_heads=4
    )
    
    print(f"  Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 4. Self-Supervised Training Example
    print("\n4. Self-Supervised Training (Demo)...")
    
    from torch.utils.data import DataLoader
    from geoscience_transformers.training import GeoscienceDataset
    
    dataset = GeoscienceDataset(X)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    trainer = SelfSupervisedTrainer(model, learning_rate=1e-4)
    
    print("  Training for 2 epochs (demo)...")
    history = trainer.fit(dataloader, num_epochs=2)
    
    print(f"  Final training loss: {history['train_loss'][-1]:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
