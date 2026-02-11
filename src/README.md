# Geoscience Transformers Package

This directory contains the complete implementation of the geoscience transformers package for predictive mapping of critical minerals.

## Package Structure

```
src/geoscience_transformers/
├── __init__.py              # Package initialization
├── text_processing.py       # Text mining and NLP
├── data_integration.py      # Multimodal data integration
├── models.py                # Transformer models
├── training.py              # Training utilities
├── prediction.py            # Prediction and visualization
├── cli.py                   # Command-line interface
└── utils.py                 # Utility functions
```

## Core Modules

### text_processing.py
- GeoscienceTextProcessor: Extract features from geoscience documents
- Text preprocessing and cleaning
- Named entity recognition for minerals
- LLM-based feature extraction

### data_integration.py
- MultimodalDataIntegrator: Integrate diverse data sources
- Geochemical feature engineering
- Geophysical feature engineering
- Spatial feature engineering
- Feature normalization and scaling

### models.py
- GeoscienceTransformer: Core transformer architecture
- ProspectivityModel: Complete prospectivity mapping model
- SelfSupervisedWrapper: Self-supervised learning wrapper

### training.py
- SelfSupervisedTrainer: Train with unlabeled data
- SupervisedTrainer: Train with labeled data
- GeoscienceDataset: Dataset class for geoscience data

### prediction.py
- ProspectivityPredictor: Make predictions and visualizations
- Uncertainty estimation
- Map generation
- High-potential area identification

### cli.py
- Command-line interface for training and prediction
- Configuration file support
- Batch processing

### utils.py
- Helper functions
- Data splitting
- Configuration management
- Random seed setting

## Development

To work on the package:

1. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

2. Run tests:
   ```bash
   pytest tests/
   ```

3. Format code:
   ```bash
   black src/
   ```

4. Type checking:
   ```bash
   mypy src/
   ```
