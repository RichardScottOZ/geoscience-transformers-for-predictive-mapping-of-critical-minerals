# Geoscience Transformers for Predictive Mapping of Critical Minerals

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the paper: **"Large Language Models and Geoscience Transformers for Predictive Mapping of Canadian Critical Minerals"** (DOI: [10.1007/s11053-025-10564-0](https://link.springer.com/article/10.1007/s11053-025-10564-0))

This Python package provides tools for mineral prospectivity mapping using transformer-based deep learning models that integrate multimodal geoscience data including:
- Textual geoscience documents (reports, papers, databases)
- Geochemical data
- Geophysical data
- Satellite/remote sensing data

## Key Features

- **Text Processing**: Extract features from unstructured geoscience documents using domain-specific language models
- **Multimodal Data Integration**: Seamlessly combine text, geochemical, and geophysical data
- **Self-Supervised Learning**: Train models with limited labeled data using contrastive learning
- **Prospectivity Mapping**: Generate high-resolution mineral prospectivity maps
- **Uncertainty Quantification**: Estimate prediction uncertainty for risk assessment
- **Visualization Tools**: Create publication-ready maps and analysis plots

## Installation

### From Source

```bash
git clone https://github.com/RichardScottOZ/geoscience-transformers-for-predictive-mapping-of-critical-minerals.git
cd geoscience-transformers-for-predictive-mapping-of-critical-minerals
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Text Processing

Extract features from geoscience documents:

```python
from geoscience_transformers import GeoscienceTextProcessor

# Initialize processor with domain-specific model
processor = GeoscienceTextProcessor(model_name="bert-base-uncased")

# Process text
text = "The lithium deposit shows high-grade mineralization..."
clean_text = processor.preprocess_text(text)
embeddings = processor.extract_features(clean_text)
minerals = processor.extract_mineral_mentions(clean_text)
```

### 2. Data Integration

Integrate multiple data sources:

```python
from geoscience_transformers import MultimodalDataIntegrator
import pandas as pd

# Initialize integrator
integrator = MultimodalDataIntegrator()

# Add geochemical features
data = integrator.add_geochemical_features(
    data, 
    element_columns=['Cu', 'Au', 'Ag', 'Pb', 'Zn']
)

# Add geophysical features
data = integrator.add_geophysical_features(
    data,
    geophysical_columns=['magnetic_intensity', 'gravity']
)

# Prepare features for modeling
X, feature_names = integrator.prepare_features(data, feature_columns)
```

### 3. Model Training

Train a prospectivity model:

```python
from geoscience_transformers import ProspectivityModel, SupervisedTrainer
from geoscience_transformers.training import GeoscienceDataset
from torch.utils.data import DataLoader

# Create model
model = ProspectivityModel(
    geochemical_dim=50,
    geophysical_dim=20,
    text_embedding_dim=768,
    hidden_dim=256,
    num_layers=4
)

# Prepare data
dataset = GeoscienceDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
trainer = SupervisedTrainer(model, learning_rate=1e-4)
history = trainer.fit(dataloader, num_epochs=10)
```

### 4. Prediction and Visualization

Make predictions and create maps:

```python
from geoscience_transformers import ProspectivityPredictor
import geopandas as gpd

# Initialize predictor
predictor = ProspectivityPredictor(model)

# Make spatial predictions
geodata = gpd.read_file("study_area.geojson")
results = predictor.predict_spatial(
    geodata, 
    feature_columns,
    return_uncertainty=True
)

# Create prospectivity map
fig = predictor.create_prospectivity_map(
    results, 
    output_path="prospectivity_map.png"
)

# Identify high-potential areas
high_potential = predictor.identify_high_potential_areas(
    results, 
    threshold=0.8,
    min_area=1000.0
)
```

## Command Line Interface

Train a model from the command line:

```bash
geoscience-train \
    --config config.yaml \
    --data training_data.csv \
    --output ./models \
    --mode supervised
```

Make predictions:

```bash
geoscience-predict \
    --model ./models/best_model.pt \
    --config config.yaml \
    --data input_data.geojson \
    --output predictions.geojson \
    --uncertainty
```

## Configuration

Example configuration file (`config.yaml`):

```yaml
# Model configuration
model_type: prospectivity
hidden_dim: 256
num_layers: 4
num_heads: 8
dropout: 0.1

# Data dimensions
geochemical_dim: 50
geophysical_dim: 20
text_embedding_dim: 768

# Training parameters
batch_size: 32
learning_rate: 0.0001
num_epochs: 10
weight_decay: 0.01

# Feature columns
feature_columns:
  - Cu
  - Au
  - Ag
  - magnetic_intensity
  - gravity
  - text_emb_0
  - text_emb_1
  # ... more features

# Label column (for supervised learning)
label_column: deposit_present
```

## Architecture

The package implements a transformer-based architecture that:

1. **Processes text** using pre-trained language models fine-tuned on geoscience corpora
2. **Integrates multimodal features** through learned projections
3. **Applies self-attention** to capture complex relationships across data modalities
4. **Supports self-supervised pre-training** to leverage unlabeled data
5. **Provides uncertainty estimates** through Monte Carlo dropout

## Methodology

Based on the paper by Parsa et al. (2025), this implementation:

- Uses NLP and LLMs to extract knowledge from geoscience text
- Employs transformer architectures for multimodal data fusion
- Implements self-supervised learning for limited labeled data scenarios
- Achieves ~87% reduction in mineral exploration search space

## Examples

See the `examples/` directory for complete workflows:

- `examples/basic_usage.py` - Basic usage examples
- `examples/full_pipeline.py` - Complete prospectivity mapping pipeline
- `examples/self_supervised.py` - Self-supervised pre-training
- `notebooks/tutorial.ipynb` - Interactive Jupyter notebook tutorial

## Testing

Run tests:

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=src/geoscience_transformers --cov-report=html
```

## Citation

If you use this package in your research, please cite:

```bibtex
@article{parsa2025geoscience,
  title={Large Language Models and Geoscience Transformers for Predictive Mapping of Canadian Critical Minerals},
  author={Parsa et al.},
  journal={Natural Resources Research},
  year={2025},
  doi={10.1007/s11053-025-10564-0}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on research by Natural Resources Canada (NRCan) and the Geological Survey of Canada. The methodology builds upon:

- [NRCan Geoscience Language Models](https://github.com/NRCan/Geoscience_Language_Models)
- The paper: "Large Language Models and Geoscience Transformers for Predictive Mapping of Canadian Critical Minerals"

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## Related Projects

- [NRCan Geoscience Language Models](https://github.com/NRCan/Geoscience_Language_Models)
- [GEOSCAN Database](https://geoscan.nrcan.gc.ca/)

## Roadmap

- [ ] Integration with more geoscience data sources
- [ ] Support for 3D geological models
- [ ] Additional pre-trained domain-specific models
- [ ] Web-based visualization dashboard
- [ ] API for cloud deployment
