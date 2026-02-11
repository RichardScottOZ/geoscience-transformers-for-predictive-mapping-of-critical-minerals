"""
Geoscience Transformers Package

Large Language Models and Geoscience Transformers for Predictive Mapping of Critical Minerals.
Based on the paper: DOI 10.1007/s11053-025-10564-0

Implements the methodology from Parsa et al. (2025) including:
- Domain-specific BERT model from NRCan (Lawley et al., 2022)
- TabTransformer and FT-Transformer for tabular feature fusion
- GEOSCAN text processing pipeline (PDF extraction, French removal, etc.)
- Self-supervised learning with contrastive learning, masked value prediction
- Pseudo-labeling for semi-supervised training
- Domain-specific GloVe embeddings (300-d, trained on geoscience corpora)

References:
    NRCan/Geoscience_Language_Models: https://github.com/NRCan/Geoscience_Language_Models
"""

__version__ = "0.1.0"
__author__ = "Geoscience Transformers Team"

# Lazy imports to avoid requiring all dependencies upfront
def __getattr__(name):
    """Lazy loading of modules."""
    if name == "GeoscienceTextProcessor":
        from .text_processing import GeoscienceTextProcessor
        return GeoscienceTextProcessor
    elif name == "GeoscanTextPipeline":
        from .text_processing import GeoscanTextPipeline
        return GeoscanTextPipeline
    elif name == "MultimodalDataIntegrator":
        from .data_integration import MultimodalDataIntegrator
        return MultimodalDataIntegrator
    elif name == "GeoscienceTransformer":
        from .models import GeoscienceTransformer
        return GeoscienceTransformer
    elif name == "TabTransformer":
        from .models import TabTransformer
        return TabTransformer
    elif name == "FTTransformer":
        from .models import FTTransformer
        return FTTransformer
    elif name == "ProspectivityModel":
        from .models import ProspectivityModel
        return ProspectivityModel
    elif name == "SelfSupervisedWrapper":
        from .models import SelfSupervisedWrapper
        return SelfSupervisedWrapper
    elif name == "SelfSupervisedTrainer":
        from .training import SelfSupervisedTrainer
        return SelfSupervisedTrainer
    elif name == "SupervisedTrainer":
        from .training import SupervisedTrainer
        return SupervisedTrainer
    elif name == "PseudoLabelTrainer":
        from .training import PseudoLabelTrainer
        return PseudoLabelTrainer
    elif name == "ProspectivityPredictor":
        from .prediction import ProspectivityPredictor
        return ProspectivityPredictor
    elif name == "GeoscienceGloVeEmbeddings":
        from .glove_embeddings import GeoscienceGloVeEmbeddings
        return GeoscienceGloVeEmbeddings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GeoscienceTextProcessor",
    "GeoscanTextPipeline",
    "MultimodalDataIntegrator",
    "GeoscienceTransformer",
    "TabTransformer",
    "FTTransformer",
    "ProspectivityModel",
    "SelfSupervisedWrapper",
    "SelfSupervisedTrainer",
    "SupervisedTrainer",
    "PseudoLabelTrainer",
    "ProspectivityPredictor",
    "GeoscienceGloVeEmbeddings",
]
