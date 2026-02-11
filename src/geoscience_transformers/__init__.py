"""
Geoscience Transformers Package

Large Language Models and Geoscience Transformers for Predictive Mapping of Critical Minerals.
Based on the paper: DOI 10.1007/s11053-025-10564-0

This package provides tools for:
- Text mining from geoscience documents
- Multimodal data integration (geochemistry, geophysics, text)
- Self-supervised transformer models for mineral prospectivity mapping
- Prediction and visualization of critical mineral deposits
"""

__version__ = "0.1.0"
__author__ = "Geoscience Transformers Team"

# Lazy imports to avoid requiring all dependencies upfront
def __getattr__(name):
    """Lazy loading of modules."""
    if name == "GeoscienceTextProcessor":
        from .text_processing import GeoscienceTextProcessor
        return GeoscienceTextProcessor
    elif name == "MultimodalDataIntegrator":
        from .data_integration import MultimodalDataIntegrator
        return MultimodalDataIntegrator
    elif name == "GeoscienceTransformer":
        from .models import GeoscienceTransformer
        return GeoscienceTransformer
    elif name == "ProspectivityModel":
        from .models import ProspectivityModel
        return ProspectivityModel
    elif name == "SelfSupervisedTrainer":
        from .training import SelfSupervisedTrainer
        return SelfSupervisedTrainer
    elif name == "SupervisedTrainer":
        from .training import SupervisedTrainer
        return SupervisedTrainer
    elif name == "ProspectivityPredictor":
        from .prediction import ProspectivityPredictor
        return ProspectivityPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GeoscienceTextProcessor",
    "MultimodalDataIntegrator",
    "GeoscienceTransformer",
    "ProspectivityModel",
    "SelfSupervisedTrainer",
    "SupervisedTrainer",
    "ProspectivityPredictor",
]
