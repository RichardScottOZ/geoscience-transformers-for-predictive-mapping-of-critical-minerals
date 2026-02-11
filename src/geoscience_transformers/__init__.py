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

from .text_processing import GeoscienceTextProcessor
from .data_integration import MultimodalDataIntegrator
from .models import GeoscienceTransformer, ProspectivityModel
from .training import SelfSupervisedTrainer
from .prediction import ProspectivityPredictor

__all__ = [
    "GeoscienceTextProcessor",
    "MultimodalDataIntegrator",
    "GeoscienceTransformer",
    "ProspectivityModel",
    "SelfSupervisedTrainer",
    "ProspectivityPredictor",
]
