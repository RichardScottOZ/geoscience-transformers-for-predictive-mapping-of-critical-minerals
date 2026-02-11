"""
Prediction Module

Handle prospectivity prediction and visualization.
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ProspectivityPredictor:
    """
    Make predictions on new areas for mineral prospectivity.
    
    Handles model inference, uncertainty estimation, and result visualization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained model
            device: Device for inference
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def predict(
        self,
        features: np.ndarray,
        batch_size: int = 32,
        return_uncertainty: bool = False
    ) -> np.ndarray:
        """
        Make predictions on features.
        
        Args:
            features: Feature array (n_samples, n_features)
            batch_size: Batch size for inference
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predictions array (n_samples,) or tuple of (predictions, uncertainties)
        """
        predictions = []
        uncertainties = []
        
        features_tensor = torch.FloatTensor(features)
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features_tensor[i:i + batch_size].to(self.device)
                
                if return_uncertainty:
                    # Enable dropout for uncertainty estimation (MC Dropout)
                    # Only enable dropout layers, keep BatchNorm in eval mode
                    for m in self.model.modules():
                        if isinstance(m, nn.Dropout):
                            m.train()
                    samples = []
                    for _ in range(10):  # Monte Carlo samples
                        output = self.model(batch)
                        samples.append(torch.sigmoid(output).cpu().numpy())
                    self.model.eval()
                    
                    samples = np.array(samples)
                    pred = samples.mean(axis=0)
                    unc = samples.std(axis=0)
                    
                    predictions.append(pred)
                    uncertainties.append(unc)
                else:
                    output = self.model(batch)
                    pred = torch.sigmoid(output).cpu().numpy()
                    predictions.append(pred)
                    
        predictions = np.concatenate(predictions, axis=0).squeeze()
        
        if return_uncertainty:
            uncertainties = np.concatenate(uncertainties, axis=0).squeeze()
            return predictions, uncertainties
        
        return predictions
    
    def predict_spatial(
        self,
        geodata: gpd.GeoDataFrame,
        feature_columns: List[str],
        batch_size: int = 32,
        return_uncertainty: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Make spatial predictions.
        
        Args:
            geodata: GeoDataFrame with features and geometry
            feature_columns: List of feature column names
            batch_size: Batch size for inference
            return_uncertainty: Whether to return uncertainty
            
        Returns:
            GeoDataFrame with predictions
        """
        # Extract features
        features = geodata[feature_columns].values
        
        # Make predictions
        if return_uncertainty:
            predictions, uncertainties = self.predict(
                features, 
                batch_size=batch_size,
                return_uncertainty=True
            )
            geodata = geodata.copy()
            geodata["prospectivity"] = predictions
            geodata["uncertainty"] = uncertainties
        else:
            predictions = self.predict(features, batch_size=batch_size)
            geodata = geodata.copy()
            geodata["prospectivity"] = predictions
            
        return geodata
    
    def create_prospectivity_map(
        self,
        geodata: gpd.GeoDataFrame,
        output_path: Optional[str] = None,
        cmap: str = "YlOrRd",
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Create a prospectivity map visualization.
        
        Args:
            geodata: GeoDataFrame with prospectivity scores
            output_path: Optional path to save figure
            cmap: Colormap name
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot prospectivity
        geodata.plot(
            column="prospectivity",
            cmap=cmap,
            legend=True,
            ax=ax,
            legend_kwds={"label": "Prospectivity Score"}
        )
        
        ax.set_title("Mineral Prospectivity Map", fontsize=16, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved map to {output_path}")
            
        return fig
    
    def create_uncertainty_map(
        self,
        geodata: gpd.GeoDataFrame,
        output_path: Optional[str] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Create an uncertainty map.
        
        Args:
            geodata: GeoDataFrame with uncertainty estimates
            output_path: Optional path to save figure
            cmap: Colormap name
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if "uncertainty" not in geodata.columns:
            raise ValueError("GeoDataFrame does not have uncertainty column")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        geodata.plot(
            column="uncertainty",
            cmap=cmap,
            legend=True,
            ax=ax,
            legend_kwds={"label": "Prediction Uncertainty"}
        )
        
        ax.set_title("Prediction Uncertainty Map", fontsize=16, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved uncertainty map to {output_path}")
            
        return fig
    
    def identify_high_potential_areas(
        self,
        geodata: gpd.GeoDataFrame,
        threshold: float = 0.8,
        min_area: float = 1000.0
    ) -> gpd.GeoDataFrame:
        """
        Identify high-potential areas for exploration.
        
        Args:
            geodata: GeoDataFrame with prospectivity scores
            threshold: Minimum prospectivity threshold
            min_area: Minimum area in square meters
            
        Returns:
            GeoDataFrame with high-potential areas
        """
        # Filter by threshold
        high_potential = geodata[geodata["prospectivity"] >= threshold].copy()
        
        # Filter by area
        if len(high_potential) > 0:
            areas = high_potential.geometry.area
            high_potential = high_potential[areas >= min_area]
            
        return high_potential
    
    def calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Predicted probabilities
            labels: True labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, average_precision_score
        )
        
        # Binary predictions
        binary_preds = (predictions >= threshold).astype(int)
        
        metrics = {
            "accuracy": accuracy_score(labels, binary_preds),
            "precision": precision_score(labels, binary_preds, zero_division=0),
            "recall": recall_score(labels, binary_preds, zero_division=0),
            "f1": f1_score(labels, binary_preds, zero_division=0),
            "roc_auc": roc_auc_score(labels, predictions),
            "pr_auc": average_precision_score(labels, predictions)
        }
        
        return metrics
    
    def plot_roc_curve(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            predictions: Predicted probabilities
            labels: True labels
            output_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label="Random classifier")
        
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            
        return fig
    
    def export_predictions(
        self,
        geodata: gpd.GeoDataFrame,
        output_path: str,
        output_format: str = "geojson"
    ):
        """
        Export predictions to file.
        
        Args:
            geodata: GeoDataFrame with predictions
            output_path: Output file path
            output_format: Output format ('geojson', 'shapefile', 'gpkg')
        """
        if output_format == "geojson":
            geodata.to_file(output_path, driver="GeoJSON")
        elif output_format == "shapefile":
            geodata.to_file(output_path, driver="ESRI Shapefile")
        elif output_format == "gpkg":
            geodata.to_file(output_path, driver="GPKG")
        else:
            raise ValueError(f"Unknown format: {output_format}")
            
        print(f"Exported predictions to {output_path}")
