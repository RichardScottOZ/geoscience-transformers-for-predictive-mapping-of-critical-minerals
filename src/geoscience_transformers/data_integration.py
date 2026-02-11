"""
Data Integration Module

Handles integration of multimodal geoscience data including:
- Geochemical data
- Geophysical data
- Satellite/remote sensing data
- Textual features from documents
"""

from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class MultimodalDataIntegrator:
    """
    Integrate multiple types of geoscience data for prospectivity mapping.
    
    This class handles spatial alignment, normalization, and feature engineering
    across different data modalities.
    """
    
    def __init__(
        self,
        spatial_resolution: float = 1000.0,  # meters
        normalize: bool = True,
        scaler_type: str = "standard"
    ):
        """
        Initialize the data integrator.
        
        Args:
            spatial_resolution: Target spatial resolution in meters
            normalize: Whether to normalize features
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.spatial_resolution = spatial_resolution
        self.normalize = normalize
        self.scaler_type = scaler_type
        
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
            
        self.feature_names = []
        self.is_fitted = False
        
    def add_geochemical_features(
        self,
        data: pd.DataFrame,
        element_columns: List[str]
    ) -> pd.DataFrame:
        """
        Add geochemical element features.
        
        Args:
            data: DataFrame with geochemical data
            element_columns: List of column names for elements
            
        Returns:
            DataFrame with added features
        """
        df = data.copy()
        
        # Calculate ratios between key elements
        if len(element_columns) >= 2:
            for i, elem1 in enumerate(element_columns[:-1]):
                for elem2 in element_columns[i+1:]:
                    if elem1 in df.columns and elem2 in df.columns:
                        ratio_name = f"{elem1}_{elem2}_ratio"
                        # Avoid division by zero
                        df[ratio_name] = df[elem1] / (df[elem2] + 1e-10)
                        
        # Log-transform highly skewed elements
        for col in element_columns:
            if col in df.columns:
                log_col = f"{col}_log"
                df[log_col] = np.log1p(df[col])
                
        return df
    
    def add_geophysical_features(
        self,
        data: pd.DataFrame,
        geophysical_columns: List[str]
    ) -> pd.DataFrame:
        """
        Add geophysical features.
        
        Args:
            data: DataFrame with geophysical data
            geophysical_columns: List of geophysical feature columns
            
        Returns:
            DataFrame with added features
        """
        df = data.copy()
        
        # Calculate spatial derivatives if data has spatial coordinates
        if "x" in df.columns and "y" in df.columns:
            for col in geophysical_columns:
                if col in df.columns:
                    # Sort by spatial coordinates
                    df_sorted = df.sort_values(['y', 'x'])
                    
                    # Simple gradient approximation
                    gradient_col = f"{col}_gradient"
                    df[gradient_col] = df_sorted[col].diff().abs()
                    
        return df
    
    def add_spatial_features(
        self,
        data: Union[pd.DataFrame, gpd.GeoDataFrame],
        x_col: str = "x",
        y_col: str = "y"
    ) -> pd.DataFrame:
        """
        Add spatial context features.
        
        Args:
            data: DataFrame with spatial coordinates
            x_col: Name of x coordinate column
            y_col: Name of y coordinate column
            
        Returns:
            DataFrame with spatial features
        """
        df = data.copy()
        
        if x_col in df.columns and y_col in df.columns:
            # Distance from major geological features (simplified)
            # In real implementation, would calculate distance to known deposits
            center_x = df[x_col].mean()
            center_y = df[y_col].mean()
            
            df["distance_from_center"] = np.sqrt(
                (df[x_col] - center_x)**2 + (df[y_col] - center_y)**2
            )
            
            # Spatial density estimation
            df["spatial_density"] = self._estimate_spatial_density(
                df[[x_col, y_col]].values
            )
            
        return df
    
    def _estimate_spatial_density(
        self,
        coordinates: np.ndarray,
        bandwidth: float = 1000.0
    ) -> np.ndarray:
        """
        Estimate spatial density using kernel density estimation.
        
        Args:
            coordinates: Array of (x, y) coordinates
            bandwidth: Bandwidth for KDE
            
        Returns:
            Array of density values
        """
        from scipy.stats import gaussian_kde
        
        # Simple density estimation
        if len(coordinates) < 2:
            return np.ones(len(coordinates))
            
        try:
            kde = gaussian_kde(coordinates.T, bw_method=bandwidth/coordinates.std())
            density = kde(coordinates.T)
        except:
            density = np.ones(len(coordinates))
            
        return density
    
    def integrate_text_features(
        self,
        data: pd.DataFrame,
        text_embeddings: np.ndarray,
        prefix: str = "text_emb"
    ) -> pd.DataFrame:
        """
        Integrate text embeddings into the data.
        
        Args:
            data: Main DataFrame
            text_embeddings: Array of text embeddings
            prefix: Prefix for embedding column names
            
        Returns:
            DataFrame with text features
        """
        df = data.copy()
        
        # Add embeddings as columns
        n_dims = text_embeddings.shape[1]
        for i in range(n_dims):
            df[f"{prefix}_{i}"] = text_embeddings[:, i]
            
        return df
    
    def prepare_features(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        fit: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare and normalize features for modeling.
        
        Args:
            data: DataFrame with features
            feature_columns: List of feature column names
            fit: Whether to fit the scaler
            
        Returns:
            Tuple of (normalized features array, feature names)
        """
        # Select features
        X = data[feature_columns].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Normalize
        if self.normalize:
            if fit:
                X = self.scaler.fit_transform(X)
                self.is_fitted = True
            else:
                if not self.is_fitted:
                    raise ValueError("Scaler not fitted. Call with fit=True first.")
                X = self.scaler.transform(X)
                
        self.feature_names = feature_columns
        
        return X, feature_columns
    
    def create_spatial_grid(
        self,
        bounds: Tuple[float, float, float, float],
        resolution: Optional[float] = None
    ) -> gpd.GeoDataFrame:
        """
        Create a spatial grid for prediction.
        
        Args:
            bounds: Tuple of (minx, miny, maxx, maxy)
            resolution: Grid resolution (uses self.spatial_resolution if None)
            
        Returns:
            GeoDataFrame with grid cells
        """
        from shapely.geometry import box
        
        resolution = resolution or self.spatial_resolution
        minx, miny, maxx, maxy = bounds
        
        # Create grid
        x_coords = np.arange(minx, maxx, resolution)
        y_coords = np.arange(miny, maxy, resolution)
        
        grid_cells = []
        for x in x_coords:
            for y in y_coords:
                cell = box(x, y, x + resolution, y + resolution)
                grid_cells.append({
                    "geometry": cell,
                    "x": x + resolution / 2,
                    "y": y + resolution / 2
                })
                
        return gpd.GeoDataFrame(grid_cells)
