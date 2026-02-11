"""
GeoTIFF Export Example

This example demonstrates how to export mineral prospectivity predictions
as GeoTIFF raster files for use in GIS software.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import torch

from geoscience_transformers import GeoscienceTransformer
from geoscience_transformers import MultimodalDataIntegrator
from geoscience_transformers import ProspectivityPredictor
from geoscience_transformers.training import SupervisedTrainer, GeoscienceDataset
from torch.utils.data import DataLoader


def create_synthetic_spatial_data(n_samples=200):
    """Create synthetic geospatial data for demonstration."""
    print("Creating synthetic spatial data...")
    np.random.seed(42)
    
    # Create spatial grid (simulating a geographic region)
    lon = np.random.uniform(-120, -115, n_samples)
    lat = np.random.uniform(35, 40, n_samples)
    
    # Create Point geometries
    geometries = [Point(x, y) for x, y in zip(lon, lat)]
    
    # Geochemical data (log-normal distributions)
    Cu = np.random.lognormal(0, 1, n_samples)
    Au = np.random.lognormal(-2, 1, n_samples)
    Ag = np.random.lognormal(-1, 1, n_samples)
    
    # Geophysical data
    magnetic = np.random.normal(50000, 5000, n_samples)
    gravity = np.random.normal(9.81, 0.05, n_samples)
    
    # Synthetic labels (deposit presence)
    deposit_prob = (
        (Cu > np.percentile(Cu, 75)) & 
        (Au > np.percentile(Au, 75)) |
        (magnetic > np.percentile(magnetic, 70))
    ).astype(float)
    
    labels = (deposit_prob + np.random.uniform(0, 0.3, n_samples)) > 0.6
    
    # Create GeoDataFrame
    data = gpd.GeoDataFrame({
        'Cu': Cu,
        'Au': Au,
        'Ag': Ag,
        'magnetic_intensity': magnetic,
        'gravity': gravity,
        'deposit_present': labels.astype(int)
    }, geometry=geometries, crs='EPSG:4326')
    
    print(f"  Created {n_samples} spatial samples")
    print(f"  Geographic extent: {lon.min():.2f}°W to {lon.max():.2f}°W, {lat.min():.2f}°N to {lat.max():.2f}°N")
    print(f"  Deposit ratio: {labels.mean():.2%}")
    
    return data


def main():
    print("=" * 70)
    print("GeoTIFF Export Example - Mineral Prospectivity Mapping")
    print("=" * 70)
    
    # 1. Create synthetic spatial data
    print("\n1. DATA PREPARATION")
    print("-" * 70)
    
    data = create_synthetic_spatial_data(n_samples=200)
    
    # Split data
    from geoscience_transformers.utils import split_data
    
    feature_cols = ['Cu', 'Au', 'Ag', 'magnetic_intensity', 'gravity']
    X = data[feature_cols].values
    y = data['deposit_present'].values
    geometries = data.geometry.values
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    
    # Keep test geometries for spatial prediction
    n_train = len(X_train)
    n_val = len(X_val)
    test_geometries = geometries[n_train + n_val:]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # 2. Train a simple model
    print("\n2. MODEL TRAINING")
    print("-" * 70)
    
    integrator = MultimodalDataIntegrator()
    
    # Prepare features
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    X_train_proc, _ = integrator.prepare_features(train_df, feature_cols, fit=True)
    
    val_df = pd.DataFrame(X_val, columns=feature_cols)
    X_val_proc, _ = integrator.prepare_features(val_df, feature_cols, fit=False)
    
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    X_test_proc, _ = integrator.prepare_features(test_df, feature_cols, fit=False)
    
    # Create and train model
    model = GeoscienceTransformer(
        input_dim=X_train_proc.shape[1],
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        dropout=0.1
    )
    
    train_dataset = GeoscienceDataset(X_train_proc, y_train)
    val_dataset = GeoscienceDataset(X_val_proc, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    trainer = SupervisedTrainer(model, learning_rate=1e-3)
    
    print("  Training model (3 epochs)...")
    history = trainer.fit(
        train_loader, 
        val_loader,
        num_epochs=3,
        checkpoint_path=None
    )
    
    # 3. Make spatial predictions
    print("\n3. SPATIAL PREDICTIONS")
    print("-" * 70)
    
    predictor = ProspectivityPredictor(model)
    
    # Create GeoDataFrame with test data
    test_gdf = gpd.GeoDataFrame(
        test_df,
        geometry=test_geometries,
        crs='EPSG:4326'
    )
    
    # Make predictions with uncertainty
    print("  Making predictions with uncertainty estimation...")
    results = predictor.predict_spatial(
        test_gdf,
        feature_cols,
        return_uncertainty=True
    )
    
    print(f"  Prospectivity range: {results['prospectivity'].min():.3f} - {results['prospectivity'].max():.3f}")
    print(f"  Mean uncertainty: {results['uncertainty'].mean():.3f}")
    
    # 4. Export predictions in multiple formats
    print("\n4. EXPORTING PREDICTIONS")
    print("-" * 70)
    
    # Export as GeoJSON (vector format)
    print("  Exporting as GeoJSON (vector)...")
    predictor.export_predictions(
        results,
        '/tmp/prospectivity_predictions.geojson',
        output_format='geojson'
    )
    
    # Export as GeoTIFF (raster format) - DEFAULT RESOLUTION
    print("\n  Exporting as GeoTIFF (raster) with default resolution...")
    predictor.export_predictions(
        results,
        '/tmp/prospectivity_map.tif',
        output_format='geotiff'
    )
    
    # Export as GeoTIFF with custom resolution
    print("\n  Exporting as GeoTIFF with custom resolution (0.01 degrees)...")
    predictor.export_predictions(
        results,
        '/tmp/prospectivity_map_highres.tif',
        output_format='geotiff',
        resolution=0.01
    )
    
    # Export as GeoTIFF using nearest neighbor interpolation
    print("\n  Exporting as GeoTIFF with nearest neighbor interpolation...")
    predictor.export_to_geotiff(
        results,
        '/tmp/prospectivity_map_nearest.tif',
        resolution=0.02,
        interpolation_method='nearest'
    )
    
    # 5. Summary
    print("\n" + "=" * 70)
    print("EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nOutput files created:")
    print("  1. /tmp/prospectivity_predictions.geojson")
    print("     - Vector format (point features)")
    print("     - Use in QGIS, ArcGIS, or web mapping")
    print()
    print("  2. /tmp/prospectivity_map.tif")
    print("     - GeoTIFF raster (default resolution)")
    print("     - Band 1: Prospectivity scores")
    print("     - Band 2: Uncertainty estimates")
    print()
    print("  3. /tmp/prospectivity_map_highres.tif")
    print("     - High-resolution GeoTIFF (0.01° grid)")
    print("     - Smoother interpolation")
    print()
    print("  4. /tmp/prospectivity_map_nearest.tif")
    print("     - GeoTIFF with nearest neighbor interpolation")
    print("     - Preserves original point values better")
    print()
    print("How to use GeoTIFF files:")
    print("  - Open in QGIS: Layer > Add Raster Layer")
    print("  - Open in ArcGIS: Add Data > Raster")
    print("  - Process with Python: rasterio.open('file.tif')")
    print("  - View both bands for prospectivity and uncertainty")
    print()
    print(f"Model performance on test set:")
    test_preds = predictor.predict(X_test_proc)
    metrics = predictor.calculate_metrics(test_preds, y_test)
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    from geoscience_transformers.utils import set_seed
    set_seed(42)
    
    main()
