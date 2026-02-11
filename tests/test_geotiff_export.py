"""
Test GeoTIFF export functionality.
"""

import numpy as np
import geopandas as gpd
import torch
import torch.nn as nn
from shapely.geometry import Point
import rasterio
import tempfile
import os

from geoscience_transformers.prediction import ProspectivityPredictor


class DummyModel(nn.Module):
    """Simple dummy model for testing."""
    def __init__(self, input_dim=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.fc(x)


def create_test_geodata(n_points=50, include_uncertainty=True):
    """Create test GeoDataFrame with predictions."""
    np.random.seed(42)
    
    # Create random points in a grid
    x = np.random.uniform(0, 100, n_points)
    y = np.random.uniform(0, 100, n_points)
    
    # Create geometries
    geometries = [Point(xi, yi) for xi, yi in zip(x, y)]
    
    # Create prospectivity scores
    prospectivity = np.random.uniform(0, 1, n_points)
    
    # Create GeoDataFrame
    data = {
        'geometry': geometries,
        'prospectivity': prospectivity
    }
    
    if include_uncertainty:
        data['uncertainty'] = np.random.uniform(0, 0.3, n_points)
    
    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    
    return gdf


def test_export_to_geotiff_basic():
    """Test basic GeoTIFF export functionality."""
    # Create test data
    geodata = create_test_geodata(n_points=100, include_uncertainty=False)
    
    # Create dummy model and predictor
    model = DummyModel()
    predictor = ProspectivityPredictor(model)
    
    # Export to GeoTIFF
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        predictor.export_to_geotiff(geodata, output_path, resolution=1.0)
        
        # Verify file exists
        assert os.path.exists(output_path), "Output file was not created"
        
        # Read and verify GeoTIFF
        with rasterio.open(output_path) as src:
            # Check basic properties
            assert src.count == 1, f"Expected 1 band, got {src.count}"
            assert src.crs is not None, "CRS is missing"
            assert src.transform is not None, "Transform is missing"
            
            # Check data
            data = src.read(1)
            assert data.shape[0] > 0, "No rows in raster"
            assert data.shape[1] > 0, "No columns in raster"
            
            # Check that we have some valid data (not all nodata)
            valid_data = data[data != src.nodata]
            assert len(valid_data) > 0, "No valid data in raster"
            assert np.all((valid_data >= 0) & (valid_data <= 1)), "Prospectivity values out of range [0, 1]"
    
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_export_to_geotiff_with_uncertainty():
    """Test GeoTIFF export with uncertainty band."""
    # Create test data with uncertainty
    geodata = create_test_geodata(n_points=100, include_uncertainty=True)
    
    # Create dummy model and predictor
    model = DummyModel()
    predictor = ProspectivityPredictor(model)
    
    # Export to GeoTIFF
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        predictor.export_to_geotiff(geodata, output_path, resolution=1.0)
        
        # Verify file exists
        assert os.path.exists(output_path), "Output file was not created"
        
        # Read and verify GeoTIFF
        with rasterio.open(output_path) as src:
            # Check basic properties
            assert src.count == 2, f"Expected 2 bands, got {src.count}"
            assert src.crs is not None, "CRS is missing"
            
            # Check band descriptions
            assert src.descriptions[0] == 'Prospectivity', f"Band 1 description incorrect: {src.descriptions[0]}"
            assert src.descriptions[1] == 'Uncertainty', f"Band 2 description incorrect: {src.descriptions[1]}"
            
            # Check data in both bands
            prospectivity_data = src.read(1)
            uncertainty_data = src.read(2)
            
            assert prospectivity_data.shape == uncertainty_data.shape, "Band shapes don't match"
            
            # Check valid data
            valid_prosp = prospectivity_data[prospectivity_data != src.nodata]
            valid_uncert = uncertainty_data[uncertainty_data != src.nodata]
            
            assert len(valid_prosp) > 0, "No valid prospectivity data"
            assert len(valid_uncert) > 0, "No valid uncertainty data"
    
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_export_to_geotiff_interpolation_methods():
    """Test different interpolation methods."""
    geodata = create_test_geodata(n_points=50, include_uncertainty=False)
    model = DummyModel()
    predictor = ProspectivityPredictor(model)
    
    methods = ['linear', 'nearest', 'cubic']
    
    for method in methods:
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            predictor.export_to_geotiff(
                geodata, 
                output_path, 
                resolution=2.0,
                interpolation_method=method
            )
            
            # Verify file was created
            assert os.path.exists(output_path), f"Output file not created for method {method}"
            
            # Verify it can be read
            with rasterio.open(output_path) as src:
                assert src.count == 1, f"Expected 1 band for method {method}"
                data = src.read(1)
                assert data.shape[0] > 0, f"No data for method {method}"
        
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


def test_export_predictions_geotiff_format():
    """Test export_predictions with GeoTIFF format."""
    geodata = create_test_geodata(n_points=75, include_uncertainty=True)
    model = DummyModel()
    predictor = ProspectivityPredictor(model)
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        predictor.export_predictions(
            geodata,
            output_path,
            output_format='geotiff',
            resolution=1.5
        )
        
        # Verify file exists
        assert os.path.exists(output_path), "Output file was not created"
        
        # Verify it's a valid GeoTIFF
        with rasterio.open(output_path) as src:
            assert src.count == 2, f"Expected 2 bands, got {src.count}"
            assert src.crs is not None, "CRS is missing"
    
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_export_to_geotiff_missing_prospectivity():
    """Test that export fails gracefully without prospectivity column."""
    # Create GeoDataFrame without prospectivity column
    geometries = [Point(0, 0), Point(1, 1)]
    gdf = gpd.GeoDataFrame({'geometry': geometries}, crs='EPSG:4326')
    
    model = DummyModel()
    predictor = ProspectivityPredictor(model)
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        try:
            predictor.export_to_geotiff(gdf, output_path)
            assert False, "Expected ValueError for missing prospectivity column"
        except ValueError as e:
            assert "prospectivity" in str(e).lower(), f"Unexpected error message: {e}"
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_export_to_geotiff_missing_crs():
    """Test that export fails gracefully without CRS."""
    # Create GeoDataFrame without CRS
    geometries = [Point(0, 0), Point(1, 1)]
    gdf = gpd.GeoDataFrame({
        'geometry': geometries,
        'prospectivity': [0.5, 0.8]
    })
    
    model = DummyModel()
    predictor = ProspectivityPredictor(model)
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        try:
            predictor.export_to_geotiff(gdf, output_path)
            assert False, "Expected ValueError for missing CRS"
        except ValueError as e:
            assert "crs" in str(e).lower(), f"Unexpected error message: {e}"
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


if __name__ == "__main__":
    # Run tests manually
    print("Running GeoTIFF export tests...")
    
    print("\n1. Testing basic GeoTIFF export...")
    test_export_to_geotiff_basic()
    print("   ✓ Passed")
    
    print("\n2. Testing GeoTIFF export with uncertainty...")
    test_export_to_geotiff_with_uncertainty()
    print("   ✓ Passed")
    
    print("\n3. Testing interpolation methods...")
    test_export_to_geotiff_interpolation_methods()
    print("   ✓ Passed")
    
    print("\n4. Testing export_predictions with GeoTIFF format...")
    test_export_predictions_geotiff_format()
    print("   ✓ Passed")
    
    print("\n5. Testing error handling for missing prospectivity...")
    test_export_to_geotiff_missing_prospectivity()
    print("   ✓ Passed")
    
    print("\n6. Testing error handling for missing CRS...")
    test_export_to_geotiff_missing_crs()
    print("   ✓ Passed")
    
    print("\n✓ All tests passed!")

