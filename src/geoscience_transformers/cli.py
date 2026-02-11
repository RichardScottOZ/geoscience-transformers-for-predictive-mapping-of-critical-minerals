"""
Command Line Interface Module

Provides CLI commands for training and prediction.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import torch

from .models import GeoscienceTransformer, ProspectivityModel, SelfSupervisedWrapper, TabTransformer, FTTransformer
from .training import SelfSupervisedTrainer, SupervisedTrainer, GeoscienceDataset, PseudoLabelTrainer
from .prediction import ProspectivityPredictor
from .data_integration import MultimodalDataIntegrator


def train():
    """Train a geoscience transformer model."""
    parser = argparse.ArgumentParser(description="Train geoscience transformer model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--mode", type=str, default="supervised", 
                       choices=["supervised", "self-supervised"],
                       help="Training mode")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = pd.read_csv(args.data)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare features
    integrator = MultimodalDataIntegrator()
    feature_columns = config.get("feature_columns", data.columns.tolist())
    features, _ = integrator.prepare_features(data, feature_columns)
    
    # Create dataset
    labels = data[config["label_column"]].values if "label_column" in config else None
    dataset = GeoscienceDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.get("batch_size", 32),
        shuffle=True
    )
    
    # Create model
    print("Creating model...")
    if config.get("model_type") == "prospectivity":
        model = ProspectivityModel(
            geochemical_dim=config.get("geochemical_dim", 50),
            geophysical_dim=config.get("geophysical_dim", 20),
            text_embedding_dim=config.get("text_embedding_dim", 768),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 4)
        )
    elif config.get("model_type") == "tab_transformer":
        model = TabTransformer(
            num_features=features.shape[1],
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 4),
            num_heads=config.get("num_heads", 8),
        )
    elif config.get("model_type") == "ft_transformer":
        model = FTTransformer(
            num_features=features.shape[1],
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 4),
            num_heads=config.get("num_heads", 8),
        )
    else:
        model = GeoscienceTransformer(
            input_dim=features.shape[1],
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 4)
        )
    
    # Train
    print(f"Training in {args.mode} mode...")
    if args.mode == "supervised":
        trainer = SupervisedTrainer(model)
    else:
        model = SelfSupervisedWrapper(model)
        trainer = SelfSupervisedTrainer(model)
    
    checkpoint_path = output_dir / "best_model.pt"
    history = trainer.fit(
        dataloader,
        num_epochs=config.get("num_epochs", 10),
        checkpoint_path=str(checkpoint_path)
    )
    
    print(f"Training complete. Model saved to {checkpoint_path}")
    
    # Save history
    history_path = output_dir / "training_history.yaml"
    with open(history_path, 'w') as f:
        yaml.dump(history, f)
    print(f"Training history saved to {history_path}")


def predict():
    """Make predictions using a trained model."""
    parser = argparse.ArgumentParser(description="Make prospectivity predictions")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to input data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--uncertainty", action="store_true", help="Estimate uncertainty")
    parser.add_argument("--format", type=str, default="geojson", 
                       choices=["geojson", "shapefile", "gpkg", "geotiff", "csv"],
                       help="Output format (default: geojson)")
    parser.add_argument("--resolution", type=float, default=None,
                       help="Grid resolution for GeoTIFF output (in CRS units)")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    print(f"Loading data from {args.data}...")
    if args.data.endswith(".geojson"):
        import geopandas as gpd
        data = gpd.read_file(args.data)
    else:
        data = pd.read_csv(args.data)
    
    # Load model
    print(f"Loading model from {args.model}...")
    if config.get("model_type") == "prospectivity":
        model = ProspectivityModel(
            geochemical_dim=config.get("geochemical_dim", 50),
            geophysical_dim=config.get("geophysical_dim", 20),
            text_embedding_dim=config.get("text_embedding_dim", 768),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 4)
        )
    else:
        input_dim = len(config.get("feature_columns", []))
        model = GeoscienceTransformer(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 4)
        )
    
    model.load_state_dict(torch.load(args.model, weights_only=True))
    
    # Prepare features
    integrator = MultimodalDataIntegrator()
    feature_columns = config.get("feature_columns", data.columns.tolist())
    features, _ = integrator.prepare_features(data, feature_columns)
    
    # Make predictions
    print("Making predictions...")
    predictor = ProspectivityPredictor(model)
    
    if isinstance(data, pd.DataFrame) and "geometry" in data.columns:
        # Spatial predictions
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(data)
        results = predictor.predict_spatial(
            gdf, 
            feature_columns,
            return_uncertainty=args.uncertainty
        )
        # Export in requested format
        predictor.export_predictions(
            results, 
            args.output, 
            output_format=args.format,
            resolution=args.resolution
        )
    else:
        # Regular predictions (CSV only for non-spatial data)
        if args.uncertainty:
            predictions, uncertainties = predictor.predict(
                features, 
                return_uncertainty=True
            )
            data["prospectivity"] = predictions
            data["uncertainty"] = uncertainties
        else:
            predictions = predictor.predict(features)
            data["prospectivity"] = predictions
        
        data.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict()
    else:
        print("Usage: python -m geoscience_transformers.cli [train|predict] [options]")
