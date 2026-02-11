"""
Training Module

Handles model training with self-supervised and supervised approaches.
"""

from typing import Optional, Dict, Any, Callable, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm


class GeoscienceDataset(Dataset):
    """Dataset for geoscience prospectivity data."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            features: Feature array (n_samples, n_features)
            labels: Optional labels (n_samples,)
            feature_names: Optional feature names
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.feature_names = feature_names
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"features": self.features[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class SelfSupervisedTrainer:
    """
    Trainer for self-supervised learning on geoscience data.
    
    Implements contrastive learning and masked feature prediction
    to learn representations from unlabeled data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
        
    def create_augmented_views(
        self,
        features: torch.Tensor,
        noise_std: float = 0.1,
        dropout_prob: float = 0.2
    ) -> tuple:
        """
        Create two augmented views of the data for contrastive learning.
        
        Args:
            features: Input features (batch_size, n_features)
            noise_std: Standard deviation for Gaussian noise
            dropout_prob: Probability for feature dropout
            
        Returns:
            Tuple of (view1, view2)
        """
        # View 1: Add Gaussian noise
        view1 = features + torch.randn_like(features) * noise_std
        
        # View 2: Feature dropout
        mask = torch.rand_like(features) > dropout_prob
        view2 = features * mask
        
        return view1, view2
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            features = batch["features"].to(self.device)
            
            # Create augmented views
            view1, view2 = self.create_augmented_views(features)
            
            # Forward pass - get embeddings (not final output)
            if hasattr(self.model, 'get_embeddings'):
                emb1 = self.model.get_embeddings(view1)
                emb2 = self.model.get_embeddings(view2)
            elif hasattr(self.model, 'transformer'):
                # For models with transformer attribute
                emb1 = self.model.transformer.get_embeddings(view1)
                emb2 = self.model.transformer.get_embeddings(view2)
            else:
                # Direct model call
                emb1 = self.model(view1)
                emb2 = self.model(view2)
            
            # Compute contrastive loss
            if hasattr(self.model, 'contrastive_loss'):
                loss = self.model.contrastive_loss(emb1, emb2)
            else:
                # Simple MSE loss if no contrastive loss
                loss = nn.functional.mse_loss(emb1, emb2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                
                # Create augmented views
                view1, view2 = self.create_augmented_views(features)
                
                # Forward pass - get embeddings (not final output)
                if hasattr(self.model, 'get_embeddings'):
                    emb1 = self.model.get_embeddings(view1)
                    emb2 = self.model.get_embeddings(view2)
                elif hasattr(self.model, 'transformer'):
                    emb1 = self.model.transformer.get_embeddings(view1)
                    emb2 = self.model.transformer.get_embeddings(view2)
                else:
                    emb1 = self.model(view1)
                    emb2 = self.model(view2)
                
                # Compute loss
                if hasattr(self.model, 'contrastive_loss'):
                    loss = self.model.contrastive_loss(emb1, emb2)
                else:
                    loss = nn.functional.mse_loss(emb1, emb2)
                
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            num_epochs: Number of epochs to train
            checkpoint_path: Path to save best model checkpoint
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_dataloader, epoch)
            self.history["train_loss"].append(train_loss)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validate
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                self.history["val_loss"].append(val_loss)
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
                
                # Save best model
                if checkpoint_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"Saved best model to {checkpoint_path}")
                    
        return self.history


class SupervisedTrainer:
    """
    Trainer for supervised prospectivity mapping.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        """
        Initialize supervised trainer.
        
        Args:
            model: Model to train
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(features).squeeze()
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                "loss": loss.item(),
                "acc": correct / total
            })
            
        return total_loss / len(dataloader), correct / total
    
    def validate(self, dataloader: DataLoader) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(features).squeeze()
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
        return total_loss / len(dataloader), correct / total
    
    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the model."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_dataloader, epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Acc = {train_acc:.4f}")
            
            if val_dataloader is not None:
                val_loss, val_acc = self.validate(val_dataloader)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, Acc = {val_acc:.4f}")
                
                if checkpoint_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"Saved best model to {checkpoint_path}")
                    
        return self.history
