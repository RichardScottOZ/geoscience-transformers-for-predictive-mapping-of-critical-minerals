"""
Training Module

Handles model training with self-supervised and supervised approaches,
as described in Parsa et al. (2025).

Self-supervised learning strategies include:
- Contrastive learning with data augmentation
- Masked value prediction (randomly mask and reconstruct feature values)
- Pseudo-labeling (generate labels from unlabeled data for semi-supervised learning)

References:
    - Parsa, M., Lawley, C.J.M., et al. (2025). Large Language Models and
      Geoscience Transformers for Predictive Mapping of Canadian Critical Minerals.
      Natural Resources Research. DOI: 10.1007/s11053-025-10564-0
    - NRCan/Geoscience_Language_Models: https://github.com/NRCan/Geoscience_Language_Models
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
    
    Implements multiple self-supervised strategies from Parsa et al. (2025):
    - Contrastive learning: learn representations by contrasting augmented views
    - Masked value prediction: mask random features and predict their values
    - Pseudo-labeling: generate labels from confident predictions on unlabeled data
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        mask_ratio: float = 0.15,
        pseudo_label_threshold: float = 0.9,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            mask_ratio: Fraction of features to mask for masked value prediction
            pseudo_label_threshold: Confidence threshold for pseudo-labeling
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.mask_ratio = mask_ratio
        self.pseudo_label_threshold = pseudo_label_threshold
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.history = {
            "train_loss": [],
            "val_loss": []
        }

        # Masked value prediction head (initialized lazily based on input dim)
        self._mask_prediction_head: Optional[nn.Module] = None
        
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

    def create_masked_input(
        self,
        features: torch.Tensor
    ) -> tuple:
        """
        Create masked input for masked value prediction.

        Randomly masks a fraction of feature values, replacing them with zeros.
        The model must predict the original values of masked features.

        This implements the masked value prediction strategy from
        Parsa et al. (2025), analogous to masked language modeling in BERT.

        Args:
            features: Input features (batch_size, n_features)

        Returns:
            Tuple of (masked_features, mask, original_values) where mask
            is True for masked positions
        """
        mask = torch.rand_like(features) < self.mask_ratio
        masked_features = features.clone()
        masked_features[mask] = 0.0
        return masked_features, mask, features

    def _get_mask_prediction_head(self, input_dim: int, feature_dim: int) -> nn.Module:
        """Lazily initialize the mask prediction head."""
        if self._mask_prediction_head is None:
            self._mask_prediction_head = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, feature_dim),
            ).to(self.device)
            # Add parameters to optimizer
            self.optimizer.add_param_group({
                'params': self._mask_prediction_head.parameters()
            })
        return self._mask_prediction_head

    def masked_value_prediction_loss(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        original_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked value prediction loss.

        Only computes loss on masked positions, so the model learns to
        reconstruct missing feature values from context.

        Args:
            embeddings: Model embeddings (batch_size, hidden_dim)
            mask: Boolean mask (batch_size, n_features) - True = masked
            original_features: Original unmasked features (batch_size, n_features)

        Returns:
            MSE loss on masked positions
        """
        pred_head = self._get_mask_prediction_head(
            embeddings.shape[-1], original_features.shape[-1]
        )
        predicted = pred_head(embeddings)

        # Only compute loss on masked positions
        if mask.any():
            loss = nn.functional.mse_loss(
                predicted[mask], original_features[mask]
            )
        else:
            loss = torch.tensor(0.0, device=self.device)

        return loss
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch with combined self-supervised objectives.

        Combines contrastive learning and masked value prediction losses
        as described in Parsa et al. (2025).
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        if self._mask_prediction_head is not None:
            self._mask_prediction_head.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            features = batch["features"].to(self.device)
            
            # === Contrastive learning ===
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
                raise TypeError(
                    "Model must have a get_embeddings() method or a transformer "
                    "attribute with get_embeddings(). Wrap your model in "
                    "SelfSupervisedWrapper or add a get_embeddings() method."
                )
            
            # Compute contrastive loss
            if hasattr(self.model, 'contrastive_loss'):
                contrastive = self.model.contrastive_loss(emb1, emb2)
            else:
                contrastive = nn.functional.mse_loss(emb1, emb2)

            # === Masked value prediction ===
            masked_features, mask, original = self.create_masked_input(features)
            if hasattr(self.model, 'get_embeddings'):
                masked_emb = self.model.get_embeddings(masked_features)
            elif hasattr(self.model, 'transformer'):
                masked_emb = self.model.transformer.get_embeddings(masked_features)
            else:
                masked_emb = emb1  # fallback

            mask_loss = self.masked_value_prediction_loss(
                masked_emb, mask, original
            )

            # Combined loss
            loss = contrastive + mask_loss
            
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
                    raise TypeError(
                        "Model must have a get_embeddings() method or a "
                        "transformer attribute with get_embeddings()."
                    )
                
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


class PseudoLabelTrainer:
    """
    Pseudo-labeling trainer for semi-supervised prospectivity mapping.

    Implements the pseudo-labeling strategy from Parsa et al. (2025):
    1. Train on labeled data
    2. Generate pseudo-labels on unlabeled data using confident predictions
    3. Retrain on combined labeled + pseudo-labeled data
    4. Repeat

    This is particularly useful in geoscience prospectivity mapping where
    labeled deposit locations are scarce but unlabeled geological data is
    abundant.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        confidence_threshold: float = 0.9,
    ):
        """
        Initialize pseudo-label trainer.

        Args:
            model: Model to train
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
            confidence_threshold: Only use pseudo-labels with prediction
                confidence above this threshold
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def generate_pseudo_labels(
        self,
        unlabeled_dataloader: DataLoader,
    ) -> tuple:
        """
        Generate pseudo-labels for unlabeled data.

        Only assigns labels to samples where model confidence exceeds
        the threshold.

        Args:
            unlabeled_dataloader: DataLoader for unlabeled data

        Returns:
            Tuple of (features, pseudo_labels, mask) where mask indicates
            which samples received pseudo-labels
        """
        self.model.eval()
        all_features = []
        all_probs = []

        with torch.no_grad():
            for batch in unlabeled_dataloader:
                features = batch["features"].to(self.device)
                outputs = self.model(features).squeeze()
                probs = torch.sigmoid(outputs)
                all_features.append(features.cpu())
                all_probs.append(probs.cpu())

        all_features = torch.cat(all_features, dim=0)
        all_probs = torch.cat(all_probs, dim=0)

        # Select confident predictions
        confident_positive = all_probs >= self.confidence_threshold
        confident_negative = all_probs <= (1.0 - self.confidence_threshold)
        confident_mask = confident_positive | confident_negative

        pseudo_labels = (all_probs >= 0.5).float()

        return all_features, pseudo_labels, confident_mask

    def fit(
        self,
        labeled_dataloader: DataLoader,
        unlabeled_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        num_pseudo_rounds: int = 3,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train with pseudo-labeling.

        Args:
            labeled_dataloader: DataLoader for labeled data
            unlabeled_dataloader: DataLoader for unlabeled data
            val_dataloader: Optional validation DataLoader
            num_epochs: Epochs per pseudo-labeling round
            num_pseudo_rounds: Number of pseudo-labeling iterations
            checkpoint_path: Path to save best model

        Returns:
            Training history
        """
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "pseudo_count": []
        }
        best_val_loss = float('inf')

        for pseudo_round in range(num_pseudo_rounds):
            print(f"\n=== Pseudo-labeling round {pseudo_round + 1}/{num_pseudo_rounds} ===")

            # Generate pseudo-labels from current model
            if pseudo_round > 0:
                features, pseudo_labels, mask = self.generate_pseudo_labels(
                    unlabeled_dataloader
                )
                n_pseudo = mask.sum().item()
                print(f"Generated {n_pseudo} pseudo-labels "
                      f"(threshold={self.confidence_threshold})")
                history["pseudo_count"].append(n_pseudo)

                # Create combined dataset
                if n_pseudo > 0:
                    pseudo_features = features[mask]
                    pseudo_labels_filtered = pseudo_labels[mask]
                    pseudo_dataset = GeoscienceDataset(
                        pseudo_features.numpy(),
                        pseudo_labels_filtered.numpy()
                    )
                    pseudo_loader = DataLoader(
                        pseudo_dataset,
                        batch_size=labeled_dataloader.batch_size,
                        shuffle=True
                    )
            else:
                pseudo_loader = None
                history["pseudo_count"].append(0)

            # Train on labeled data + pseudo-labeled data
            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0.0
                n_batches = 0

                # Train on labeled data
                for batch in labeled_dataloader:
                    loss = self._train_step(batch)
                    total_loss += loss
                    n_batches += 1

                # Train on pseudo-labeled data
                if pseudo_round > 0 and pseudo_loader is not None:
                    for batch in pseudo_loader:
                        loss = self._train_step(batch)
                        total_loss += loss
                        n_batches += 1

                avg_loss = total_loss / max(n_batches, 1)
                history["train_loss"].append(avg_loss)

                # Validate
                if val_dataloader is not None:
                    val_loss = self._validate(val_dataloader)
                    history["val_loss"].append(val_loss)

                    if checkpoint_path and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), checkpoint_path)

        return history

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        features = batch["features"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.model(features).squeeze()
        loss = self.criterion(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(features).squeeze()
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(dataloader)
