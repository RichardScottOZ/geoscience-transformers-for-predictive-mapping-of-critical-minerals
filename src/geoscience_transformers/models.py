"""
Model Module

Core transformer models for geoscience prospectivity mapping.

Implements TabTransformer and FT-Transformer architectures for tabular data,
as described in Parsa et al. (2025). These architectures apply self-attention
across tabular feature columns rather than treating features as a flat sequence.

References:
    - Parsa, M., Lawley, C.J.M., et al. (2025). Large Language Models and
      Geoscience Transformers for Predictive Mapping of Canadian Critical Minerals.
      Natural Resources Research. DOI: 10.1007/s11053-025-10564-0
    - Huang, X., et al. (2020). TabTransformer: Tabular Data Modeling Using
      Contextual Embeddings. arXiv:2012.06678
    - Gorishniy, Y., et al. (2021). Revisiting Deep Learning Models for Tabular
      Data. NeurIPS 2021 (FT-Transformer).
    - NRCan/Geoscience_Language_Models: https://github.com/NRCan/Geoscience_Language_Models
"""

from typing import Optional, Dict, Any, List
import math
import torch
import torch.nn as nn


class GeoscienceTransformer(nn.Module):
    """
    Transformer model for geoscience multimodal data.
    
    This model uses self-attention to integrate features from multiple
    data sources (text, geochemical, geophysical) for prospectivity prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        """
        Initialize the geoscience transformer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            output_dim: Output dimension (1 for binary, N for N classes)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Handle 2D input (single sample)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        # Project input
        x = self.input_projection(x)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Global pooling (mean)
        x = x.mean(dim=1)
        
        # Output projection
        output = self.output_projection(x)
        
        return output

    def get_embeddings(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get intermediate embeddings (before output projection).

        Args:
            x: Input tensor (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            mask: Optional attention mask

        Returns:
            Embedding tensor (batch_size, hidden_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        return x


class TabTransformer(nn.Module):
    """
    TabTransformer for tabular geoscience data.

    Applies self-attention across individual tabular feature columns. Each
    continuous feature is projected to an embedding, then transformer layers
    apply column-wise attention so that features can interact contextually.

    This is the architecture described in Parsa et al. (2025) for integrating
    geophysical, geochemical, geochronological, and text-derived features.

    Reference:
        Huang, X., et al. (2020). TabTransformer: Tabular Data Modeling Using
        Contextual Embeddings. arXiv:2012.06678
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        """
        Initialize TabTransformer.

        Args:
            num_features: Number of input feature columns
            hidden_dim: Dimension for each column embedding
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            output_dim: Output dimension (1 for binary prospectivity)
        """
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Per-column linear embeddings (each feature gets its own projection)
        self.column_embeddings = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])

        # Learnable column-type embeddings (analogous to positional encoding)
        self.column_type_embedding = nn.Parameter(
            torch.randn(num_features, hidden_dim) * 0.02
        )

        # Transformer encoder for column-wise self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output MLP
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * num_features),
            nn.Linear(hidden_dim * num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tabular features (batch_size, num_features)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Embed each column separately: (batch, num_features, hidden_dim)
        column_tensors = []
        for i in range(self.num_features):
            col_val = x[:, i:i+1]  # (batch, 1)
            col_emb = self.column_embeddings[i](col_val)  # (batch, hidden_dim)
            column_tensors.append(col_emb)
        embedded = torch.stack(column_tensors, dim=1)  # (batch, num_feat, hidden)

        # Add column-type embeddings
        embedded = embedded + self.column_type_embedding.unsqueeze(0)

        # Apply transformer (column-wise self-attention)
        contextual = self.transformer(embedded)  # (batch, num_feat, hidden)

        # Flatten and predict
        flat = contextual.reshape(batch_size, -1)
        output = self.output_head(flat)

        return output

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get contextual column embeddings (before output head).

        Args:
            x: Tabular features (batch_size, num_features)

        Returns:
            Embedding tensor (batch_size, hidden_dim * num_features)
        """
        batch_size = x.shape[0]
        column_tensors = []
        for i in range(self.num_features):
            col_val = x[:, i:i+1]
            col_emb = self.column_embeddings[i](col_val)
            column_tensors.append(col_emb)
        embedded = torch.stack(column_tensors, dim=1)
        embedded = embedded + self.column_type_embedding.unsqueeze(0)
        contextual = self.transformer(embedded)
        return contextual.reshape(batch_size, -1)


class FTTransformer(nn.Module):
    """
    FT-Transformer (Feature Tokenizer + Transformer) for tabular data.

    Each numerical feature is tokenized into an embedding using a learned linear
    projection, then a [CLS] token is prepended and transformer layers apply
    self-attention across all feature tokens. The [CLS] representation is used
    for prediction.

    This is the architecture described in Parsa et al. (2025) for mineral
    prospectivity mapping with multimodal geoscience features.

    Reference:
        Gorishniy, Y., et al. (2021). Revisiting Deep Learning Models for
        Tabular Data. NeurIPS 2021.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        """
        Initialize FT-Transformer.

        Args:
            num_features: Number of input feature columns
            hidden_dim: Dimension for each feature token
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            output_dim: Output dimension
        """
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Feature tokenizer: per-column linear projection + bias
        self.feature_tokenizer = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Positional/column embeddings for features + CLS
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_features + 1, hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Layer norm and output head on [CLS] token
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tabular features (batch_size, num_features)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Tokenize each feature column
        tokens = []
        for i in range(self.num_features):
            col_val = x[:, i:i+1]  # (batch, 1)
            token = self.feature_tokenizer[i](col_val)  # (batch, hidden_dim)
            tokens.append(token)
        feature_tokens = torch.stack(tokens, dim=1)  # (batch, num_feat, hidden)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, feature_tokens], dim=1)

        # Add positional embeddings
        sequence = sequence + self.position_embedding

        # Apply transformer
        transformed = self.transformer(sequence)

        # Extract [CLS] token output
        cls_output = self.norm(transformed[:, 0, :])

        return self.output_head(cls_output)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get [CLS] token embedding (before output head).

        Args:
            x: Tabular features (batch_size, num_features)

        Returns:
            Embedding tensor (batch_size, hidden_dim)
        """
        batch_size = x.shape[0]
        tokens = []
        for i in range(self.num_features):
            col_val = x[:, i:i+1]
            token = self.feature_tokenizer[i](col_val)
            tokens.append(token)
        feature_tokens = torch.stack(tokens, dim=1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, feature_tokens], dim=1)
        sequence = sequence + self.position_embedding
        transformed = self.transformer(sequence)
        return self.norm(transformed[:, 0, :])


class ProspectivityModel(nn.Module):
    """
    Complete prospectivity mapping model.
    
    This model combines text embeddings, geochemical, and geophysical features
    using a transformer architecture for mineral prospectivity prediction.
    """
    
    def __init__(
        self,
        geochemical_dim: int = 50,
        geophysical_dim: int = 20,
        text_embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_text: bool = True,
        use_geochemical: bool = True,
        use_geophysical: bool = True
    ):
        """
        Initialize the prospectivity model.
        
        Args:
            geochemical_dim: Dimension of geochemical features
            geophysical_dim: Dimension of geophysical features
            text_embedding_dim: Dimension of text embeddings
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_text: Whether to use text features
            use_geochemical: Whether to use geochemical features
            use_geophysical: Whether to use geophysical features
        """
        super().__init__()
        
        self.use_text = use_text
        self.use_geochemical = use_geochemical
        self.use_geophysical = use_geophysical
        self.hidden_dim = hidden_dim
        
        # Feature projections
        self.feature_projections = nn.ModuleDict()
        
        if use_geochemical:
            self.feature_projections["geochemical"] = nn.Linear(geochemical_dim, hidden_dim)
            
        if use_geophysical:
            self.feature_projections["geophysical"] = nn.Linear(geophysical_dim, hidden_dim)
            
        if use_text:
            self.feature_projections["text"] = nn.Linear(text_embedding_dim, hidden_dim)
            
        # Transformer for feature fusion
        self.transformer = GeoscienceTransformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            output_dim=1
        )
        
    def forward(
        self,
        geochemical: Optional[torch.Tensor] = None,
        geophysical: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with multimodal inputs.
        
        Args:
            geochemical: Geochemical features (batch_size, geochemical_dim)
            geophysical: Geophysical features (batch_size, geophysical_dim)
            text: Text embeddings (batch_size, text_embedding_dim)
            
        Returns:
            Prospectivity scores (batch_size, 1)
        """
        features = []
        
        if self.use_geochemical and geochemical is not None:
            geo_chem_proj = self.feature_projections["geochemical"](geochemical)
            features.append(geo_chem_proj)
            
        if self.use_geophysical and geophysical is not None:
            geo_phys_proj = self.feature_projections["geophysical"](geophysical)
            features.append(geo_phys_proj)
            
        if self.use_text and text is not None:
            text_proj = self.feature_projections["text"](text)
            features.append(text_proj)
            
        if not features:
            raise ValueError("No input features provided")
            
        # Stack features as sequence
        x = torch.stack(features, dim=1)  # (batch_size, num_modalities, hidden_dim)
        
        # Apply transformer
        output = self.transformer(x)
        
        return output


class SelfSupervisedWrapper(nn.Module):
    """
    Wrapper for self-supervised pre-training of geoscience models.
    
    Implements contrastive learning and masked feature prediction.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        projection_dim: int = 128,
        temperature: float = 0.07
    ):
        """
        Initialize self-supervised wrapper.
        
        Args:
            base_model: Base model to wrap
            projection_dim: Dimension for projection head
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        
        self.base_model = base_model
        self.temperature = temperature
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base model."""
        return self.base_model(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings from base model (before output projection).

        Args:
            x: Input tensor

        Returns:
            Embedding tensor (batch_size, hidden_dim)
        """
        if hasattr(self.base_model, 'get_embeddings'):
            return self.base_model.get_embeddings(x)
        return self.base_model(x)
    
    def contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two augmented views.
        
        Args:
            z1: First view embeddings (batch_size, hidden_dim)
            z2: Second view embeddings (batch_size, hidden_dim)
            
        Returns:
            Contrastive loss
        """
        # Project embeddings
        z1_proj = self.projection_head(z1)
        z2_proj = self.projection_head(z2)
        
        # Normalize
        z1_norm = nn.functional.normalize(z1_proj, dim=1)
        z2_norm = nn.functional.normalize(z2_proj, dim=1)
        
        # Compute similarity matrix
        batch_size = z1_norm.shape[0]
        similarity_matrix = torch.mm(z1_norm, z2_norm.t()) / self.temperature
        
        # Create labels (positive pairs are on diagonal)
        labels = torch.arange(batch_size, device=z1.device)
        
        # Compute loss (symmetric)
        loss_1 = nn.functional.cross_entropy(similarity_matrix, labels)
        loss_2 = nn.functional.cross_entropy(similarity_matrix.t(), labels)
        
        return (loss_1 + loss_2) / 2
