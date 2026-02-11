"""
Text Processing Module

Handles preprocessing and feature extraction from geoscience text documents
using NLP and domain-specific language models.
"""

import re
import string
from typing import List, Dict, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


class GeoscienceTextProcessor:
    """
    Process geoscience text documents and extract features using LLMs.
    
    This class handles:
    - Text cleaning and preprocessing
    - Feature extraction using domain-specific BERT models
    - Named entity recognition for minerals and geological features
    """
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize the text processor.
        
        Args:
            model_name: Name of the transformer model to use
            max_length: Maximum sequence length for tokenization
            device: Device to run the model on (cuda/cpu)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def preprocess_text(
        self, 
        text: str, 
        remove_special_chars: bool = True,
        lowercase: bool = True
    ) -> str:
        """
        Preprocess geoscience text.
        
        Args:
            text: Input text to preprocess
            remove_special_chars: Whether to remove special characters
            lowercase: Whether to convert to lowercase
            
        Returns:
            Preprocessed text
        """
        if lowercase:
            text = text.lower()
            
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        if remove_special_chars:
            # Keep only alphanumeric and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\-]', ' ', text)
            
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_features(
        self, 
        texts: Union[str, List[str]],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Extract embeddings from text using the transformer model.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get model outputs
                outputs = self.model(**inputs)
                
                # Use CLS token embedding or mean pooling
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
                
        return np.vstack(embeddings)
    
    def extract_mineral_mentions(self, text: str) -> List[str]:
        """
        Extract mentions of minerals and elements from text.
        
        Args:
            text: Input text
            
        Returns:
            List of mineral mentions
        """
        # Common critical minerals
        minerals = [
            'lithium', 'cobalt', 'nickel', 'copper', 'rare earth',
            'graphite', 'manganese', 'vanadium', 'zinc', 'aluminum',
            'platinum', 'palladium', 'rhodium', 'iridium', 'gold',
            'silver', 'uranium', 'tungsten', 'chromium', 'molybdenum'
        ]
        
        text_lower = text.lower()
        found_minerals = []
        
        for mineral in minerals:
            if mineral in text_lower:
                found_minerals.append(mineral)
                
        return list(set(found_minerals))
    
    def process_document_batch(
        self,
        documents: List[Dict[str, str]],
        text_key: str = "text"
    ) -> pd.DataFrame:
        """
        Process a batch of documents and extract features.
        
        Args:
            documents: List of document dictionaries
            text_key: Key in document dict containing text
            
        Returns:
            DataFrame with document features
        """
        processed_data = []
        
        for doc in documents:
            text = doc.get(text_key, "")
            
            # Preprocess
            clean_text = self.preprocess_text(text)
            
            # Extract features
            embedding = self.extract_features(clean_text)[0]
            
            # Extract minerals
            minerals = self.extract_mineral_mentions(clean_text)
            
            processed_data.append({
                "original_text": text,
                "clean_text": clean_text,
                "embedding": embedding,
                "minerals_mentioned": minerals,
                "num_minerals": len(minerals)
            })
            
        return pd.DataFrame(processed_data)
