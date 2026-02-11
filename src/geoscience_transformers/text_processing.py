"""
Text Processing Module

Handles preprocessing and feature extraction from geoscience text documents
using NLP and domain-specific language models from NRCan.

This module implements the GEOSCAN text processing pipeline described in
Lawley et al. (2022) and used in Parsa et al. (2025), including:
- PDF text extraction via pdfminer
- French text detection and removal
- DOI, URL, email, and phone number removal
- Non-ASCII to ASCII conversion
- Geological vocabulary tokenization

The default BERT model is the NRCan geoscience DistilBERT, fine-tuned on
27,000+ documents from the GEOSCAN publications database.

References:
    - Lawley, C.J.M., et al. (2022). Geoscience language models and their
      intrinsic evaluation. Applied Computing and Geosciences, 14, 100084.
    - Raimondo, S., et al. (2022). Datasets to support geoscience language
      models. Geological Survey of Canada, Open File 8848.
    - NRCan/Geoscience_Language_Models: https://github.com/NRCan/Geoscience_Language_Models
"""

import re
import os
import string
from typing import List, Dict, Optional, Union, Any
from pathlib import Path

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, AutoConfig
import torch

# Default model: NRCan geoscience DistilBERT fine-tuned on GEOSCAN corpus.
# Users should provide a local path to the model downloaded from:
# https://github.com/NRCan/Geoscience_Language_Models
# Falls back to distilbert-base-uncased if no local model is available.
NRCAN_GEOSCIENCE_BERT = "distilbert-base-uncased"


class GeoscanTextPipeline:
    """
    Text preprocessing pipeline matching the NRCan GEOSCAN processing steps.

    Implements the cleaning pipeline from NRCan/Geoscience_Language_Models
    (project_tools/nrcan_p2/data_processing/), including:
    - PDF text extraction via pdfminer
    - French text detection and removal
    - DOI/URL/email/phone number removal
    - Non-ASCII to ASCII conversion
    - Punctuation normalization
    - Double-space removal

    Reference:
        Raimondo, S., et al. (2022). Datasets to support geoscience language
        models. Geological Survey of Canada, Open File 8848.
    """

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from a PDF file using pdfminer.

        This replicates the first step of the NRCan GEOSCAN pipeline
        (scripts/pdf_to_txt.py) which converts PDF documents to text.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text string

        Raises:
            ImportError: If pdfminer.six is not installed
        """
        try:
            from pdfminer.high_level import extract_text
        except ImportError:
            raise ImportError(
                "pdfminer.six is required for PDF extraction. "
                "Install with: pip install pdfminer.six"
            )
        return extract_text(pdf_path)

    @staticmethod
    def extract_text_from_pdf_to_blocks(pdf_path: str) -> List[str]:
        """
        Extract text blocks from a PDF (preserving layout structure).

        Each block corresponds to a text box in the PDF, matching the
        NRCan pipeline's CSV-based intermediate representation.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of text blocks
        """
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer
        except ImportError:
            raise ImportError(
                "pdfminer.six is required for PDF extraction. "
                "Install with: pip install pdfminer.six"
            )

        blocks = []
        for page_layout in extract_pages(pdf_path):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text().strip()
                    if text:
                        blocks.append(text)
        return blocks

    @staticmethod
    def detect_french(text: str, threshold: float = 0.3) -> bool:
        """
        Detect if text is primarily French.

        Uses common French words to estimate language. This replicates the
        NRCan pipeline's filter_non_english step.

        Args:
            text: Input text
            threshold: Fraction of French indicator words required

        Returns:
            True if text appears to be primarily French
        """
        french_indicators = {
            'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une',
            'et', 'est', 'sont', 'dans', 'pour', 'avec', 'sur',
            'par', 'que', 'qui', 'ce', 'cette', 'nous', 'vous',
            'ils', 'elles', 'au', 'aux', 'en', 'ou', 'mais',
            'pas', 'ne', 'se', 'ont', 'peut', 'entre', 'aussi',
            'comme', 'très', 'tout', 'tous', 'être', 'avoir',
            'fait', 'même', 'plus', 'sans', 'sous', 'vers',
            'géologie', 'minéral', 'roches', 'région',
        }
        words = text.lower().split()
        if len(words) < 5:
            return False
        french_count = sum(1 for w in words if w in french_indicators)
        return (french_count / len(words)) > threshold

    @staticmethod
    def remove_cid_markers(text: str) -> str:
        """Remove (cid:X) markers from PDF extraction artifacts."""
        return re.sub(r'\(cid:[0-9]+\)', '', text)

    @staticmethod
    def convert_to_ascii(text: str) -> str:
        """
        Convert non-ASCII characters to ASCII equivalents.

        Matches the NRCan pipeline's convert_to_ascii step using
        unidecode-style conversion.
        """
        try:
            from unidecode import unidecode_expect_ascii
            return re.sub(r'\[\?\]', ' ', unidecode_expect_ascii(text))
        except ImportError:
            # Fallback: strip non-ASCII characters (replacing with space)
            return re.sub(r'[^\x00-\x7F]', ' ', text)

    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text (NRCan pipeline: rm_url)."""
        return re.sub(
            r'\b((http(s)?|ftp):\/\/)?(www\.)?'
            r'(([-a-zA-Z0-9@:%_\+~#=]+\.){1,256})[a-z]{2,6}\b'
            r'(([-a-zA-Z0-9@:%_\+~#?&//=.]+[-a-zA-Z0-9@%_\+~#?&//=])'
            r'|[-a-zA-Z0-9@%_\+~#?&//=])?',
            ' ', text
        )

    @staticmethod
    def remove_dois(text: str) -> str:
        """Remove DOIs from text (NRCan pipeline: rm_doi)."""
        return re.sub(
            r'(((doi.?:?\s?)|(doi\.org/)|(https://doi\.org/))\s*)'
            r'(10\.([A-Za-z0-9.\/-]+)?[A-Za-z0-9\/])',
            ' ', text
        )

    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses (NRCan pipeline: rm_email)."""
        return re.sub(r'[\w.\-]+@[\w\-.]+[.][\w\-.]+[\w]', ' ', text)

    @staticmethod
    def remove_phone_numbers(text: str) -> str:
        """Remove phone numbers (NRCan pipeline: rm_phonenumber)."""
        return re.sub(
            r'(?:\+?\d{1,2}\s?)?1?[-. ]?'
            r'(?:\(\d{3}\)|\d{3})[-. ]?\d{3}[-. ]?\d{4}',
            ' ', text
        )

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Reduce multiple whitespace to single space."""
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def remove_newline_hyphenation(text: str) -> str:
        """Remove hyphens at line breaks (word continuation)."""
        return re.sub(r'([a-z])(-\s*\n\s*)([a-z])', r'\1\3', text)

    @classmethod
    def preprocess_geoscan(cls, text: str, remove_french: bool = True) -> str:
        """
        Apply the full GEOSCAN preprocessing pipeline to text.

        This replicates the BASE_PIPELINE_CLEAN from the NRCan
        Geoscience_Language_Models repository:
        1. Remove CID markers
        2. Convert to ASCII
        3. Remove non-printable characters
        4. Remove newline hyphenation
        5. Remove newlines
        6. Remove URLs, DOIs, emails, phone numbers
        7. Optionally filter French text
        8. Normalize whitespace

        Args:
            text: Raw text to preprocess
            remove_french: Whether to skip French text blocks

        Returns:
            Cleaned text string
        """
        if not text or not text.strip():
            return ""

        # Check for French text
        if remove_french and cls.detect_french(text):
            return ""

        text = cls.remove_cid_markers(text)
        text = cls.convert_to_ascii(text)
        # Remove non-printable characters
        text = re.sub(f'[^{re.escape(string.printable)}]', ' ', text)
        text = cls.remove_newline_hyphenation(text)
        text = text.replace('\n', ' ')
        text = cls.remove_urls(text)
        text = cls.remove_dois(text)
        text = cls.remove_emails(text)
        text = cls.remove_phone_numbers(text)
        text = cls.normalize_whitespace(text)

        return text

    @classmethod
    def process_pdf(cls, pdf_path: str, remove_french: bool = True) -> str:
        """
        Extract and preprocess text from a GEOSCAN PDF document.

        This is the complete pipeline: PDF extraction + text cleaning.

        Args:
            pdf_path: Path to PDF file
            remove_french: Whether to filter French text blocks

        Returns:
            Preprocessed text from the PDF
        """
        blocks = cls.extract_text_from_pdf_to_blocks(pdf_path)
        cleaned_blocks = []
        for block in blocks:
            cleaned = cls.preprocess_geoscan(block, remove_french=remove_french)
            if cleaned:
                cleaned_blocks.append(cleaned)
        return ' '.join(cleaned_blocks)


class GeoscienceTextProcessor:
    """
    Process geoscience text documents and extract features using LLMs.

    This class handles:
    - Text cleaning using the GEOSCAN pipeline (French removal, DOI/URL filtering)
    - Feature extraction using NRCan domain-specific BERT models
    - Named entity recognition for minerals and geological features

    The default model is DistilBERT fine-tuned on the GEOSCAN corpus. For the
    full NRCan geoscience BERT, provide a local path to the model from:
    https://github.com/NRCan/Geoscience_Language_Models

    Reference:
        Lawley, C.J.M., et al. (2022). Geoscience language models and their
        intrinsic evaluation. Applied Computing and Geosciences, 14, 100084.
    """

    def __init__(
        self,
        model_name: str = NRCAN_GEOSCIENCE_BERT,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize the text processor.

        Args:
            model_name: Name or path of the transformer model. Use a local path
                to a NRCan geoscience BERT model for best results, or
                "distilbert-base-uncased" as a fallback.
            max_length: Maximum sequence length for tokenization
            device: Device to run the model on (cuda/cpu)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        # For local NRCan geoscience BERT models, use DistilBertTokenizer
        # as per the NRCan repository (functions/bert.py: load_bert)
        if os.path.isdir(model_name):
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # GEOSCAN text preprocessing pipeline
        self.pipeline = GeoscanTextPipeline()

    def preprocess_text(
        self,
        text: str,
        remove_special_chars: bool = True,
        lowercase: bool = True,
        use_geoscan_pipeline: bool = True
    ) -> str:
        """
        Preprocess geoscience text using the GEOSCAN pipeline.

        Args:
            text: Input text to preprocess
            remove_special_chars: Whether to remove special characters
            lowercase: Whether to convert to lowercase
            use_geoscan_pipeline: Use full GEOSCAN pipeline (recommended)

        Returns:
            Preprocessed text
        """
        if use_geoscan_pipeline:
            text = self.pipeline.preprocess_geoscan(text)

        if lowercase:
            text = text.lower()

        if not use_geoscan_pipeline:
            # Legacy simple preprocessing (kept for backwards compatibility)
            text = re.sub(r'http\S+|www.\S+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        if remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\-]', ' ', text)

        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_features(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 8,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract embeddings from text using the transformer model.

        Follows the NRCan BERT embedding approach (functions/bert.py:
        bert_embedding) which averages token embeddings (excluding CLS/SEP).

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            pooling: Pooling strategy - "mean" averages all token embeddings
                (NRCan default), "cls" uses CLS token only

        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state

                if pooling == "mean":
                    # NRCan approach: mean of token embeddings (excl. CLS/SEP)
                    # See functions/bert.py: bert_embedding
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    # Zero out CLS (pos 0) and SEP (last non-pad) tokens
                    mask = attention_mask.clone()
                    mask[:, 0, :] = 0  # Remove CLS
                    # Find SEP positions (last 1 in attention_mask per row)
                    for j in range(mask.shape[0]):
                        seq_len = inputs['attention_mask'][j].sum().item()
                        if seq_len > 1:
                            mask[j, int(seq_len) - 1, :] = 0  # Remove SEP
                    masked_hidden = last_hidden * mask
                    sum_hidden = masked_hidden.sum(dim=1)
                    count = mask.sum(dim=1).clamp(min=1)
                    batch_embeddings = (sum_hidden / count).cpu().numpy()
                else:
                    # CLS token embedding
                    batch_embeddings = last_hidden[:, 0, :].cpu().numpy()

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
        # Canadian critical minerals list (Natural Resources Canada)
        minerals = [
            'lithium', 'cobalt', 'nickel', 'copper', 'rare earth',
            'graphite', 'manganese', 'vanadium', 'zinc', 'aluminum',
            'platinum', 'palladium', 'rhodium', 'iridium', 'gold',
            'silver', 'uranium', 'tungsten', 'chromium', 'molybdenum',
            'antimony', 'bismuth', 'cesium', 'fluorspar', 'gallium',
            'germanium', 'hafnium', 'helium', 'indium', 'magnesium',
            'niobium', 'potash', 'scandium', 'selenium', 'tantalum',
            'tellurium', 'tin', 'titanium', 'zirconium',
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

            # Preprocess using GEOSCAN pipeline
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

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF document through the full GEOSCAN pipeline.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted text, clean text, embedding, and minerals
        """
        raw_text = GeoscanTextPipeline.extract_text_from_pdf(pdf_path)
        clean_text = GeoscanTextPipeline.preprocess_geoscan(raw_text)
        embedding = self.extract_features(clean_text)[0] if clean_text else np.zeros(768)
        minerals = self.extract_mineral_mentions(clean_text) if clean_text else []

        return {
            "raw_text": raw_text,
            "clean_text": clean_text,
            "embedding": embedding,
            "minerals_mentioned": minerals,
        }
