"""
detection/text/classifier.py

DharmaShield - Advanced Multi-Label/Multi-Task Text Classifier
--------------------------------------------------------------
• Production-grade multi-label classification with Gemma 3n/MatFormer support
• Multi-task learning architecture for scam detection, intent, and threat assessment
• Cross-task attention mechanism for enhanced performance
• Fully offline, cross-platform optimized for mobile/desktop deployment
• Industry-standard explainability and performance monitoring

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import asyncio
import threading
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from pathlib import Path
import numpy as np
import json
import pickle
from collections import defaultdict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    classification_report, 
    multilabel_confusion_matrix,
    hamming_loss,
    jaccard_score
)

# Project imports
from ...utils.logger import get_logger
from ...core.config_loader import load_config
from ...utils.language import detect_language, get_language_name
from .vectorize import vectorize_batch
from .clean_text import clean_text

logger = get_logger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class ClassificationResult:
    """
    Comprehensive result object for multi-label/multi-task classification.
    """
    # Primary results
    predicted_labels: List[str] = field(default_factory=list)
    label_scores: Dict[str, float] = field(default_factory=dict)
    
    # Multi-task results
    auxiliary_predictions: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence and uncertainty
    confidence: float = 0.0
    uncertainty: float = 0.0
    
    # Explainability
    explanations: Dict[str, str] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    processing_time: float = 0.0
    model_version: str = ""
    language: str = "en"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'predicted_labels': self.predicted_labels,
            'label_scores': self.label_scores,
            'auxiliary_predictions': self.auxiliary_predictions,
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'explanations': self.explanations,
            'feature_importance': self.feature_importance,
            'processing_time': self.processing_time,
            'model_version': self.model_version,
            'language': self.language
        }
    
    @property
    def is_scam(self) -> bool:
        """Check if any scam-related label is predicted."""
        scam_labels = {'scam', 'phishing', 'fraud', 'spam', 'malware'}
        return bool(set(self.predicted_labels) & scam_labels)
    
    @property
    def threat_level(self) -> int:
        """Calculate threat level based on predictions."""
        if not self.is_scam:
            return 0
        
        max_score = max([self.label_scores.get(label, 0.0) 
                        for label in self.predicted_labels if label in 
                        {'scam', 'phishing', 'fraud', 'spam', 'malware'}] or [0.0])
        
        if max_score >= 0.9:
            return 4  # Critical
        elif max_score >= 0.75:
            return 3  # High
        elif max_score >= 0.5:
            return 2  # Medium
        else:
            return 1  # Low


class ClassifierConfig:
    """Configuration class for the multi-label classifier."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        classifier_config = self.config.get('classifier', {})
        
        # Model configuration
        self.model_type = classifier_config.get('model_type', 'gemma3n')  # gemma3n, matformer, transformer
        self.model_path = classifier_config.get('model_path', 'models/gemma3n_classifier.pth')
        self.tokenizer_path = classifier_config.get('tokenizer_path', 'google/gemma-2b')
        
        # Label configuration
        self.primary_labels = classifier_config.get('primary_labels', [
            'not_scam', 'scam', 'phishing', 'fraud', 'spam', 'urgent', 'malware'
        ])
        self.auxiliary_labels = classifier_config.get('auxiliary_labels', [
            'intent_purchase', 'intent_info', 'intent_support', 'emotion_fear', 'emotion_greed'
        ])
        
        # Multi-task settings
        self.enable_multi_task = classifier_config.get('enable_multi_task', True)
        self.cross_task_attention = classifier_config.get('cross_task_attention', True)
        
        # Training parameters
        self.max_sequence_length = classifier_config.get('max_sequence_length', 512)
        self.batch_size = classifier_config.get('batch_size', 16)
        self.learning_rate = classifier_config.get('learning_rate', 2e-5)
        self.dropout_rate = classifier_config.get('dropout_rate', 0.1)
        
        # Threshold settings
        self.classification_threshold = classifier_config.get('classification_threshold', 0.5)
        self.confidence_threshold = classifier_config.get('confidence_threshold', 0.7)
        
        # Performance settings
        self.device = classifier_config.get('device', 'auto')
        self.use_mixed_precision = classifier_config.get('use_mixed_precision', True)
        self.enable_caching = classifier_config.get('enable_caching', True)
        self.cache_size = classifier_config.get('cache_size', 1000)
        
        # Language support
        self.supported_languages = classifier_config.get('supported_languages', 
                                                        ['en', 'hi', 'es', 'fr', 'de'])


class CrossTaskAttention(nn.Module):
    """
    Cross-task attention mechanism for multi-task learning.
    Enables information sharing between different classification tasks.
    """
    
    def __init__(self, hidden_size: int, num_attention_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query_task: torch.Tensor, key_value_tasks: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply cross-task attention.
        
        Args:
            query_task: Task representation seeking information [batch_size, hidden_size]
            key_value_tasks: List of other task representations [batch_size, hidden_size]
        
        Returns:
            Enhanced task representation with cross-task information
        """
        batch_size = query_task.size(0)
        
        # Transform query
        query_layer = self.query(query_task)
        query_layer = query_layer.view(batch_size, self.num_attention_heads, self.attention_head_size)
        
        # Combine all key-value tasks
        if not key_value_tasks:
            return query_task
        
        combined_kv = torch.stack(key_value_tasks, dim=1)  # [batch_size, num_tasks, hidden_size]
        
        key_layer = self.key(combined_kv)
        value_layer = self.value(combined_kv)
        
        key_layer = key_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        value_layer = value_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        
        # Compute attention scores
        attention_scores = torch.matmul(
            query_layer.unsqueeze(2), 
            key_layer.transpose(-1, -2)
        )
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        
        # Apply attention weights
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.view(batch_size, self.hidden_size)
        
        # Residual connection
        return query_task + context_layer


class GemmaMultiTaskClassifier(nn.Module):
    """
    Advanced multi-task classifier based on Gemma 3n architecture.
    Supports multi-label classification with cross-task attention.
    """
    
    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.config = config
        
        # Load base model
        if config.model_type == 'gemma3n':
            # In production, load actual Gemma 3n model
            # For now, using a compatible transformer as proxy
            self.base_model = AutoModel.from_pretrained('google/gemma-2b')
        else:
            self.base_model = AutoModel.from_pretrained(config.tokenizer_path)
        
        self.hidden_size = self.base_model.config.hidden_size
        
        # Task-specific heads
        self.primary_classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_size // 2, len(config.primary_labels))
        )
        
        if config.enable_multi_task:
            self.auxiliary_classifier = nn.Sequential(
                nn.Dropout(config.dropout_rate),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(self.hidden_size // 2, len(config.auxiliary_labels))
            )
            
            # Cross-task attention
            if config.cross_task_attention:
                self.cross_attention = CrossTaskAttention(self.hidden_size)
        
        # Feature extractor for interpretability
        self.feature_extractor = nn.Linear(self.hidden_size, self.hidden_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task outputs.
        
        Returns:
            Dictionary containing primary and auxiliary predictions
        """
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else sequence_output.mean(dim=1)
        
        # Extract features for interpretability
        features = self.feature_extractor(pooled_output)
        
        results = {
            'features': features,
            'primary_logits': self.primary_classifier(pooled_output)
        }
        
        if self.config.enable_multi_task:
            auxiliary_features = pooled_output
            
            # Apply cross-task attention if enabled
            if self.config.cross_task_attention and hasattr(self, 'cross_attention'):
                # Use primary task features to enhance auxiliary task
                auxiliary_features = self.cross_attention(pooled_output, [features])
            
            results['auxiliary_logits'] = self.auxiliary_classifier(auxiliary_features)
        
        return results


class AdvancedMultiLabelClassifier:
    """
    Production-ready multi-label, multi-task classifier for DharmaShield.
    
    Features:
    - Gemma 3n / MatFormer support with fallback to standard transformers
    - Multi-label classification with configurable thresholds
    - Multi-task learning with cross-task attention
    - Real-time performance monitoring and caching
    - Comprehensive explainability features
    - Thread-safe operation with singleton pattern
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if getattr(self, '_initialized', False):
            return
        
        self.config = ClassifierConfig(config_path)
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self.label_encoders = {}
        self.performance_metrics = defaultdict(list)
        self.prediction_cache = {} if self.config.enable_caching else None
        
        # Initialize model
        self._load_model()
        self._initialized = True
        
        logger.info("Advanced Multi-Label Classifier initialized")
    
    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')  # Apple Silicon
            else:
                return torch.device('cpu')
        return torch.device(self.config.device)
    
    def _load_model(self):
        """Load and initialize the classification model."""
        try:
            logger.info(f"Loading classifier model: {self.config.model_type}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            if Path(self.config.model_path).exists():
                # Load fine-tuned model
                self.model = torch.load(self.config.model_path, map_location=self.device)
                logger.info("Loaded fine-tuned model from checkpoint")
            else:
                # Initialize new model
                self.model = GemmaMultiTaskClassifier(self.config)
                logger.info("Initialized new model (no checkpoint found)")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Setup label encoders
            self._setup_label_encoders()
            
        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            raise
    
    def _setup_label_encoders(self):
        """Setup label encoding mappings."""
        self.label_encoders['primary'] = {
            label: idx for idx, label in enumerate(self.config.primary_labels)
        }
        
        if self.config.enable_multi_task:
            self.label_encoders['auxiliary'] = {
                label: idx for idx, label in enumerate(self.config.auxiliary_labels)
            }
    
    def _get_cache_key(self, text: str, language: str) -> str:
        """Generate cache key for predictions."""
        import hashlib
        content = f"{text}_{language}_{self.config.classification_threshold}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _apply_threshold(self, probabilities: np.ndarray, threshold: float) -> List[int]:
        """Apply classification threshold to probabilities."""
        return (probabilities >= threshold).astype(int).tolist()
    
    def _calculate_uncertainty(self, probabilities: np.ndarray) -> float:
        """Calculate prediction uncertainty using entropy."""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        probs = np.clip(probabilities, epsilon, 1 - epsilon)
        entropy = -np.sum(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
        return float(entropy / len(probabilities))  # Normalize by number of labels
    
    def _generate_explanations(self, 
                             text: str, 
                             predictions: Dict[str, np.ndarray],
                             predicted_labels: List[str]) -> Dict[str, str]:
        """Generate explanations for predictions."""
        explanations = {}
        
        for label in predicted_labels:
            if label in self.config.primary_labels:
                score = predictions['primary'][self.label_encoders['primary'][label]]
                explanations[label] = (
                    f"Classified as '{label}' with {score:.1%} confidence. "
                    f"Based on text patterns and contextual analysis."
                )
        
        return explanations
    
    def predict(self, 
                text: str,
                language: Optional[str] = None,
                return_probabilities: bool = True,
                explain: bool = False) -> ClassificationResult:
        """
        Predict labels for a single text input.
        
        Args:
            text: Input text to classify
            language: Language code (auto-detected if None)
            return_probabilities: Whether to return probability scores
            explain: Whether to generate explanations
            
        Returns:
            ClassificationResult with predictions and metadata
        """
        start_time = time.time()
        
        # Input validation and preprocessing
        if not text or not text.strip():
            return ClassificationResult(
                processing_time=time.time() - start_time,
                model_version=self.config.model_type
            )
        
        # Language detection
        if language is None:
            language = detect_language(text)
        
        # Check cache
        cache_key = None
        if self.prediction_cache is not None:
            cache_key = self._get_cache_key(text, language)
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                cached_result.processing_time = time.time() - start_time
                return cached_result
        
        try:
            # Clean and prepare text
            cleaned_text = clean_text(text, language=language)
            
            # Tokenize input
            encoding = self.tokenizer(
                cleaned_text,
                truncation=True,
                padding=True,
                max_length=self.config.max_sequence_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Model inference
            with torch.no_grad():
                if self.config.use_mixed_precision:
                    with torch.autocast(device_type=str(self.device).split(':')[0]):
                        model_outputs = self.model(input_ids, attention_mask)
                else:
                    model_outputs = self.model(input_ids, attention_mask)
            
            # Process predictions
            primary_probs = torch.sigmoid(model_outputs['primary_logits']).cpu().numpy().flatten()
            
            auxiliary_probs = None
            auxiliary_predictions = {}
            if self.config.enable_multi_task and 'auxiliary_logits' in model_outputs:
                auxiliary_probs = torch.sigmoid(model_outputs['auxiliary_logits']).cpu().numpy().flatten()
                aux_binary = self._apply_threshold(auxiliary_pro
