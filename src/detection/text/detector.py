"""
detection/text/detector.py

DharmaShield - Advanced Text Scam Detection Engine
--------------------------------------------------
• Production-grade scam detection with threat level assessment and confidence scoring
• Google Gemma 3n / MatFormer integration with Per-Layer Embedding (PLE) optimization
• Multi-language support with adaptive threat calculation
• Industry-standard confidence scoring and uncertainty quantification
• Real-time inference optimization for cross-platform deployment

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import asyncio
import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
import hashlib
import json
from pathlib import Path
from collections import defaultdict, deque

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, pipeline

# Project imports
from ...utils.logger import get_logger
from ...core.config_loader import load_config
from ...utils.language import detect_language, get_language_name
from .clean_text import clean_text
from .vectorize import vectorize_batch, get_vectorizer_stats
from .classifier import classify_text, ClassificationResult

logger = get_logger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class ThreatLevel(IntEnum):
    """
    Standardized threat levels for scam detection.
    Based on industry best practices for cybersecurity threat assessment.
    """
    NONE = 0      # No threat detected
    LOW = 1       # Minimal risk, stay alert
    MEDIUM = 2    # Moderate risk, exercise caution
    HIGH = 3      # Significant risk, do not proceed
    CRITICAL = 4  # Extreme risk, immediate action required
    
    @classmethod
    def from_confidence(cls, confidence: float, is_scam: bool) -> 'ThreatLevel':
        """Calculate threat level from confidence score and scam prediction."""
        if not is_scam:
            return cls.NONE
        
        if confidence >= 0.90:
            return cls.CRITICAL
        elif confidence >= 0.75:
            return cls.HIGH
        elif confidence >= 0.55:
            return cls.MEDIUM
        elif confidence >= 0.35:
            return cls.LOW
        else:
            return cls.NONE
    
    def description(self) -> str:
        """Get human-readable description of threat level."""
        descriptions = {
            self.NONE: "No threat detected",
            self.LOW: "Low threat - Stay alert",
            self.MEDIUM: "Medium threat - Exercise caution", 
            self.HIGH: "High threat - Do not proceed",
            self.CRITICAL: "Critical threat - Immediate action required"
        }
        return descriptions.get(self, "Unknown threat level")
    
    def color_code(self) -> str:
        """Get color code for UI display."""
        colors = {
            self.NONE: "#28a745",      # Green
            self.LOW: "#ffc107",       # Yellow
            self.MEDIUM: "#fd7e14",    # Orange
            self.HIGH: "#dc3545",      # Red
            self.CRITICAL: "#6f42c1"   # Purple
        }
        return colors.get(self, "#6c757d")  # Gray default


@dataclass
class DetectionResult:
    """
    Comprehensive detection result with threat assessment and explainability.
    """
    # Core detection results
    is_scam: bool = False
    confidence: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.NONE
    
    # Classification details
    predicted_labels: List[str] = None
    label_scores: Dict[str, float] = None
    
    # Uncertainty and quality metrics
    uncertainty: float = 0.0
    prediction_quality: str = "unknown"  # high, medium, low, unknown
    
    # Explainability
    explanation: str = ""
    risk_factors: List[str] = None
    protective_factors: List[str] = None
    
    # Context and metadata
    language: str = "en"
    processing_time: float = 0.0
    model_version: str = ""
    confidence_breakdown: Dict[str, Any] = None
    
    # Additional insights
    similar_patterns: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.predicted_labels is None:
            self.predicted_labels = []
        if self.label_scores is None:
            self.label_scores = {}
        if self.risk_factors is None:
            self.risk_factors = []
        if self.protective_factors is None:
            self.protective_factors = []
        if self.confidence_breakdown is None:
            self.confidence_breakdown = {}
        if self.similar_patterns is None:
            self.similar_patterns = []
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format for API responses."""
        return {
            'is_scam': self.is_scam,
            'confidence': round(self.confidence, 4),
            'threat_level': {
                'value': int(self.threat_level),
                'name': self.threat_level.name,
                'description': self.threat_level.description(),
                'color': self.threat_level.color_code()
            },
            'predicted_labels': self.predicted_labels,
            'label_scores': {k: round(v, 4) for k, v in self.label_scores.items()},
            'uncertainty': round(self.uncertainty, 4),
            'prediction_quality': self.prediction_quality,
            'explanation': self.explanation,
            'risk_factors': self.risk_factors,
            'protective_factors': self.protective_factors,
            'language': self.language,
            'processing_time': round(self.processing_time * 1000, 2),  # Convert to ms
            'model_version': self.model_version,
            'confidence_breakdown': self.confidence_breakdown,
            'similar_patterns': self.similar_patterns,
            'recommendations': self.recommendations
        }
    
    @property
    def summary(self) -> str:
        """Get a brief summary of the detection result."""
        if self.is_scam:
            return f"⚠️ SCAM DETECTED ({self.threat_level.name}) - {self.confidence:.1%} confidence"
        else:
            return f"✅ No scam detected - {self.confidence:.1%} confidence"


class DetectorConfig:
    """Configuration class for the text detector."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        detector_config = self.config.get('detector', {})
        
        # Model configuration
        self.model_type = detector_config.get('model_type', 'gemma3n')
        self.model_path = detector_config.get('model_path', 'models/gemma3n_detector.pth')
        self.use_classifier = detector_config.get('use_classifier', True)
        self.enable_ple_optimization = detector_config.get('enable_ple_optimization', True)
        
        # Confidence and threshold settings
        self.confidence_threshold = detector_config.get('confidence_threshold', 0.5)
        self.uncertainty_threshold = detector_config.get('uncertainty_threshold', 0.3)
        self.quality_thresholds = detector_config.get('quality_thresholds', {
            'high': 0.8, 'medium': 0.6, 'low': 0.4
        })
        
        # Performance settings
        self.enable_caching = detector_config.get('enable_caching', True)
        self.cache_size = detector_config.get('cache_size', 1000)
        self.batch_optimization = detector_config.get('batch_optimization', True)
        self.async_processing = detector_config.get('async_processing', True)
        
        # Explainability settings
        self.enable_explanations = detector_config.get('enable_explanations', True)
        self.explanation_detail_level = detector_config.get('explanation_detail_level', 'medium')
        self.include_recommendations = detector_config.get('include_recommendations', True)
        
        # Language support
        self.supported_languages = detector_config.get('supported_languages', 
                                                      ['en', 'hi', 'es', 'fr', 'de', 'zh'])
        self.auto_language_detection = detector_config.get('auto_language_detection', True)
        
        # Monitoring and metrics
        self.enable_metrics = detector_config.get('enable_metrics', True)
        self.metrics_window_size = detector_config.get('metrics_window_size', 1000)


class ThreatAssessmentEngine:
    """
    Advanced threat assessment engine for calculating threat levels and confidence scores.
    Implements industry-standard threat intelligence practices.
    """
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.threat_patterns = self._load_threat_patterns()
        self.protective_patterns = self._load_protective_patterns()
        
    def _load_threat_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load known threat patterns and their weights."""
        return {
            'urgency_indicators': {
                'patterns': ['urgent', 'immediate', 'expires', 'limited time', 'act now', 
                           'hurry', 'don\'t miss', 'last chance', 'deadline'],
                'weight': 0.3,
                'languages': {
                    'hi': ['तुरंत', 'जल्दी', 'अंतिम', 'समय सीमा'],
                    'es': ['urgente', 'inmediato', 'expire', 'último'],
                    'fr': ['urgent', 'immédiat', 'expire', 'dernier'],
                    'de': ['dringend', 'sofort', 'läuft ab', 'letzte']
                }
            },
            'financial_indicators': {
                'patterns': ['money', 'prize', 'winner', 'lottery', 'inheritance', 
                           'investment', 'profit', 'earnings', 'refund', 'tax'],
                'weight': 0.4,
                'languages': {
                    'hi': ['पैसे', 'पुरस्कार', 'विजेता', 'लॉटरी', 'निवेश'],
                    'es': ['dinero', 'premio', 'ganador', 'lotería', 'inversión'],
                    'fr': ['argent', 'prix', 'gagnant', 'loterie', 'investissement'],
                    'de': ['geld', 'preis', 'gewinner', 'lotterie', 'investition']
                }
            },
            'authority_impersonation': {
                'patterns': ['bank', 'government', 'tax office', 'police', 'court',
                           'irs', 'fbi', 'paypal', 'amazon', 'microsoft'],
                'weight': 0.5,
                'languages': {
                    'hi': ['बैंक', 'सरकार', 'पुलिस', 'अदालत'],
                    'es': ['banco', 'gobierno', 'policía', 'tribunal'],
                    'fr': ['banque', 'gouvernement', 'police', 'tribunal'],
                    'de': ['bank', 'regierung', 'polizei', 'gericht']
                }
            },
            'personal_info_request': {
                'patterns': ['password', 'ssn', 'social security', 'credit card',
                           'account number', 'pin', 'verify', 'confirm', 'update'],
                'weight': 0.6,
                'languages': {
                    'hi': ['पासवर्ड', 'खाता', 'पुष्टि करें', 'अपडेट'],
                    'es': ['contraseña', 'cuenta', 'verificar', 'actualizar'],
                    'fr': ['mot de passe', 'compte', 'vérifier', 'mettre à jour'],
                    'de': ['passwort', 'konto', 'bestätigen', 'aktualisieren']
                }
            },
            'threat_indicators': {
                'patterns': ['suspend', 'freeze', 'close', 'terminate', 'legal action',
                           'arrest', 'fine', 'penalty', 'consequences'],
                'weight': 0.4,
                'languages': {
                    'hi': ['बंद', 'जमा', 'कानूनी कार्रवाई', 'जुर्माना'],
                    'es': ['suspender', 'congelar', 'cerrar', 'multa'],
                    'fr': ['suspendre', 'geler', 'fermer', 'amende'],
                    'de': ['sperren', 'einfrieren', 'schließen', 'strafe']
                }
            }
        }
    
    def _load_protective_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load patterns that indicate legitimate communication."""
        return {
            'legitimate_business': {
                'patterns': ['unsubscribe', 'privacy policy', 'terms of service',
                           'customer service', 'help desk', 'support'],
                'weight': -0.2
            },
            'personal_communication': {
                'patterns': ['hi', 'hello', 'how are you', 'hope you are well',
                           'see you soon', 'talk to you later'],
                'weight': -0.3
            },
            'transaction_receipts': {
                'patterns': ['receipt', 'order confirmation', 'invoice',
                           'transaction id', 'order number'],
                'weight': -0.4
            }
        }
    
    def calculate_pattern_score(self, text: str, language: str) -> Tuple[float, List[str], List[str]]:
        """
        Calculate pattern-based threat score.
        
        Returns:
            Tuple of (pattern_score, risk_factors, protective_factors)
        """
        text_lower = text.lower()
        risk_factors = []
        protective_factors = []
        total_threat_score = 0.0
        total_protective_score = 0.0
        
        # Check threat patterns
        for category, config in self.threat_patterns.items():
            patterns = config['patterns']
            weight = config['weight']
            
            # Add language-specific patterns
            if language in config.get('languages', {}):
                patterns.extend(config['languages'][language])
            
            matches = [pattern for pattern in patterns if pattern in text_lower]
            if matches:
                category_score = len(matches) * weight
                total_threat_score += category_score
                risk_factors.extend([f"{category}: {match}" for match in matches])
        
        # Check protective patterns
        for category, config in self.protective_patterns.items():
            patterns = config['patterns']
            weight = config['weight']  # Negative weight
            
            matches = [pattern for pattern in patterns if pattern in text_lower]
            if matches:
                category_score = len(matches) * abs(weight)
                total_protective_score += category_score
                protective_factors.extend([f"{category}: {match}" for match in matches])
        
        # Combine scores (protective factors reduce threat)
        final_score = max(0.0, total_threat_score - total_protective_score)
        
        return final_score, risk_factors, protective_factors
    
    def calculate_confidence_quality(self, confidence: float, uncertainty: float) -> str:
        """Determine prediction quality based on confidence and uncertainty."""
        if confidence >= self.config.quality_thresholds['high'] and uncertainty <= 0.2:
            return 'high'
        elif confidence >= self.config.quality_thresholds['medium'] and uncertainty <= 0.4:
            return 'medium'
        elif confidence >= self.config.quality_thresholds['low']:
            return 'low'
        else:
            return 'unknown'
    
    def generate_recommendations(self, 
                               is_scam: bool, 
                               threat_level: ThreatLevel,
                               risk_factors: List[str],
                               language: str) -> List[str]:
        """Generate actionable recommendations based on detection results."""
        recommendations = []
        
        if is_scam:
            if threat_level >= ThreatLevel.HIGH:
                recommendations.extend([
                    "🚨 Do not click any links or download attachments",
                    "🚨 Do not provide any personal or financial information",
                    "🚨 Report this message to relevant authorities"
                ])
            elif threat_level >= ThreatLevel.MEDIUM:
                recommendations.extend([
                    "⚠️ Verify sender identity through official channels",
                    "⚠️ Be cautious with any requested actions",
                    "⚠️ Consider this message suspicious"
                ])
            else:
                recommendations.extend([
                    "👀 Exercise caution and verify information",
                    "👀 Double-check sender identity if unsure"
                ])
            
            # Specific recommendations based on risk factors
            if any('financial' in factor.lower() for factor in risk_factors):
                recommendations.append("💰 Never share financial information via unsolicited messages")
            
            if any('personal_info' in factor.lower() for factor in risk_factors):
                recommendations.append("🔒 Legitimate organizations never ask for passwords via email/text")
            
            if any('urgency' in factor.lower() for factor in risk_factors):
                recommendations.append("⏰ Scammers use urgency to pressure quick decisions - take your time")
        
        else:
            recommendations.append("✅ Message appears legitimate, but always stay vigilant")
        
        return recommendations


class AdvancedTextDetector:
    """
    Production-grade text scam detector with comprehensive threat assessment.
    
    Features:
    - Google Gemma 3n integration with MatFormer optimization
    - Multi-language support with language-specific threat patterns
    - Advanced confidence scoring and uncertainty quantification
    - Industry-standard threat level assessment
    - Real-time performance monitoring and caching
    - Comprehensive explainability and recommendations
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
        
        self.config = DetectorConfig(config_path)
        self.threat_engine = ThreatAssessmentEngine(self.config)
        
        # Performance monitoring
        self.metrics = defaultdict(list)
        self.prediction_cache = {} if self.config.enable_caching else None
        self.recent_predictions = deque(maxlen=self.config.metrics_window_size)
        
        # Initialize components
        self._initialize_components()
        self._initialized = True
        
        logger.info(f"Advanced Text Detector initialized with {self.config.model_type}")
    
    def _initialize_components(self):
        """Initialize detection components and models."""
        try:
            # Initialize classifier if enabled
            if self.config.use_classifier:
                from .classifier import get_classifier
                self.classifier = get_classifier()
                logger.info("Multi-label classifier initialized")
            else:
                self.classifier = None
                logger.info("Using pattern-based detection only")
            
            # Initialize vectorizer for embedding-based features
            from .vectorize import initialize_vectorizer
            initialize_vectorizer()
            logger.info("Text vectorizer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _get_cache_key(self, text: str, language: str) -> str:
        """Generate cache key for predictions."""
        content = f"{tex
