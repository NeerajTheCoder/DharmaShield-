"""
detection/audio/analyze_speech.py

DharmaShield - Advanced Speech Transcription Analysis Engine
----------------------------------------------------------
• Production-grade speech analysis with text-level scam detection on transcribed content
• Multi-modal audio-text analysis pipeline using Gemma 3n architecture
• Advanced NLP processing with context-aware threat assessment
• Real-time speech pattern analysis and linguistic fraud detection
• Cross-platform deployment ready for Android, iOS, and desktop environments

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import asyncio
import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
import numpy as np
import json
import hashlib
from pathlib import Path
from collections import defaultdict, deque
import re

# NLP and ML imports
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel, pipeline
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available - advanced NLP analysis disabled")

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    warnings.warn("NLTK not available - basic NLP features disabled")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("Scikit-learn not available - advanced text analysis disabled")

# Project imports
from ...utils.logger import get_logger
from ...core.config_loader import load_config
from ...utils.language import detect_language, get_language_name
from .transcribe import get_transcriber, TranscriptionResult
from ..text.detector import detect_scam, DetectionResult
from ..text.clean_text import clean_text
from ..text.classifier import classify_text, ClassificationResult

logger = get_logger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class SpeechAnalysisType(Enum):
    """Types of speech analysis performed."""
    TRANSCRIPTION_QUALITY = "transcription_quality"
    LINGUISTIC_PATTERNS = "linguistic_patterns"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    INTENT_DETECTION = "intent_detection"
    EMOTION_ANALYSIS = "emotion_analysis"
    COHERENCE_ANALYSIS = "coherence_analysis"
    CONTEXTUAL_ANALYSIS = "contextual_analysis"

class SpeechThreatLevel(IntEnum):
    """Speech-specific threat levels based on linguistic analysis."""
    NONE = 0          # No linguistic threats detected
    LOW = 1           # Minor suspicious patterns
    MEDIUM = 2        # Moderate linguistic anomalies
    HIGH = 3          # Strong fraud indicators
    CRITICAL = 4      # Definitive scam patterns

    def description(self) -> str:
        """Get human-readable description."""
        descriptions = {
            self.NONE: "No linguistic threats detected in speech",
            self.LOW: "Minor suspicious speech patterns detected",
            self.MEDIUM: "Moderate linguistic fraud indicators present",
            self.HIGH: "Strong scam patterns in speech content",
            self.CRITICAL: "Critical fraud indicators - definitive scam speech"
        }
        return descriptions.get(self, "Unknown threat level")

@dataclass
class SpeechAnalysisResult:
    """
    Comprehensive speech analysis result combining transcription and text-level detection.
    """
    # Transcription results
    transcription_result: TranscriptionResult = None
    transcription_quality: str = "unknown"  # excellent, good, fair, poor, unknown
    
    # Text-level detection results
    text_detection_result: DetectionResult = None
    classification_result: ClassificationResult = None
    
    # Speech-specific linguistic analysis
    linguistic_patterns: Dict[str, Any] = None
    semantic_features: Dict[str, float] = None
    intent_analysis: Dict[str, Any] = None
    emotion_indicators: Dict[str, float] = None
    
    # Coherence and flow analysis
    speech_coherence: float = 0.0
    narrative_flow: str = "unknown"
    topic_consistency: float = 0.0
    
    # Context and metadata
    language: str = "en"
    confidence: float = 0.0
    threat_level: SpeechThreatLevel = SpeechThreatLevel.NONE
    
    # Processing metadata
    processing_time: float = 0.0
    analysis_types_performed: List[SpeechAnalysisType] = None
    
    # Comprehensive assessment
    overall_assessment: str = ""
    risk_factors: List[str] = None
    protective_factors: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.linguistic_patterns is None:
            self.linguistic_patterns = {}
        if self.semantic_features is None:
            self.semantic_features = {}
        if self.intent_analysis is None:
            self.intent_analysis = {}
        if self.emotion_indicators is None:
            self.emotion_indicators = {}
        if self.analysis_types_performed is None:
            self.analysis_types_performed = []
        if self.risk_factors is None:
            self.risk_factors = []
        if self.protective_factors is None:
            self.protective_factors = []
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'transcription_result': self.transcription_result.to_dict() if self.transcription_result else None,
            'transcription_quality': self.transcription_quality,
            'text_detection_result': self.text_detection_result.to_dict() if self.text_detection_result else None,
            'classification_result': self.classification_result.to_dict() if self.classification_result else None,
            'linguistic_patterns': self.linguistic_patterns,
            'semantic_features': {k: round(v, 4) for k, v in self.semantic_features.items()},
            'intent_analysis': self.intent_analysis,
            'emotion_indicators': {k: round(v, 4) for k, v in self.emotion_indicators.items()},
            'speech_coherence': round(self.speech_coherence, 4),
            'narrative_flow': self.narrative_flow,
            'topic_consistency': round(self.topic_consistency, 4),
            'language': self.language,
            'confidence': round(self.confidence, 4),
            'threat_level': {
                'value': int(self.threat_level),
                'name': self.threat_level.name,
                'description': self.threat_level.description()
            },
            'processing_time': round(self.processing_time * 1000, 2),
            'analysis_types_performed': [t.value for t in self.analysis_types_performed],
            'overall_assessment': self.overall_assessment,
            'risk_factors': self.risk_factors,
            'protective_factors': self.protective_factors,
            'recommendations': self.recommendations
        }
    
    @property
    def is_scam_speech(self) -> bool:
        """Check if speech content indicates scam."""
        return (self.text_detection_result and self.text_detection_result.is_scam) or \
               (self.classification_result and self.classification_result.is_scam) or \
               self.threat_level >= SpeechThreatLevel.HIGH
    
    @property
    def summary(self) -> str:
        """Get a brief summary of the speech analysis result."""
        if self.is_scam_speech:
            return f"⚠️ SCAM SPEECH DETECTED ({self.threat_level.name}) - {self.confidence:.1%} confidence"
        else:
            return f"✅ Speech appears legitimate - {self.confidence:.1%} confidence"


class SpeechAnalysisConfig:
    """Configuration class for speech analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        analysis_config = self.config.get('speech_analysis', {})
        
        # Analysis modules to enable
        self.enabled_analyses = analysis_config.get('enabled_analyses', [
            'transcription_quality', 'linguistic_patterns', 'semantic_analysis',
            'intent_detection', 'emotion_analysis', 'coherence_analysis'
        ])
        
        # Transcription settings
        self.transcription_engine = analysis_config.get('transcription_engine', 'auto')
        self.transcription_language = analysis_config.get('transcription_language', 'auto')
        self.require_transcription = analysis_config.get('require_transcription', True)
        
        # Text analysis settings
        self.enable_text_detection = analysis_config.get('enable_text_detection', True)
        self.enable_classification = analysis_config.get('enable_classification', True)
        self.text_cleaning_enabled = analysis_config.get('text_cleaning_enabled', True)
        
        # NLP processing settings
        self.min_text_length = analysis_config.get('min_text_length', 10)
        self.max_text_length = analysis_config.get('max_text_length', 10000)
        self.sentence_tokenization = analysis_config.get('sentence_tokenization', True)
        
        # Linguistic analysis thresholds
        self.coherence_threshold = analysis_config.get('coherence_threshold', 0.6)
        self.topic_consistency_threshold = analysis_config.get('topic_consistency_threshold', 0.7)
        self.confidence_threshold = analysis_config.get('confidence_threshold', 0.75)
        
        # Advanced features
        self.enable_semantic_similarity = analysis_config.get('enable_semantic_similarity', True)
        self.enable_emotion_detection = analysis_config.get('enable_emotion_detection', True)
        self.enable_intent_classification = analysis_config.get('enable_intent_classification', True)
        
        # Performance settings
        self.enable_caching = analysis_config.get('enable_caching', True)
        self.cache_size = analysis_config.get('cache_size', 500)
        self.batch_processing = analysis_config.get('batch_processing', True)
        
        # Language support
        self.supported_languages = analysis_config.get('supported_languages', [
            'en', 'hi', 'es', 'fr', 'de', 'zh', 'ar', 'ru'
        ])


class TranscriptionQualityAnalyzer:
    """
    Analyzes the quality of transcription results for downstream processing.
    """
    
    def __init__(self, config: SpeechAnalysisConfig):
        self.config = config
    
    def assess_transcription_quality(self, transcription_result: TranscriptionResult) -> Tuple[str, Dict[str, Any]]:
        """Assess the quality of transcription for analysis reliability."""
        if not transcription_result or not transcription_result.is_successful:
            return "poor", {"error": "Transcription failed or empty"}
        
        quality_metrics = {}
        quality_score = 0.0
        total_checks = 0
        
        # Check confidence score
        if transcription_result.confidence > 0:
            confidence_score = transcription_result.confidence
            quality_metrics['confidence_score'] = confidence_score
            quality_score += confidence_score
            total_checks += 1
        
        # Check text length and completeness
        text_length = len(transcription_result.text.strip())
        quality_metrics['text_length'] = text_length
        
        if text_length >= self.config.min_text_length:
            length_score = min(1.0, text_length / 100)  # Normalize to reasonable length
            quality_score += length_score
            total_checks += 1
        
        # Check for transcription artifacts
        text = transcription_result.text.lower()
        artifacts = []
        
        # Common transcription errors
        if '[inaudible]' in text or '[unclear]' in text or '***' in text:
            artifacts.append("Contains inaudible markers")
        
        if len(re.findall(r'\b\w{1,2}\b', text)) > len(text.split()) * 0.3:
            artifacts.append("High ratio of very short words")
        
        if len(re.findall(r'[aeiou]{4,}', text)) > 3:
            artifacts.append("Unusual vowel sequences")
        
        quality_metrics['artifacts'] = artifacts
        artifact_penalty = len(artifacts) * 0.1
        quality_score = max(0.0, quality_score - artifact_penalty)
        
        # Check language consistency
        if transcription_result.language:
            detected_lang = detect_language(transcription_result.text)
            if detected_lang == transcription_result.language:
                quality_score += 0.2
                total_checks += 1
                quality_metrics['language_consistent'] = True
            else:
                quality_metrics['language_consistent'] = False
        
        # Calculate final quality
        if total_checks > 0:
            final_score = quality_score / total_checks
        else:
            final_score = 0.0
        
        quality_metrics['final_score'] = final_score
        
        # Determine quality level
        if final_score >= 0.85:
            quality_level = "excellent"
        elif final_score >= 0.7:
            quality_level = "good"
        elif final_score >= 0.5:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return quality_level, quality_metrics


class LinguisticPatternAnalyzer:
    """
    Advanced linguistic pattern analysis for detecting scam-specific language patterns.
    """
    
    def __init__(self, config: SpeechAnalysisConfig):
        self.config = config
        self._load_scam_patterns()
    
    def _load_scam_patterns(self):
        """Load scam-specific linguistic patterns."""
        self.urgency_patterns = [
            r'\b(urgent|immediately|right now|asap|time is running out)\b',
            r'\b(act now|don\'t wait|limited time|expires soon)\b',
            r'\b(hurry|quick|fast|rush)\b'
        ]
        
        self.pressure_patterns = [
            r'\b(must|have to|need to|required to)\b',
            r'\b(or else|otherwise|consequences|penalty)\b',
            r'\b(final notice|last chance|deadline)\b'
        ]
        
        self.authority_patterns = [
            r'\b(government|official|authority|agency)\b',
            r'\b(police|court|legal|lawsuit)\b',
            r'\b(bank|financial|credit|account)\b',
            r'\b(microsoft|apple|google|amazon)\b'
        ]
        
        self.reward_patterns = [
            r'\b(win|won|winner|prize|reward)\b',
            r'\b(free|gift|bonus|special offer)\b',
            r'\b(money|cash|dollars|pounds|euros)\b',
            r'\b(lottery|sweepstakes|contest)\b'
        ]
        
        self.personal_info_patterns = [
            r'\b(social security|ssn|credit card|bank account)\b',
            r'\b(password|pin|verification|confirm)\b',
            r'\b(personal information|details|data)\b'
        ]
        
        self.threat_patterns = [
            r'\b(suspend|close|terminate|cancel)\b',
            r'\b(arrest|jail|prison|legal action)\b',
            r'\b(fine|penalty|charges|fee)\b'
        ]
    
    def analyze_linguistic_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Comprehensive linguistic pattern analysis."""
        if not text or len(text.strip()) < self.config.min_text_length:
            return {}
        
        text_lower = text.lower()
        patterns = {}
        
        # Analyze each pattern category
        patterns['urgency_score'] = self._calculate_pattern_score(text_lower, self.urgency_patterns)
        patterns['pressure_score'] = self._calculate_pattern_score(text_lower, self.pressure_patterns)
        patterns['authority_score'] = self._calculate_pattern_score(text_lower, self.authority_patterns)
        patterns['reward_score'] = self._calculate_pattern_score(text_lower, self.reward_patterns)
        patterns['personal_info_score'] = self._calculate_pattern_score(text_lower, self.personal_info_patterns)
        patterns['threat_score'] = self._calculate_pattern_score(text_lower, self.threat_patterns)
        
        # Calculate aggregate scores
        patterns['total_scam_score'] = sum(patterns.values())
        patterns['normalized_score'] = min(1.0, patterns['total_scam_score'] / 6.0)
        
        # Analyze sentence structure
        if HAS_NLTK:
            patterns.update(self._analyze_sentence_structure(text))
        
        # Analyze vocabulary complexity
        patterns.update(self._analyze_vocabulary(text_lower))
        
        return patterns
    
    def _calculate_pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate score for a specific pattern category."""
        matches = 0
        for pattern in patterns:
            matches += len(re.findall(pattern, text, re.IGNORECASE))
        return min(1.0, matches / max(1, len(text.split()) / 10))  # Normalize by text length
    
    def _analyze_sentence_structure(self, text: str) -> Dict[str, float]:
        """Analyze sentence structure patterns."""
        try:
            sentences = sent_tokenize(text)
            
            # Basic structure metrics
            avg_sentence_length = np.mean([len(sent.split()) for sent in sentences])
            sentence_count = len(sentences)
            
            # Question/exclamation ratio
            questions = sum(1 for sent in sentences if '?' in sent)
            exclamations = sum(1 for sent in sentences if '!' in sent)
            
            return {
                'avg_sentence_length': avg_sentence_length,
                'sentence_count': sentence_count,
                'question_ratio': questions / max(1, sentence_count),
                'exclamation_ratio': exclamations / max(1, sentence_count)
            }
        except Exception as e:
            logger.warning(f"Sentence structure analysis failed: {e}")
            return {}
    
    def _analyze_vocabulary(self, text: str) -> Dict[str, float]:
        """Analyze vocabulary characteristics."""
        words = text.split()
        if not words:
            return {}
        
        # Basic vocabulary metrics
        unique_words = set(words)
        vocab_diversity = len(unique_words) / len(words)
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Complexity indicators
        long_words = sum(1 for word in words if len(word) > 6)
        complexity_ratio = long_words / len(words)
        
        return {
            'vocab_diversity': vocab_diversity,
            'avg_word_length': avg_word_length,
            'complexity_ratio': complexity_ratio,
            'total_words': len(words),
            'unique_words': len(unique_words)
        }


class SemanticAnalyzer:
    """
    Advanced semantic analysis using embeddings and similarity metrics.
    """
    
    def __init__(self, config: SpeechAnalysisConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize semantic analysis models."""
        if not HAS_TORCH:
            return
        
        try:
            # Use a lightweight model for semantic analysis
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model
