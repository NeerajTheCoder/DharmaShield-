"""
detection/audio/voice_clone_detector.py

DharmaShield - Advanced Voice Clone & Deepfake Detection Engine
---------------------------------------------------------------
• Production-grade voice authenticity verification using Gemma 3n audio pipeline
• Multi-modal deepfake detection with advanced neural network architectures
• Real-time voice liveness detection and spoofing countermeasures
• Industry-standard biometric anti-spoofing with comprehensive threat assessment
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
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
import json
import hashlib
from pathlib import Path
from collections import defaultdict, deque
import io

# Audio processing imports
try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("Librosa not available - advanced audio analysis disabled")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoProcessor, AutoModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available - neural network detection disabled")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("Scikit-learn not available - traditional ML detection disabled")

# Project imports
from ...utils.logger import get_logger
from ...core.config_loader import load_config
from ...utils.language import detect_language, get_language_name
from .transcribe import get_transcriber, TranscriptionResult

logger = get_logger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class VoiceAuthenticityLevel(IntEnum):
    """
    Voice authenticity assessment levels based on industry standards.
    """
    UNKNOWN = 0         # Cannot determine authenticity
    SYNTHETIC = 1       # Clearly artificial/generated
    SUSPICIOUS = 2      # Likely cloned or manipulated
    NATURAL = 3         # Appears genuine with minor artifacts
    AUTHENTIC = 4       # High confidence genuine human voice
    
    def description(self) -> str:
        """Get human-readable description."""
        descriptions = {
            self.UNKNOWN: "Cannot determine authenticity",
            self.SYNTHETIC: "Synthetic/AI-generated voice detected",
            self.SUSPICIOUS: "Voice shows signs of cloning or manipulation",
            self.NATURAL: "Voice appears natural with minor artifacts",
            self.AUTHENTIC: "High confidence authentic human voice"
        }
        return descriptions.get(self, "Unknown authenticity level")
    
    def color_code(self) -> str:
        """Get color code for UI display."""
        colors = {
            self.UNKNOWN: "#6c757d",      # Gray
            self.SYNTHETIC: "#dc3545",    # Red
            self.SUSPICIOUS: "#fd7e14",   # Orange
            self.NATURAL: "#ffc107",      # Yellow
            self.AUTHENTIC: "#28a745"     # Green
        }
        return colors.get(self, "#6c757d")

class SpoofingType(Enum):
    """Types of voice spoofing attacks."""
    REPLAY = "replay"                    # Pre-recorded audio playback
    TEXT_TO_SPEECH = "tts"              # Synthetic speech generation
    VOICE_CONVERSION = "vc"             # Voice characteristics conversion
    DEEPFAKE = "deepfake"               # AI-generated voice cloning
    IMPERSONATION = "impersonation"     # Human vocal mimicry
    UNKNOWN = "unknown"                 # Unidentified spoofing type

@dataclass
class VoiceCloneDetectionResult:
    """
    Comprehensive voice clone detection result with detailed analysis.
    """
    # Core detection results
    is_authentic: bool = True
    authenticity_level: VoiceAuthenticityLevel = VoiceAuthenticityLevel.UNKNOWN
    confidence: float = 0.0
    
    # Spoofing analysis
    detected_spoofing_types: List[SpoofingType] = None
    spoofing_probabilities: Dict[str, float] = None
    
    # Audio quality and characteristics
    audio_quality: str = "unknown"      # excellent, good, fair, poor, unknown
    signal_artifacts: List[str] = None
    prosody_analysis: Dict[str, Any] = None
    
    # Biometric features
    vocal_characteristics: Dict[str, float] = None
    liveness_score: float = 0.0
    
    # Technical analysis
    spectral_anomalies: List[str] = None
    temporal_inconsistencies: List[str] = None
    
    # Processing metadata
    processing_time: float = 0.0
    model_version: str = ""
    language: str = "en"
    
    # Explanation and recommendations
    explanation: str = ""
    risk_assessment: str = ""
    recommendations: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.detected_spoofing_types is None:
            self.detected_spoofing_types = []
        if self.spoofing_probabilities is None:
            self.spoofing_probabilities = {}
        if self.signal_artifacts is None:
            self.signal_artifacts = []
        if self.prosody_analysis is None:
            self.prosody_analysis = {}
        if self.vocal_characteristics is None:
            self.vocal_characteristics = {}
        if self.spectral_anomalies is None:
            self.spectral_anomalies = []
        if self.temporal_inconsistencies is None:
            self.temporal_inconsistencies = []
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'is_authentic': self.is_authentic,
            'authenticity_level': {
                'value': int(self.authenticity_level),
                'name': self.authenticity_level.name,
                'description': self.authenticity_level.description(),
                'color': self.authenticity_level.color_code()
            },
            'confidence': round(self.confidence, 4),
            'detected_spoofing_types': [t.value for t in self.detected_spoofing_types],
            'spoofing_probabilities': {k: round(v, 4) for k, v in self.spoofing_probabilities.items()},
            'audio_quality': self.audio_quality,
            'signal_artifacts': self.signal_artifacts,
            'prosody_analysis': self.prosody_analysis,
            'vocal_characteristics': {k: round(v, 4) for k, v in self.vocal_characteristics.items()},
            'liveness_score': round(self.liveness_score, 4),
            'spectral_anomalies': self.spectral_anomalies,
            'temporal_inconsistencies': self.temporal_inconsistencies,
            'processing_time': round(self.processing_time * 1000, 2),  # Convert to ms
            'model_version': self.model_version,
            'language': self.language,
            'explanation': self.explanation,
            'risk_assessment': self.risk_assessment,
            'recommendations': self.recommendations
        }
    
    @property
    def summary(self) -> str:
        """Get a brief summary of the detection result."""
        if self.is_authentic:
            return f"✅ Authentic voice ({self.authenticity_level.name}) - {self.confidence:.1%} confidence"
        else:
            spoofing_types = ", ".join([t.value for t in self.detected_spoofing_types])
            return f"⚠️ Voice spoofing detected ({spoofing_types}) - {self.confidence:.1%} confidence"


class VoiceCloneDetectorConfig:
    """Configuration class for voice clone detection."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        detector_config = self.config.get('voice_clone_detector', {})
        
        # Model configuration
        self.gemma_model_path = detector_config.get('gemma_model_path', 'google/gemma-3n-e2b-it')
        self.enable_gemma_pipeline = detector_config.get('enable_gemma_pipeline', True)
        self.use_multimodal_analysis = detector_config.get('use_multimodal_analysis', True)
        
        # Detection thresholds
        self.authenticity_threshold = detector_config.get('authenticity_threshold', 0.7)
        self.liveness_threshold = detector_config.get('liveness_threshold', 0.6)
        self.confidence_threshold = detector_config.get('confidence_threshold', 0.8)
        
        # Audio processing settings
        self.sample_rate = detector_config.get('sample_rate', 16000)
        self.window_size = detector_config.get('window_size', 2048)
        self.hop_length = detector_config.get('hop_length', 512)
        self.n_mfcc = detector_config.get('n_mfcc', 13)
        self.n_mels = detector_config.get('n_mels', 128)
        
        # Feature extraction settings
        self.enable_spectral_analysis = detector_config.get('enable_spectral_analysis', True)
        self.enable_prosody_analysis = detector_config.get('enable_prosody_analysis', True)
        self.enable_biometric_features = detector_config.get('enable_biometric_features', True)
        
        # Detection algorithms
        self.detection_methods = detector_config.get('detection_methods', [
            'spectral_artifacts', 'temporal_consistency', 'prosody_analysis',
            'biometric_verification', 'neural_detection'
        ])
        
        # Performance settings
        self.enable_caching = detector_config.get('enable_caching', True)
        self.cache_size = detector_config.get('cache_size', 500)
        self.batch_processing = detector_config.get('batch_processing', True)
        
        # Language support
        self.supported_languages = detector_config.get('supported_languages', [
            'en', 'hi', 'es', 'fr', 'de', 'zh', 'ar', 'ru'
        ])


class SpectralArtifactDetector:
    """
    Advanced spectral analysis for detecting synthesis artifacts.
    Uses frequency domain analysis to identify artificial voice characteristics.
    """
    
    def __init__(self, config: VoiceCloneDetectorConfig):
        self.config = config
        
    def extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract comprehensive spectral features for analysis."""
        if not HAS_LIBROSA:
            return {}
        
        try:
            features = {}
            
            # MFCC features (Mel-Frequency Cepstral Coefficients)
            mfcc = librosa.feature.mfcc(
                y=audio, sr=sr, 
                n_mfcc=self.config.n_mfcc,
                hop_length=self.config.hop_length
            )
            features['mfcc'] = {
                'mean': np.mean(mfcc, axis=1),
                'std': np.std(mfcc, axis=1),
                'delta': np.mean(librosa.feature.delta(mfcc), axis=1)
            }
            
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr,
                n_mels=self.config.n_mels,
                hop_length=self.config.hop_length
            )
            features['mel_spectrogram'] = {
                'mean': np.mean(mel_spec, axis=1),
                'std': np.std(mel_spec, axis=1)
            }
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma'] = {
                'mean': np.mean(chroma, axis=1),
                'std': np.std(chroma, axis=1)
            }
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            
            features['spectral'] = {
                'centroid_mean': np.mean(spectral_centroids),
                'centroid_std': np.std(spectral_centroids),
                'bandwidth_mean': np.mean(spectral_bandwidth),
                'bandwidth_std': np.std(spectral_bandwidth),
                'rolloff_mean': np.mean(spectral_rolloff),
                'rolloff_std': np.std(spectral_rolloff)
            }
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr'] = {
                'mean': np.mean(zcr),
                'std': np.std(zcr)
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
            return {}
    
    def detect_synthesis_artifacts(self, features: Dict[str, Any]) -> Tuple[List[str], float]:
        """Detect synthesis artifacts in spectral features."""
        artifacts = []
        anomaly_score = 0.0
        
        try:
            # Check MFCC anomalies
            if 'mfcc' in features:
                mfcc_mean = features['mfcc']['mean']
                mfcc_std = features['mfcc']['std']
                
                # Unusual MFCC patterns typical in synthetic speech
                if np.any(mfcc_std < 0.1):
                    artifacts.append("Low MFCC variance (unnatural consistency)")
                    anomaly_score += 0.3
                
                if np.any(mfcc_mean > 50) or np.any(mfcc_mean < -50):
                    artifacts.append("Extreme MFCC values (synthesis artifacts)")
                    anomaly_score += 0.2
            
            # Check spectral anomalies
            if 'spectral' in features:
                spectral = features['spectral']
                
                # Unnatural spectral centroid behavior
                if spectral['centroid_std'] < 100:
                    artifacts.append("Unnaturally stable spectral centroid")
                    anomaly_score += 0.25
                
                # Unusual bandwidth patterns
                if spectral['bandwidth_std'] < 50:
                    artifacts.append("Consistent spectral bandwidth (synthetic pattern)")
                    anomaly_score += 0.2
            
            # Check zero crossing rate anomalies
            if 'zcr' in features:
                zcr = features['zcr']
                
                if zcr['std'] < 0.01:
                    artifacts.append("Unnaturally consistent zero crossing rate")
                    anomaly_score += 0.15
            
            # Check chroma anomalies
            if 'chroma' in features:
                chroma_std = features['chroma']['std']
                
                if np.all(chroma_std < 0.05):
                    artifacts.append("Flat chroma distribution (synthetic characteristic)")
                    anomaly_score += 0.1
            
            return artifacts, min(1.0, anomaly_score)
            
        except Exception as e:
            logger.warning(f"Artifact detection failed: {e}")
            return [], 0.0


class ProsodyAnalyzer:
    """
    Advanced prosody analysis for detecting unnatural speech patterns.
    Analyzes rhythm, stress, and intonation patterns typical in human speech.
    """
    
    def __init__(self, config: VoiceCloneDetectorConfig):
        self.config = config
    
    def analyze_prosody(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Comprehensive prosody analysis."""
        if not HAS_LIBROSA:
            return {}
        
        try:
            prosody_features = {}
            
            # Pitch analysis
            f0 = librosa.yin(audio, fmin=80, fmax=400, sr=sr)
            f0_clean = f0[f0 > 0]  # Remove unvoiced segments
            
            if len(f0_clean) > 0:
                prosody_features['pitch'] = {
                    'mean': np.mean(f0_clean),
                    'std': np.std(f0_clean),
                    'range': np.max(f0_clean) - np.min(f0_clean),
                    'contour_variation': np.std(np.diff(f0_clean))
                }
            
            # Rhythm analysis
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            if len(onset_frames) > 1:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                inter_onset_intervals = np.diff(onset_times)
                
                prosody_features['rhythm'] = {
                    'onset_rate': len(onset_frames) / (len(audio) / sr),
                    'ioi_mean': np.mean(inter_onset_intervals),
                    'ioi_std': np.std(inter_onset_intervals),
                    'rhythm_regularity': 1.0 / (1.0 + np.std(inter_onset_intervals))
                }
            
            # Energy dynamics
            rms_energy = librosa.feature.rms(y=audio)[0]
            prosody_features['energy'] = {
                'mean': np.mean(rms_energy),
                'std': np.std(rms_energy),
                'dynamic_range': np.max(rms_energy) - np.min(rms_energy),
                'energy_variation': np.std(np.diff(rms_energy))
            }
            
            # Temporal features
            prosody_features['temporal'] = {
                'duration': len(audio) / sr,
                'speech_rate': len(onset_frames) / (len(audio) / sr) if len(onset_frames) > 0 else 0,
                'pause_ratio': self._calculate_pause_ratio(audio, sr)
            }
            
            return prosody_features
            
        except Exception as e:
            logger.warning(f"Prosody analysis failed: {e}")
            return {}
    
    def _calculate_pause_ratio(self, audio: np.ndarray, sr: int) -> float:
        """Calculate the ratio of silence/pause to total duration."""
        try:
            # Simple energy-based voice activity detection
            frame_length = 2048
            hop_length = 512
            energy_threshold = 0.01
            
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            frame_energies = np.sum(frames ** 2, axis=0)
            
            silence_frames = np.sum(frame_energies < energy_threshold)
            total_frames = len(frame_energies)
            
            return silence_frames / total_frames if total_frames > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def detect_prosody_anomalies(self, prosody_features: Dict[str, Any]) -> Tuple[List[str], float]:
        """Detect prosodic anomalies typical in synthetic speech."""
        anomalies = []
        anomaly_score = 0.0
        
        try:
            # Pitch anomalies
            if 'pitch' in prosody_features:
                pitch = prosody_features['pitch']
                
                # Unnaturally stable pitch
                if pitch['std'] < 10:
                    anomalies.append("Unnaturally stable pitch (monotonic speech)")
                    anomaly_score += 0.3
                
                # Extreme pitch values
                if pitch['mean'] < 80 or pitch['mean'] > 300:
                    anomalies.append("Extreme pitch values")
                    anomaly_score += 0.2
                
                # Low pitch contour variation
                if pitch['contour_variation'] < 5:
                    anomalies.append("Flat pitch contour (synthetic characteristic)")
                    anomaly_score += 0.25
            
            # Rhythm anomalies
            if 'rhythm' in prosody_features:
                rhythm = prosody_features['rhythm']
                
                # Overly regular rhythm
                if rhythm['rhythm_regularity'] > 0.8:
                    anomalies.append("Unnaturally regular rhythm")
 
