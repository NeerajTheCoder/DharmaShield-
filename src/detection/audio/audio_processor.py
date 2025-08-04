"""
detection/audio/audio_processor.py

DharmaShield - Advanced Audio Preprocessing & Voice Activity Detection Engine
---------------------------------------------------------------------------
• Production-grade audio preprocessing pipeline with VAD, normalization, and segmentation
• Multi-modal audio enhancement optimized for cross-platform deployment
• Real-time voice activity detection with advanced noise suppression
• Industry-standard audio processing with mobile-optimized performance
• Comprehensive audio quality assessment and adaptive filtering

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
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import json
import hashlib
from pathlib import Path
from collections import defaultdict, deque
import io
import wave

# Audio processing imports
try:
    import librosa
    import soundfile as sf
    from librosa.util import normalize
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("Librosa not available - advanced audio processing disabled")

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False
    warnings.warn("WebRTC VAD not available - fallback VAD will be used")

try:
    import scipy.signal
    from scipy.signal import butter, filtfilt, medfilt, savgol_filter
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available - limited signal processing")

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    warnings.warn("NoiseReduce not available - basic noise suppression only")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available - no neural VAD")

# Project imports
from ...utils.logger import get_logger
from ...core.config_loader import load_config
from ...utils.language import detect_language, get_language_name

logger = get_logger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class AudioQuality(IntEnum):
    """Audio quality levels for processed audio."""
    UNKNOWN = 0
    POOR = 1
    FAIR = 2
    GOOD = 3
    EXCELLENT = 4
    
    def description(self) -> str:
        descriptions = {
            self.UNKNOWN: "Unknown audio quality",
            self.POOR: "Poor quality - significant issues detected",
            self.FAIR: "Fair quality - minor issues present",
            self.GOOD: "Good quality - suitable for processing",
            self.EXCELLENT: "Excellent quality - optimal for analysis"
        }
        return descriptions.get(self, "Unknown quality level")

class ProcessingStage(Enum):
    """Audio processing stages."""
    INPUT_VALIDATION = "input_validation"
    NORMALIZATION = "normalization"
    NOISE_REDUCTION = "noise_reduction"
    VAD_PROCESSING = "vad_processing"
    SEGMENTATION = "segmentation"
    ENHANCEMENT = "enhancement"
    OUTPUT_FORMATTING = "output_formatting"

@dataclass
class AudioSegment:
    """Representation of an audio segment with metadata."""
    audio_data: np.ndarray
    start_time: float
    end_time: float
    duration: float
    sample_rate: int
    is_speech: bool = True
    confidence: float = 1.0
    quality_score: float = 0.0
    noise_level: float = 0.0
    
    def to_bytes(self) -> bytes:
        """Convert audio segment to bytes."""
        try:
            # Convert to 16-bit PCM
            audio_int16 = (self.audio_data * 32767).astype(np.int16)
            return audio_int16.tobytes()
        except Exception as e:
            logger.error(f"Failed to convert segment to bytes: {e}")
            return b''
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary format."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'is_speech': self.is_speech,
            'confidence': round(self.confidence, 4),
            'quality_score': round(self.quality_score, 4),
            'noise_level': round(self.noise_level, 4),
            'audio_length': len(self.audio_data)
        }

@dataclass
class ProcessingResult:
    """Comprehensive audio processing result."""
    # Processed audio data
    processed_audio: np.ndarray = None
    sample_rate: int = 16000
    
    # Segmentation results
    speech_segments: List[AudioSegment] = None
    non_speech_segments: List[AudioSegment] = None
    
    # Quality metrics
    overall_quality: AudioQuality = AudioQuality.UNKNOWN
    signal_to_noise_ratio: float = 0.0
    voice_activity_ratio: float = 0.0
    
    # Processing metadata
    processing_stages_completed: List[ProcessingStage] = None
    processing_time: float = 0.0
    original_duration: float = 0.0
    processed_duration: float = 0.0
    
    # Enhancement metrics
    noise_reduction_db: float = 0.0
    dynamic_range: float = 0.0
    spectral_centroid: float = 0.0
    
    # Error handling
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.speech_segments is None:
            self.speech_segments = []
        if self.non_speech_segments is None:
            self.non_speech_segments = []
        if self.processing_stages_completed is None:
            self.processing_stages_completed = []
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'sample_rate': self.sample_rate,
            'speech_segments': [seg.to_dict() for seg in self.speech_segments],
            'non_speech_segments': [seg.to_dict() for seg in self.non_speech_segments],
            'overall_quality': {
                'value': int(self.overall_quality),
                'name': self.overall_quality.name,
                'description': self.overall_quality.description()
            },
            'signal_to_noise_ratio': round(self.signal_to_noise_ratio, 2),
            'voice_activity_ratio': round(self.voice_activity_ratio, 4),
            'processing_stages_completed': [stage.value for stage in self.processing_stages_completed],
            'processing_time': round(self.processing_time * 1000, 2),
            'original_duration': round(self.original_duration, 3),
            'processed_duration': round(self.processed_duration, 3),
            'noise_reduction_db': round(self.noise_reduction_db, 2),
            'dynamic_range': round(self.dynamic_range, 2),
            'spectral_centroid': round(self.spectral_centroid, 2),
            'warnings': self.warnings,
            'errors': self.errors,
            'processed_audio_length': len(self.processed_audio) if self.processed_audio is not None else 0
        }
    
    @property
    def is_successful(self) -> bool:
        """Check if processing was successful."""
        return (self.processed_audio is not None and 
               len(self.errors) == 0 and
               self.overall_quality > AudioQuality.POOR)
    
    @property
    def total_speech_duration(self) -> float:
        """Get total duration of speech segments."""
        return sum(seg.duration for seg in self.speech_segments)
    
    @property
    def summary(self) -> str:
        """Get a brief summary of processing results."""
        if self.is_successful:
            speech_count = len(self.speech_segments)
            return f"✅ Audio processed successfully - {speech_count} speech segments, {self.overall_quality.name} quality"
        else:
            return f"❌ Audio processing failed - {len(self.errors)} errors, {self.overall_quality.name} quality"


class AudioProcessorConfig:
    """Configuration class for audio processor."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        processor_config = self.config.get('audio_processor', {})
        
        # Audio format settings
        self.target_sample_rate = processor_config.get('target_sample_rate', 16000)
        self.target_channels = processor_config.get('target_channels', 1)  # Mono
        self.target_dtype = processor_config.get('target_dtype', np.float32)
        
        # Processing pipeline settings
        self.enable_normalization = processor_config.get('enable_normalization', True)
        self.enable_noise_reduction = processor_config.get('enable_noise_reduction', True)
        self.enable_vad = processor_config.get('enable_vad', True)
        self.enable_segmentation = processor_config.get('enable_segmentation', True)
        self.enable_enhancement = processor_config.get('enable_enhancement', True)
        
        # VAD settings
        self.vad_engine = processor_config.get('vad_engine', 'auto')  # 'webrtc', 'energy', 'spectral', 'auto'
        self.vad_aggressiveness = processor_config.get('vad_aggressiveness', 2)  # 0-3 for WebRTC VAD
        self.vad_frame_duration = processor_config.get('vad_frame_duration', 30)  # ms
        self.min_speech_duration = processor_config.get('min_speech_duration', 0.1)  # seconds
        self.max_silence_duration = processor_config.get('max_silence_duration', 2.0)  # seconds
        
        # Noise reduction settings
        self.noise_reduction_strength = processor_config.get('noise_reduction_strength', 0.5)
        self.noise_estimation_duration = processor_config.get('noise_estimation_duration', 0.5)  # seconds
        self.enable_spectral_subtraction = processor_config.get('enable_spectral_subtraction', True)
        
        # Quality thresholds
        self.snr_excellent_threshold = processor_config.get('snr_excellent_threshold', 20.0)
        self.snr_good_threshold = processor_config.get('snr_good_threshold', 10.0)
        self.snr_fair_threshold = processor_config.get('snr_fair_threshold', 5.0)
        
        # Enhancement settings
        self.enable_dynamic_range_compression = processor_config.get('enable_dynamic_range_compression', True)
        self.enable_high_pass_filter = processor_config.get('enable_high_pass_filter', True)
        self.high_pass_cutoff = processor_config.get('high_pass_cutoff', 80)  # Hz
        self.enable_low_pass_filter = processor_config.get('enable_low_pass_filter', False)
        self.low_pass_cutoff = processor_config.get('low_pass_cutoff', 8000)  # Hz
        
        # Performance settings
        self.enable_multiprocessing = processor_config.get('enable_multiprocessing', True)
        self.chunk_size = processor_config.get('chunk_size', 4096)
        self.overlap_ratio = processor_config.get('overlap_ratio', 0.25)
        
        # Validation settings
        self.min_audio_duration = processor_config.get('min_audio_duration', 0.1)  # seconds
        self.max_audio_duration = processor_config.get('max_audio_duration', 300.0)  # seconds
        self.max_file_size_mb = processor_config.get('max_file_size_mb', 100)


class BaseVAD(ABC):
    """Abstract base class for Voice Activity Detection implementations."""
    
    def __init__(self, config: AudioProcessorConfig):
        self.config = config
    
    @abstractmethod
    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float, float]]:
        """
        Detect speech segments in audio.
        
        Returns:
            List of tuples (start_time, end_time, confidence)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this VAD implementation is available."""
        pass


class WebRTCVAD(BaseVAD):
    """WebRTC-based Voice Activity Detection."""
    
    def __init__(self, config: AudioProcessorConfig):
        super().__init__(config)
        self.vad = None
        if HAS_WEBRTCVAD:
            try:
                self.vad = webrtcvad.Vad(config.vad_aggressiveness)
            except Exception as e:
                logger.warning(f"Failed to initialize WebRTC VAD: {e}")
    
    def is_available(self) -> bool:
        return self.vad is not None
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float, float]]:
        """Detect speech using WebRTC VAD."""
        if not self.is_available():
            return []
        
        try:
            # WebRTC VAD requires specific sample rates
            if sample_rate not in [8000, 16000, 32000, 48000]:
                if HAS_LIBROSA:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                else:
                    logger.warning("Cannot resample for WebRTC VAD without librosa")
                    return []
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Frame duration in samples
            frame_duration_ms = self.config.vad_frame_duration
            frame_length = int(sample_rate * frame_duration_ms / 1000)
            
            # Ensure frame length is compatible with WebRTC VAD
            if frame_duration_ms not in [10, 20, 30]:
                frame_duration_ms = 30
                frame_length = int(sample_rate * frame_duration_ms / 1000)
            
            speech_segments = []
            current_speech_start = None
            
            for i in range(0, len(audio_int16) - frame_length + 1, frame_length):
                frame = audio_int16[i:i + frame_length].tobytes()
                
                try:
                    is_speech = self.vad.is_speech(frame, sample_rate)
                    frame_time = i / sample_rate
                    
                    if is_speech and current_speech_start is None:
                        current_speech_start = frame_time
                    elif not is_speech and current_speech_start is not None:
                        duration = frame_time - current_speech_start
                        if duration >= self.config.min_speech_duration:
                            speech_segments.append((current_speech_start, frame_time, 0.8))
                        current_speech_start = None
                
                except Exception as e:
                    logger.warning(f"WebRTC VAD frame processing failed: {e}")
                    continue
            
            # Handle case where speech continues to end of audio
            if current_speech_start is not None:
                end_time = len(audio_int16) / sample_rate
                duration = end_time - current_speech_start
                if duration >= self.config.min_speech_duration:
                    speech_segments.append((current_speech_start, end_time, 0.8))
            
            return speech_segments
            
        except Exception as e:
            logger.error(f"WebRTC VAD detection failed: {e}")
            return []


class EnergyVAD(BaseVAD):
    """Energy-based Voice Activity Detection."""
    
    def __init__(self, config: AudioProcessorConfig):
        super().__init__(config)
    
    def is_available(self) -> bool:
        return True
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float, float]]:
        """Detect speech using energy-based method."""
        try:
            # Frame-based analysis
            frame_length = int(sample_rate * self.config.vad_frame_duration / 1000)
            hop_length = frame_length // 2
            
            # Calculate frame energy
            frame_energies = []
            frame_times = []
            
            for i in range(0, len(audio) - frame_length + 1, hop_length):
                frame = audio[i:i + frame_length]
                energy = np.sum(frame ** 2)
                frame_energies.append(energy)
                frame_times.append(i / sample_rate)
            
            if not frame_energies:
                return []
            
            # Adaptive threshold based on noise floor
            sorted_energies = sorted(frame_energies)
            noise_floor = np.mean(sorted_energies[:len(sorted_energies)//4])  # Bottom 25%
            
            # Dynamic threshold
            energy_threshold = noise_floor * 3.0  # Adjust multiplier as needed
            
            # Find speech segments
            speech_frames = [energy > energy_threshold for energy in frame_energies]
            
            # Convert frame-based decisions to time segments
            speech_segments = []
            current_speech_start = None
            
            for i, is_speech in enumerate(speech_frames):
                frame_time = frame_times[i]
                
                if is_speech and current_speech_start is None:
                    current_speech_start = frame_time
                elif not is_speech and current_speech_start is not None:
                    duration = frame_time - current_speech_start
                    if duration >= self.config.min_speech_duration:
                        # Calculate confidence based on average energy in segment
                        start_frame = int(current_speech_start * sample_rate / hop_length)
                        end_frame = i
                        avg_energy = np.mean(frame_energies[start_frame:end_frame])
                        confidence = min(1.0, avg_energy / (noise_floor * 10))
                        speech_segments.append((current_speech_start, frame_time, confidence))
                    current_speech_start = None
            
            # Handle case where speech continues to end
            if current_speech_start is not None:
                end_time = len(audio) / sample_rate
                duration = end_time - current_speech_start
                if duration >= self.config.min_speech_duration:
                    speech_segments.append((current_speech_start, end_time, 0.7))
            
            return speech_segments
            
        except Exception as e:
            logger.error(f"Energy VAD detection failed: {e}")
            return []


class SpectralVAD(BaseVAD):
    """Spectral-based Voice Activity Detection."""
    
    def __init__(self, config: AudioProcessorConfig):
        super().__init__(config)
    
    def is_available(self) -> bool:
        return HAS_LIBROSA
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float, float]]:
        """Detect speech using spectral features."""
        if not HAS_LIBROSA:
            return []
        
        try:
            # Extract spectral features
            hop_length = 512
            n_fft = 2048
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13, 
                                       hop_length=hop_length, n_fft=n_fft)
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, 
                                                                 hop_length=hop_length)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
            
            # Frame times
            frame_times = librosa.frames_to_time(np.arange(len(spectral_centroids)), 
                                               sr=sample_rate, hop_length=hop_length)
            
            # Combine features for speech detection
      
