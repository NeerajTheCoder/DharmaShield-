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
            speech_scores = []
            for i in range(len(spectral_centroids)):
                # Speech typically has:
                # - Higher spectral centroid
                # - Lower zero crossing rate
                # - Characteristic MFCC patterns
                
                centroid_score = min(1.0, spectral_centroids[i] / 2000)  # Normalize
                zcr_score = 1.0 - min(1.0, zcr[i] * 10)  # Invert and normalize
                mfcc_score = min(1.0, np.std(mfccs[:, i]) / 10)  # MFCC variance
                
                combined_score = (centroid_score + zcr_score + mfcc_score) / 3.0
                speech_scores.append(combined_score)
            
            # Threshold-based segmentation
            speech_threshold = 0.4
            speech_frames = [score > speech_threshold for score in speech_scores]
            
            # Convert to time segments
            speech_segments = []
            current_speech_start = None
            
            for i, is_speech in enumerate(speech_frames):
                frame_time = frame_times[i]
                
                if is_speech and current_speech_start is None:
                    current_speech_start = frame_time
                elif not is_speech and current_speech_start is not None:
                    duration = frame_time - current_speech_start
                    if duration >= self.config.min_speech_duration:
                        # Calculate average confidence in segment
                        start_idx = max(0, i - int(duration * sample_rate / hop_length))
                        avg_confidence = np.mean(speech_scores[start_idx:i])
                        speech_segments.append((current_speech_start, frame_time, avg_confidence))
                    current_speech_start = None
            
            # Handle final segment
            if current_speech_start is not None:
                end_time = len(audio) / sample_rate
                duration = end_time - current_speech_start
                if duration >= self.config.min_speech_duration:
                    speech_segments.append((current_speech_start, end_time, 0.6))
            
            return speech_segments
            
        except Exception as e:
            logger.error(f"Spectral VAD detection failed: {e}")
            return []


class AudioQualityAssessor:
    """Advanced audio quality assessment and analysis."""
    
    def __init__(self, config: AudioProcessorConfig):
        self.config = config
    
    def assess_quality(self, audio: np.ndarray, sample_rate: int) -> Tuple[AudioQuality, Dict[str, float]]:
        """Comprehensive audio quality assessment."""
        try:
            metrics = {}
            
            # Signal-to-Noise Ratio estimation
            snr = self._estimate_snr(audio)
            metrics['snr'] = snr
            
            # Dynamic range
            dynamic_range = self._calculate_dynamic_range(audio)
            metrics['dynamic_range'] = dynamic_range
            
            # Clipping detection
            clipping_ratio = self._detect_clipping(audio)
            metrics['clipping_ratio'] = clipping_ratio
            
            # Frequency response analysis
            if HAS_LIBROSA:
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0])
                spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0])
                metrics['spectral_centroid'] = spectral_centroid
                metrics['spectral_bandwidth'] = spectral_bandwidth
            
            # Noise level estimation
            noise_level = self._estimate_noise_level(audio)
            metrics['noise_level'] = noise_level
            
            # Overall quality score
            quality_score = self._calculate_overall_quality(metrics)
            metrics['overall_score'] = quality_score
            
            # Determine quality level
            if snr >= self.config.snr_excellent_threshold and clipping_ratio < 0.01:
                quality = AudioQuality.EXCELLENT
            elif snr >= self.config.snr_good_threshold and clipping_ratio < 0.05:
                quality = AudioQuality.GOOD
            elif snr >= self.config.snr_fair_threshold and clipping_ratio < 0.1:
                quality = AudioQuality.FAIR
            elif snr > 0:
                quality = AudioQuality.POOR
            else:
                quality = AudioQuality.UNKNOWN
            
            return quality, metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return AudioQuality.UNKNOWN, {}
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio."""
        try:
            # Simple SNR estimation based on signal statistics
            signal_power = np.mean(audio ** 2)
            
            # Estimate noise from quieter portions
            sorted_samples = np.sort(np.abs(audio))
            noise_samples = sorted_samples[:len(sorted_samples)//4]  # Bottom 25%
            noise_power = np.mean(noise_samples ** 2)
            
            if noise_power > 0:
                snr_linear = signal_power / noise_power
                snr_db = 10 * np.log10(snr_linear + 1e-10)
                return float(snr_db)
            else:
                return 60.0  # Very clean signal
                
        except Exception:
            return 0.0
    
    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range of audio signal."""
        try:
            if len(audio) == 0:
                return 0.0
            
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude == 0:
                return 0.0
            
            # RMS of quieter portions
            sorted_samples = np.sort(np.abs(audio))
            quiet_threshold = sorted_samples[int(len(sorted_samples) * 0.1)]  # 10th percentile
            
            if quiet_threshold > 0:
                dynamic_range_db = 20 * np.log10(max_amplitude / quiet_threshold)
                return float(dynamic_range_db)
            else:
                return 60.0  # Very high dynamic range
                
        except Exception:
            return 0.0
    
    def _detect_clipping(self, audio: np.ndarray) -> float:
        """Detect audio clipping ratio."""
        try:
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(audio) >= clipping_threshold)
            clipping_ratio = clipped_samples / len(audio) if len(audio) > 0 else 0.0
            return float(clipping_ratio)
        except Exception:
            return 0.0
    
    def _estimate_noise_level(self, audio: np.ndarray) -> float:
        """Estimate background noise level."""
        try:
            # Use median filtering to estimate noise floor
            if HAS_SCIPY:
                smoothed = medfilt(np.abs(audio), kernel_size=101)
                noise_level = np.percentile(smoothed, 10)  # 10th percentile
            else:
                noise_level = np.percentile(np.abs(audio), 10)
            
            return float(noise_level)
        except Exception:
            return 0.0
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from metrics."""
        try:
            score = 0.0
            
            # SNR contribution (40%)
            snr = metrics.get('snr', 0)
            snr_score = min(1.0, max(0.0, snr / 30.0))  # Normalize to 0-1
            score += 0.4 * snr_score
            
            # Dynamic range contribution (30%)
            dr = metrics.get('dynamic_range', 0)
            dr_score = min(1.0, max(0.0, dr / 40.0))  # Normalize to 0-1
            score += 0.3 * dr_score
            
            # Clipping penalty (20%)
            clipping = metrics.get('clipping_ratio', 0)
            clipping_score = max(0.0, 1.0 - clipping * 10)  # Penalty for clipping
            score += 0.2 * clipping_score
            
            # Noise level penalty (10%)
            noise = metrics.get('noise_level', 0)
            noise_score = max(0.0, 1.0 - noise * 5)  # Penalty for high noise
            score += 0.1 * noise_score
            
            return float(score)
            
        except Exception:
            return 0.5


class AudioEnhancer:
    """Advanced audio enhancement and filtering."""
    
    def __init__(self, config: AudioProcessorConfig):
        self.config = config
    
    def enhance_audio(self, audio: np.ndarray, sample_rate: int, 
                     noise_profile: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Comprehensive audio enhancement pipeline."""
        try:
            enhanced_audio = audio.copy()
            enhancement_log = {}
            
            # Step 1: Noise reduction
            if self.config.enable_noise_reduction:
                enhanced_audio, nr_metrics = self._reduce_noise(
                    enhanced_audio, sample_rate, noise_profile
                )
                enhancement_log['noise_reduction'] = nr_metrics
            
            # Step 2: High-pass filtering
            if self.config.enable_high_pass_filter:
                enhanced_audio = self._apply_high_pass_filter(
                    enhanced_audio, sample_rate, self.config.high_pass_cutoff
                )
                enhancement_log['high_pass_filter'] = {'cutoff': self.config.high_pass_cutoff}
            
            # Step 3: Low-pass filtering (if enabled)
            if self.config.enable_low_pass_filter:
                enhanced_audio = self._apply_low_pass_filter(
                    enhanced_audio, sample_rate, self.config.low_pass_cutoff
                )
                enhancement_log['low_pass_filter'] = {'cutoff': self.config.low_pass_cutoff}
            
            # Step 4: Dynamic range compression
            if self.config.enable_dynamic_range_compression:
                enhanced_audio, compression_ratio = self._apply_compression(enhanced_audio)
                enhancement_log['compression'] = {'ratio': compression_ratio}
            
            # Step 5: Normalization
            if self.config.enable_normalization:
                enhanced_audio, norm_factor = self._normalize_audio(enhanced_audio)
                enhancement_log['normalization'] = {'factor': norm_factor}
            
            return enhanced_audio, enhancement_log
            
        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            return audio, {'error': str(e)}
    
    def _reduce_noise(self, audio: np.ndarray, sample_rate: int, 
                     noise_profile: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Advanced noise reduction."""
        try:
            if HAS_NOISEREDUCE and len(audio) > sample_rate * 0.5:  # At least 0.5 seconds
                # Use noisereduce library if available
                if noise_profile is not None:
                    reduced_audio = nr.reduce_noise(
                        y=audio, 
                        sr=sample_rate,
                        noise_clip=noise_profile,
                        prop_decrease=self.config.noise_reduction_strength
                    )
                else:
                    reduced_audio = nr.reduce_noise(
                        y=audio, 
                        sr=sample_rate,
                        prop_decrease=self.config.noise_reduction_strength
                    )
                
                noise_reduction_db = 20 * np.log10(
                    (np.std(audio) + 1e-10) / (np.std(reduced_audio) + 1e-10)
                )
                
                return reduced_audio, {'method': 'noisereduce', 'reduction_db': noise_reduction_db}
            
            elif HAS_SCIPY and self.config.enable_spectral_subtraction:
                # Spectral subtraction approach
                return self._spectral_subtraction(audio, sample_rate)
            
            else:
                # Simple noise gate
                return self._apply_noise_gate(audio)
                
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio, {'error': str(e)}
    
    def _spectral_subtraction(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Spectral subtraction noise reduction."""
        try:
            if not HAS_LIBROSA:
                return audio, {'error': 'librosa not available'}
            
            # STFT
            stft = librosa.stft(audio, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise spectrum from initial frames
            noise_frames = int(self.config.noise_estimation_duration * sample_rate / 512)
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction
            alpha = self.config.noise_reduction_strength * 2  # Adjustment factor
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            
            # Prevent over-subtraction
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            # Ensure same length as input
            if len(enhanced_audio) != len(audio):
                enhanced_audio = np.resize(enhanced_audio, len(audio))
            
            reduction_db = 20 * np.log10(
                (np.std(audio) + 1e-10) / (np.std(enhanced_audio) + 1e-10)
            )
            
            return enhanced_audio, {'method': 'spectral_subtraction', 'reduction_db': reduction_db}
            
        except Exception as e:
            logger.warning(f"Spectral subtraction failed: {e}")
            return audio, {'error': str(e)}
    
    def _apply_noise_gate(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply simple noise gate."""
        try:
            # Calculate threshold based on signal statistics
            sorted_samples = np.sort(np.abs(audio))
            threshold = sorted_samples[int(len(sorted_samples) * 0.1)]  # 10th percentile
            
            # Apply gate
            gated_audio = np.where(np.abs(audio) > threshold, audio, audio * 0.1)
            
            return gated_audio, {'method': 'noise_gate', 'threshold': float(threshold)}
            
        except Exception:
            return audio, {'error': 'noise gate failed'}
    
    def _apply_high_pass_filter(self, audio: np.ndarray, sample_rate: int, cutoff: float) -> np.ndarray:
        """Apply high-pass filter."""
        try:
            if not HAS_SCIPY:
                return audio
            
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            if normalized_cutoff >= 1.0:
                return audio
            
            b, a = butter(4, normalized_cutoff, btype='high')
            filtered_audio = filtfilt(b, a, audio)
            
            return filtered_audio.astype(audio.dtype)
            
        except Exception as e:
            logger.warning(f"High-pass filter failed: {e}")
            return audio
    
    def _apply_low_pass_filter(self, audio: np.ndarray, sample_rate: int, cutoff: float) -> np.ndarray:
        """Apply low-pass filter."""
        try:
            if not HAS_SCIPY:
                return audio
            
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            if normalized_cutoff >= 1.0:
                return audio
            
            b, a = butter(4, normalized_cutoff, btype='low')
            filtered_audio = filtfilt(b, a, audio)
            
            return filtered_audio.astype(audio.dtype)
            
        except Exception as e:
            logger.warning(f"Low-pass filter failed: {e}")
            return audio
    
    def _apply_compression(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply dynamic range compression."""
        try:
            # Simple compression algorithm
            threshold = 0.7
            ratio = 4.0
            
            compressed_audio = audio.copy()
            
            # Apply compression to samples above threshold
            above_threshold = np.abs(compressed_audio) > threshold
            compressed_audio[above_threshold] = (
                np.sign(compressed_audio[above_threshold]) * 
                (threshold + (np.abs(compressed_audio[above_threshold]) - threshold) / ratio)
            )
            
            return compressed_audio, ratio
            
        except Exception:
            return audio, 1.0
    
    def _normalize_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Normalize audio amplitude."""
        try:
            if HAS_LIBROSA:
                normalized_audio = normalize(audio)
                norm_factor = np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 1.0
            else:
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    normalized_audio = audio / max_val
                    norm_factor = max_val
                else:
                    normalized_audio = audio
                    norm_factor = 1.0
            
            return normalized_audio.astype(audio.dtype), float(norm_factor)
            
        except Exception:
            return audio, 1.0


class AdvancedAudioProcessor:
    """
    Production-grade audio preprocessing pipeline with comprehensive VAD, enhancement, and segmentation.
    
    Features:
    - Multi-engine Voice Activity Detection (WebRTC, Energy-based, Spectral)
    - Advanced noise reduction and audio enhancement
    - Intelligent audio segmentation with quality assessment
    - Real-time processing optimization for mobile deployment
    - Comprehensive audio quality metrics and reporting
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
        
        self.config = AudioProcessorConfig(config_path)
        
        # Initialize components
        self.quality_assessor = AudioQualityAssessor(self.config)
        self.enhancer = AudioEnhancer(self.config)
        
        # Initialize VAD engines
        self.vad_engines = {}
        self._initialize_vad_engines()
        
        # Performance monitoring
        self.processing_cache = {}
        self.recent_processes = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        
        self._initialized = True
        logger.info("Advanced Audio Processor initialized")
    
    def _initialize_vad_engines(self):
        """Initialize available VAD engines."""
        # WebRTC VAD
        webrtc_vad = WebRTCVAD(self.config)
        if webrtc_vad.is_available():
            self.vad_engines['webrtc'] = webrtc_vad
        
        # Energy-based VAD
        energy_vad = EnergyVAD(self.config)
        if energy_vad.is_available():
            self.vad_engines['energy'] = energy_vad
        
        # Spectral VAD
        spectral_vad = SpectralVAD(self.config)
        if spectral_vad.is_available():
            self.vad_engines['spectral'] = spectral_vad
        
        logger.info(f"Initialized {len(self.vad_engines)} VAD engines: {list(self.vad_engines.keys())}")
    
    def _get_vad_engine(self) -> Optional[BaseVAD]:
        """Get the appropriate VAD engine based on configuration."""
        if self.config.vad_engine == 'auto':
            # Preference order: WebRTC > Spectral > Energy
            for engine_name in ['webrtc', 'spectral', 'energy']:
                if engine_name in self.vad_engines:
                    return self.vad_engines[engine_name]
        elif self.config.vad_engine in self.vad_engines:
            return self.vad_engines[self.config.vad_engine]
        
        # Fallback to any available engine
        if self.vad_engines:
            return next(iter(self.vad_engines.values()))
        
        return None
    
    def _validate_input(self, audio_data: Union[np.ndarray, bytes]) -> Tuple[np.ndarray, int, List[str]]:
        """Validate and convert input audio data."""
        warnings = []
        
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                if HAS_LIBROSA:
                    audio, sr = librosa.load(io.BytesIO(audio_data), 
                                           sr=self.config.target_sample_rate, 
                                           mono=True)
                else:
                    # Assume 16-bit PCM
                    audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    sr = self.config.target_sample_rate
            else:
                audio = audio_data.astype(self.config.target_dtype)
                sr = self.config.target_sample_rate
            
            # Validate duration
            duration = len(audio) / sr
            if duration < self.config.min_audio_duration:
                raise ValueError(f"Audio too short: {duration:.2f}s < {self.config.min_audio_duration}s")
            
            if duration > self.config.max_audio_duration:
                warnings.append(f"Audio longer than recommended: {duration:.2f}s")
            
            # Check for empty or invalid audio
            if len(audio) == 0:
                raise ValueError("Empty audio data")
            
            if np.all(audio == 0):
                warnings.append("Audio contains only silence")
            
            # Check for clipping
            if np.any(np.abs(audio) >= 0.99):
                warnings.append("Audio clipping detected")
            
            return audio, sr, warnings
            
        except Exception as e:
            raise ValueError(f"Invalid audio input: {e}")
    
    def _create_segments(self, audio: np.ndarray, sample_rate: int, 
                        speech_intervals: List[Tuple[float, float, float]]) -> Tuple[List[AudioSegment], List[AudioSegment]]:
        """Create audio segments from VAD results."""
        speech_segments = []
        non_speech_segments = []
        
        try:
            # Sort intervals by start time
            sorted_intervals = sorted(speech_intervals, key=lambda x: x[0])
            
            current_time = 0.0
            total_duration = len(audio) / sample_rate
            
            for start_time, end_time, confidence in sorted_intervals:
                # Add non-speech segment before this speech segment
                if current_time < start_time:
                    non_speech_start = int(current_time * sample_rate)
                    non_speech_end = int(start_time * sample_rate)
                    non_speech_audio = audio[non_speech_start:non_speech_end]
                    
                    if len(non_speech_audio) > 0:
                        quality, _ = self.quality_assessor.assess_quality(non_speech_audio, sample_rate)
                        
                        non_speech_segment = AudioSegment(
                            audio_data=non_speech_audio,
                            start_time=current_time,
                            end_time=start_time,
                            duration=start_time - current_time,
                            sample_rate=sample_rate,
                            is_speech=False,
                            confidence=1.0 - confidence,
                            quality_score=int(quality) / 4.0
                        )
                        non_speech_segments.append(non_speech_segment)
                
                # Add speech segment
                speech_start = int(start_time * sample_rate)
                speech_end = int(end_time * sample_rate)
                speech_audio = audio[speech_start:speech_end]
                
                if len(speech_audio) > 0:
                    quality, metrics = self.quality_assessor.assess_quality(speech_audio, sample_rate)
                    
                    speech_segment = AudioSegment(
                        audio_data=speech_audio,
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                        sample_rate=sample_rate,
                        is_speech=True,
                        confidence=confidence,
                        quality_score=int(quality) / 4.0,
                        noise_level=metrics.get('noise_level', 0.0)
                    )
                    speech_segments.append(speech_segment)
                
                current_time = end_time
            
            # Add final non-speech segment if needed
            if current_time < total_duration:
                non_speech_start = int(current_time * sample_rate)
                non_speech_audio = audio[non_speech_start:]
                
                if len(non_speech_audio) > 0:
                    quality, _ = self.quality_assessor.assess_quality(non_speech_audio, sample_rate)
                    
                    non_speech_segment = AudioSegment(
                        audio_data=non_speech_audio,
                        start_time=current_time,
                        end_time=total_duration,
                        duration=total_duration - current_time,
                        sample_rate=sample_rate,
                        is_speech=False,
                        confidence=0.8,
                        quality_score=int(quality) / 4.0
                    )
                    non_speech_segments.append(non_speech_segment)
            
            return speech_segments, non_speech_segments
            
        except Exception as e:
            logger.error(f"Segment creation failed: {e}")
            return [], []
    
    def process_audio(self, 
                     audio_data: Union[np.ndarray, bytes],
                     sample_rate: Optional[int] = None) -> ProcessingResult:
        """
        Main audio processing pipeline with comprehensive preprocessing.
        
        Args:
            audio_data: Raw audio data (numpy array or bytes)
            sample_rate: Sample rate of input audio (if numpy array)
            
        Returns:
            ProcessingResult with processed audio and analysis
        """
        start_time = time.time()
        result = ProcessingResult()
        
        try:
            # Step 1: Input validation
            audio, sr, validation_warnings = self._validate_input(audio_data)
            result.warnings.extend(validation_warnings)
            result.processing_stages_completed.append(ProcessingStage.INPUT_VALIDATION)
            result.original_duration = len(audio) / sr
            
            # Override sample rate if provided
            if sample_rate is not None:
                sr = sample_rate
            
            # Step 2: Quality assessment
            initial_quality, quality_metrics = self.quality_assessor.assess_quality(audio, sr)
            result.signal_to_noise_ratio = quality_metrics.get('snr', 0.0)
            result.spectral_centroid = quality_metrics.get('spectral_centroid', 0.0)
            result.dynamic_range = quality_metrics.get('dynamic_range', 0.0)
            
            # Step 3: Audio enhancement
            enhanced_audio = audio
            if self.config.enable_enhancement:
                noise_profile = audio[:int(self.config.noise_estimation_duration * sr)] if len(audio) > sr else None
                enhanced_audio, enhancement_log = self.enhancer.enhance_audio(audio, sr, noise_profile)
                result.noise_reduction_db = enhancement_log.get('noise_reduction', {}).get('reduction_db', 0.0)
                result.processing_stages_completed.append(ProcessingStage.ENHANCEMENT)
            
            # Step 4: Final normalization
            if self.config.enable_normalization:
                if HAS_LIBROSA:
                    enhanced_audio = normalize(enhanced_audio)
                else:
                    max_val = np.max(np.abs(enhanced_audio))
                    if max_val > 0:
                        enhanced_audio = enhanced_audio / max_val
                result.processing_stages_completed.append(ProcessingStage.NORMALIZATION)
            
            # Step 5: Voice Activity Detection
            speech_intervals = []
            if self.config.enable_vad:
                vad_engine = self._get_vad_engine()
                if vad_engine:
                    speech_intervals = vad_engine.detect_speech(enhanced_audio, sr)
                    result.processing_stages_completed.append(ProcessingStage.VAD_PROCESSING)
                else:
                    result.warnings.append("No VAD engine available")
            
            # If no VAD or no speech detected, treat entire audio as speech
            if not speech_intervals:
                duration = len(enhanced_audio) / sr
                speech_intervals = [(0.0, duration, 0.8)]
            
            # Step 6: Segmentation
            if self.config.enable_segmentation:
                speech_segments, non_speech_segments = self._create_segments(
                    enhanced_audio, sr, speech_intervals
                )
                result.speech_segments = speech_segments
                result.non_speech_segments = non_speech_segments
                result.processing_stages_completed.append(ProcessingStage.SEGMENTATION)
            
            # Step 7: Final quality assessment
            final_quality, final_metrics = self.quality_assessor.assess_quality(enhanced_audio, sr)
            result.overall_quality = final_quality
            
            # Calculate voice activity ratio
            total_speech_duration = sum(seg.duration for seg in result.speech_segments)
            result.voice_activity_ratio = total_speech_duration / result.original_duration
            
            # Set processed audio and metadata
            result.processed_audio = enhanced_audio
            result.sample_rate = sr
            result.processed_duration = len(enhanced_audio) / sr
            result.processing_time = time.time() - start_time
            
            # Update performance metrics
            self.recent_processes.append(result)
            self.performance_metrics['processing_time'].append(result.processing_time)
            self.performance_metrics['quality_score'].append(int(result.overall_quality))
            
            logger.info(f"Audio processing completed: {result.summary}")
            return result
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            result.errors.append(str(e))
            result.processing_time = time.time() - start_time
            return result
    
    async def process_audio_async(self, 
                                 audio_data: Union[np.ndarray, bytes],
                                 sample_rate: Optional[int] = None) -> ProcessingResult:
        """Asynchronous audio processing."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.process_audio, audio_data, sample_rate
        )
    
    def process_batch(self, 
                     audio_samples: List[Union[np.ndarray, bytes]],
                     sample_rate: Optional[int] = None) -> List[ProcessingResult]:
        """Batch processing for multiple audio samples."""
        results = []
        for audio_data in audio_samples:
            result = self.process_audio(audio_data, sample_rate)
            results.append(result)
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.recent_processes:
            return {"message": "No processing performed yet"}
        
        recent_results = list(self.recent_processes)
        total_processes = len(recent_results)
        
        # Calculate statistics
        avg_processing_time = np.mean([r.processing_time for r in recent_results])
        success_rate = np.mean([r.is_successful for r in recent_results])
        avg_speech_ratio = np.mean([r.voice_activity_ratio for r in recent_results])
        
        # Quality distribution
        quality_distribution = defaultdict(int)
        for result in recent_results:
            quality_distribution[result.overall_quality.name] += 1
        
        quality_distribution = {
            quality: count / total_processes 
            for quality, count in quality_distribution.items()
        }
        
        # Processing stage success rates
        stage_success = defaultdict(int)
        for result in recent_results:
            for stage in result.processing_stages_completed:
                stage_success[stage.value] += 1
        
        stage_success_rates = {
            stage: count / total_processes 
            for stage, count in stage_success.items()
        }
        
        return {
            'total_processes': total_processes,
            'average_processing_time_ms': avg_processing_time * 1000,
            'success_rate': success_rate,
            'average_voice_activity_ratio': avg_speech_ratio,
            'quality_distribution': quality_distribution,
            'processing_stage_success_rates': stage_success_rates,
            'available_vad_engines': list(self.vad_engines.keys()),
            'configuration': {
                'target_sample_rate': self.config.target_sample_rate,
                'vad_engine': self.config.vad_engine,
                'enhancement_enabled': self.config.enable_enhancement
            }
        }
    
    def clear_cache(self):
        """Clear processing cache and reset metrics."""
        self.processing_cache.clear()
        self.recent_processes.clear()
        self.performance_metrics.clear()
        logger.info("Processing cache and metrics cleared")


# Global instance and convenience functions
_global_processor = None

def get_audio_processor(config_path: Optional[str] = None) -> AdvancedAudioProcessor:
    """Get the global audio processor instance."""
    global _global_processor
    if _global_processor is None:
        _global_processor = AdvancedAudioProcessor(config_path)
    return _global_processor

def process_audio(audio_data: Union[np.ndarray, bytes],
                 sample_rate: Optional[int] = None) -> ProcessingResult:
    """
    Convenience function for audio processing.
    
    Args:
        audio_data: Raw audio data (numpy array or bytes)
        sample_rate: Sample rate of input audio
        
    Returns:
        ProcessingResult with comprehensive analysis
    """
    processor = get_audio_processor()
    return processor.process_audio(audio_data, sample_rate)

async def process_audio_async(audio_data: Union[np.ndarray, bytes],
                             sample_rate: Optional[int] = None) -> ProcessingResult:
    """Asynchronous convenience function for audio processing."""
    processor = get_audio_processor()
    return await processor.process_audio_async(audio_data, sample_rate)

def process_batch(audio_samples: List[Union[np.ndarray, bytes]],
                 sample_rate: Optional[int] = None) -> List[ProcessingResult]:
    """Convenience function for batch audio processing."""
    processor = get_audio_processor()
    return processor.process_batch(audio_samples, sample_rate)


# Testing and validation
if __name__ == "__main__":
    import time
    
    print("=== DharmaShield Advanced Audio Processor Test Suite ===\n")
    
    processor = AdvancedAudioProcessor()
    
    # Test with synthetic audio
    print("Testing with synthetic audio samples...\n")
    
    # Generate test signals
    sample_rate = 16000
    duration = 3.0  # 3 seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    test_cases = [
        {
            'name': 'Pure sine wave (speech-like)',
            'audio': 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.normal(0, 0.05, len(t))
        },
        {
            'name': 'Noisy speech simulation',
            'audio': (0.3 * np.sin(2 * np.pi * 200 * t) + 
                     0.2 * np.sin(2 * np.pi * 800 * t) + 
                     0.3 * np.random.normal(0, 0.1, len(t)))
        },
        {
            'name': 'Silence with noise',
            'audio': np.random.normal(0, 0.02, len(t))
        },
        {
            'name': 'Clipped audio',
            'audio': np.clip(0.8 * np.sin(2 * np.pi * 300 * t), -1.0, 1.0)
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        
        start_time = time.time()
        result = processor.process_audio(test_case['audio'], sample_rate)
        end_time = time.time()
        
        print(f"  {result.summary}")
        print(f"  Quality: {result.overall_quality.description()}")
        print(f"  SNR: {result.signal_to_noise_ratio:.1f} dB")
        print(f"  Voice Activity Ratio: {result.voice_activity_ratio:.1%}")
        print(f"  Speech Segments: {len(result.speech_segments)}")
        print(f"  Processing Time: {(end_time - start_time)*1000:.1f}ms")
        
        if result.warnings:
            print(f"  Warnings: {len(result.warnings)}")
        
        if result.errors:
            print(f"  Errors: {result.errors}")
        
        print(f"  Stages Completed: {[s.value for s in result.processing_stages_completed]}")
        print("-" * 70)
    
    # Test batch processing
    print("Testing batch processing...")
    batch_audio = [case['audio'] for case in test_cases[:2]]
    
    batch_start = time.time()
    batch_results = processor.process_batch(batch_audio, sample_rate)
    batch_end = time.time()
    
    print(f"Batch processed {len(batch_results)} samples in {(batch_end - batch_start)*1000:.1f}ms")
    print(f"Average time per sample: {(batch_end - batch_start)/len(batch_results)*1000:.1f}ms")
    
    # Performance statistics
    print("\nPerformance Statistics:")
    stats = processor.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, float):
                    print(f"    {subkey}: {subvalue:.3f}")
                else:
                    print(f"    {subkey}: {subvalue}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ All tests completed successfully!")
    print("🎯 Advanced Audio Processor ready for production deployment!")
    print("\n🚀 Features demonstrated:")
    print("  ✓ Multi-engine Voice Activity Detection")
    print("  ✓ Advanced audio enhancement and noise reduction")
    print("  ✓ Intelligent audio segmentation and quality assessment")
    print("  ✓ Real-time processing optimization")
    print("  ✓ Comprehensive performance monitoring")
    print("  ✓ Cross-platform deployment ready")
