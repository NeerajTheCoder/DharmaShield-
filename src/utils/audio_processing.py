"""
src/utils/audio_processing.py

DharmaShield - Advanced Audio Processing Engine (File Loading, Normalization, ASR Preprocessing)
-----------------------------------------------------------------------------------------------
• Industry-grade audio processing utility for cross-platform (Android/iOS/Desktop) with Kivy/Buildozer support
• Advanced file loading, downsampling, normalization, spectral analysis for ASR engines and threat detection
• Support for WAV, MP3, FLAC, M4A formats with automatic format detection and conversion
• Fully offline, optimized for voice-first operation with Google Gemma 3n integration

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import os
import sys
import io
import wave
import struct
import threading
import time
from typing import Optional, Union, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import warnings
import numpy as np

# Audio processing libraries with fallback handling
try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa not available. Advanced audio processing will be limited.", ImportWarning)

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    warnings.warn("soundfile not available. Some audio formats may not be supported.", ImportWarning)

try:
    from pydub import AudioSegment
    from pydub.effects import normalize as pydub_normalize
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    warnings.warn("pydub not available. MP3 and advanced format support limited.", ImportWarning)

try:
    from scipy import signal
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available. Some signal processing features will be limited.", ImportWarning)

# Project imports
from .logger import get_logger

logger = get_logger(__name__)

# -------------------------------
# Constants and Configuration
# -------------------------------

# Standard audio configurations for ASR
DEFAULT_SAMPLE_RATE = 16000  # Optimal for speech recognition
DEFAULT_CHANNELS = 1  # Mono audio for ASR
DEFAULT_BIT_DEPTH = 16
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_TIMEOUT = 30.0  # Max processing time per file

# Supported audio formats
SUPPORTED_FORMATS = {
    '.wav': 'WAV',
    '.mp3': 'MP3', 
    '.flac': 'FLAC',
    '.m4a': 'MP4',
    '.aac': 'AAC',
    '.ogg': 'OGG',
    '.wma': 'WMA'
}

# Normalization methods
class NormalizationMethod(Enum):
    PEAK = "peak"           # Peak normalization
    RMS = "rms"            # RMS normalization  
    LUFS = "lufs"          # Loudness normalization
    MIN_MAX = "min_max"    # Min-max scaling
    Z_SCORE = "z_score"    # Z-score normalization

# Audio quality levels
class AudioQuality(Enum):
    LOW = "low"       # 8kHz, 8-bit
    MEDIUM = "medium" # 16kHz, 16-bit
    HIGH = "high"     # 44.1kHz, 16-bit
    ULTRA = "ultra"   # 48kHz, 24-bit

# -------------------------------
# Data Structures
# -------------------------------

@dataclass
class AudioMetadata:
    """Comprehensive audio file metadata."""
    filename: str
    file_size: int
    duration: float
    sample_rate: int
    channels: int
    bit_depth: int
    format: str
    encoding: str = "unknown"
    is_mono: bool = True
    peak_amplitude: float = 0.0
    rms_level: float = 0.0
    dynamic_range: float = 0.0
    spectral_centroid: float = 0.0
    zero_crossing_rate: float = 0.0

@dataclass  
class AudioProcessingConfig:
    """Configuration for audio processing operations."""
    target_sample_rate: int = DEFAULT_SAMPLE_RATE
    target_channels: int = DEFAULT_CHANNELS
    target_bit_depth: int = DEFAULT_BIT_DEPTH
    normalization_method: NormalizationMethod = NormalizationMethod.PEAK
    quality_level: AudioQuality = AudioQuality.MEDIUM
    apply_pre_emphasis: bool = True
    apply_noise_reduction: bool = True
    frame_length: int = 2048
    hop_length: int = 512
    window_type: str = "hann"
    trim_silence: bool = True
    silence_threshold: float = 0.01
    max_duration: Optional[float] = None
    enable_caching: bool = True

@dataclass
class ProcessingResult:
    """Result of audio processing operation."""
    success: bool
    audio_data: Optional[np.ndarray] = None
    sample_rate: int = 0
    metadata: Optional[AudioMetadata] = None
    processing_time: float = 0.0
    error_message: str = ""
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

# -------------------------------
# Core Audio Processing Engine
# -------------------------------

class AudioProcessor:
    """
    Advanced audio processing engine for DharmaShield.
    
    Features:
    - Multi-format audio file loading with automatic format detection
    - Advanced resampling and format conversion
    - Multiple normalization algorithms
    - Spectral analysis and feature extraction  
    - Noise reduction and enhancement
    - ASR-optimized preprocessing
    - Caching for performance optimization
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[AudioProcessingConfig] = None):
        self.config = config or AudioProcessingConfig()
        self.cache = {}
        self.cache_lock = threading.RLock()
        self.processing_stats = {
            'files_processed': 0,
            'total_duration': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        logger.info(f"AudioProcessor initialized with config: {self.config}")
    
    def load_audio(
        self,
        file_path: Union[str, Path],
        target_sr: Optional[int] = None,
        mono: Optional[bool] = None,
        normalize: bool = True
    ) -> ProcessingResult:
        """
        Load audio file with advanced preprocessing.
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (default: config value)
            mono: Convert to mono (default: config value)  
            normalize: Apply normalization
            
        Returns:
            ProcessingResult with loaded audio data
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        # Input validation
        if not file_path.exists():
            return ProcessingResult(
                success=False,
                error_message=f"File not found: {file_path}",
                processing_time=time.time() - start_time
            )
        
        if not self._is_supported_format(file_path):
            return ProcessingResult(
                success=False,
                error_message=f"Unsupported format: {file_path.suffix}",
                processing_time=time.time() - start_time
            )
        
        # Check cache
        cache_key = self._get_cache_key(file_path, target_sr, mono, normalize)
        if self.config.enable_caching and cache_key in self.cache:
            self.processing_stats['cache_hits'] += 1
            cached_result = self.cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        self.processing_stats['cache_misses'] += 1
        
        try:
            # Load with appropriate backend
            result = self._load_with_best_backend(
                file_path, 
                target_sr or self.config.target_sample_rate,
                mono if mono is not None else (self.config.target_channels == 1),
                normalize
            )
            
            if result.success:
                self.processing_stats['files_processed'] += 1
                if result.metadata:
                    self.processing_stats['total_duration'] += result.metadata.duration
                
                # Cache result
                if self.config.enable_caching:
                    with self.cache_lock:
                        self.cache[cache_key] = result
            else:
                self.processing_stats['errors'] += 1
            
            result.processing_time = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Audio loading failed for {file_path}: {e}")
            self.processing_stats['errors'] += 1
            return ProcessingResult(
                success=False,
                error_message=f"Loading failed: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _load_with_best_backend(
        self,
        file_path: Path,
        target_sr: int,
        mono: bool,
        normalize: bool
    ) -> ProcessingResult:
        """Load audio using the best available backend."""
        
        # Try librosa first (most capable)
        if HAS_LIBROSA:
            try:
                return self._load_with_librosa(file_path, target_sr, mono, normalize)
            except Exception as e:
                logger.warning(f"Librosa loading failed for {file_path}: {e}")
        
        # Try soundfile
        if HAS_SOUNDFILE:
            try:
                return self._load_with_soundfile(file_path, target_sr, mono, normalize)
            except Exception as e:
                logger.warning(f"Soundfile loading failed for {file_path}: {e}")
        
        # Try pydub for MP3/other formats
        if HAS_PYDUB:
            try:
                return self._load_with_pydub(file_path, target_sr, mono, normalize)
            except Exception as e:
                logger.warning(f"Pydub loading failed for {file_path}: {e}")
        
        # Fallback to wave module for WAV files
        if file_path.suffix.lower() == '.wav':
            try:
                return self._load_with_wave(file_path, target_sr, mono, normalize)
            except Exception as e:
                logger.warning(f"Wave loading failed for {file_path}: {e}")
        
        return ProcessingResult(
            success=False,
            error_message="No suitable audio backend available"
        )
    
    def _load_with_librosa(
        self,
        file_path: Path,
        target_sr: int,
        mono: bool,
        normalize: bool
    ) -> ProcessingResult:
        """Load audio using librosa (preferred method)."""
        
        # Load audio with librosa
        audio_data, sample_rate = librosa.load(
            str(file_path),
            sr=target_sr,
            mono=mono,
            res_type='kaiser_fast'  # Fast, high-quality resampling
        )
        
        # Extract metadata
        metadata = self._extract_metadata_librosa(file_path, audio_data, sample_rate)
        
        # Apply preprocessing
        if normalize:
            audio_data = self._normalize_audio(audio_data, self.config.normalization_method)
        
        if self.config.apply_pre_emphasis:
            audio_data = self._apply_pre_emphasis(audio_data)
        
        if self.config.trim_silence:
            audio_data = self._trim_silence_librosa(audio_data, sample_rate)
        
        if self.config.apply_noise_reduction:
            audio_data = self._reduce_noise_spectral(audio_data, sample_rate)
        
        return ProcessingResult(
            success=True,
            audio_data=audio_data,
            sample_rate=sample_rate,
            metadata=metadata
        )
    
    def _load_with_soundfile(
        self,
        file_path: Path,
        target_sr: int,
        mono: bool,
        normalize: bool
    ) -> ProcessingResult:
        """Load audio using soundfile."""
        
        # Load audio
        audio_data, sample_rate = sf.read(str(file_path))
        
        # Convert to mono if needed
        if mono and len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sample_rate != target_sr and HAS_LIBROSA:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        
        # Extract basic metadata
        metadata = self._extract_basic_metadata(file_path, audio_data, sample_rate)
        
        # Apply processing
        if normalize:
            audio_data = self._normalize_audio(audio_data, self.config.normalization_method)
        
        return ProcessingResult(
            success=True,
            audio_data=audio_data,
            sample_rate=sample_rate,
            metadata=metadata
        )
    
    def _load_with_pydub(
        self,
        file_path: Path,
        target_sr: int,
        mono: bool,
        normalize: bool
    ) -> ProcessingResult:
        """Load audio using pydub."""
        
        # Load with pydub
        audio_segment = AudioSegment.from_file(str(file_path))
        
        # Convert to mono if needed
        if mono and audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        
        # Resample if needed
        if audio_segment.frame_rate != target_sr:
            audio_segment = audio_segment.set_frame_rate(target_sr)
        
        # Normalize with pydub if requested
        if normalize:
            audio_segment = pydub_normalize(audio_segment)
        
        # Convert to numpy array
        audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        audio_data = audio_data / (2**15)  # Convert from int16 to float32
        
        # Extract metadata
        metadata = AudioMetadata(
            filename=file_path.name,
            file_size=file_path.stat().st_size,
            duration=len(audio_segment) / 1000.0,
            sample_rate=audio_segment.frame_rate,
            channels=audio_segment.channels,
            bit_depth=audio_segment.sample_width * 8,
            format=file_path.suffix.upper(),
            is_mono=audio_segment.channels == 1
        )
        
        return ProcessingResult(
            success=True,
            audio_data=audio_data,
            sample_rate=target_sr,
            metadata=metadata
        )
    
    def _load_with_wave(
        self,
        file_path: Path,
        target_sr: int,
        mono: bool,
        normalize: bool
    ) -> ProcessingResult:
        """Load WAV file using wave module (fallback)."""
        
        with wave.open(str(file_path), 'rb') as wav_file:
            # Get parameters
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            
            # Read audio data
            frames = wav_file.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 1:
                dtype = np.uint8
                offset = 128
            elif sample_width == 2:
                dtype = np.int16
                offset = 0
            else:
                dtype = np.int32
                offset = 0
            
            audio_data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            audio_data = (audio_data - offset) / (2**(8*sample_width-1))
            
            # Handle multichannel
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
                if mono:
                    audio_data = np.mean(audio_data, axis=1)
            
            # Basic resampling if needed (simple decimation/interpolation)
            if sample_rate != target_sr:
                if HAS_SCIPY:
                    # Use scipy for resampling
                    num_samples = int(len(audio_data) * target_sr / sample_rate)
                    audio_data = signal.resample(audio_data, num_samples)
                else:
                    # Simple decimation/interpolation
                    ratio = target_sr / sample_rate
                    if ratio < 1:  # Downsample
                        step = int(1 / ratio)
                        audio_data = audio_data[::step]
                    else:  # Upsample
                        audio_data = np.repeat(audio_data, int(ratio))
                
                sample_rate = target_sr
            
            # Extract metadata
            metadata = AudioMetadata(
                filename=file_path.name,
                file_size=file_path.stat().st_size,
                duration=n_frames / wav_file.getframerate(),
                sample_rate=sample_rate,
                channels=1 if mono else channels,
                bit_depth=sample_width * 8,
                format="WAV",
                is_mono=mono or channels == 1
            )
            
            # Apply normalization
            if normalize:
                audio_data = self._normalize_audio(audio_data, self.config.normalization_method)
            
            return ProcessingResult(
                success=True,
                audio_data=audio_data,
                sample_rate=sample_rate,
                metadata=metadata
            )
    
    def _normalize_audio(
        self,
        audio_data: np.ndarray,
        method: NormalizationMethod
    ) -> np.ndarray:
        """Apply various normalization methods."""
        
        if len(audio_data) == 0:
            return audio_data
        
        if method == NormalizationMethod.PEAK:
            # Peak normalization to [-1, 1]
            peak = np.max(np.abs(audio_data))
            if peak > 0:
                return audio_data / peak
            return audio_data
        
        elif method == NormalizationMethod.RMS:
            # RMS normalization
            rms = np.sqrt(np.mean(audio_data**2))
            target_rms = 0.1  # Target RMS level
            if rms > 0:
                return audio_data * (target_rms / rms)
            return audio_data
        
        elif method == NormalizationMethod.MIN_MAX:
            # Min-max scaling to [-1, 1]
            min_val = np.min(audio_data)
            max_val = np.max(audio_data)
            if max_val > min_val:
                return 2 * (audio_data - min_val) / (max_val - min_val) - 1
            return audio_data
        
        elif method == NormalizationMethod.Z_SCORE:
            # Z-score normalization
            mean = np.mean(audio_data)
            std = np.std(audio_data)
            if std > 0:
                return (audio_data - mean) / std
            return audio_data - mean
        
        else:
            # Default to peak normalization
            return self._normalize_audio(audio_data, NormalizationMethod.PEAK)
    
    def _apply_pre_emphasis(self, audio_data: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to enhance high frequencies."""
        if len(audio_data) == 0:
            return audio_data
        
        return np.append(audio_data[0], audio_data[1:] - alpha * audio_data[:-1])
    
    def _trim_silence_librosa(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        threshold: float = None
    ) -> np.ndarray:
        """Trim silence from audio using librosa."""
        if not HAS_LIBROSA:
            return audio_data
        
        threshold = threshold or self.config.silence_threshold
        
        try:
            # Use librosa's trim function
            trimmed_audio, _ = librosa.effects.trim(
                audio_data,
                top_db=20 * np.log10(threshold) if threshold > 0 else 20
            )
            return trimmed_audio
        except Exception:
            return audio_data
    
    def _reduce_noise_spectral(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Simple spectral noise reduction."""
        if not HAS_LIBROSA or len(audio_data) == 0:
            return audio_data
        
        try:
            # Compute spectrogram
            stft = librosa.stft(audio_data, hop_length=self.config.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor from first few frames
            noise_floor = np.median(magnitude[:, :10], axis=1, keepdims=True)
            
            # Apply spectral subtraction
            clean_magnitude = magnitude - 0.5 * noise_floor
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
            
            # Reconstruct audio
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft, hop_length=self.config.hop_length)
            
            return clean_audio
        except Exception:
            return audio_data
    
    def _extract_metadata_librosa(
        self,
        file_path: Path,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> AudioMetadata:
        """Extract comprehensive metadata using librosa."""
        
        duration = len(audio_data) / sample_rate
        peak_amplitude = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0.0
        rms_level = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0.0
        
        # Advanced features with librosa
        spectral_centroid = 0.0
        zero_crossing_rate = 0.0
        
        try:
            if len(audio_data) > 0:
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                    y=audio_data, sr=sample_rate
                )[0])
                zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
        except Exception as e:
            logger.debug(f"Advanced feature extraction failed: {e}")
        
        return AudioMetadata(
            filename=file_path.name,
            file_size=file_path.stat().st_size,
            duration=duration,
            sample_rate=sample_rate,
            channels=1,  # Assuming mono after processing
            bit_depth=32,  # Float32
            format=file_path.suffix.upper(),
            is_mono=True,
            peak_amplitude=peak_amplitude,
            rms_level=rms_level,
            dynamic_range=peak_amplitude - rms_level,
            spectral_centroid=spectral_centroid,
            zero_crossing_rate=zero_crossing_rate
        )
    
    def _extract_basic_metadata(
        self,
        file_path: Path,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> AudioMetadata:
        """Extract basic metadata without advanced features."""
        
        duration = len(audio_data) / sample_rate if sample_rate > 0 else 0.0
        peak_amplitude = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0.0
        rms_level = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0.0
        
        return AudioMetadata(
            filename=file_path.name,
            file_size=file_path.stat().st_size if file_path.exists() else 0,
            duration=duration,
            sample_rate=sample_rate,
            channels=1,
            bit_depth=32,
            format=file_path.suffix.upper(),
            is_mono=True,
            peak_amplitude=peak_amplitude,
            rms_level=rms_level,
            dynamic_range=peak_amplitude - rms_level
        )
    
    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        return file_path.suffix.lower() in SUPPORTED_FORMATS
    
    def _get_cache_key(
        self,
        file_path: Path,
        target_sr: Optional[int],
        mono: Optional[bool],
        normalize: bool
    ) -> str:
        """Generate cache key for processed audio."""
        stat = file_path.stat()
        return f"{file_path}_{stat.st_mtime}_{stat.st_size}_{target_sr}_{mono}_{normalize}"
    
    def save_audio(
        self,
        audio_data: np.ndarray,
        output_path: Union[str, Path],
        sample_rate: int,
        format: str = "wav"
    ) -> bool:
        """Save processed audio to file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if HAS_SOUNDFILE:
                sf.write(str(output_path), audio_data, sample_rate)
                return True
            elif format.lower() == "wav":
                # Fallback to wave module
                with wave.open(str(output_path), 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    
                    # Convert to int16
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                return True
            else:
                logger.error(f"Cannot save format {format} without soundfile")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def clear_cache(self):
        """Clear the processing cache."""
        with self.cache_lock:
            self.cache.clear()
        logger.info("Audio processing cache cleared")

# -------------------------------
# Utility Functions
# -------------------------------

def load_audio_file(
    file_path: Union[str, Path],
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
    normalize: bool = True,
    config: Optional[AudioProcessingConfig] = None
) -> ProcessingResult:
    """
    Convenience function to load audio file.
    
    Args:
        file_path: Path to audio file
        target_sample_rate: Target sample rate for output
        mono: Convert to mono audio
        normalize: Apply normalization
        config: Optional processing configuration
        
    Returns:
        ProcessingResult with loaded audio data
    """
    processor = AudioProcessor(config)
    return processor.load_audio(file_path, target_sample_rate, mono, normalize)

def convert_sample_rate(
    audio_data: np.ndarray,
    original_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Convert audio sample rate using best available method.
    
    Args:
        audio_data: Input audio data
        original_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio data
    """
    if original_sr == target_sr:
        return audio_data
    
    if HAS_LIBROSA:
        return librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
    elif HAS_SCIPY:
        num_samples = int(len(audio_data) * target_sr / original_sr)
        return signal.resample(audio_data, num_samples)
    else:
        # Simple linear interpolation fallback
        ratio = target_sr / original_sr
        if ratio < 1:  # Downsample
            step = int(1 / ratio)
            return audio_data[::step]
        else:  # Upsample
            return np.repeat(audio_data, int(ratio))

def detect_audio_format(file_path: Union[str, Path]) -> str:
    """Detect audio file format from extension."""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    return SUPPORTED_FORMATS.get(extension, "UNKNOWN")

def validate_audio_data(audio_data: np.ndarray, sample_rate: int) -> List[str]:
    """
    Validate audio data and return list of issues.
    
    Args:
        audio_data: Audio data to validate
        sample_rate: Sample rate
        
    Returns:
        List of validation warnings/errors
    """
    issues = []
    
    if len(audio_data) == 0:
        issues.append("Audio data is empty")
    
    if sample_rate <= 0:
        issues.append("Invalid sample rate")
    
    if np.max(np.abs(audio_data)) > 1.0:
        issues.append("Audio data exceeds [-1, 1] range")
    
    if np.any(np.isnan(audio_data)):
        issues.append("Audio data contains NaN values")
    
    if np.any(np.isinf(audio_data)):
        issues.append("Audio data contains infinite values")
    
    duration = len(audio_data) / sample_rate if sample_rate > 0 else 0
    if duration > 300:  # 5 minutes
        issues.append("Audio duration is very long (>5 minutes)")
    
    return issues

# -------------------------------
# Testing and Demo
# -------------------------------

if __name__ == "__main__":
    # Demo mode
    print("=== DharmaShield Audio Processing Engine Demo ===")
    
    # Create test configuration
    config = AudioProcessingConfig(
        target_sample_rate=16000,
        normalization_method=NormalizationMethod.PEAK,
        apply_pre_emphasis=True,
        trim_silence=True
    )
    
    processor = AudioProcessor(config)
    
    print("Audio Processing Engine Features:")
    print("✓ Multi-format audio loading (WAV, MP3, FLAC, M4A)")
    print("✓ Advanced resampling and format conversion")
    print("✓ Multiple normalization algorithms")
    print("✓ Pre-emphasis and noise reduction")
    print("✓ Silence trimming and enhancement")
    print("✓ ASR-optimized preprocessing")
    print("✓ Metadata extraction and analysis")
    print("✓ Caching for performance optimization")
    print("✓ Thread-safe operations")
    
    # Show available backends
    print(f"\nAvailable backends:")
    print(f"  Librosa: {'✓' if HAS_LIBROSA else '✗'}")
    print(f"  SoundFile: {'✓' if HAS_SOUNDFILE else '✗'}")
    print(f"  PyDub: {'✓' if HAS_PYDUB else '✗'}")
    print(f"  SciPy: {'✓' if HAS_SCIPY else '✗'}")
    
    # Show supported formats
    print(f"\nSupported formats: {', '.join(SUPPORTED_FORMATS.keys())}")
    
    # Processing stats
    stats = processor.get_stats()
    print(f"\nProcessing statistics: {stats}")
    
    print("\n✅ Audio Processing Engine ready for production!")
    print("🎵 Ready to process audio for:")
    print("  • Speech recognition (ASR) preprocessing")
    print("  • Voice command analysis")
    print("  • Audio threat detection")
    print("  • Real-time audio enhancement")
    print("  • Cross-platform audio support (Android/iOS/Desktop)")

