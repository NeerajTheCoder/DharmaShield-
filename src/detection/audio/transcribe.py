"""
detection/audio/transcribe.py

DharmaShield - Advanced Multi-Language Audio Speech Recognition Engine
----------------------------------------------------------------------
• Production-grade ASR wrapper supporting Vosk, Google Speech-to-Text, and Whisper
• Fully offline-capable with intelligent fallback mechanisms
• Multi-language support with automatic language detection
• Industry-standard error handling and performance optimization
• Cross-platform deployment ready for Android, iOS, and desktop

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import asyncio
import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import json
import io
import wave
import tempfile
import os
from pathlib import Path
from collections import defaultdict, deque

# Audio processing imports
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    warnings.warn("PyAudio not available - microphone input disabled")

try:
    import speech_recognition as sr
    HAS_SPEECHRECOGNITION = True
except ImportError:
    HAS_SPEECHRECOGNITION = False
    warnings.warn("SpeechRecognition not available - Google ASR disabled")

try:
    from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel
    HAS_VOSK = True
except ImportError:
    HAS_VOSK = False
    warnings.warn("Vosk not available - offline ASR disabled")

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    warnings.warn("Whisper not available - OpenAI ASR disabled")

try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("Librosa not available - advanced audio processing disabled")

# Project imports
from ...utils.logger import get_logger
from ...core.config_loader import load_config
from ...utils.language import detect_language, get_language_name, get_google_lang_code

logger = get_logger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class ASREngine(Enum):
    """Supported ASR engines."""
    VOSK = "vosk"
    GOOGLE = "google"
    WHISPER = "whisper"
    AUTO = "auto"

class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    FLAC = "flac"
    MP3 = "mp3"
    M4A = "m4a"
    OGG = "ogg"

@dataclass
class TranscriptionResult:
    """
    Comprehensive transcription result with metadata and confidence scoring.
    """
    # Core transcription
    text: str = ""
    language: str = "en"
    confidence: float = 0.0
    
    # Engine and processing info
    engine_used: ASREngine = ASREngine.AUTO
    processing_time: float = 0.0
    
    # Audio metadata
    audio_duration: float = 0.0
    sample_rate: int = 16000
    audio_format: str = "wav"
    
    # Advanced features
    word_timestamps: List[Dict[str, Any]] = None
    alternative_transcripts: List[str] = None
    
    # Quality metrics
    signal_quality: str = "unknown"  # excellent, good, fair, poor, unknown
    noise_level: float = 0.0
    
    # Error handling
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.word_timestamps is None:
            self.word_timestamps = []
        if self.alternative_transcripts is None:
            self.alternative_transcripts = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'text': self.text,
            'language': self.language,
            'confidence': round(self.confidence, 4),
            'engine_used': self.engine_used.value,
            'processing_time': round(self.processing_time * 1000, 2),  # Convert to ms
            'audio_duration': round(self.audio_duration, 2),
            'sample_rate': self.sample_rate,
            'audio_format': self.audio_format,
            'word_timestamps': self.word_timestamps,
            'alternative_transcripts': self.alternative_transcripts,
            'signal_quality': self.signal_quality,
            'noise_level': round(self.noise_level, 4),
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    @property
    def is_successful(self) -> bool:
        """Check if transcription was successful."""
        return bool(self.text.strip()) and not self.errors
    
    @property
    def words_per_minute(self) -> float:
        """Calculate speaking rate in words per minute."""
        if self.audio_duration <= 0:
            return 0.0
        word_count = len(self.text.split())
        return (word_count / self.audio_duration) * 60.0


class TranscriptionConfig:
    """Configuration class for audio transcription."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        transcription_config = self.config.get('transcription', {})
        
        # Engine preferences (in order of preference)
        self.engine_preference = transcription_config.get('engine_preference', [
            ASREngine.VOSK.value, ASREngine.WHISPER.value, ASREngine.GOOGLE.value
        ])
        self.fallback_enabled = transcription_config.get('fallback_enabled', True)
        
        # Audio settings
        self.sample_rate = transcription_config.get('sample_rate', 16000)
        self.chunk_size = transcription_config.get('chunk_size', 4096)
        self.audio_timeout = transcription_config.get('audio_timeout', 10.0)
        self.phrase_timeout = transcription_config.get('phrase_timeout', 1.0)
        
        # Language settings
        self.default_language = transcription_config.get('default_language', 'en')
        self.supported_languages = transcription_config.get('supported_languages', [
            'en', 'hi', 'es', 'fr', 'de', 'zh', 'ar', 'ru', 'bn', 'ur', 'ta', 'te', 'mr'
        ])
        self.auto_language_detection = transcription_config.get('auto_language_detection', True)
        
        # Quality settings
        self.enable_noise_reduction = transcription_config.get('enable_noise_reduction', True)
        self.enable_voice_activity_detection = transcription_config.get('enable_vad', True)
        self.confidence_threshold = transcription_config.get('confidence_threshold', 0.7)
        
        # Advanced features
        self.enable_word_timestamps = transcription_config.get('enable_word_timestamps', True)
        self.enable_alternative_transcripts = transcription_config.get('enable_alternatives', True)
        self.max_alternatives = transcription_config.get('max_alternatives', 3)
        
        # Performance settings
        self.enable_caching = transcription_config.get('enable_caching', True)
        self.cache_size = transcription_config.get('cache_size', 100)
        self.batch_processing = transcription_config.get('batch_processing', True)
        
        # Model paths
        self.vosk_model_dir = transcription_config.get('vosk_model_dir', 'models/vosk')
        self.whisper_model_size = transcription_config.get('whisper_model_size', 'base')
        
        # Error handling
        self.max_retries = transcription_config.get('max_retries', 3)
        self.retry_delay = transcription_config.get('retry_delay', 1.0)


class BaseASREngine(ABC):
    """Abstract base class for all ASR engines."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.is_initialized = False
        self._lock = threading.Lock()
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the ASR engine. Returns True if successful."""
        pass
    
    @abstractmethod
    def transcribe_audio(self, 
                        audio_data: bytes, 
                        language: Optional[str] = None,
                        **kwargs) -> TranscriptionResult:
        """Transcribe audio data. Returns TranscriptionResult."""
        pass
    
    @abstractmethod
    def transcribe_file(self, 
                       file_path: str, 
                       language: Optional[str] = None,
                       **kwargs) -> TranscriptionResult:
        """Transcribe audio file. Returns TranscriptionResult."""
        pass
    
    def is_available(self) -> bool:
        """Check if this engine is available and can be used."""
        return self.is_initialized


class VoskASREngine(BaseASREngine):
    """
    Vosk offline ASR engine implementation.
    Provides robust offline speech recognition with multi-language support.
    """
    
    def __init__(self, config: TranscriptionConfig):
        super().__init__(config)
        self.models = {}  # Language -> Model mapping
        self.recognizers = {}  # Language -> Recognizer mapping
        
    def initialize(self) -> bool:
        """Initialize Vosk models for supported languages."""
        if not HAS_VOSK:
            logger.error("Vosk not available - cannot initialize VoskASREngine")
            return False
            
        try:
            with self._lock:
                if self.is_initialized:
                    return True
                
                SetLogLevel(-1)  # Suppress Vosk logs
                
                # Load models for supported languages
                models_loaded = 0
                for lang in self.config.supported_languages:
                    model_path = Path(self.config.vosk_model_dir) / f"vosk-model-{lang}"
                    
                    if model_path.exists():
                        try:
                            self.models[lang] = VoskModel(str(model_path))
                            self.recognizers[lang] = KaldiRecognizer(
                                self.models[lang], 
                                self.config.sample_rate
                            )
                            models_loaded += 1
                            logger.info(f"Loaded Vosk model for {lang}")
                        except Exception as e:
                            logger.warning(f"Failed to load Vosk model for {lang}: {e}")
                
                if models_loaded > 0:
                    self.is_initialized = True
                    logger.info(f"Vosk ASR initialized with {models_loaded} language models")
                    return True
                else:
                    logger.error("No Vosk models could be loaded")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to initialize Vosk ASR: {e}")
            return False
    
    def _get_recognizer(self, language: str) -> Optional[KaldiRecognizer]:
        """Get recognizer for specified language."""
        if language in self.recognizers:
            return self.recognizers[language]
        
        # Try default language
        if self.config.default_language in self.recognizers:
            return self.recognizers[self.config.default_language]
        
        # Return any available recognizer
        if self.recognizers:
            return next(iter(self.recognizers.values()))
        
        return None
    
    def transcribe_audio(self, 
                        audio_data: bytes, 
                        language: Optional[str] = None,
                        **kwargs) -> TranscriptionResult:
        """Transcribe audio data using Vosk."""
        start_time = time.time()
        
        if not self.is_initialized:
            return TranscriptionResult(
                errors=["Vosk ASR not initialized"],
                processing_time=time.time() - start_time,
                engine_used=ASREngine.VOSK
            )
        
        language = language or self.config.default_language
        recognizer = self._get_recognizer(language)
        
        if not recognizer:
            return TranscriptionResult(
                errors=[f"No Vosk model available for language: {language}"],
                processing_time=time.time() - start_time,
                engine_used=ASREngine.VOSK
            )
        
        try:
            # Process audio data
            wf = wave.open(io.BytesIO(audio_data), 'rb')
            
            # Validate audio format
            if wf.getframerate() != self.config.sample_rate:
                return TranscriptionResult(
                    errors=[f"Audio sample rate {wf.getframerate()} doesn't match expected {self.config.sample_rate}"],
                    processing_time=time.time() - start_time,
                    engine_used=ASREngine.VOSK
                )
            
            # Process audio in chunks
            transcription_parts = []
            audio_duration = wf.getnframes() / wf.getframerate()
            
            while True:
                data = wf.readframes(self.config.chunk_size)
                if len(data) == 0:
                    break
                
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if result.get('text'):
                        transcription_parts.append(result['text'])
            
            # Get final result
            final_result = json.loads(recognizer.FinalResult())
            if final_result.get('text'):
                transcription_parts.append(final_result['text'])
            
            # Combine transcription parts
            full_text = ' '.join(transcription_parts).strip()
            
            # Calculate confidence (Vosk doesn't provide confidence scores directly)
            confidence = 0.8 if full_text else 0.0
            
            return TranscriptionResult(
                text=full_text,
                language=language,
                confidence=confidence,
                engine_used=ASREngine.VOSK,
                processing_time=time.time() - start_time,
                audio_duration=audio_duration,
                sample_rate=wf.getframerate(),
                audio_format="wav"
            )
            
        except Exception as e:
            logger.error(f"Vosk transcription failed: {e}")
            return TranscriptionResult(
                errors=[f"Vosk transcription error: {str(e)}"],
                processing_time=time.time() - start_time,
                engine_used=ASREngine.VOSK
            )
    
    def transcribe_file(self, 
                       file_path: str, 
                       language: Optional[str] = None,
                       **kwargs) -> TranscriptionResult:
        """Transcribe audio file using Vosk."""
        try:
            # Convert file to appropriate format if needed
            processed_audio = self._preprocess_audio_file(file_path)
            return self.transcribe_audio(processed_audio, language, **kwargs)
            
        except Exception as e:
            logger.error(f"Failed to transcribe file {file_path}: {e}")
            return TranscriptionResult(
                errors=[f"File transcription error: {str(e)}"],
                engine_used=ASREngine.VOSK
            )
    
    def _preprocess_audio_file(self, file_path: str) -> bytes:
        """Preprocess audio file to required format."""
        if HAS_LIBROSA:
            # Use librosa for robust audio loading and conversion
            audio, sr = librosa.load(file_path, sr=self.config.sample_rate, mono=True)
            
            # Convert to bytes
            with io.BytesIO() as wav_buffer:
                sf.write(wav_buffer, audio, sr, format='WAV', subtype='PCM_16')
                return wav_buffer.getvalue()
        else:
            # Fallback: read file directly (assumes correct format)
            with open(file_path, 'rb') as f:
                return f.read()


class GoogleASREngine(BaseASREngine):
    """
    Google Cloud Speech-to-Text ASR engine implementation.
    Provides high-accuracy cloud-based speech recognition.
    """
    
    def __init__(self, config: TranscriptionConfig):
        super().__init__(config)
        self.recognizer = None
        
    def initialize(self) -> bool:
        """Initialize Google ASR engine."""
        if not HAS_SPEECHRECOGNITION:
            logger.error("SpeechRecognition library not available")
            return False
            
        try:
            with self._lock:
                if self.is_initialized:
                    return True
                
                self.recognizer = sr.Recognizer()
                
                # Configure recognition parameters
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = self.config.phrase_timeout
                self.recognizer.phrase_threshold = 0.3
                
                self.is_initialized = True
                logger.info("Google ASR engine initialized")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize Google ASR: {e}")
            return False
    
    def transcribe_audio(self, 
                        audio_data: bytes, 
                        language: Optional[str] = None,
                        **kwargs) -> TranscriptionResult:
        """Transcribe audio data using Google Speech-to-Text."""
        start_time = time.time()
        
        if not self.is_initialized:
            return TranscriptionResult(
                errors=["Google ASR not initialized"],
                processing_time=time.time() - start_time,
                engine_used=ASREngine.GOOGLE
            )
        
        language = language or self.config.default_language
        google_lang_code = get_google_lang_code(language)
        
        try:
            # Create AudioData object from bytes
            audio_source = sr.AudioData(audio_data, self.config.sample_rate, 2)  # 16-bit samples
            
            # Perform recognition with alternatives if enabled
            show_all = self.config.enable_alternative_transcripts
            
            result = self.recognizer.recognize_google(
                audio_source,
                language=google_lang_code,
                show_all=show_all
            )
            
            # Process results
            if show_all and isinstance(result, list) and result:
                # Multiple alternatives returned
                primary_result = result[0]
                text = primary_result.get('transcript', '')
                confidence = primary_result.get('confidence', 0.0)
                alternatives = [alt.get('transcript', '') for alt in result[1:self.config.max_alternatives + 1]]
            elif isinstance(result, str):
                # Single result returned
                text = result
                confidence = 0.85  # Google doesn't always provide confidence
                alternatives = []
            else:
                text = ""
                confidence = 0.0
                alternatives = []
            
            return TranscriptionResult(
                text=text,
                language=language,
                confidence=confidence,
                engine_used=ASREngine.GOOGLE,
                processing_time=time.time() - start_time,
                alternative_transcripts=alternatives,
                audio_format="wav"
            )
            
        except sr.UnknownValueError:
            return TranscriptionResult(
                errors=["Google ASR could not understand audio"],
                processing_time=time.time() - start_time,
                engine_used=ASREngine.GOOGLE
            )
        except sr
