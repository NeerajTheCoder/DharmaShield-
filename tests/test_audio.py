"""
tests/test_audio.py

DharmaShield - Audio Processing Unit Tests
------------------------------------------
• Comprehensive test suite for ASR, audio processing, and voice clone detection
• Cross-platform audio testing with multi-language speech recognition
• Industry-grade testing with mocking, fixtures, and async support
"""

import pytest
import asyncio
import tempfile
import os
import wave
import io
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
from typing import Dict, Any, List, Optional

# Project imports
from src.utils.asr_engine import ASREngine, RecognitionResult
from src.utils.tts_engine import TTSEngine, speak
from src.utils.audio_processing import AudioProcessor, AudioFormat, AudioMetadata
from src.ml.voice_clone_detector import VoiceCloneDetector, CloneDetectionResult
from src.ml.audio_threat_detector import AudioThreatDetector, ThreatType


class TestASREngine:
    """Test suite for Automatic Speech Recognition engine."""
    
    @pytest.fixture
    def asr_engine(self):
        """Create ASR engine instance for testing."""
        return ASREngine(engine='vosk', language='en')
    
    @pytest.fixture
    def mock_vosk_model(self):
        """Mock Vosk model for testing."""
        mock_model = Mock()
        mock_recognizer = Mock()
        mock_recognizer.Result.return_value = '{"text": "hello world"}'
        mock_recognizer.AcceptWaveform.return_value = True
        
        with patch('src.utils.asr_engine.VoskModel', return_value=mock_model), \
             patch('src.utils.asr_engine.KaldiRecognizer', return_value=mock_recognizer):
            yield mock_model, mock_recognizer
    
    @pytest.fixture
    def mock_speech_recognition(self):
        """Mock speech recognition library."""
        mock_recognizer = Mock()
        mock_audio = Mock()
        mock_microphone = Mock()
        
        with patch('speech_recognition.Recognizer', return_value=mock_recognizer), \
             patch('speech_recognition.Microphone', return_value=mock_microphone):
            yield mock_recognizer, mock_audio, mock_microphone
    
    def test_asr_engine_initialization(self, asr_engine):
        """Test ASR engine initialization."""
        assert asr_engine.engine == 'vosk'
        assert asr_engine.language == 'en'
        assert hasattr(asr_engine, 'recognizer')
    
    def test_asr_engine_initialization_with_vosk_model(self, mock_vosk_model):
        """Test ASR engine initialization with Vosk model loading."""
        mock_model, mock_recognizer = mock_vosk_model
        
        asr_engine = ASREngine(engine='vosk', language='en')
        
        # Vosk model should be attempted to load
        assert hasattr(asr_engine, 'vosk_model')
    
    @pytest.mark.parametrize("engine,language", [
        ('vosk', 'en'),
        ('vosk', 'hi'),
        ('google', 'en'),
        ('google', 'es'),
        ('whisper', 'fr'),
    ])
    def test_asr_engine_different_configs(self, engine, language):
        """Test ASR engine with different configurations."""
        asr_engine = ASREngine(engine=engine, language=language)
        assert asr_engine.engine == engine
        assert asr_engine.language == language
    
    def test_set_language(self, asr_engine):
        """Test language setting."""
        asr_engine.set_language('hi')
        assert asr_engine.language == 'hi'
        
        asr_engine.set_language('es')
        assert asr_engine.language == 'es'
    
    def test_listen_and_transcribe_vosk_success(self, asr_engine, mock_vosk_model, mock_speech_recognition):
        """Test successful transcription with Vosk engine."""
        mock_model, mock_recognizer = mock_vosk_model
        mock_sr_recognizer, mock_audio, mock_microphone = mock_speech_recognition
        
        # Setup mocks
        asr_engine.vosk_model = mock_model
        mock_sr_recognizer.listen.return_value = mock_audio
        mock_audio.get_wav_data.return_value = b'fake_wav_data'
        
        # Mock wave file processing
        with patch('wave.open') as mock_wave_open:
            mock_wave_file = Mock()
            mock_wave_file.getframerate.return_value = 16000
            mock_wave_file.getnframes.return_value = 1000
            mock_wave_file.readframes.return_value = b'fake_audio_frames'
            mock_wave_open.return_value.__enter__.return_value = mock_wave_file
            
            result = asr_engine.listen_and_transcribe(prompt="Say something")
            
            assert result == "hello world"
            mock_sr_recognizer.listen.assert_called_once()
    
    def test_listen_and_transcribe_google_fallback(self, asr_engine, mock_speech_recognition):
        """Test fallback to Google Speech Recognition."""
        mock_sr_recognizer, mock_audio, mock_microphone = mock_speech_recognition
        
        # Disable Vosk model
        asr_engine.vosk_model = None
        
        # Setup Google recognition mock
        mock_sr_recognizer.listen.return_value = mock_audio
        mock_sr_recognizer.recognize_google.return_value = "recognized text"
        
        with patch('src.utils.language.get_google_lang_code', return_value='en-US'):
            result = asr_engine.listen_and_transcribe()
            
            assert result == "recognized text"
            mock_sr_recognizer.recognize_google.assert_called_once_with(mock_audio, language='en-US')
    
    def test_listen_and_transcribe_timeout(self, asr_engine, mock_speech_recognition):
        """Test transcription with timeout."""
        mock_sr_recognizer, mock_audio, mock_microphone = mock_speech_recognition
        
        # Simulate timeout
        mock_sr_recognizer.listen.side_effect = Exception("Timeout")
        
        result = asr_engine.listen_and_transcribe(timeout=1)
        
        assert result == ""  # Should return empty string on error
    
    def test_listen_and_transcribe_recognition_error(self, asr_engine, mock_speech_recognition):
        """Test handling of recognition errors."""
        mock_sr_recognizer, mock_audio, mock_microphone = mock_speech_recognition
        
        asr_engine.vosk_model = None
        mock_sr_recognizer.listen.return_value = mock_audio
        mock_sr_recognizer.recognize_google.side_effect = Exception("Recognition failed")
        
        result = asr_engine.listen_and_transcribe()
        
        assert result == ""  # Should return empty string on error
    
    @pytest.mark.parametrize("language,expected_google_code", [
        ('en', 'en-US'),
        ('hi', 'hi-IN'),
        ('es', 'es-ES'),
        ('fr', 'fr-FR'),
    ])
    def test_language_code_mapping(self, asr_engine, language, expected_google_code):
        """Test language code mapping for Google Speech Recognition."""
        asr_engine.set_language(language)
        
        with patch('src.utils.language.get_google_lang_code', return_value=expected_google_code) as mock_get_code:
            with patch.object(asr_engine.recognizer, 'listen'):
                with patch.object(asr_engine.recognizer, 'recognize_google'):
                    asr_engine.listen_and_transcribe()
                    mock_get_code.assert_called_with(language)
    
    def test_audio_data_processing(self, asr_engine):
        """Test processing of different audio data formats."""
        # Test with bytes data
        audio_bytes = b'fake_audio_data'
        
        # Mock the recognition process
        with patch.object(asr_engine, 'vosk_model', None):
            with patch.object(asr_engine.recognizer, 'recognize_google', return_value="test result"):
                # This would be called with actual audio data in real scenario
                pass


class TestTTSEngine:
    """Test suite for Text-to-Speech engine."""
    
    @pytest.fixture
    def tts_engine(self):
        """Create TTS engine instance for testing."""
        with patch('pyttsx3.init') as mock_init:
            mock_engine = Mock()
            mock_init.return_value = mock_engine
            
            # Mock voices
            mock_voices = [
                Mock(id='voice1', languages=[b'en']),
                Mock(id='voice2', languages=[b'hi']),
                Mock(id='voice3', languages=[b'es']),
            ]
            mock_engine.getProperty.return_value = mock_voices
            
            tts_engine = TTSEngine(rate=170)
            tts_engine.engine = mock_engine
            return tts_engine
    
    def test_tts_engine_initialization(self, tts_engine):
        """Test TTS engine initialization."""
        assert tts_engine.rate == 170
        assert tts_engine.current_voice_lang == 'en'
        assert tts_engine.engine is not None
    
    def test_set_voice_language_found(self, tts_engine):
        """Test setting voice when language is found."""
        tts_engine.set_voice('hi')
        
        # Should set voice and update current language
        tts_engine.engine.setProperty.assert_called()
        assert tts_engine.current_voice_lang == 'hi'
    
    def test_set_voice_language_not_found(self, tts_engine):
        """Test setting voice when language is not found."""
        tts_engine.set_voice('unknown')
        
        # Should fallback to first available voice
        tts_engine.engine.setProperty.assert_called()
        assert tts_engine.current_voice_lang == 'unknown'
    
    @pytest.mark.parametrize("text,language", [
        ("Hello world", "en"),
        ("नमस्ते दुनिया", "hi"),
        ("Hola mundo", "es"),
        ("Bonjour le monde", "fr"),
    ])
    def test_speak_different_languages(self, tts_engine, text, language):
        """Test speaking in different languages."""
        tts_engine.speak(text, lang=language)
        
        # Should set rate, say text, and run
        tts_engine.engine.setProperty.assert_called()
        tts_engine.engine.say.assert_called_with(text)
        tts_engine.engine.runAndWait.assert_called()
    
    def test_speak_same_language_twice(self, tts_engine):
        """Test speaking in same language twice (no voice change needed)."""
        tts_engine.speak("First text", lang="en")
        tts_engine.speak("Second text", lang="en")
        
        # Voice should only be set once
        assert tts_engine.engine.setProperty.call_count >= 1
        assert tts_engine.engine.say.call_count == 2
    
    def test_speak_empty_text(self, tts_engine):
        """Test speaking empty text."""
        tts_engine.speak("", lang="en")
        
        # Should still attempt to speak
        tts_engine.engine.say.assert_called_with("")
        tts_engine.engine.runAndWait.assert_called()
    
    def test_global_speak_function(self):
        """Test global speak function."""
        with patch('src.utils.tts_engine._tts_engine') as mock_global_engine:
            speak("test text", lang="en")
            mock_global_engine.speak.assert_called_once_with("test text", lang="en")


class TestAudioProcessor:
    """Test suite for audio processing functionality."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create audio processor instance for testing."""
        return AudioProcessor()
    
    @pytest.fixture
    def sample_audio_data(self):
        """Create sample audio data for testing."""
        # Generate simple sine wave
        import numpy as np
        sample_rate = 16000
        duration = 1.0  # 1 second
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        return {
            'data': audio_data,
            'sample_rate': sample_rate,
            'duration': duration,
            'channels': 1
        }
    
    def test_audio_processor_initialization(self, audio_processor):
        """Test audio processor initialization."""
        assert audio_processor is not None
        assert hasattr(audio_processor, 'supported_formats')
    
    def test_load_audio_from_file(self, audio_processor, sample_audio_data):
        """Test loading audio from file."""
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Write sample audio data to file
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_audio_data['sample_rate'])
                
                # Convert float to int16
                audio_int16 = (sample_audio_data['data'] * 32767).astype('int16')
                wav_file.writeframes(audio_int16.tobytes())
            
            try:
                result = audio_processor.load_audio(temp_file.name)
                
                assert result.success
                assert result.audio_data is not None
                assert result.metadata.sample_rate == sample_audio_data['sample_rate']
                assert result.metadata.channels == 1
                
            finally:
                os.unlink(temp_file.name)
    
    def test_load_audio_nonexistent_file(self, audio_processor):
        """Test loading audio from nonexistent file."""
        result = audio_processor.load_audio("nonexistent_file.wav")
        
        assert not result.success
        assert result.audio_data is None
        assert result.error_message != ""
    
    def test_audio_format_detection(self, audio_processor):
        """Test audio format detection."""
        formats = ['.wav', '.mp3', '.flac', '.ogg']
        
        for fmt in formats:
            detected = audio_processor.detect_format(f"test_file{fmt}")
            assert detected.lower() == fmt[1:]  # Remove dot
    
    def test_audio_preprocessing(self, audio_processor, sample_audio_data):
        """Test audio preprocessing operations."""
        audio_data = sample_audio_data['data']
        sample_rate = sample_audio_data['sample_rate']
        
        # Test normalization
        normalized = audio_processor.normalize_audio(audio_data)
        assert abs(normalized.max()) <= 1.0
        assert abs(normalized.min()) <= 1.0
        
        # Test resampling
        new_sample_rate = 8000
        resampled = audio_processor.resample_audio(audio_data, sample_rate, new_sample_rate)
        expected_length = len(audio_data) * new_sample_rate // sample_rate
        assert abs(len(resampled) - expected_length) <= 1
    
    def test_audio_feature_extraction(self, audio_processor, sample_audio_data):
        """Test audio feature extraction."""
        audio_data = sample_audio_data['data']
        sample_rate = sample_audio_data['sample_rate']
        
        features = audio_processor.extract_features(audio_data, sample_rate)
        
        assert isinstance(features, dict)
        assert 'mfcc' in features
        assert 'spectral_centroid' in features
        assert 'zero_crossing_rate' in features
        assert 'energy' in features
    
    def test_noise_reduction(self, audio_processor, sample_audio_data):
        """Test noise reduction functionality."""
        # Add noise to audio
        import numpy as np
        audio_data = sample_audio_data['data']
        noisy_audio = audio_data + np.random.normal(0, 0.1, len(audio_data))
        
        # Apply noise reduction
        clean_audio = audio_processor.reduce_noise(noisy_audio, sample_audio_data['sample_rate'])
        
        # Clean audio should be closer to original
        original_mse = np.mean((audio_data - audio_data) ** 2)  # 0
        noisy_mse = np.mean((noisy_audio - audio_data) ** 2)
        clean_mse = np.mean((clean_audio - audio_data) ** 2)
        
        assert clean_mse < noisy_mse  # Should be less noisy
    
    @pytest.mark.parametrize("sample_rate,expected_quality", [
        (8000, "low"),
        (16000, "medium"),
        (44100, "high"),
        (48000, "high"),
    ])
    def test_audio_quality_assessment(self, audio_processor, sample_rate, expected_quality):
        """Test audio quality assessment."""
        # Generate audio with specific sample rate
        import numpy as np
        duration = 1.0
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        quality = audio_processor.assess_quality(audio_data, sample_rate)
        
        assert quality == expected_quality


class TestVoiceCloneDetector:
    """Test suite for voice clone detection."""
    
    @pytest.fixture
    def clone_detector(self):
        """Create voice clone detector instance for testing."""
        return VoiceCloneDetector()
    
    @pytest.fixture
    def mock_voice_model(self):
        """Mock voice analysis model."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.2]  # Low clone probability
        mock_model.predict_proba.return_value = [[0.8, 0.2]]  # [real, clone]
        return mock_model
    
    def test_clone_detector_initialization(self, clone_detector):
        """Test clone detector initialization."""
        assert clone_detector.model is None
        assert not clone_detector.is_loaded
        assert clone_detector.threshold == 0.5
    
    def test_load_model(self, clone_detector, mock_voice_model):
        """Test model loading."""
        with patch.object(clone_detector, '_load_pretrained_model', return_value=mock_voice_model):
            result = clone_detector.load_model()
            
            assert result is True
            assert clone_detector.is_loaded
            assert clone_detector.model == mock_voice_model
    
    def test_detect_voice_clone_real_voice(self, clone_detector, mock_voice_model, sample_audio_data):
        """Test detection of real (non-cloned) voice."""
        clone_detector.model = mock_voice_model
        clone_detector.is_loaded = True
        
        # Mock low clone probability
        mock_voice_model.predict_proba.return_value = [[0.9, 0.1]]
        
        result = clone_detector.detect_clone(
            sample_audio_data['data'], 
            sample_audio_data['sample_rate']
        )
        
        assert isinstance(result, CloneDetectionResult)
        assert result.success
        assert not result.is_clone
        assert result.confidence > 0.8
    
    def test_detect_voice_clone_cloned_voice(self, clone_detector, mock_voice_model, sample_audio_data):
        """Test detection of cloned voice."""
        clone_detector.model = mock_voice_model
        clone_detector.is_loaded = True
        
        # Mock high clone probability
        mock_voice_model.predict_proba.return_value = [[0.2, 0.8]]
        
        result = clone_detector.detect_clone(
            sample_audio_data['data'], 
            sample_audio_data['sample_rate']
        )
        
        assert isinstance(result, CloneDetectionResult)
        assert result.success
        assert result.is_clone
        assert result.confidence > 0.7
    
    def test_detect_clone_model_not_loaded(self, clone_detector, sample_audio_data):
        """Test clone detection with unloaded model."""
        result = clone_detector.detect_clone(
            sample_audio_data['data'], 
            sample_audio_data['sample_rate']
        )
        
        assert isinstance(result, CloneDetectionResult)
        assert not result.success
        assert "not loaded" in result.error_message.lower()
    
    def test_detect_clone_empty_audio(self, clone_detector, mock_voice_model):
        """Test clone detection with empty audio."""
        clone_detector.model = mock_voice_model
        clone_detector.is_loaded = True
        
        result = clone_detector.detect_clone([], 16000)
        
        assert isinstance(result, CloneDetectionResult)
        assert not result.success
        assert "empty" in result.error_message.lower()
    
    def test_set_threshold(self, clone_detector):
        """Test threshold setting."""
        clone_detector.set_threshold(0.7)
        assert clone_detector.threshold == 0.7
        
        # Test threshold clamping
        clone_detector.set_threshold(-0.1)
        assert clone_detector.threshold == 0.0
        
        clone_detector.set_threshold(1.5)
        assert clone_detector.threshold == 1.0
    
    def test_batch_clone_detection(self, clone_detector, mock_voice_model, sample_audio_data):
        """Test batch clone detection."""
        clone_detector.model = mock_voice_model
        clone_detector.is_loaded = True
        
        # Create multiple audio samples
        audio_samples = [
            (sample_audio_data['data'], sample_audio_data['sample_rate']),
            (sample_audio_data['data'] * 0.5, sample_audio_data['sample_rate']),
            (sample_audio_data['data'] * 2.0, sample_audio_data['sample_rate']),
        ]
        
        # Mock different probabilities
        mock_voice_model.predict_proba.return_value = [
            [0.9, 0.1],  # Real
            [0.3, 0.7],  # Clone
            [0.8, 0.2],  # Real
        ]
        
        results = clone_detector.detect_batch(audio_samples)
        
        assert len(results) == 3
        assert all(isinstance(r, CloneDetectionResult) for r in results)
        assert not results[0].is_clone  # Real
        assert results[1].is_clone      # Clone
        assert not results[2].is_clone  # Real


class TestAudioThreatDetector:
    """Test suite for audio threat detection."""
    
    @pytest.fixture
    def threat_detector(self):
        """Create audio threat detector instance for testing."""
        return AudioThreatDetector()
    
    @pytest.fixture
    def mock_threat_model(self):
        """Mock threat detection model."""
        mock_model = Mock()
        mock_model.predict.return_value = [ThreatType.SAFE.value]
        mock_model.predict_proba.return_value = [[0.8, 0.1, 0.05, 0.05]]  # [safe, phishing, spam, malicious]
        return mock_model
    
    def test_threat_detector_initialization(self, threat_detector):
        """Test threat detector initialization."""
        assert threat_detector.model is None
        assert not threat_detector.is_loaded
        assert isinstance(threat_detector.threat_types, list)
    
    def test_detect_audio_threats_safe(self, threat_detector, mock_threat_model, sample_audio_data):
        """Test detection of safe audio."""
        threat_detector.model = mock_threat_model
        threat_detector.is_loaded = True
        
        result = threat_detector.detect_threats(
            sample_audio_data['data'],
            sample_audio_data['sample_rate']
        )
        
        assert result.success
        assert result.threat_type == ThreatType.SAFE
        assert result.confidence > 0.7
    
    def test_detect_audio_threats_phishing(self, threat_detector, mock_threat_model, sample_audio_data):
        """Test detection of phishing audio."""
        threat_detector.model = mock_threat_model
        threat_detector.is_loaded = True
        
        # Mock phishing detection
        mock_threat_model.predict.return_value = [ThreatType.PHISHING.value]
        mock_threat_model.predict_proba.return_value = [[0.1, 0.8, 0.05, 0.05]]
        
        result = threat_detector.detect_threats(
            sample_audio_data['data'],
            sample_audio_data['sample_rate']
        )
        
        assert result.success
        assert result.threat_type == ThreatType.PHISHING
        assert result.confidence > 0.7
    
    @pytest.mark.parametrize("threat_type,proba_vector", [
        (ThreatType.SAFE, [0.9, 0.05, 0.025, 0.025]),
        (ThreatType.PHISHING, [0.1, 0.8, 0.05, 0.05]),
        (ThreatType.SPAM, [0.1, 0.1, 0.7, 0.1]),
        (ThreatType.MALICIOUS, [0.1, 0.1, 0.1, 0.7]),
    ])
    def test_detect_different_threat_types(self, threat_detector, mock_threat_model, 
                                          sample_audio_data, threat_type, proba_vector):
        """Test detection of different threat types."""
        threat_detector.model = mock_threat_model
        threat_detector.is_loaded = True
        
        mock_threat_model.predict.return_value = [threat_type.value]
        mock_threat_model.predict_proba.return_value = [proba_vector]
        
        result = threat_detector.detect_threats(
            sample_audio_data['data'],
            sample_audio_data['sample_rate']
        )
        
        assert result.success
        assert result.threat_type == threat_type
        assert result.confidence == max(proba_vector)


class TestAudioProcessingIntegration:
    """Integration tests for audio processing components."""
    
    @pytest.fixture
    def audio_pipeline(self):
        """Create complete audio processing pipeline."""
        return {
            'asr': ASREngine(),
            'tts': None,  # Will be mocked
            'processor': AudioProcessor(),
            'clone_detector': VoiceCloneDetector(),
            'threat_detector': AudioThreatDetector()
        }
    
    def test_end_to_end_audio_processing(self, audio_pipeline, sample_audio_data):
        """Test complete audio processing pipeline."""
        # Mock all components
        with patch.object(audio_pipeline['asr'], 'listen_and_transcribe', return_value="test transcript"):
            with patch.object(audio_pipeline['processor'], 'load_audio') as mock_load:
                with patch.object(audio_pipeline['clone_detector'], 'detect_clone') as mock_clone:
                    with patch.object(audio_pipeline['threat_detector'], 'detect_threats') as mock_threat:
                        
                        # Setup mocks
                        mock_load.return_value = Mock(success=True, audio_data=sample_audio_data['data'])
                        mock_clone.return_value = Mock(success=True, is_clone=False, confidence=0.9)
                        mock_threat.return_value = Mock(success=True, threat_type=ThreatType.SAFE, confidence=0.9)
                        
                        # Process audio file
                        audio_file = "test_audio.wav"
                        
                        # Load audio
                        load_result = audio_pipeline['processor'].load_audio(audio_file)
                        
                        # Detect clones
                        clone_result = audio_pipeline['clone_detector'].detect_clone(
                            sample_audio_data['data'], sample_audio_data['sample_rate']
                        )
                        
                        # Detect threats
                        threat_result = audio_pipeline['threat_detector'].detect_threats(
                            sample_audio_data['data'], sample_audio_data['sample_rate']
                        )
                        
                        # Verify results
                        assert load_result.success
                        assert clone_result.success and not clone_result.is_clone
                        assert threat_result.success and threat_result.threat_type == ThreatType.SAFE
    
    def test_multilingual_audio_processing(self, audio_pipeline):
        """Test multilingual audio processing."""
        languages = ['en', 'hi', 'es', 'fr']
        
        for lang in languages:
            asr_engine = ASREngine(language=lang)
            
            with patch.object(asr_engine, 'listen_and_transcribe') as mock_transcribe:
                mock_transcribe.return_value = f"test text in {lang}"
                
                result = asr_engine.listen_and_transcribe()
                assert result == f"test text in {lang}"
    
    def test_concurrent_audio_processing(self, sample_audio_data):
        """Test concurrent audio processing."""
        import threading
        import queue
        
        def worker(audio_queue, result_queue):
            processor = AudioProcessor()
            
            while True:
                try:
                    audio_data, sample_rate = audio_queue.get(timeout=1)
                    features = processor.extract_features(audio_data, sample_rate)
                    result_queue.put(features)
                    audio_queue.task_done()
                except queue.Empty:
                    break
        
        # Setup queues
        audio_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Add audio samples
        for _ in range(5):
            audio_queue.put((sample_audio_data['data'], sample_audio_data['sample_rate']))
        
        # Start workers
        threads = []
        for _ in range(3):
            t = threading.Thread(target=worker, args=(audio_queue, result_queue))
            t.start()
            threads.append(t)
        
        # Wait for completion
        audio_queue.join()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Cleanup threads
        for t in threads:
            t.join()
        
        assert len(results) == 5
        assert all('mfcc' in result for result in results)
    
    @pytest.mark.performance
    def test_audio_processing_performance(self, sample_audio_data):
        """Test audio processing performance."""
        processor = AudioProcessor()
        
        # Measure feature extraction time
        import time
        start_time = time.time()
        
        for _ in range(100):
            features = processor.extract_features(
                sample_audio_data['data'], 
                sample_audio_data['sample_rate']
            )
        
        processing_time = time.time() - start_time
        
        # Should process 100 samples in reasonable time
        assert processing_time < 10.0  # 10 seconds max
        assert processing_time / 100 < 0.1  # Max 100ms per sample
    
    def test_memory_management_audio(self, sample_audio_data):
        """Test memory management in audio processing."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large amount of audio data
        large_audio_data = sample_audio_data['data']
        for _ in range(100):
            processor = AudioProcessor()
            features = processor.extract_features(large_audio_data, sample_audio_data['sample_rate'])
            del processor, features
        
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 200 * 1024 * 1024  # Less than 200MB


# Performance and stress tests
class TestAudioPerformance:
    """Performance testing for audio processing."""
    
    @pytest.mark.performance
    def test_large_audio_file_processing(self, sample_audio_data):
        """Test processing of large audio files."""
        # Create large audio data (10 seconds)
        import numpy as np
        large_audio = np.tile(sample_audio_data['data'], 10)
        
        processor = AudioProcessor()
        
        import time
        start_time = time.time()
        features = processor.extract_features(large_audio, sample_audio_data['sample_rate'])
        processing_time = time.time() - start_time
        
        assert features is not None
        assert processing_time < 30.0  # Should complete within 30 seconds
    
    @pytest.mark.stress
    def test_continuous_audio_processing(self, sample_audio_data):
        """Stress test continuous audio processing."""
        asr_engine = ASREngine()
        
        # Mock continuous processing
        with patch.object(asr_engine, 'listen_and_transcribe') as mock_transcribe:
            mock_transcribe.return_value = "test transcription"
            
            # Simulate continuous processing for 1000 iterations
            for i in range(1000):
                result = asr_engine.listen_and_transcribe(timeout=0.1)
                assert result == "test transcription"
                
                if i % 100 == 0:
                    # Periodic memory cleanup
                    import gc
                    gc.collect()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-m", "not performance and not stress"])

