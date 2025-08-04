"""
tests/test_text.py

DharmaShield - Text Processing Unit Tests
-----------------------------------------
• Comprehensive test suite for text detection, language processing, and vectorization modules
• Cross-platform compatibility testing with multi-language support
• Industry-grade test patterns with mocking, fixtures, and parametrization
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
from typing import Dict, Any, List

# Project imports
from src.utils.language import (
    detect_language, get_language_name, get_google_lang_code, 
    list_supported, LANGUAGE_NAMES, GOOGLE_LANG_CODES
)
from src.ml.text_vectorizer import TextVectorizer, VectorizationResult
from src.ml.scam_classifier import ScamClassifier, ClassificationResult
from src.core.privacy_engine import PrivacyEngine, DataClassification


class TestLanguageUtils:
    """Test suite for language utility functions."""
    
    @pytest.fixture(autouse=True)
    def setup_language_detection(self):
        """Setup language detection with deterministic seed."""
        with patch('src.utils.language.DetectorFactory') as mock_factory:
            mock_factory.seed = 0
            yield
    
    @pytest.mark.parametrize("text,expected_lang", [
        ("Hello world", "en"),
        ("Hola mundo", "es"),
        ("Bonjour le monde", "fr"),
        ("Hallo Welt", "de"),
        ("नमस्ते दुनिया", "hi"),
        ("你好世界", "zh"),
        ("", "en"),  # Empty text should default to English
        ("123456", "en"),  # Numbers should default to English
    ])
    def test_detect_language(self, text, expected_lang):
        """Test language detection with various inputs."""
        with patch('src.utils.language.detect') as mock_detect:
            mock_detect.return_value = expected_lang
            result = detect_language(text)
            assert result == expected_lang
    
    def test_detect_language_exception_handling(self):
        """Test language detection exception handling."""
        with patch('src.utils.language.detect', side_effect=Exception("Detection failed")):
            result = detect_language("some text")
            assert result == "en"  # Should default to English on exception
    
    @pytest.mark.parametrize("lang_code,expected_name", [
        ("en", "English"),
        ("hi", "Hindi"),
        ("es", "Spanish"),
        ("fr", "French"),
        ("de", "German"),
        ("zh", "Chinese"),
        ("ar", "Arabic"),
        ("ru", "Russian"),
        ("unknown", "unknown"),  # Unknown language should return as-is
    ])
    def test_get_language_name(self, lang_code, expected_name):
        """Test language name retrieval."""
        result = get_language_name(lang_code)
        assert result == expected_name
    
    @pytest.mark.parametrize("lang_code,expected_google_code", [
        ("en", "en-US"),
        ("hi", "hi-IN"),
        ("es", "es-ES"),
        ("fr", "fr-FR"),
        ("de", "de-DE"),
        ("zh", "zh-CN"),
        ("ar", "ar-SA"),
        ("ru", "ru-RU"),
        ("unknown", "en-US"),  # Unknown should default to en-US
    ])
    def test_get_google_lang_code(self, lang_code, expected_google_code):
        """Test Google language code mapping."""
        result = get_google_lang_code(lang_code)
        assert result == expected_google_code
    
    def test_list_supported_languages(self):
        """Test supported languages list."""
        supported = list_supported()
        assert isinstance(supported, list)
        assert len(supported) > 0
        assert "en" in supported
        assert "hi" in supported
        assert all(isinstance(lang, str) for lang in supported)
    
    def test_language_constants_consistency(self):
        """Test consistency between language constants."""
        # All languages in LANGUAGE_NAMES should have Google codes
        for lang_code in LANGUAGE_NAMES.keys():
            assert lang_code in GOOGLE_LANG_CODES, f"Missing Google code for {lang_code}"
        
        # All language names should be strings
        for name in LANGUAGE_NAMES.values():
            assert isinstance(name, str) and len(name) > 0
        
        # All Google codes should be valid format
        for code in GOOGLE_LANG_CODES.values():
            assert isinstance(code, str)
            assert "-" in code  # Should be in format like "en-US"


class TestTextVectorizer:
    """Test suite for text vectorization functionality."""
    
    @pytest.fixture
    def vectorizer(self):
        """Create text vectorizer instance for testing."""
        return TextVectorizer(model_name="test_model", embedding_dim=384)
    
    @pytest.fixture
    def mock_transformer_model(self):
        """Mock transformer model for testing."""
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4]] * 384  # 384-dim vector
        return mock_model
    
    def test_vectorizer_initialization(self, vectorizer):
        """Test vectorizer initialization."""
        assert vectorizer.model_name == "test_model"
        assert vectorizer.embedding_dim == 384
        assert vectorizer.model is None  # Not loaded yet
        assert not vectorizer.is_loaded
    
    def test_vectorizer_load_model(self, vectorizer, mock_transformer_model):
        """Test model loading."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_transformer_model
            
            result = vectorizer.load_model()
            
            assert result is True
            assert vectorizer.is_loaded
            assert vectorizer.model is not None
            mock_st.assert_called_once_with("test_model")
    
    def test_vectorizer_load_model_failure(self, vectorizer):
        """Test model loading failure handling."""
        with patch('sentence_transformers.SentenceTransformer', side_effect=Exception("Load failed")):
            result = vectorizer.load_model()
            
            assert result is False
            assert not vectorizer.is_loaded
            assert vectorizer.model is None
    
    @pytest.mark.parametrize("text,expected_success", [
        ("Hello, this is a test message.", True),
        ("", False),  # Empty text should fail
        ("   ", False),  # Whitespace only should fail
        ("A" * 10000, True),  # Very long text should still work
        ("Multi\nline\ntext", True),  # Multi-line text
        ("Text with émojis 😀🎉", True),  # Unicode text
    ])
    def test_vectorize_text(self, vectorizer, mock_transformer_model, text, expected_success):
        """Test text vectorization."""
        vectorizer.model = mock_transformer_model
        vectorizer.is_loaded = True
        
        if expected_success:
            mock_transformer_model.encode.return_value = [[0.1] * 384]
        
        result = vectorizer.vectorize_text(text)
        
        assert isinstance(result, VectorizationResult)
        assert result.success == expected_success
        
        if expected_success:
            assert result.vector is not None
            assert len(result.vector) == 384
            assert all(isinstance(x, float) for x in result.vector)
            mock_transformer_model.encode.assert_called_once()
        else:
            assert result.vector is None
            assert result.error_message != ""
    
    def test_vectorize_text_model_not_loaded(self, vectorizer):
        """Test vectorization with unloaded model."""
        result = vectorizer.vectorize_text("test text")
        
        assert isinstance(result, VectorizationResult)
        assert not result.success
        assert result.vector is None
        assert "not loaded" in result.error_message.lower()
    
    def test_vectorize_batch(self, vectorizer, mock_transformer_model):
        """Test batch text vectorization."""
        vectorizer.model = mock_transformer_model
        vectorizer.is_loaded = True
        
        texts = ["Text 1", "Text 2", "Text 3"]
        mock_transformer_model.encode.return_value = [[0.1] * 384] * len(texts)
        
        results = vectorizer.vectorize_batch(texts)
        
        assert len(results) == len(texts)
        assert all(isinstance(r, VectorizationResult) for r in results)
        assert all(r.success for r in results)
        assert all(len(r.vector) == 384 for r in results)
        mock_transformer_model.encode.assert_called_once_with(texts)
    
    def test_vectorize_batch_empty_list(self, vectorizer):
        """Test batch vectorization with empty list."""
        results = vectorizer.vectorize_batch([])
        assert results == []
    
    def test_vectorize_batch_mixed_content(self, vectorizer, mock_transformer_model):
        """Test batch vectorization with mixed valid/invalid content."""
        vectorizer.model = mock_transformer_model
        vectorizer.is_loaded = True
        
        texts = ["Valid text", "", "Another valid text", "   "]
        valid_texts = ["Valid text", "Another valid text"]
        mock_transformer_model.encode.return_value = [[0.1] * 384] * len(valid_texts)
        
        results = vectorizer.vectorize_batch(texts)
        
        assert len(results) == len(texts)
        assert results[0].success  # Valid text
        assert not results[1].success  # Empty text
        assert results[2].success  # Valid text
        assert not results[3].success  # Whitespace only
    
    def test_similarity_calculation(self, vectorizer):
        """Test vector similarity calculation."""
        vector1 = [1.0, 0.0, 0.0]
        vector2 = [0.0, 1.0, 0.0]
        vector3 = [1.0, 0.0, 0.0]  # Same as vector1
        
        # Test orthogonal vectors (similarity = 0)
        similarity = vectorizer.calculate_similarity(vector1, vector2)
        assert abs(similarity - 0.0) < 1e-6
        
        # Test identical vectors (similarity = 1)
        similarity = vectorizer.calculate_similarity(vector1, vector3)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_similarity_calculation_edge_cases(self, vectorizer):
        """Test similarity calculation edge cases."""
        # Empty vectors
        assert vectorizer.calculate_similarity([], []) == 0.0
        
        # Different length vectors
        assert vectorizer.calculate_similarity([1, 2], [1, 2, 3]) == 0.0
        
        # Zero vectors
        assert vectorizer.calculate_similarity([0, 0, 0], [0, 0, 0]) == 0.0


class TestScamClassifier:
    """Test suite for scam classification functionality."""
    
    @pytest.fixture
    def classifier(self):
        """Create scam classifier instance for testing."""
        return ScamClassifier()
    
    @pytest.fixture
    def mock_model(self):
        """Mock ML model for testing."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.8]  # High scam probability
        mock_model.predict_proba.return_value = [[0.2, 0.8]]  # [not_scam, scam]
        return mock_model
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.model is None
        assert not classifier.is_loaded
        assert classifier.threshold == 0.5  # Default threshold
    
    def test_load_default_model(self, classifier, mock_model):
        """Test loading default model."""
        with patch.object(classifier, '_load_model_from_path', return_value=mock_model):
            result = classifier.load_default()
            
            assert result is not None
            assert isinstance(result, ScamClassifier)
            assert result.is_loaded
            assert result.model is not None
    
    def test_load_model_from_file(self, classifier, mock_model):
        """Test loading model from file."""
        with patch('joblib.load', return_value=mock_model):
            result = classifier.load_model("test_model.pkl")
            
            assert result is True
            assert classifier.is_loaded
            assert classifier.model == mock_model
    
    def test_load_model_file_not_found(self, classifier):
        """Test loading model when file doesn't exist."""
        with patch('joblib.load', side_effect=FileNotFoundError("File not found")):
            result = classifier.load_model("nonexistent_model.pkl")
            
            assert result is False
            assert not classifier.is_loaded
            assert classifier.model is None
    
    @pytest.mark.parametrize("text,expected_scam", [
        ("Congratulations! You've won $1000000! Click here to claim!", True),
        ("Your account will be suspended unless you verify immediately", True),
        ("URGENT: Send your bank details to claim lottery prize", True),
        ("Hello, how are you today?", False),
        ("Meeting scheduled for tomorrow at 2 PM", False),
        ("Your order has been shipped and will arrive tomorrow", False),
    ])
    def test_classify_text(self, classifier, mock_model, text, expected_scam):
        """Test text classification."""
        classifier.model = mock_model
        classifier.is_loaded = True
        
        # Mock different probabilities based on expected result
        if expected_scam:
            mock_model.predict_proba.return_value = [[0.2, 0.8]]  # High scam probability
        else:
            mock_model.predict_proba.return_value = [[0.8, 0.2]]  # Low scam probability
        
        result = classifier.classify_text(text)
        
        assert isinstance(result, ClassificationResult)
        assert result.success
        assert result.is_scam == expected_scam
        assert 0.0 <= result.confidence <= 1.0
        mock_model.predict_proba.assert_called_once()
    
    def test_classify_text_model_not_loaded(self, classifier):
        """Test classification with unloaded model."""
        result = classifier.classify_text("test text")
        
        assert isinstance(result, ClassificationResult)
        assert not result.success
        assert result.error_message != ""
    
    def test_classify_empty_text(self, classifier, mock_model):
        """Test classification of empty text."""
        classifier.model = mock_model
        classifier.is_loaded = True
        
        result = classifier.classify_text("")
        
        assert isinstance(result, ClassificationResult)
        assert not result.success
        assert "empty" in result.error_message.lower()
    
    def test_predict_proba(self, classifier, mock_model):
        """Test probability prediction."""
        classifier.model = mock_model
        classifier.is_loaded = True
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        
        proba = classifier.predict_proba("test text")
        
        assert proba == 0.7  # Scam probability
        mock_model.predict_proba.assert_called_once()
    
    def test_set_threshold(self, classifier):
        """Test threshold setting."""
        assert classifier.threshold == 0.5  # Default
        
        classifier.set_threshold(0.7)
        assert classifier.threshold == 0.7
        
        # Test invalid thresholds
        classifier.set_threshold(-0.1)
        assert classifier.threshold == 0.0  # Clamped to 0
        
        classifier.set_threshold(1.5)
        assert classifier.threshold == 1.0  # Clamped to 1
    
    def test_batch_classification(self, classifier, mock_model):
        """Test batch text classification."""
        classifier.model = mock_model
        classifier.is_loaded = True
        
        texts = ["Scam text", "Normal text", "Another scam"]
        mock_model.predict_proba.return_value = [
            [0.1, 0.9],  # Scam
            [0.8, 0.2],  # Not scam
            [0.2, 0.8]   # Scam
        ]
        
        results = classifier.classify_batch(texts)
        
        assert len(results) == len(texts)
        assert all(isinstance(r, ClassificationResult) for r in results)
        assert results[0].is_scam  # First is scam
        assert not results[1].is_scam  # Second is not scam
        assert results[2].is_scam  # Third is scam
    
    def test_model_performance_metrics(self, classifier, mock_model):
        """Test model performance evaluation."""
        classifier.model = mock_model
        classifier.is_loaded = True
        
        # Mock test data
        test_texts = ["scam1", "normal1", "scam2", "normal2"]
        true_labels = [1, 0, 1, 0]  # 1 = scam, 0 = normal
        
        mock_model.predict_proba.return_value = [
            [0.2, 0.8],  # Correct scam prediction
            [0.7, 0.3],  # Correct normal prediction
            [0.1, 0.9],  # Correct scam prediction
            [0.6, 0.4],  # Correct normal prediction
        ]
        
        metrics = classifier.evaluate_performance(test_texts, true_labels)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert all(0.0 <= metrics[key] <= 1.0 for key in metrics)


class TestTextProcessingIntegration:
    """Integration tests for text processing components."""
    
    @pytest.fixture
    def privacy_engine(self):
        """Create privacy engine for testing."""
        return PrivacyEngine()
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return {
            "scam_texts": [
                "URGENT: Your account will be suspended! Click here immediately!",
                "Congratulations! You've won $50,000! Send your details now!",
                "FINAL NOTICE: Pay now or face legal action!"
            ],
            "normal_texts": [
                "Hello, how are you doing today?",
                "The meeting is scheduled for 3 PM tomorrow.",
                "Thank you for your order. It will be delivered soon."
            ],
            "multilingual_texts": {
                "en": "Hello world",
                "hi": "नमस्ते दुनिया",
                "es": "Hola mundo",
                "fr": "Bonjour le monde"
            }
        }
    
    def test_end_to_end_text_processing(self, sample_texts, privacy_engine):
        """Test complete text processing pipeline."""
        for text in sample_texts["scam_texts"]:
            # Test language detection
            detected_lang = detect_language(text)
            assert detected_lang in list_supported()
            
            # Test privacy protection
            data_id = f"test_message_{hash(text)}"
            privacy_engine.register_data_context(
                data_id=data_id,
                classification=DataClassification.CONFIDENTIAL,
                source="user_input",
                processing_purpose="scam_analysis"
            )
            
            processed_data, privacy_metadata = privacy_engine.process_sensitive_data(
                data=text,
                data_id=data_id,
                operation="analyze"
            )
            
            assert processed_data is not None
            assert isinstance(privacy_metadata, dict)
            
            # Cleanup
            privacy_engine.secure_delete_data(data_id)
    
    def test_multilingual_processing(self, sample_texts):
        """Test multilingual text processing."""
        for lang_code, text in sample_texts["multilingual_texts"].items():
            # Test language detection
            with patch('src.utils.language.detect', return_value=lang_code):
                detected = detect_language(text)
                assert detected == lang_code
            
            # Test language name retrieval
            lang_name = get_language_name(lang_code)
            assert lang_name in LANGUAGE_NAMES.values()
            
            # Test Google language code mapping
            google_code = get_google_lang_code(lang_code)
            assert "-" in google_code
            assert google_code.startswith(lang_code)
    
    def test_text_vectorization_with_classification(self, sample_texts):
        """Test integration between vectorization and classification."""
        vectorizer = TextVectorizer()
        classifier = ScamClassifier()
        
        # Mock loaded models
        mock_transformer = Mock()
        mock_transformer.encode.return_value = [[0.1] * 384]
        vectorizer.model = mock_transformer
        vectorizer.is_loaded = True
        
        mock_classifier = Mock()
        mock_classifier.predict_proba.return_value = [[0.2, 0.8]]
        classifier.model = mock_classifier
        classifier.is_loaded = True
        
        for text in sample_texts["scam_texts"]:
            # Vectorize text
            vector_result = vectorizer.vectorize_text(text)
            assert vector_result.success
            assert len(vector_result.vector) == 384
            
            # Classify text
            class_result = classifier.classify_text(text)
            assert class_result.success
            # For scam texts, we expect high confidence
            assert class_result.confidence > 0.5
    
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 50])
    def test_batch_processing_performance(self, sample_texts, batch_size):
        """Test batch processing performance with different sizes."""
        texts = (sample_texts["scam_texts"] + sample_texts["normal_texts"]) * (batch_size // 6 + 1)
        texts = texts[:batch_size]
        
        vectorizer = TextVectorizer()
        mock_transformer = Mock()
        mock_transformer.encode.return_value = [[0.1] * 384] * len(texts)
        vectorizer.model = mock_transformer
        vectorizer.is_loaded = True
        
        import time
        start_time = time.time()
        results = vectorizer.vectorize_batch(texts)
        processing_time = time.time() - start_time
        
        assert len(results) == len(texts)
        assert all(r.success for r in results)
        # Performance assertion - should process reasonably fast
        assert processing_time < (batch_size * 0.1)  # Max 100ms per text
    
    def test_error_handling_and_recovery(self):
        """Test error handling in text processing components."""
        vectorizer = TextVectorizer()
        classifier = ScamClassifier()
        
        # Test unloaded model errors
        vector_result = vectorizer.vectorize_text("test")
        assert not vector_result.success
        assert "not loaded" in vector_result.error_message.lower()
        
        class_result = classifier.classify_text("test")
        assert not class_result.success
        assert class_result.error_message != ""
        
        # Test invalid input handling
        vector_result = vectorizer.vectorize_text("")
        assert not vector_result.success
        
        class_result = classifier.classify_text("")
        assert not class_result.success
    
    def test_memory_management(self, sample_texts):
        """Test memory management in text processing."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large amount of text
        large_text_batch = sample_texts["scam_texts"] * 100
        
        vectorizer = TextVectorizer()
        mock_transformer = Mock()
        mock_transformer.encode.return_value = [[0.1] * 384] * len(large_text_batch)
        vectorizer.model = mock_transformer
        vectorizer.is_loaded = True
        
        # Process batch
        results = vectorizer.vectorize_batch(large_text_batch)
        
        # Clear results and force garbage collection
        del results
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_concurrent_processing(self, sample_texts):
        """Test concurrent text processing."""
        import threading
        import queue
        
        def worker(text_queue, result_queue):
            vectorizer = TextVectorizer()
            mock_transformer = Mock()
            mock_transformer.encode.return_value = [[0.1] * 384]
            vectorizer.model = mock_transformer
            vectorizer.is_loaded = True
            
            while True:
                try:
                    text = text_queue.get(timeout=1)
                    result = vectorizer.vectorize_text(text)
                    result_queue.put(result)
                    text_queue.task_done()
                except queue.Empty:
                    break
        
        # Setup queues and threads
        text_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Add texts to queue
        all_texts = sample_texts["scam_texts"] + sample_texts["normal_texts"]
        for text in all_texts:
            text_queue.put(text)
        
        # Start worker threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=worker, args=(text_queue, result_queue))
            t.start()
            threads.append(t)
        
        # Wait for completion
        text_queue.join()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Wait for threads to finish
        for t in threads:
            t.join()
        
        assert len(results) == len(all_texts)
        assert all(r.success for r in results)


# Test configuration and fixtures
@pytest.fixture(scope="session")
def test_config():
    """Test configuration for the entire test session."""
    return {
        "test_data_dir": Path(__file__).parent / "data",
        "temp_dir": tempfile.mkdtemp(),
        "mock_models": True,
        "performance_testing": True
    }

@pytest.fixture
def cleanup_temp_files():
    """Cleanup temporary files after test."""
    temp_files = []
    
    def _create_temp_file(content="", suffix=".txt"):
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        temp_files.append(temp_file.name)
        return temp_file.name
    
    yield _create_temp_file
    
    # Cleanup
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except OSError:
            pass

# Performance and stress tests
class TestTextProcessingPerformance:
    """Performance and stress testing for text processing."""
    
    @pytest.mark.performance
    def test_large_text_processing(self):
        """Test processing of very large texts."""
        large_text = "This is a test sentence. " * 10000  # ~250KB text
        
        vectorizer = TextVectorizer()
        mock_transformer = Mock()
        mock_transformer.encode.return_value = [[0.1] * 384]
        vectorizer.model = mock_transformer
        vectorizer.is_loaded = True
        
        import time
        start_time = time.time()
        result = vectorizer.vectorize_text(large_text)
        processing_time = time.time() - start_time
        
        assert result.success
        assert processing_time < 10.0  # Should complete within 10 seconds
    
    @pytest.mark.stress
    def test_memory_stress(self):
        """Stress test memory usage with many concurrent operations."""
        vectorizer = TextVectorizer()
        mock_transformer = Mock()
        mock_transformer.encode.return_value = [[0.1] * 384]
        vectorizer.model = mock_transformer
        vectorizer.is_loaded = True
        
        # Process many texts simultaneously
        texts = [f"Test message number {i}" for i in range(1000)]
        
        results = []
        for text in texts:
            result = vectorizer.vectorize_text(text)
            results.append(result)
        
        assert len(results) == 1000
        assert all(r.success for r in results)
        
        # Memory cleanup
        del results
        import gc
        gc.collect()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])

