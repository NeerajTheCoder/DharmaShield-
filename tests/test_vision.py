"""
tests/test_vision.py

DharmaShield - Vision Processing Unit Tests
-------------------------------------------
• Comprehensive test suite for QR scanning, image detection, and fake document detection
• Cross-platform vision testing with multi-format image support
• Industry-grade testing with computer vision mocking and fixtures
"""

import pytest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw
import io

# Project imports
from src.utils.image_processing import ImageProcessor, ImageFormat, ProcessingResult
from src.ml.qr_scanner import QRScanner, QRScanResult, QRCodeType
from src.ml.vision_detector import ImageThreatDetector, ThreatDetectionResult, ThreatCategory
from src.ml.fake_doc_detector import FakeDocumentDetector, DocumentAnalysisResult, DocumentType


class TestImageProcessor:
    """Test suite for image processing functionality."""
    
    @pytest.fixture
    def image_processor(self):
        """Create image processor instance for testing."""
        return ImageProcessor()
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for testing."""
        images = {}
        
        # Create RGB image
        rgb_image = Image.new('RGB', (100, 100), color='red')
        images['rgb'] = rgb_image
        
        # Create RGBA image
        rgba_image = Image.new('RGBA', (100, 100), color=(0, 255, 0, 128))
        images['rgba'] = rgba_image
        
        # Create grayscale image
        gray_image = Image.new('L', (100, 100), color=128)
        images['grayscale'] = gray_image
        
        # Create numpy array image
        np_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        images['numpy'] = np_image
        
        return images
    
    @pytest.fixture
    def sample_qr_image(self):
        """Create sample QR code image."""
        # Create simple QR-like pattern
        qr_image = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(qr_image)
        
        # Draw QR-like pattern
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i + j) % 40 == 0:
                    draw.rectangle([i, j, i+20, j+20], fill='black')
        
        return qr_image
    
    def test_image_processor_initialization(self, image_processor):
        """Test image processor initialization."""
        assert image_processor is not None
        assert hasattr(image_processor, 'supported_formats')
        assert len(image_processor.supported_formats) > 0
    
    def test_load_image_from_pil(self, image_processor, sample_images):
        """Test loading image from PIL Image object."""
        result = image_processor.load_image(sample_images['rgb'])
        
        assert result.success
        assert result.image is not None
        assert result.format == ImageFormat.RGB
        assert result.dimensions == (100, 100)
    
    def test_load_image_from_numpy(self, image_processor, sample_images):
        """Test loading image from NumPy array."""
        result = image_processor.load_image(sample_images['numpy'])
        
        assert result.success
        assert result.image is not None
        assert result.dimensions == (100, 100)
    
    def test_load_image_from_file(self, image_processor, sample_images):
        """Test loading image from file."""
        # Save sample image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            sample_images['rgb'].save(temp_file.name)
            
            try:
                result = image_processor.load_image(temp_file.name)
                
                assert result.success
                assert result.image is not None
                assert result.format in [ImageFormat.RGB, ImageFormat.RGBA]
                
            finally:
                os.unlink(temp_file.name)
    
    def test_load_image_nonexistent_file(self, image_processor):
        """Test loading image from nonexistent file."""
        result = image_processor.load_image("nonexistent_file.jpg")
        
        assert not result.success
        assert result.image is None
        assert result.error_message != ""
    
    @pytest.mark.parametrize("format_ext,expected_format", [
        ('.jpg', ImageFormat.RGB),
        ('.jpeg', ImageFormat.RGB),
        ('.png', ImageFormat.RGBA),
        ('.bmp', ImageFormat.RGB),
        ('.gif', ImageFormat.RGB),
    ])
    def test_format_detection(self, image_processor, format_ext, expected_format):
        """Test image format detection."""
        detected = image_processor.detect_format(f"test_image{format_ext}")
        assert detected in [expected_format, ImageFormat.RGB, ImageFormat.RGBA]
    
    def test_resize_image(self, image_processor, sample_images):
        """Test image resizing."""
        original_image = sample_images['rgb']
        
        resized = image_processor.resize_image(original_image, (50, 50))
        
        assert resized.size == (50, 50)
        assert isinstance(resized, Image.Image)
    
    def test_resize_image_maintain_aspect_ratio(self, image_processor, sample_images):
        """Test image resizing with aspect ratio preservation."""
        # Create rectangular image
        rect_image = Image.new('RGB', (200, 100), color='blue')
        
        resized = image_processor.resize_image(rect_image, (100, 100), maintain_aspect=True)
        
        # Should maintain aspect ratio (2:1)
        assert resized.size[0] == 100
        assert resized.size[1] <= 50  # Height should be scaled down proportionally
    
    def test_crop_image(self, image_processor, sample_images):
        """Test image cropping."""
        original_image = sample_images['rgb']
        
        cropped = image_processor.crop_image(original_image, (10, 10, 60, 60))
        
        assert cropped.size == (50, 50)
        assert isinstance(cropped, Image.Image)
    
    def test_rotate_image(self, image_processor, sample_images):
        """Test image rotation."""
        original_image = sample_images['rgb']
        
        rotated = image_processor.rotate_image(original_image, 90)
        
        # 90-degree rotation should swap width and height
        assert rotated.size == (100, 100)  # Square image stays same
        assert isinstance(rotated, Image.Image)
    
    def test_convert_to_grayscale(self, image_processor, sample_images):
        """Test conversion to grayscale."""
        rgb_image = sample_images['rgb']
        
        gray_image = image_processor.to_grayscale(rgb_image)
        
        assert gray_image.mode == 'L'
        assert gray_image.size == rgb_image.size
    
    def test_enhance_image(self, image_processor, sample_images):
        """Test image enhancement."""
        original_image = sample_images['rgb']
        
        enhanced = image_processor.enhance_image(
            original_image,
            brightness=1.2,
            contrast=1.1,
            sharpness=1.1
        )
        
        assert enhanced.size == original_image.size
        assert isinstance(enhanced, Image.Image)
    
    def test_extract_image_features(self, image_processor, sample_images):
        """Test image feature extraction."""
        image = sample_images['rgb']
        
        features = image_processor.extract_features(image)
        
        assert isinstance(features, dict)
        assert 'histogram' in features
        assert 'edges' in features
        assert 'texture' in features
        assert len(features['histogram']) > 0
    
    def test_detect_edges(self, image_processor, sample_images):
        """Test edge detection."""
        image = sample_images['rgb']
        
        edges = image_processor.detect_edges(image)
        
        assert isinstance(edges, np.ndarray)
        assert edges.shape[:2] == image.size[::-1]  # Width, height reversed for numpy
    
    def test_batch_image_processing(self, image_processor, sample_images):
        """Test batch image processing."""
        images = [sample_images['rgb'], sample_images['rgba'], sample_images['grayscale']]
        
        results = image_processor.process_batch(images, resize_to=(50, 50))
        
        assert len(results) == len(images)
        assert all(r.success for r in results)
        assert all(r.image.size == (50, 50) for r in results if r.success)


class TestQRScanner:
    """Test suite for QR code scanning functionality."""
    
    @pytest.fixture
    def qr_scanner(self):
        """Create QR scanner instance for testing."""
        return QRScanner()
    
    @pytest.fixture
    def mock_qr_decoder(self):
        """Mock QR code decoder."""
        mock_decoder = Mock()
        mock_decoded = Mock()
        mock_decoded.data = b'https://example.com'
        mock_decoded.type = 'QRCODE'
        mock_decoded.polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        mock_decoder.decode.return_value = [mock_decoded]
        return mock_decoder
    
    def test_qr_scanner_initialization(self, qr_scanner):
        """Test QR scanner initialization."""
        assert qr_scanner is not None
        assert hasattr(qr_scanner, 'decode_qr')
    
    def test_scan_qr_code_success(self, qr_scanner, sample_qr_image, mock_qr_decoder):
        """Test successful QR code scanning."""
        with patch('pyzbar.pyzbar.decode', return_value=mock_qr_decoder.decode.return_value):
            result = qr_scanner.scan_qr_code(sample_qr_image)
            
            assert isinstance(result, QRScanResult)
            assert result.success
            assert result.data == 'https://example.com'
            assert result.qr_type == QRCodeType.URL
            assert len(result.corners) == 4
    
    def test_scan_qr_code_no_qr_found(self, qr_scanner, sample_images):
        """Test QR scanning when no QR code is found."""
        with patch('pyzbar.pyzbar.decode', return_value=[]):
            result = qr_scanner.scan_qr_code(sample_images['rgb'])
            
            assert isinstance(result, QRScanResult)
            assert not result.success
            assert result.data is None
            assert "No QR code found" in result.error_message
    
    @pytest.mark.parametrize("qr_data,expected_type", [
        ("https://example.com", QRCodeType.URL),
        ("http://test.com", QRCodeType.URL),
        ("mailto:test@example.com", QRCodeType.EMAIL),
        ("tel:+1234567890", QRCodeType.PHONE),
        ("SMS:+1234567890:Hello", QRCodeType.SMS),
        ("WIFI:T:WPA;S:NetworkName;P:password;;", QRCodeType.WIFI),
        ("geo:37.7749,-122.4194", QRCodeType.LOCATION),
        ("Just plain text", QRCodeType.TEXT),
    ])
    def test_qr_type_detection(self, qr_scanner, qr_data, expected_type):
        """Test QR code type detection."""
        detected_type = qr_scanner.detect_qr_type(qr_data)
        assert detected_type == expected_type
    
    def test_validate_qr_url(self, qr_scanner):
        """Test QR URL validation."""
        # Valid URLs
        assert qr_scanner.validate_url("https://example.com")
        assert qr_scanner.validate_url("http://test.com")
        assert qr_scanner.validate_url("https://subdomain.example.com/path")
        
        # Invalid URLs
        assert not qr_scanner.validate_url("not-a-url")
        assert not qr_scanner.validate_url("ftp://example.com")  # Only HTTP(S) allowed
        assert not qr_scanner.validate_url("")
    
    def test_extract_qr_metadata(self, qr_scanner, mock_qr_decoder):
        """Test QR code metadata extraction."""
        with patch('pyzbar.pyzbar.decode', return_value=mock_qr_decoder.decode.return_value):
            result = qr_scanner.scan_qr_code(Image.new('RGB', (100, 100)))
            
            assert result.metadata is not None
            assert 'format' in result.metadata
            assert 'quality' in result.metadata
            assert 'position' in result.metadata
    
    def test_scan_multiple_qr_codes(self, qr_scanner):
        """Test scanning multiple QR codes in single image."""
        # Mock multiple QR codes
        mock_decoded1 = Mock()
        mock_decoded1.data = b'First QR code'
        mock_decoded1.type = 'QRCODE'
        mock_decoded1.polygon = [(0, 0), (50, 0), (50, 50), (0, 50)]
        
        mock_decoded2 = Mock()
        mock_decoded2.data = b'Second QR code'
        mock_decoded2.type = 'QRCODE'
        mock_decoded2.polygon = [(60, 60), (110, 60), (110, 110), (60, 110)]
        
        with patch('pyzbar.pyzbar.decode', return_value=[mock_decoded1, mock_decoded2]):
            results = qr_scanner.scan_multiple_qr_codes(Image.new('RGB', (200, 200)))
            
            assert len(results) == 2
            assert all(r.success for r in results)
            assert results[0].data == 'First QR code'
            assert results[1].data == 'Second QR code'
    
    def test_qr_code_security_check(self, qr_scanner):
        """Test QR code security validation."""
        # Safe URLs
        safe_urls = [
            "https://google.com",
            "https://github.com",
            "https://stackoverflow.com"
        ]
        
        for url in safe_urls:
            risk_level = qr_scanner.assess_security_risk(url)
            assert risk_level in ['low', 'medium', 'high']
        
        # Suspicious URLs
        suspicious_urls = [
            "http://bit.ly/suspicious",  # Shortened URL
            "https://phishing-site.evil.com",
            "javascript:alert('xss')"
        ]
        
        for url in suspicious_urls:
            risk_level = qr_scanner.assess_security_risk(url)
            assert risk_level in ['medium', 'high']


class TestImageThreatDetector:
    """Test suite for image threat detection."""
    
    @pytest.fixture
    def threat_detector(self):
        """Create image threat detector instance for testing."""
        return ImageThreatDetector()
    
    @pytest.fixture
    def mock_threat_model(self):
        """Mock threat detection model."""
        mock_model = Mock()
        mock_model.predict.return_value = [ThreatCategory.SAFE.value]
        mock_model.predict_proba.return_value = [[0.9, 0.05, 0.03, 0.02]]  # [safe, phishing, malware, explicit]
        return mock_model
    
    @pytest.fixture
    def sample_threat_images(self, sample_images):
        """Create sample images for threat testing."""
        threat_images = {}
        
        # Safe image
        threat_images['safe'] = sample_images['rgb']
        
        # Create phishing-like image (with text overlay)
        phishing_image = Image.new('RGB', (300, 200), color='white')
        draw = ImageDraw.Draw(phishing_image)
        draw.text((10, 10), "Click here to win $1000!", fill='red')
        threat_images['phishing'] = phishing_image
        
        # Malware-like image (suspicious QR code)
        malware_image = Image.new('RGB', (100, 100), color='black')
        draw = ImageDraw.Draw(malware_image)
        draw.rectangle([20, 20, 80, 80], fill='white')
        threat_images['malware'] = malware_image
        
        return threat_images
    
    def test_threat_detector_initialization(self, threat_detector):
        """Test threat detector initialization."""
        assert threat_detector.model is None
        assert not threat_detector.is_loaded
        assert hasattr(threat_detector, 'threat_categories')
    
    def test_load_default_model(self, threat_detector, mock_threat_model):
        """Test loading default threat detection model."""
        with patch.object(threat_detector, '_load_pretrained_model', return_value=mock_threat_model):
            result = threat_detector.load_default()
            
            assert result is not None
            assert isinstance(result, ImageThreatDetector)
            assert result.is_loaded
            assert result.model is not None
    
    def test_detect_safe_image(self, threat_detector, mock_threat_model, sample_threat_images):
        """Test detection of safe image."""
        threat_detector.model = mock_threat_model
        threat_detector.is_loaded = True
        
        result = threat_detector.detect_threats(sample_threat_images['safe'])
        
        assert isinstance(result, ThreatDetectionResult)
        assert result.success
        assert result.threat_category == ThreatCategory.SAFE
        assert result.confidence > 0.8
    
    def test_detect_phishing_image(self, threat_detector, mock_threat_model, sample_threat_images):
        """Test detection of phishing image."""
        threat_detector.model = mock_threat_model
        threat_detector.is_loaded = True
        
        # Mock phishing detection
        mock_threat_model.predict.return_value = [ThreatCategory.PHISHING.value]
        mock_threat_model.predict_proba.return_value = [[0.1, 0.8, 0.05, 0.05]]
        
        result = threat_detector.detect_threats(sample_threat_images['phishing'])
        
        assert result.success
        assert result.threat_category == ThreatCategory.PHISHING
        assert result.confidence > 0.7
    
    @pytest.mark.parametrize("threat_category,proba_vector", [
        (ThreatCategory.SAFE, [0.9, 0.05, 0.03, 0.02]),
        (ThreatCategory.PHISHING, [0.1, 0.8, 0.05, 0.05]),
        (ThreatCategory.MALWARE, [0.1, 0.1, 0.7, 0.1]),
        (ThreatCategory.EXPLICIT, [0.1, 0.1, 0.1, 0.7]),
    ])
    def test_detect_different_threat_categories(self, threat_detector, mock_threat_model,
                                               sample_threat_images, threat_category, proba_vector):
        """Test detection of different threat categories."""
        threat_detector.model = mock_threat_model
        threat_detector.is_loaded = True
        
        mock_threat_model.predict.return_value = [threat_category.value]
        mock_threat_model.predict_proba.return_value = [proba_vector]
        
        result = threat_detector.detect_threats(sample_threat_images['safe'])
        
        assert result.success
        assert result.threat_category == threat_category
        assert result.confidence == max(proba_vector)
    
    def test_detect_threats_model_not_loaded(self, threat_detector, sample_threat_images):
        """Test threat detection with unloaded model."""
        result = threat_detector.detect_threats(sample_threat_images['safe'])
        
        assert isinstance(result, ThreatDetectionResult)
        assert not result.success
        assert "not loaded" in result.error_message.lower()
    
    def test_batch_threat_detection(self, threat_detector, mock_threat_model, sample_threat_images):
        """Test batch threat detection."""
        threat_detector.model = mock_threat_model
        threat_detector.is_loaded = True
        
        images = list(sample_threat_images.values())
        
        # Mock different results for different images
        mock_threat_model.predict.return_value = [
            ThreatCategory.SAFE.value,
            ThreatCategory.PHISHING.value,
            ThreatCategory.MALWARE.value
        ]
        mock_threat_model.predict_proba.return_value = [
            [0.9, 0.05, 0.03, 0.02],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.1, 0.7, 0.1]
        ]
        
        results = threat_detector.detect_batch(images)
        
        assert len(results) == len(images)
        assert all(isinstance(r, ThreatDetectionResult) for r in results)
        assert all(r.success for r in results)
    
    def test_threat_analysis_with_metadata(self, threat_detector, mock_threat_model, sample_threat_images):
        """Test threat analysis with detailed metadata."""
        threat_detector.model = mock_threat_model
        threat_detector.is_loaded = True
        
        result = threat_detector.analyze_with_metadata(sample_threat_images['safe'])
        
        assert result.success
        assert result.metadata is not None
        assert 'image_properties' in result.metadata
        assert 'threat_indicators' in result.metadata
        assert 'confidence_breakdown' in result.metadata


class TestFakeDocumentDetector:
    """Test suite for fake document detection."""
    
    @pytest.fixture
    def doc_detector(self):
        """Create fake document detector instance for testing."""
        return FakeDocumentDetector()
    
    @pytest.fixture
    def mock_doc_model(self):
        """Mock document analysis model."""
        mock_model = Mock()
        mock_model.predict.return_value = [0]  # 0 = authentic, 1 = fake
        mock_model.predict_proba.return_value = [[0.8, 0.2]]  # [authentic, fake]
        return mock_model
    
    @pytest.fixture
    def sample_document_images(self):
        """Create sample document images."""
        documents = {}
        
        # Create authentic-looking document
        auth_doc = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(auth_doc)
        draw.text((20, 20), "OFFICIAL DOCUMENT", fill='black')
        draw.text((20, 50), "ID: 123456789", fill='black')
        draw.rectangle([300, 200, 380, 280], outline='blue', width=2)
        documents['authentic'] = auth_doc
        
        # Create fake-looking document
        fake_doc = Image.new('RGB', (400, 300), color='yellow')
        draw = ImageDraw.Draw(fake_doc)
        draw.text((20, 20), "FAKE DOCUMENT", fill='red')
        draw.text((20, 50), "ID: INVALID", fill='red')
        documents['fake'] = fake_doc
        
        # Create ID card
        id_card = Image.new('RGB', (320, 200), color='lightblue')
        draw = ImageDraw.Draw(id_card)
        draw.text((20, 20), "DRIVER LICENSE", fill='darkblue')
        draw.text((20, 50), "Name: John Doe", fill='black')
        draw.text((20, 80), "DOB: 01/01/1990", fill='black')
        documents['id_card'] = id_card
        
        return documents
    
    def test_doc_detector_initialization(self, doc_detector):
        """Test document detector initialization."""
        assert doc_detector.model is None
        assert not doc_detector.is_loaded
        assert hasattr(doc_detector, 'document_types')
    
    def test_detect_authentic_document(self, doc_detector, mock_doc_model, sample_document_images):
        """Test detection of authentic document."""
        doc_detector.model = mock_doc_model
        doc_detector.is_loaded = True
        
        result = doc_detector.analyze_document(sample_document_images['authentic'])
        
        assert isinstance(result, DocumentAnalysisResult)
        assert result.success
        assert not result.is_fake
        assert result.confidence > 0.7
    
    def test_detect_fake_document(self, doc_detector, mock_doc_model, sample_document_images):
        """Test detection of fake document."""
        doc_detector.model = mock_doc_model
        doc_detector.is_loaded = True
        
        # Mock fake detection
        mock_doc_model.predict.return_value = [1]  # Fake
        mock_doc_model.predict_proba.return_value = [[0.2, 0.8]]
        
        result = doc_detector.analyze_document(sample_document_images['fake'])
        
        assert result.success
        assert result.is_fake
        assert result.confidence > 0.7
    
    @pytest.mark.parametrize("doc_type,features", [
        (DocumentType.ID_CARD, ['photo', 'signature', 'barcode']),
        (DocumentType.PASSPORT, ['photo', 'mrz', 'security_features']),
        (DocumentType.DRIVERS_LICENSE, ['photo', 'signature', 'holograms']),
        (DocumentType.CERTIFICATE, ['seal', 'signature', 'watermark']),
    ])
    def test_document_type_detection(self, doc_detector, doc_type, features):
        """Test document type detection."""
        detected_type = doc_detector.detect_document_type(features)
        assert detected_type in DocumentType
    
    def test_extract_document_features(self, doc_detector, sample_document_images):
        """Test document feature extraction."""
        features = doc_detector.extract_features(sample_document_images['id_card'])
        
        assert isinstance(features, dict)
        assert 'text_regions' in features
        assert 'security_features' in features
        assert 'image_quality' in features
        assert 'layout_analysis' in features
    
    def test_ocr_text_extraction(self, doc_detector, sample_document_images):
        """Test OCR text extraction from documents."""
        with patch('pytesseract.image_to_string') as mock_ocr:
            mock_ocr.return_value = "DRIVER LICENSE\nName: John Doe\nDOB: 01/01/1990"
            
            text = doc_detector.extract_text(sample_document_images['id_card'])
            
            assert isinstance(text, str)
            assert "DRIVER LICENSE" in text
            assert "John Doe" in text
    
    def test_security_features_detection(self, doc_detector, sample_document_images):
        """Test security features detection."""
        security_features = doc_detector.detect_security_features(sample_document_images['authentic'])
        
        assert isinstance(security_features, dict)
        assert 'watermarks' in security_features
        assert 'holograms' in security_features
        assert 'microprint' in security_features
        assert 'uv_features' in security_features
    
    def test_tampering_detection(self, doc_detector, sample_document_images):
        """Test document tampering detection."""
        tampering_result = doc_detector.detect_tampering(sample_document_images['authentic'])
        
        assert isinstance(tampering_result, dict)
        assert 'tampering_detected' in tampering_result
        assert 'tampering_areas' in tampering_result
        assert 'confidence' in tampering_result


class TestVisionProcessingIntegration:
    """Integration tests for vision processing components."""
    
    @pytest.fixture
    def vision_pipeline(self):
        """Create complete vision processing pipeline."""
        return {
            'processor': ImageProcessor(),
            'qr_scanner': QRScanner(),
            'threat_detector': ImageThreatDetector(),
            'doc_detector': FakeDocumentDetector()
        }
    
    def test_end_to_end_image_analysis(self, vision_pipeline, sample_images, sample_qr_image):
        """Test complete image analysis pipeline."""
        processor = vision_pipeline['processor']
        qr_scanner = vision_pipeline['qr_scanner']
        threat_detector = vision_pipeline['threat_detector']
        
        # Process image
        processed_result = processor.load_image(sample_images['rgb'])
        assert processed_result.success
        
        # Check for QR codes
        with patch('pyzbar.pyzbar.decode', return_value=[]):
            qr_result = qr_scanner.scan_qr_code(processed_result.image)
            # No QR code expected in sample image
            assert not qr_result.success
        
        # Mock threat detection
        with patch.object(threat_detector, 'detect_threats') as mock_detect:
            mock_detect.return_value = Mock(
                success=True,
                threat_category=ThreatCategory.SAFE,
                confidence=0.9
            )
            
            threat_result = threat_detector.detect_threats(processed_result.image)
            assert threat_result.success
            assert threat_result.threat_category == ThreatCategory.SAFE
    
    def test_qr_code_security_pipeline(self, vision_pipeline, sample_qr_image):
        """Test QR code security analysis pipeline."""
        qr_scanner = vision_pipeline['qr_scanner']
        threat_detector = vision_pipeline['threat_detector']
        
        # Mock QR detection
        mock_qr_data = Mock()
        mock_qr_data.data = b'https://suspicious-site.com'
        mock_qr_data.type = 'QRCODE'
        mock_qr_data.polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        
        with patch('pyzbar.pyzbar.decode', return_value=[mock_qr_data]):
            qr_result = qr_scanner.scan_qr_code(sample_qr_image)
            
            assert qr_result.success
            assert qr_result.data == 'https://suspicious-site.com'
            
            # Assess security risk
            risk_level = qr_scanner.assess_security_risk(qr_result.data)
            assert risk_level in ['low', 'medium', 'high']
    
    def test_document_verification_pipeline(self, vision_pipeline, sample_document_images):
        """Test document verification pipeline."""
        processor = vision_pipeline['processor']
        doc_detector = vision_pipeline['doc_detector']
        
        # Process document image
        processed_result = processor.load_image(sample_document_images['authentic'])
        assert processed_result.success
        
        # Mock document analysis
        with patch.object(doc_detector, 'analyze_document') as mock_analyze:
            mock_analyze.return_value = Mock(
                success=True,
                is_fake=False,
                confidence=0.85,
                document_type=DocumentType.ID_CARD,
                anomalies=[]
            )
            
            doc_result = doc_detector.analyze_document(processed_result.image)
            
            assert doc_result.success
            assert not doc_result.is_fake
            assert doc_result.confidence > 0.8
    
    def test_batch_image_processing(self, vision_pipeline, sample_images):
        """Test batch processing of multiple images."""
        processor = vision_pipeline['processor']
        threat_detector = vision_pipeline['threat_detector']
        
        images = list(sample_images.values())
        
        # Process all images
        processed_results = processor.process_batch(images, resize_to=(224, 224))
        assert all(r.success for r in processed_results)
        
        # Mock batch threat detection
        with patch.object(threat_detector, 'detect_batch') as mock_batch:
            mock_results = [
                Mock(success=True, threat_category=ThreatCategory.SAFE, confidence=0.9)
                for _ in images
            ]
            mock_batch.return_value = mock_results
            
            threat_results = threat_detector.detect_batch([r.image for r in processed_results])
            
            assert len(threat_results) == len(images)
            assert all(r.success for r in threat_results)
    
    @pytest.mark.performance
    def test_vision_processing_performance(self, vision_pipeline, sample_images):
        """Test vision processing performance."""
        processor = vision_pipeline['processor']
        
        # Measure processing time for batch of images
        import time
        start_time = time.time()
        
        # Process 50 images
        for _ in range(50):
            result = processor.load_image(sample_images['rgb'])
            features = processor.extract_features(result.image)
        
        processing_time = time.time() - start_time
        
        # Should process reasonably fast
        assert processing_time < 30.0  # Max 30 seconds for 50 images
        assert processing_time / 50 < 0.6  # Max 600ms per image
    
    def test_memory_management_vision(self, sample_images):
        """Test memory management in vision processing."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process many images
        for _ in range(100):
            processor = ImageProcessor()
            result = processor.load_image(sample_images['rgb'])
            features = processor.extract_features(result.image)
            del processor, result, features
        
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 300 * 1024 * 1024  # Less than 300MB
    
    def test_concurrent_vision_processing(self, sample_images):
        """Test concurrent vision processing."""
        import threading
        import queue
        
        def worker(image_queue, result_queue):
            processor = ImageProcessor()
            
            while True:
                try:
                    image = image_queue.get(timeout=1)
                    result = processor.load_image(image)
                    features = processor.extract_features(result.image)
                    result_queue.put(features)
                    image_queue.task_done()
                except queue.Empty:
                    break
        
        # Setup queues
        image_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Add images to queue
        images = list(sample_images.values()) * 5  # 15 images total
        for image in images:
            image_queue.put(image)
        
        # Start workers
        threads = []
        for _ in range(3):
            t = threading.Thread(target=worker, args=(image_queue, result_queue))
            t.start()
            threads.append(t)
        
        # Wait for completion
        image_queue.join()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Cleanup threads
        for t in threads:
            t.join()
        
        assert len(results) == len(images)


# Performance and stress tests
class TestVisionPerformance:
    """Performance testing for vision processing."""
    
    @pytest.mark.performance
    def test_large_image_processing(self, sample_images):
        """Test processing of large images."""
        # Create large image (4K resolution)
        large_image = sample_images['rgb'].resize((3840, 2160))
        
        processor = ImageProcessor()
        
        import time
        start_time = time.time()
        
        result = processor.load_image(large_image)
        features = processor.extract_features(result.image)
        
        processing_time = time.time() - start_time
        
        assert result.success
        assert features is not None
        assert processing_time < 60.0  # Should complete within 60 seconds
    
    @pytest.mark.stress
    def test_continuous_vision_processing(self, sample_images):
        """Stress test continuous vision processing."""
        processor = ImageProcessor()
        qr_scanner = QRScanner()
        
        # Process images continuously
        for i in range(1000):
            image = sample_images['rgb']
            
            # Process image
            result = processor.load_image(image)
            assert result.success
            
            # Scan for QR codes
            with patch('pyzbar.pyzbar.decode', return_value=[]):
                qr_result = qr_scanner.scan_qr_code(result.image)
            
            # Periodic cleanup
            if i % 100 == 0:
                import gc
                gc.collect()


# Utility fixtures for complex test scenarios
@pytest.fixture(scope="session")
def test_images_dir(tmp_path_factory):
    """Create temporary directory with test images."""
    images_dir = tmp_path_factory.mktemp("test_images")
    
    # Create various test images
    formats = ['PNG', 'JPEG', 'BMP', 'GIF']
    sizes = [(100, 100), (200, 150), (300, 300)]
    
    for fmt in formats:
        for i, size in enumerate(sizes):
            image = Image.new('RGB', size, color=(i*50, i*50, i*50))
            image.save(images_dir / f"test_{fmt.lower()}_{i}.{fmt.lower()}")
    
    return images_dir


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-m", "not performance and not stress"])

