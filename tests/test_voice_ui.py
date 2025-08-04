"""
tests/test_voice_ui.py

DharmaShield - Voice UI End-to-End Tests
----------------------------------------
• Comprehensive end-to-end tests for voice interface and multi-language support
• Voice command recognition, language switching, and scam analysis workflow testing
• Cross-platform voice UI testing with mocking and async support
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from pathlib import Path
from typing import Dict, Any, List, Optional

# Project imports
from src.ui.voice_interface import VoiceInterface, VOICE_TAGLINE
from src.core.orchestrator import DharmaShieldCore, AnalysisResult, ThreatLevel
from src.utils.asr_engine import ASREngine
from src.utils.tts_engine import TTSEngine, speak
from src.utils.language import detect_language, get_language_name, list_supported


class TestVoiceInterface:
    """Test suite for voice interface functionality."""
    
    @pytest.fixture
    def mock_core(self):
        """Create mock DharmaShieldCore for testing."""
        mock_core = AsyncMock(spec=DharmaShieldCore)
        
        # Mock analysis result
        mock_result = AnalysisResult(
            threat_level=ThreatLevel.LOW,
            recommendations=["Stay alert", "Verify sender"],
            spiritual_guidance="Trust your intuition"
        )
        mock_core.run_multimodal_analysis.return_value = mock_result
        
        return mock_core
    
    @pytest.fixture
    def mock_asr_engine(self):
        """Create mock ASR engine for testing."""
        mock_asr = Mock(spec=ASREngine)
        mock_asr.listen_and_transcribe.return_value = "hello world"
        mock_asr.set_language = Mock()
        return mock_asr
    
    @pytest.fixture
    def voice_interface(self, mock_core):
        """Create voice interface instance for testing."""
        with patch('src.ui.voice_interface.ASREngine') as mock_asr_class:
            mock_asr = Mock()
            mock_asr.listen_and_transcribe.return_value = "test command"
            mock_asr_class.return_value = mock_asr
            
            interface = VoiceInterface(mock_core, language='en')
            interface.asr = mock_asr
            return interface
    
    def test_voice_interface_initialization(self, mock_core):
        """Test voice interface initialization."""
        with patch('src.ui.voice_interface.ASREngine') as mock_asr_class:
            interface = VoiceInterface(mock_core, language='hi')
            
            assert interface.core == mock_core
            assert interface.language == 'hi'
            assert interface.running is True
            mock_asr_class.assert_called_once_with(language='hi')
    
    @pytest.mark.asyncio
    async def test_voice_interface_startup(self, voice_interface):
        """Test voice interface startup sequence."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            with patch('src.ui.voice_interface.list_supported', return_value=['en', 'hi', 'es']):
                with patch('src.ui.voice_interface.get_language_name') as mock_get_name:
                    mock_get_name.side_effect = lambda x: {'en': 'English', 'hi': 'Hindi', 'es': 'Spanish'}[x]
                    
                    # Mock the ASR to return exit command immediately
                    voice_interface.asr.listen_and_transcribe.return_value = "exit"
                    
                    await voice_interface.run()
                    
                    # Should speak the tagline
                    mock_speak.assert_any_call(VOICE_TAGLINE['en'], lang='en')
    
    @pytest.mark.asyncio
    async def test_voice_command_exit(self, voice_interface):
        """Test exit voice commands."""
        exit_commands = ["exit", "quit", "close", "stop"]
        
        for command in exit_commands:
            voice_interface.running = True
            voice_interface.asr.listen_and_transcribe.return_value = command
            
            with patch('src.ui.voice_interface.speak') as mock_speak:
                await voice_interface.run()
                
                assert voice_interface.running is False
                mock_speak.assert_any_call("Goodbye. Stay safe from scams!", lang='en')
    
    @pytest.mark.asyncio
    async def test_language_switching_command(self, voice_interface):
        """Test language switching functionality."""
        with patch('src.ui.voice_interface.list_supported', return_value=['en', 'hi', 'es']):
            with patch('src.ui.voice_interface.speak') as mock_speak:
                with patch('src.ui.voice_interface.get_language_name', return_value='Hindi'):
                    
                    # Simulate language switch command followed by exit
                    voice_interface.asr.listen_and_transcribe.side_effect = [
                        "switch language to hi",
                        "exit"
                    ]
                    
                    await voice_interface.run()
                    
                    # Should switch language and speak confirmation
                    assert voice_interface.language == 'hi'
                    voice_interface.asr.set_language.assert_called_with('hi')
                    mock_speak.assert_any_call("Language switched to Hindi.", lang='hi')
    
    @pytest.mark.asyncio
    async def test_invalid_language_switching(self, voice_interface):
        """Test invalid language switching."""
        with patch('src.ui.voice_interface.list_supported', return_value=['en', 'hi', 'es']):
            with patch('src.ui.voice_interface.speak') as mock_speak:
                
                # Simulate invalid language switch command followed by exit
                voice_interface.asr.listen_and_transcribe.side_effect = [
                    "switch language to invalid",
                    "exit"
                ]
                
                await voice_interface.run()
                
                # Should not switch language and speak error message
                assert voice_interface.language == 'en'  # Should remain English
                mock_speak.assert_any_call("Sorry, language not supported.", lang='en')
    
    @pytest.mark.asyncio
    async def test_empty_voice_input(self, voice_interface):
        """Test handling of empty voice input."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            
            # Simulate empty input followed by exit
            voice_interface.asr.listen_and_transcribe.side_effect = [
                "",  # Empty input
                "exit"
            ]
            
            await voice_interface.run()
            
            # Should speak error message for empty input
            mock_speak.assert_any_call("Sorry, I did not understand. Please repeat.", lang='en')
    
    @pytest.mark.asyncio
    async def test_scan_message_command(self, voice_interface, mock_core):
        """Test scan message voice command."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            
            # Mock the text scanning flow
            voice_interface.asr.listen_and_transcribe.side_effect = [
                "scan message",  # Initial command
                "This is a test message to analyze",  # Message to scan
                "exit"  # Exit
            ]
            
            # Mock language detection
            with patch('src.ui.voice_interface.detect_language', return_value='en'):
                await voice_interface.run()
            
            # Should call multimodal analysis
            mock_core.run_multimodal_analysis.assert_called_once_with(
                text="This is a test message to analyze"
            )
            
            # Should speak analysis results
            mock_speak.assert_any_call("Analyzing your message, please wait.", lang='en')
    
    @pytest.mark.asyncio
    async def test_is_this_scam_command(self, voice_interface, mock_core):
        """Test 'is this a scam' voice command."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            
            # Mock the scam analysis flow
            voice_interface.asr.listen_and_transcribe.side_effect = [
                "is this a scam",  # Initial command
                "You have won $1000000! Click here to claim now!",  # Suspicious message
                "exit"  # Exit
            ]
            
            # Mock high threat level result
            mock_result = AnalysisResult(
                threat_level=ThreatLevel.CRITICAL,
                recommendations=["Do not click", "Block sender"],
                spiritual_guidance="Trust your instincts"
            )
            mock_core.run_multimodal_analysis.return_value = mock_result
            
            with patch('src.ui.voice_interface.detect_language', return_value='en'):
                await voice_interface.run()
            
            # Should call multimodal analysis
            mock_core.run_multimodal_analysis.assert_called_once()
            
            # Should speak critical threat warning
            mock_speak.assert_any_call(
                "Threat level critical. Please do not proceed! This is a scam.", 
                lang='en'
            )
    
    @pytest.mark.asyncio
    async def test_fraud_detection_command(self, voice_interface, mock_core):
        """Test fraud detection voice command."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            
            voice_interface.asr.listen_and_transcribe.side_effect = [
                "fraud detection",  # Command that contains "fraud"
                "Urgent: Your account has been compromised",  # Message to analyze
                "exit"
            ]
            
            with patch('src.ui.voice_interface.detect_language', return_value='en'):
                await voice_interface.run()
            
            # Should trigger text scan handler
            mock_core.run_multimodal_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_tagline_command(self, voice_interface):
        """Test tagline voice command."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            
            voice_interface.asr.listen_and_transcribe.side_effect = [
                "tagline",  # Request tagline
                "exit"
            ]
            
            await voice_interface.run()
            
            # Should speak the tagline
            mock_speak.assert_any_call(VOICE_TAGLINE['en'], lang='en')
    
    @pytest.mark.asyncio
    async def test_about_you_command(self, voice_interface):
        """Test 'about you' voice command."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            
            voice_interface.asr.listen_and_transcribe.side_effect = [
                "about you",  # Request information
                "exit"
            ]
            
            await voice_interface.run()
            
            # Should speak the tagline
            mock_speak.assert_any_call(VOICE_TAGLINE['en'], lang='en')
    
    @pytest.mark.asyncio
    async def test_unknown_command(self, voice_interface):
        """Test unknown voice command."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            
            voice_interface.asr.listen_and_transcribe.side_effect = [
                "unknown command",  # Unknown command
                "exit"
            ]
            
            await voice_interface.run()
            
            # Should speak help message
            mock_speak.assert_any_call(
                "Please say: 'Scan this message' or 'Is this a scam?'.", 
                lang='en'
            )
    
    @pytest.mark.asyncio
    async def test_language_detection_during_scan(self, voice_interface, mock_core):
        """Test automatic language detection during message scanning."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            with patch('src.ui.voice_interface.detect_language', return_value='hi') as mock_detect:
                with patch('src.ui.voice_interface.get_language_name', return_value='Hindi'):
                    
                    voice_interface.asr.listen_and_transcribe.side_effect = [
                        "scan message",
                        "यह एक परीक्षण संदेश है",  # Hindi message
                        "exit"
                    ]
                    
                    await voice_interface.run()
                    
                    # Should detect language and switch
                    mock_detect.assert_called_with("यह एक परीक्षण संदेश है")
                    assert voice_interface.language == 'hi'
                    voice_interface.asr.set_language.assert_called_with('hi')
                    mock_speak.assert_any_call("Detected language: Hindi", lang='hi')
    
    @pytest.mark.asyncio
    async def test_no_message_received(self, voice_interface):
        """Test handling when no message is received during scan."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            
            voice_interface.asr.listen_and_transcribe.side_effect = [
                "scan message",
                "",  # Empty message
                "exit"
            ]
            
            await voice_interface.run()
            
            # Should speak error message
            mock_speak.assert_any_call(
                "I did not receive any message. Please try again.", 
                lang='en'
            )
    
    @pytest.mark.parametrize("threat_level,expected_message", [
        (ThreatLevel.SAFE, "No scam detected. Your message seems safe."),
        (ThreatLevel.LOW, "Threat level low. Probably safe, but stay alert."),
        (ThreatLevel.MEDIUM, "Threat level medium. Caution advised."),
        (ThreatLevel.HIGH, "Threat level high. Please do not proceed."),
        (ThreatLevel.CRITICAL, "Threat level critical. Please do not proceed! This is a scam."),
    ])
    @pytest.mark.asyncio
    async def test_different_threat_levels(self, voice_interface, mock_core, threat_level, expected_message):
        """Test voice responses for different threat levels."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            
            # Mock specific threat level result
            mock_result = AnalysisResult(
                threat_level=threat_level,
                recommendations=["Test recommendation"],
                spiritual_guidance="Test guidance"
            )
            mock_core.run_multimodal_analysis.return_value = mock_result
            
            voice_interface.asr.listen_and_transcribe.side_effect = [
                "scan message",
                "test message",
                "exit"
            ]
            
            with patch('src.ui.voice_interface.detect_language', return_value='en'):
                await voice_interface.run()
            
            # Should speak the appropriate threat level message
            mock_speak.assert_any_call(expected_message, lang='en')
    
    @pytest.mark.asyncio
    async def test_recommendations_speaking(self, voice_interface, mock_core):
        """Test speaking of recommendations."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            
            # Mock result with recommendations
            mock_result = AnalysisResult(
                threat_level=ThreatLevel.MEDIUM,
                recommendations=["Verify sender", "Don't click links", "Check URL carefully"],
                spiritual_guidance="Stay mindful"
            )
            mock_core.run_multimodal_analysis.return_value = mock_result
            
            voice_interface.asr.listen_and_transcribe.side_effect = [
                "scan message",
                "suspicious message",
                "exit"
            ]
            
            with patch('src.ui.voice_interface.detect_language', return_value='en'):
                await voice_interface.run()
            
            # Should speak recommendations header and first two recommendations
            mock_speak.assert_any_call("Recommendations:", lang='en')
            mock_speak.assert_any_call("Verify sender", lang='en')
            mock_speak.assert_any_call("Don't click links", lang='en')
    
    @pytest.mark.asyncio
    async def test_spiritual_guidance_speaking(self, voice_interface, mock_core):
        """Test speaking of spiritual guidance."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            
            # Mock result with spiritual guidance
            mock_result = AnalysisResult(
                threat_level=ThreatLevel.LOW,
                recommendations=["Stay alert"],
                spiritual_guidance="Trust your inner wisdom"
            )
            mock_core.run_multimodal_analysis.return_value = mock_result
            
            voice_interface.asr.listen_and_transcribe.side_effect = [
                "scan message",
                "test message",
                "exit"
            ]
            
            with patch('src.ui.voice_interface.detect_language', return_value='en'):
                await voice_interface.run()
            
            # Should speak spiritual guidance
            mock_speak.assert_any_call("Spiritual Guidance:", lang='en')
            mock_speak.assert_any_call("Trust your inner wisdom", lang='en')


class TestMultiLanguageVoiceUI:
    """Test suite for multi-language voice UI functionality."""
    
    @pytest.fixture
    def multilingual_responses(self):
        """Multilingual response templates for testing."""
        return {
            'threat_messages': {
                4: {
                    "en": "Threat level critical. Please do not proceed! This is a scam.",
                    "hi": "खतरा अत्यंत गंभीर है, कृपया आगे ना बढ़ें – ये घोटाला है!",
                    "es": "Nivel de amenaza crítico. Por favor, no continúe."
                },
                3: {
                    "en": "Threat level high. Please do not proceed.",
                    "hi": "खतरा उच्च स्तर का है। कृपया आगे ना बढ़ें।"
                },
                2: {
                    "en": "Threat level medium. Caution advised.",
                    "hi": "मध्यम खतरा। सतर्क रहें।"
                },
                1: {
                    "en": "Threat level low. Probably safe, but stay alert.",
                    "hi": "न्यून खतरा, शायद सुरक्षित है, लेकिन सतर्क रहें।"
                },
                0: {
                    "en": "No scam detected. Your message seems safe.",
                    "hi": "कोई घोटाला नहीं मिला, संदेश सुरक्षित लगता है।"
                }
            },
            'section_headers': {
                "en": {"recommendations": "Recommendations:", "spiritual": "Spiritual Guidance:"},
                "hi": {"recommendations": "सिफारिशें:", "spiritual": "आध्यात्मिक मार्गदर्शन:"}
            }
        }
    
    @pytest.mark.parametrize("language,expected_tagline", [
        ("en", "Let's fight scams and secure the world with DharmaShield, powered by Google Gemma 3n."),
        ("hi", "आओ धोखेबाजियों से लड़े और दुनिया को सुरक्षित बनायें – DharmaShield, Google Gemma 3n द्वारा संचालित।"),
    ])
    @pytest.mark.asyncio
    async def test_taglines_different_languages(self, mock_core, language, expected_tagline):
        """Test taglines in different languages."""
        with patch('src.ui.voice_interface.ASREngine') as mock_asr_class:
            with patch('src.ui.voice_interface.speak') as mock_speak:
                
                mock_asr = Mock()
                mock_asr.listen_and_transcribe.return_value = "exit"
                mock_asr_class.return_value = mock_asr
                
                interface = VoiceInterface(mock_core, language=language)
                interface.asr = mock_asr
                
                await interface.run()
                
                # Should speak tagline in correct language
                mock_speak.assert_any_call(expected_tagline, lang=language)
    
    @pytest.mark.parametrize("language,threat_level,expected_message", [
        ("en", ThreatLevel.CRITICAL, "Threat level critical. Please do not proceed! This is a scam."),
        ("hi", ThreatLevel.CRITICAL, "खतरा अत्यंत गंभीर है, कृपया आगे ना बढ़ें – ये घोटाला है!"),
        ("en", ThreatLevel.MEDIUM, "Threat level medium. Caution advised."),
        ("hi", ThreatLevel.MEDIUM, "मध्यम खतरा। सतर्क रहें।"),
    ])
    @pytest.mark.asyncio
    async def test_threat_messages_different_languages(self, mock_core, language, threat_level, expected_message):
        """Test threat level messages in different languages."""
        with patch('src.ui.voice_interface.ASREngine') as mock_asr_class:
            with patch('src.ui.voice_interface.speak') as mock_speak:
                with patch('src.ui.voice_interface.detect_language', return_value=language):
                    
                    mock_asr = Mock()
                    mock_asr.listen_and_transcribe.side_effect = [
                        "scan message",
                        "test message",
                        "exit"
                    ]
                    mock_asr_class.return_value = mock_asr
                    
                    # Mock threat result
                    mock_result = AnalysisResult(
                        threat_level=threat_level,
                        recommendations=[],
                        spiritual_guidance=""
                    )
                    mock_core.run_multimodal_analysis.return_value = mock_result
                    
                    interface = VoiceInterface(mock_core, language=language)
                    interface.asr = mock_asr
                    
                    await interface.run()
                    
                    # Should speak threat message in correct language
                    mock_speak.assert_any_call(expected_message, lang=language)
    
    @pytest.mark.parametrize("language,expected_rec_header", [
        ("en", "Recommendations:"),
        ("hi", "सिफारिशें:"),
    ])
    @pytest.mark.asyncio
    async def test_recommendation_headers_different_languages(self, mock_core, language, expected_rec_header):
        """Test recommendation headers in different languages."""
        with patch('src.ui.voice_interface.ASREngine') as mock_asr_class:
            with patch('src.ui.voice_interface.speak') as mock_speak:
                with patch('src.ui.voice_interface.detect_language', return_value=language):
                    
                    mock_asr = Mock()
                    mock_asr.listen_and_transcribe.side_effect = [
                        "scan message",
                        "test message",
                        "exit"
                    ]
                    mock_asr_class.return_value = mock_asr
                    
                    # Mock result with recommendations
                    mock_result = AnalysisResult(
                        threat_level=ThreatLevel.MEDIUM,
                        recommendations=["Test recommendation"],
                        spiritual_guidance=""
                    )
                    mock_core.run_multimodal_analysis.return_value = mock_result
                    
                    interface = VoiceInterface(mock_core, language=language)
                    interface.asr = mock_asr
                    
                    await interface.run()
                    
                    # Should speak recommendation header in correct language
                    mock_speak.assert_any_call(expected_rec_header, lang=language)


class TestVoiceUIErrorHandling:
    """Test suite for voice UI error handling."""
    
    @pytest.fixture
    def voice_interface_with_errors(self, mock_core):
        """Create voice interface with error-prone components."""
        with patch('src.ui.voice_interface.ASREngine') as mock_asr_class:
            interface = VoiceInterface(mock_core, language='en')
            mock_asr = Mock()
            mock_asr_class.return_value = mock_asr
            interface.asr = mock_asr
            return interface
    
    @pytest.mark.asyncio
    async def test_asr_timeout_handling(self, voice_interface_with_errors):
        """Test handling of ASR timeout errors."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            
            # Mock ASR timeout
            voice_interface_with_errors.asr.listen_and_transcribe.side_effect = [
                Exception("Timeout"),
                "exit"
            ]
            
            await voice_interface_with_errors.run()
            
            # Should handle timeout gracefully
            mock_speak.assert_any_call("Sorry, I did not understand. Please repeat.", lang='en')
    
    @pytest.mark.asyncio
    async def test_core_analysis_error(self, voice_interface_with_errors, mock_core):
        """Test handling of core analysis errors."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            with patch('src.ui.voice_interface.detect_language', return_value='en'):
                
                # Mock core analysis error
                mock_core.run_multimodal_analysis.side_effect = Exception("Analysis failed")
                
                voice_interface_with_errors.asr.listen_and_transcribe.side_effect = [
                    "scan message",
                    "test message",
                    "exit"
                ]
                
                await voice_interface_with_errors.run()
                
                # Should handle analysis error gracefully
                # The interface should continue running and eventually exit
                assert not voice_interface_with_errors.running
    
    @pytest.mark.asyncio
    async def test_tts_engine_error(self, voice_interface_with_errors):
        """Test handling of TTS engine errors."""
        with patch('src.ui.voice_interface.speak', side_effect=Exception("TTS failed")):
            
            voice_interface_with_errors.asr.listen_and_transcribe.return_value = "exit"
            
            # Should not crash even if TTS fails
            await voice_interface_with_errors.run()
            
            assert not voice_interface_with_errors.running
    
    @pytest.mark.asyncio
    async def test_language_detection_error(self, voice_interface_with_errors, mock_core):
        """Test handling of language detection errors."""
        with patch('src.ui.voice_interface.speak') as mock_speak:
            with patch('src.ui.voice_interface.detect_language', side_effect=Exception("Detection failed")):
                
                voice_interface_with_errors.asr.listen_and_transcribe.side_effect = [
                    "scan message",
                    "test message",
                    "exit"
                ]
                
                await voice_interface_with_errors.run()
                
                # Should still proceed with analysis even if language detection fails
                mock_core.run_multimodal_analysis.assert_called_once()


class TestVoiceUIPerformance:
    """Performance tests for voice UI."""
    
    @pytest.mark.asyncio
    async def test_response_time(self, mock_core):
        """Test voice UI response time."""
        with patch('src.ui.voice_interface.ASREngine') as mock_asr_class:
            with patch('src.ui.voice_interface.speak'):
                with patch('src.ui.voice_interface.detect_language', return_value='en'):
                    
                    mock_asr = Mock()
                    mock_asr.listen_and_transcribe.side_effect = [
                        "scan message",
                        "test message",
                        "exit"
                    ]
                    mock_asr_class.return_value = mock_asr
                    
                    interface = VoiceInterface(mock_core, language='en')
                    interface.asr = mock_asr
                    
                    import time
                    start_time = time.time()
                    
                    await interface.run()
                    
                    processing_time = time.time() - start_time
                    
                    # Should complete within reasonable time
                    assert processing_time < 5.0  # Max 5 seconds for this test
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, mock_core):
        """Test voice UI memory usage."""
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run multiple voice UI sessions
        for _ in range(10):
            with patch('src.ui.voice_interface.ASREngine') as mock_asr_class:
                with patch('src.ui.voice_interface.speak'):
                    
                    mock_asr = Mock()
                    mock_asr.listen_and_transcribe.return_value = "exit"
                    mock_asr_class.return_value = mock_asr
                    
                    interface = VoiceInterface(mock_core, language='en')
                    interface.asr = mock_asr
                    
                    await interface.run()
                    
                    del interface
        
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB


class TestVoiceUIIntegration:
    """Integration tests for voice UI with real components."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_voice_workflow(self):
        """Test complete voice workflow integration."""
        # This test would use real components in integration environment
        # For unit testing, we mock the components
        
        mock_core = AsyncMock(spec=DharmaShieldCore)
        mock_result = AnalysisResult(
            threat_level=ThreatLevel.HIGH,
            recommendations=["Verify sender", "Don't click links"],
            spiritual_guidance="Stay vigilant"
        )
        mock_core.run_multimodal_analysis.return_value = mock_result
        
        with patch('src.ui.voice_interface.ASREngine') as mock_asr_class:
            with patch('src.ui.voice_interface.speak') as mock_speak:
                with patch('src.ui.voice_interface.detect_language', return_value='en'):
                    
                    mock_asr = Mock()
                    mock_asr.listen_and_transcribe.side_effect = [
                        "scan this message",
                        "URGENT: Your account will be suspended! Click here now!",
                        "exit"
                    ]
                    mock_asr_class.return_value = mock_asr
                    
                    interface = VoiceInterface(mock_core, language='en')
                    interface.asr = mock_asr
                    
                    await interface.run()
                    
                    # Verify complete workflow
                    mock_core.run_multimodal_analysis.assert_called_once_with(
                        text="URGENT: Your account will be suspended! Click here now!"
                    )
                    
                    # Verify appropriate responses were spoken
                    mock_speak.assert_any_call(
                        "Threat level high. Please do not proceed.", 
                        lang='en'
                    )
                    mock_speak.assert_any_call("Recommendations:", lang='en')
                    mock_speak.assert_any_call("Verify sender", lang='en')
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cross_language_detection(self):
        """Test cross-language detection and switching."""
        mock_core = AsyncMock(spec=DharmaShieldCore)
        mock_result = AnalysisResult(
            threat_level=ThreatLevel.SAFE,
            recommendations=[],
            spiritual_guidance=""
        )
        mock_core.run_multimodal_analysis.return_value = mock_result
        
        with patch('src.ui.voice_interface.ASREngine') as mock_asr_class:
            with patch('src.ui.voice_interface.speak') as mock_speak:
                with patch('src.ui.voice_interface.get_language_name', return_value='Hindi'):
                    
                    mock_asr = Mock()
                    mock_asr.listen_and_transcribe.side_effect = [
                        "scan message",
                        "यह एक सुरक्षित संदेश है",  # Hindi message
                        "exit"
                    ]
                    mock_asr_class.return_value = mock_asr
                    
                    # Mock language detection to return Hindi
                    with patch('src.ui.voice_interface.detect_language', return_value='hi'):
                        interface = VoiceInterface(mock_core, language='en')
                        interface.asr = mock_asr
                        
                        await interface.run()
                        
                        # Should detect language and switch
                        assert interface.language == 'hi'
                        mock_speak.assert_any_call("Detected language: Hindi", lang='hi')


# Utility functions for testing
def create_mock_analysis_result(threat_level: ThreatLevel, language: str = 'en') -> AnalysisResult:
    """Create mock analysis result for testing."""
    recommendations_map = {
        'en': ["Stay alert", "Verify sender", "Don't click links"],
        'hi': ["सतर्क रहें", "भेजने वाले को सत्यापित करें", "लिंक पर क्लिक न करें"]
    }
    
    guidance_map = {
        'en': "Trust your intuition",
        'hi': "अपनी अंतरात्मा पर भरोसा करें"
    }
    
    return AnalysisResult(
        threat_level=threat_level,
        recommendations=recommendations_map.get(language, recommendations_map['en']),
        spiritual_guidance=guidance_map.get(language, guidance_map['en'])
    )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-m", "not integration"])
```# DharmaShield Test Suite - Industry-Grade Testing Framework


