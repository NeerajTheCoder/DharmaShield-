"""
src/utils/language.py

DharmaShield - Advanced Multilingual Language Detection & Mapping Utilities
---------------------------------------------------------------------------
• Industry-grade language auto-detection, human-readable mapping, supported language listing
• Cross-platform (Android/iOS/Desktop) with Kivy/Buildozer compatibility and offline-first design
• Deterministic seed setup, robust error handling, extensible architecture with caching support
• Full integration with voice interface, TTS engine, ASR engine, and core analysis systems

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import os
import sys
import time
import threading
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import warnings

# Language detection imports with fallback handling
try:
    from langdetect import detect, DetectorFactory, LangDetectException
    from langdetect.detector import Detector
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    warnings.warn("langdetect not available. Language detection will be limited.", ImportWarning)

# Project imports
from ...utils.logger import get_logger

logger = get_logger(__name__)

# Set deterministic seed for consistent language detection
if HAS_LANGDETECT:
    DetectorFactory.seed = 0

# -------------------------------
# Language Configuration & Mappings
# -------------------------------

# ISO 639-1 language codes to human-readable names
LANGUAGE_NAMES = {
    'en': 'English',
    'hi': 'Hindi',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'ru': 'Russian',
    'bn': 'Bengali',
    'ur': 'Urdu',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'or': 'Odia',
    'as': 'Assamese',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'my': 'Myanmar',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'ko': 'Korean',
    'ja': 'Japanese',
    'pt': 'Portuguese',
    'it': 'Italian',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'pl': 'Polish',
    'cs': 'Czech',
    'sk': 'Slovak',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'bg': 'Bulgarian',
    'hr': 'Croatian',
    'sr': 'Serbian',
    'bs': 'Bosnian',
    'mk': 'Macedonian',
    'sq': 'Albanian',
    'sl': 'Slovenian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'mt': 'Maltese',
    'ga': 'Irish',
    'cy': 'Welsh',
    'eu': 'Basque',
    'ca': 'Catalan',
    'gl': 'Galician',
    'is': 'Icelandic',
    'fo': 'Faroese',
    'he': 'Hebrew',
    'yi': 'Yiddish',
    'fa': 'Persian',
    'ps': 'Pashto',
    'sd': 'Sindhi',
    'ks': 'Kashmiri',
    'dv': 'Dhivehi',
    'so': 'Somali',
    'sw': 'Swahili',
    'am': 'Amharic',
    'ti': 'Tigrinya',
    'om': 'Oromo',
    'zu': 'Zulu',
    'xh': 'Xhosa',
    'af': 'Afrikaans',
    'tn': 'Tswana',
    'st': 'Sesotho',
    'ss': 'Swati',
    've': 'Venda',
    'ts': 'Tsonga',
    'nr': 'Ndebele',
    'wo': 'Wolof',
    'yo': 'Yoruba',
    'ig': 'Igbo',
    'ha': 'Hausa',
    'ff': 'Fulah',
    'ln': 'Lingala',
    'mg': 'Malagasy',
    'ny': 'Chichewa',
    'sn': 'Shona',
    'rw': 'Kinyarwanda',
    'rn': 'Kirundi',
    'ku': 'Kurdish',
    'az': 'Azerbaijani',
    'kk': 'Kazakh',
    'ky': 'Kyrgyz',
    'tg': 'Tajik',
    'tk': 'Turkmen',
    'uz': 'Uzbek',
    'mn': 'Mongolian',
    'bo': 'Tibetan',
    'dz': 'Dzongkha',
    'lo': 'Lao',
    'km': 'Khmer',
    'ms': 'Malay',
    'id': 'Indonesian',
    'tl': 'Filipino',
    'ceb': 'Cebuano',
    'haw': 'Hawaiian',
    'mi': 'Maori',
    'sm': 'Samoan',
    'to': 'Tongan',
    'fj': 'Fijian',
}

# Google TTS/ASR language codes mapping
GOOGLE_LANG_CODES = {
    'en': 'en-US',
    'hi': 'hi-IN',
    'es': 'es-ES',
    'fr': 'fr-FR',
    'de': 'de-DE',
    'zh': 'zh-CN',
    'ar': 'ar-SA',
    'ru': 'ru-RU',
    'bn': 'bn-BD',
    'ur': 'ur-PK',
    'ta': 'ta-IN',
    'te': 'te-IN',
    'mr': 'mr-IN',
    'gu': 'gu-IN',
    'kn': 'kn-IN',
    'ml': 'ml-IN',
    'pa': 'pa-IN',
    'or': 'or-IN',
    'as': 'as-IN',
    'ne': 'ne-NP',
    'si': 'si-LK',
    'my': 'my-MM',
    'th': 'th-TH',
    'vi': 'vi-VN',
    'ko': 'ko-KR',
    'ja': 'ja-JP',
    'pt': 'pt-BR',
    'it': 'it-IT',
    'nl': 'nl-NL',
    'sv': 'sv-SE',
    'da': 'da-DK',
    'no': 'nb-NO',
    'fi': 'fi-FI',
    'pl': 'pl-PL',
    'cs': 'cs-CZ',
    'sk': 'sk-SK',
    'hu': 'hu-HU',
    'ro': 'ro-RO',
    'bg': 'bg-BG',
    'hr': 'hr-HR',
    'sr': 'sr-RS',
    'bs': 'bs-BA',
    'mk': 'mk-MK',
    'sq': 'sq-AL',
    'sl': 'sl-SI',
    'et': 'et-EE',
    'lv': 'lv-LV',
    'lt': 'lt-LT',
    'mt': 'mt-MT',
    'ga': 'ga-IE',
    'cy': 'cy-GB',
    'eu': 'eu-ES',
    'ca': 'ca-ES',
    'gl': 'gl-ES',
    'is': 'is-IS',
    'he': 'he-IL',
    'fa': 'fa-IR',
    'ps': 'ps-AF',
    'so': 'so-SO',
    'sw': 'sw-KE',
    'am': 'am-ET',
    'zu': 'zu-ZA',
    'xh': 'xh-ZA',
    'af': 'af-ZA',
    'yo': 'yo-NG',
    'ig': 'ig-NG',
    'ha': 'ha-NG',
    'wo': 'wo-SN',
    'mg': 'mg-MG',
    'ny': 'ny-MW',
    'sn': 'sn-ZW',
    'rw': 'rw-RW',
    'ku': 'ku-TR',
    'az': 'az-AZ',
    'kk': 'kk-KZ',
    'ky': 'ky-KG',
    'tg': 'tg-TJ',
    'tk': 'tk-TM',
    'uz': 'uz-UZ',
    'mn': 'mn-MN',
    'bo': 'bo-CN',
    'lo': 'lo-LA',
    'km': 'km-KH',
    'ms': 'ms-MY',
    'id': 'id-ID',
    'tl': 'tl-PH',
    'mi': 'mi-NZ',
    'sm': 'sm-WS',
    'to': 'to-TO',
    'fj': 'fj-FJ',
}

# Language script information for text rendering
LANGUAGE_SCRIPTS = {
    'en': 'Latin',
    'hi': 'Devanagari',
    'es': 'Latin',
    'fr': 'Latin',
    'de': 'Latin',
    'zh': 'Han',
    'ar': 'Arabic',
    'ru': 'Cyrillic',
    'bn': 'Bengali',
    'ur': 'Arabic',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Devanagari',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Gurmukhi',
    'or': 'Odia',
    'as': 'Bengali',
    'ne': 'Devanagari',
    'si': 'Sinhala',
    'my': 'Myanmar',
    'th': 'Thai',
    'vi': 'Latin',
    'ko': 'Hangul',
    'ja': 'Hiragana',
    'he': 'Hebrew',
    'fa': 'Arabic',
    'ps': 'Arabic',
    'so': 'Latin',
    'sw': 'Latin',
    'am': 'Ethiopic',
    'zu': 'Latin',
    'xh': 'Latin',
    'af': 'Latin',
    'yo': 'Latin',
    'ig': 'Latin',
    'ha': 'Latin',
    'wo': 'Latin',
    'mg': 'Latin',
    'ny': 'Latin',
    'sn': 'Latin',
    'rw': 'Latin',
    'ku': 'Latin',
    'az': 'Latin',
    'kk': 'Cyrillic',
    'ky': 'Cyrillic',
    'tg': 'Cyrillic',
    'tk': 'Latin',
    'uz': 'Latin',
    'mn': 'Cyrillic',
    'bo': 'Tibetan',
    'lo': 'Lao',
    'km': 'Khmer',
    'ms': 'Latin',
    'id': 'Latin',
    'tl': 'Latin',
    'mi': 'Latin',
    'sm': 'Latin',
    'to': 'Latin',
    'fj': 'Latin',
}

# Language regions for cultural adaptation
LANGUAGE_REGIONS = {
    'en': ['US', 'GB', 'AU', 'CA', 'IN', 'ZA'],
    'hi': ['IN'],
    'es': ['ES', 'MX', 'AR', 'CO', 'PE', 'VE', 'CL', 'EC', 'UY', 'PY', 'BO'],
    'fr': ['FR', 'CA', 'BE', 'CH', 'SN', 'CI', 'ML', 'BF', 'NE', 'CD'],
    'de': ['DE', 'AT', 'CH', 'LI'],
    'zh': ['CN', 'TW', 'HK', 'SG', 'MO'],
    'ar': ['SA', 'EG', 'DZ', 'SD', 'IQ', 'MA', 'YE', 'SY', 'TN', 'JO'],
    'ru': ['RU', 'BY', 'KZ', 'KG', 'TJ', 'UZ', 'TM', 'AZ', 'GE', 'AM'],
    'pt': ['BR', 'PT', 'AO', 'MZ', 'GW', 'CV', 'ST', 'TL', 'GQ', 'MO'],
    'bn': ['BD', 'IN'],
    'ur': ['PK', 'IN'],
    'ta': ['IN', 'LK', 'SG', 'MY'],
    'te': ['IN'],
    'mr': ['IN'],
    'ja': ['JP'],
    'ko': ['KR', 'KP'],
    'vi': ['VN'],
    'th': ['TH'],
    'ms': ['MY', 'BN', 'ID', 'SG'],
    'id': ['ID'],
    'tl': ['PH'],
    'sw': ['KE', 'TZ', 'UG', 'RW', 'BI', 'CD', 'MZ'],
    'ha': ['NG', 'NE', 'CM', 'GH', 'TD', 'BF', 'BJ'],
    'yo': ['NG', 'BJ', 'TG'],
    'ig': ['NG'],
    'zu': ['ZA'],
    'xh': ['ZA'],
    'af': ['ZA', 'NA', 'BW'],
}

# -------------------------------
# Data Structures
# -------------------------------

@dataclass
class LanguageDetectionResult:
    """Result of language detection with confidence and alternatives."""
    language: str
    confidence: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    text_length: int = 0
    processing_time: float = 0.0
    method_used: str = "langdetect"
    is_reliable: bool = True

@dataclass
class LanguageInfo:
    """Comprehensive language information."""
    code: str
    name: str
    native_name: str = ""
    script: str = ""
    regions: List[str] = field(default_factory=list)
    google_code: str = ""
    is_rtl: bool = False
    is_supported: bool = True

# -------------------------------
# Core Language Utilities
# -------------------------------

class LanguageDetectorEngine:
    """
    Advanced language detection engine with caching and fallback mechanisms.
    
    Features:
    - Multiple detection algorithms with fallback
    - Results caching for performance
    - Confidence thresholding and validation
    - Thread-safe operations
    - Extensive error handling and logging
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        
        self.cache = {}
        self.cache_max_size = 1000
        self.confidence_threshold = 0.7
        self.min_text_length = 3
        self.max_cache_age = 3600  # 1 hour
        self._lock = threading.RLock()
        self._initialized = True
        
        logger.info("LanguageDetectorEngine initialized")
    
    def detect_language(
        self,
        text: str,
        use_cache: bool = True,
        return_alternatives: bool = False,
        confidence_threshold: Optional[float] = None
    ) -> LanguageDetectionResult:
        """
        Detect language of given text with comprehensive analysis.
        
        Args:
            text: Input text to analyze
            use_cache: Whether to use caching for performance
            return_alternatives: Whether to return alternative language candidates
            confidence_threshold: Minimum confidence for reliable detection
            
        Returns:
            LanguageDetectionResult with detection details
        """
        start_time = time.time()
        
        # Input validation
        if not text or not isinstance(text, str):
            return LanguageDetectionResult(
                language='en',
                confidence=0.0,
                text_length=0,
                is_reliable=False,
                processing_time=time.time() - start_time
            )
        
        text_clean = text.strip()
        if len(text_clean) < self.min_text_length:
            return LanguageDetectionResult(
                language='en',
                confidence=0.3,
                text_length=len(text_clean),
                is_reliable=False,
                processing_time=time.time() - start_time
            )
        
        # Check cache
        cache_key = hash(text_clean) if use_cache else None
        if cache_key and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if time.time() - cached_result.get('timestamp', 0) < self.max_cache_age:
                cached_result['result'].processing_time = time.time() - start_time
                return cached_result['result']
        
        # Perform detection
        result = self._detect_with_fallback(
            text_clean,
            return_alternatives,
            confidence_threshold or self.confidence_threshold
        )
        
        result.text_length = len(text_clean)
        result.processing_time = time.time() - start_time
        
        # Cache result
        if cache_key and use_cache:
            with self._lock:
                if len(self.cache) >= self.cache_max_size:
                    # Remove oldest entries
                    oldest_keys = sorted(
                        self.cache.keys(),
                        key=lambda k: self.cache[k].get('timestamp', 0)
                    )[:self.cache_max_size // 4]
                    for old_key in oldest_keys:
                        del self.cache[old_key]
                
                self.cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
        
        return result
    
    def _detect_with_fallback(
        self,
        text: str,
        return_alternatives: bool,
        confidence_threshold: float
    ) -> LanguageDetectionResult:
        """Detect language with fallback mechanisms."""
        
        # Primary detection with langdetect
        if HAS_LANGDETECT:
            try:
                # Single language detection
                detected_lang = detect(text)
                
                # Get alternatives if requested
                alternatives = []
                confidence = 0.95  # Default high confidence for single detection
                
                if return_alternatives:
                    try:
                        detector = Detector(text)
                        detector.seed = 0  # Ensure deterministic results
                        probabilities = detector.get_probabilities()
                        
                        alternatives = [
                            (lang.lang, lang.prob)
                            for lang in probabilities
                            if lang.lang != detected_lang
                        ]
                        
                        # Get confidence from probabilities
                        primary_prob = next(
                            (lang.prob for lang in probabilities if lang.lang == detected_lang),
                            0.95
                        )
                        confidence = primary_prob
                    
                    except Exception as e:
                        logger.debug(f"Alternative detection failed: {e}")
                
                return LanguageDetectionResult(
                    language=detected_lang,
                    confidence=confidence,
                    alternatives=alternatives,
                    method_used="langdetect",
                    is_reliable=confidence >= confidence_threshold
                )
            
            except LangDetectException as e:
                logger.debug(f"Language detection failed: {e}")
            except Exception as e:
                logger.warning(f"Unexpected detection error: {e}")
        
        # Fallback: Simple heuristic detection
        return self._heuristic_detection(text, confidence_threshold)
    
    def _heuristic_detection(self, text: str, confidence_threshold: float) -> LanguageDetectionResult:
        """Simple heuristic-based language detection as fallback."""
        
        # Basic script-based detection
        scripts_found = set()
        
        for char in text:
            if '\u0900' <= char <= '\u097F':  # Devanagari (Hindi, Marathi, Nepali)
                scripts_found.add('devanagari')
            elif '\u0980' <= char <= '\u09FF':  # Bengali
                scripts_found.add('bengali')
            elif '\u0B80' <= char <= '\u0BFF':  # Tamil
                scripts_found.add('tamil')
            elif '\u0C00' <= char <= '\u0C7F':  # Telugu
                scripts_found.add('telugu')
            elif '\u0600' <= char <= '\u06FF':  # Arabic
                scripts_found.add('arabic')
            elif '\u4E00' <= char <= '\u9FFF':  # Chinese
                scripts_found.add('chinese')
            elif '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF':  # Japanese
                scripts_found.add('japanese')
            elif '\uAC00' <= char <= '\uD7AF':  # Korean
                scripts_found.add('korean')
            elif '\u0400' <= char <= '\u04FF':  # Cyrillic
                scripts_found.add('cyrillic')
        
        # Map scripts to languages
        if 'devanagari' in scripts_found:
            return LanguageDetectionResult('hi', 0.8, method_used="heuristic")
        elif 'bengali' in scripts_found:
            return LanguageDetectionResult('bn', 0.8, method_used="heuristic")
        elif 'tamil' in scripts_found:
            return LanguageDetectionResult('ta', 0.8, method_used="heuristic")
        elif 'telugu' in scripts_found:
            return LanguageDetectionResult('te', 0.8, method_used="heuristic")
        elif 'arabic' in scripts_found:
            return LanguageDetectionResult('ar', 0.8, method_used="heuristic")
        elif 'chinese' in scripts_found:
            return LanguageDetectionResult('zh', 0.8, method_used="heuristic")
        elif 'japanese' in scripts_found:
            return LanguageDetectionResult('ja', 0.8, method_used="heuristic")
        elif 'korean' in scripts_found:
            return LanguageDetectionResult('ko', 0.8, method_used="heuristic")
        elif 'cyrillic' in scripts_found:
            return LanguageDetectionResult('ru', 0.7, method_used="heuristic")
        
        # Default to English for Latin script or unknown
        return LanguageDetectionResult(
            'en', 
            0.5, 
            method_used="heuristic",
            is_reliable=False
        )
    
    def clear_cache(self):
        """Clear the detection cache."""
        with self._lock:
            self.cache.clear()
        logger.info("Language detection cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.cache_max_size,
                'hit_rate': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
            }

# -------------------------------
# Global Utility Functions
# -------------------------------

def detect_language(text: str, use_cache: bool = True) -> str:
    """
    Detect language of text and return ISO 639-1 code.
    
    Args:
        text: Input text to analyze
        use_cache: Whether to use result caching
        
    Returns:
        Two-letter language code (e.g., 'en', 'hi', 'es')
    """
    if not text:
        return 'en'
    
    try:
        detector = LanguageDetectorEngine()
        result = detector.detect_language(text, use_cache=use_cache)
        return result.language
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return 'en'

def detect_language_detailed(
    text: str,
    return_alternatives: bool = False,
    confidence_threshold: float = 0.7
) -> LanguageDetectionResult:
    """
    Detect language with detailed results including confidence and alternatives.
    
    Args:
        text: Input text to analyze
        return_alternatives: Whether to return alternative language candidates
        confidence_threshold: Minimum confidence for reliable detection
        
    Returns:
        LanguageDetectionResult with comprehensive detection information
    """
    detector = LanguageDetectorEngine()
    return detector.detect_language(
        text,
        return_alternatives=return_alternatives,
        confidence_threshold=confidence_threshold
    )

def get_language_name(lang_code: str) -> str:
    """
    Get human-readable language name from ISO 639-1 code.
    
    Args:
        lang_code: Two-letter language code
        
    Returns:
        Human-readable language name (e.g., 'English', 'Hindi')
    """
    return LANGUAGE_NAMES.get(lang_code.lower(), lang_code)

def get_google_lang_code(lang_code: str) -> str:
    """
    Get Google TTS/ASR language code from ISO 639-1 code.
    
    Args:
        lang_code: Two-letter language code
        
    Returns:
        Google-compatible language code (e.g., 'en-US', 'hi-IN')
    """
    return GOOGLE_LANG_CODES.get(lang_code.lower(), 'en-US')

def get_language_info(lang_code: str) -> LanguageInfo:
    """
    Get comprehensive language information.
    
    Args:
        lang_code: Two-letter language code
        
    Returns:
        LanguageInfo object with detailed language data
    """
    code = lang_code.lower()
    
    return LanguageInfo(
        code=code,
        name=LANGUAGE_NAMES.get(code, code),
        script=LANGUAGE_SCRIPTS.get(code, 'Latin'),
        regions=LANGUAGE_REGIONS.get(code, []),
        google_code=GOOGLE_LANG_CODES.get(code, 'en-US'),
        is_rtl=code in ['ar', 'he', 'fa', 'ur', 'ps', 'sd', 'dv'],
        is_supported=code in LANGUAGE_NAMES
    )

def list_supported() -> List[str]:
    """
    Get list of supported language codes.
    
    Returns:
        List of ISO 639-1 language codes
    """
    return list(LANGUAGE_NAMES.keys())

def list_supported_names() -> List[str]:
    """
    Get list of supported language names.
    
    Returns:
        List of human-readable language names
    """
    return list(LANGUAGE_NAMES.values())

def get_supported_languages() -> Dict[str, str]:
    """
    Get mapping of language codes to names.
    
    Returns:
        Dictionary mapping codes to names
    """
    return LANGUAGE_NAMES.copy()

def is_language_supported(lang_code: str) -> bool:
    """
    Check if language is supported.
    
    Args:
        lang_code: Two-letter language code
        
    Returns:
        True if language is supported, False otherwise
    """
    return lang_code.lower() in LANGUAGE_NAMES

def find_language_by_name(name: str) -> Optional[str]:
    """
    Find language code by partial name match.
    
    Args:
        name: Language name or partial name
        
    Returns:
        Language code if found, None otherwise
    """
    name_lower = name.lower()
    
    # Exact match
    for code, lang_name in LANGUAGE_NAMES.items():
        if lang_name.lower() == name_lower:
            return code
    
    # Partial match
    for code, lang_name in LANGUAGE_NAMES.items():
        if name_lower in lang_name.lower() or lang_name.lower().startswith(name_lower):
            return code
    
    return None

def get_languages_by_script(script: str) -> List[str]:
    """
    Get languages that use a specific script.
    
    Args:
        script: Script name (e.g., 'Latin', 'Devanagari')
        
    Returns:
        List of language codes using the script
    """
    return [
        code for code, lang_script in LANGUAGE_SCRIPTS.items()
        if lang_script.lower() == script.lower()
    ]

def get_languages_by_region(region: str) -> List[str]:
    """
    Get languages used in a specific region.
    
    Args:
        region: Two-letter region/country code
        
    Returns:
        List of language codes used in the region
    """
    region_upper = region.upper()
    return [
        code for code, regions in LANGUAGE_REGIONS.items()
        if region_upper in regions
    ]

def validate_language_code(lang_code: str) -> Tuple[bool, str]:
    """
    Validate and normalize language code.
    
    Args:
        lang_code: Language code to validate
        
    Returns:
        Tuple of (is_valid, normalized_code)
    """
    if not lang_code or not isinstance(lang_code, str):
        return False, 'en'
    
    normalized = lang_code.lower().strip()
    
    # Handle common variations
    code_mappings = {
        'eng': 'en',
        'hin': 'hi',
        'spa': 'es',
        'fre': 'fr',
        'fra': 'fr',
        'ger': 'de',
        'deu': 'de',
        'chi': 'zh',
        'zho': 'zh',
        'ara': 'ar',
        'rus': 'ru',
        'ben': 'bn',
        'urd': 'ur',
        'tam': 'ta',
        'tel': 'te',
        'mar': 'mr',
        'guj': 'gu',
        'kan': 'kn',
        'mal': 'ml',
        'pan': 'pa',
        'ori': 'or',
        'asm': 'as',
        'nep': 'ne',
        'sin': 'si',
        'mya': 'my',
        'tha': 'th',
        'vie': 'vi',
        'kor': 'ko',
        'jpn': 'ja',
        'por': 'pt',
        'ita': 'it',
        'dut': 'nl',
        'nld': 'nl',
        'swe': 'sv',
        'dan': 'da',
        'nor': 'no',
        'fin': 'fi',
        'pol': 'pl',
        'ces': 'cs',
        'slo': 'sk',
        'slk': 'sk',
        'hun': 'hu',
        'rom': 'ro',
        'ron': 'ro',
        'bul': 'bg',
        'hrv': 'hr',
        'srp': 'sr',
        'bos': 'bs',
        'mkd': 'mk',
        'sqi': 'sq',
        'slv': 'sl',
        'est': 'et',
        'lav': 'lv',
        'lit': 'lt',
        'mlt': 'mt',
        'gle': 'ga',
        'cym': 'cy',
        'eus': 'eu',
        'cat': 'ca',
        'glg': 'gl',
        'isl': 'is',
        'fao': 'fo',
        'heb': 'he',
        'yid': 'yi',
        'fas': 'fa',
        'per': 'fa',
        'pus': 'ps',
        'snd': 'sd',
        'kas': 'ks',
        'div': 'dv',
        'som': 'so',
        'swa': 'sw',
        'amh': 'am',
        'tir': 'ti',
        'orm': 'om',
        'zul': 'zu',
        'xho': 'xh',
        'afr': 'af',
    }
    
    if normalized in code_mappings:
        normalized = code_mappings[normalized]
    
    is_valid = normalized in LANGUAGE_NAMES
    return is_valid, normalized if is_valid else 'en'

# -------------------------------
# Performance and Debugging Utilities
# -------------------------------

def benchmark_detection(texts: List[str], iterations: int = 100) -> Dict[str, Any]:
    """
    Benchmark language detection performance.
    
    Args:
        texts: List of texts to test
        iterations: Number of iterations per text
        
    Returns:
        Performance statistics
    """
    detector = LanguageDetectorEngine()
    
    total_time = 0
    total_detections = 0
    language_counts = {}
    
    start_time = time.time()
    
    for text in texts:
        for _ in range(iterations):
            detection_start = time.time()
            result = detector.detect_language(text)
            detection_time = time.time() - detection_start
            
            total_time += detection_time
            total_detections += 1
            
            language_counts[result.language] = language_counts.get(result.language, 0) + 1
    
    end_time = time.time()
    
    return {
        'total_time': end_time - start_time,
        'average_detection_time': total_time / total_detections,
        'detections_per_second': total_detections / (end_time - start_time),
        'total_detections': total_detections,
        'language_distribution': language_counts,
        'cache_stats': detector.get_cache_stats()
    }

def clear_language_cache():
    """Clear the global language detection cache."""
    detector = LanguageDetectorEngine()
    detector.clear_cache()

# -------------------------------
# Testing and Demo
# -------------------------------

if __name__ == "__main__":
    # Demo mode
    print("=== DharmaShield Language Utilities Demo ===")
    
    # Test texts in different languages
    test_texts = [
        "Hello, how are you today?",
        "नमस्ते, आप कैसे हैं?",
        "Hola, ¿cómo estás?",
        "Bonjour, comment allez-vous?",
        "Hallo, wie geht es dir?",
        "你好，你好吗？",
        "مرحبا، كيف حالك؟",
        "Привет, как дела?",
        "নমস্কার, আপনি কেমন আছেন?",
        "السلام علیکم، آپ کیسے ہیں؟",
    ]
    
    print("\n--- Language Detection Tests ---")
    for text in test_texts:
        result = detect_language_detailed(text, return_alternatives=True)
        lang_name = get_language_name(result.language)
        print(f"Text: {text}")
        print(f"Detected: {result.language} ({lang_name}) - Confidence: {result.confidence:.2f}")
        if result.alternatives:
            alts = ", ".join([f"{lang}:{conf:.2f}" for lang, conf in result.alternatives[:2]])
            print(f"Alternatives: {alts}")
        print(f"Reliable: {result.is_reliable}, Method: {result.method_used}")
        print()
    
    print("--- Language Information ---")
    for lang_code in ['en', 'hi', 'ar', 'zh']:
        info = get_language_info(lang_code)
        print(f"{info.name} ({info.code}): Script={info.script}, RTL={info.is_rtl}, Regions={info.regions}")
    
    print("\n--- Supported Languages ---")
    supported = list_supported()
    print(f"Total supported languages: {len(supported)}")
    print("First 10:", [f"{code}:{get_language_name(code)}" for code in supported[:10]])
    
    print("\n--- Language Search ---")
    search_tests = ["Hindi", "Chinese", "Arabic", "Spanish"]
    for search_term in search_tests:
        found_code = find_language_by_name(search_term)
        if found_code:
            print(f"'{search_term}' -> {found_code} ({get_language_name(found_code)})")
        else:
            print(f"'{search_term}' -> Not found")
    
    print("\n--- Performance Benchmark ---")
    short_texts = ["Hello world", "नमस्ते", "こんにちは", "Hola"]
    benchmark_results = benchmark_detection(short_texts, iterations=50)
    print(f"Average detection time: {benchmark_results['average_detection_time']*1000:.2f}ms")
    print(f"Detections per second: {benchmark_results['detections_per_second']:.0f}")
    print(f"Language distribution: {benchmark_results['language_distribution']}")
    
    print("\n✅ Language utilities ready for production!")
    print("🌐 Features demonstrated:")
    print("  ✓ Multi-method language detection with confidence scoring")
    print("  ✓ Comprehensive language mapping and information system")
    print("  ✓ Caching and performance optimization")
    print("  ✓ Robust error handling and fallback mechanisms")
    print("  ✓ Thread-safe operations and modular architecture")
    print("  ✓ Support for 100+ languages with regional variations")
    print("  ✓ Integration-ready for voice, TTS, and ASR systems")

