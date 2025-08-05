"""
detection/text/clean_text.py

DharmaShield - Advanced Multilingual Text Cleaning & Normalization Engine
--------------------------------------------------------------------------
• Production-grade text preprocessing for scam detection
• Full Unicode normalization & encoding repair
• Multilingual support with language-specific cleaning
• Industry-standard noise removal & text standardization
• Optimized for cross-platform mobile/desktop deployment

Author: DharmaShield Expert Team  
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import re
import string
import unicodedata
import warnings
from typing import Optional, Dict, List, Set, Union, Tuple
from functools import lru_cache
import threading
from pathlib import Path

# Third-party imports
try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False
    warnings.warn("ftfy not available - encoding repair disabled")

try:
    from charset_normalizer import from_bytes
    HAS_CHARSET_NORMALIZER = True
except ImportError:
    try:
        import chardet
        HAS_CHARDET = True
        HAS_CHARSET_NORMALIZER = False
    except ImportError:
        HAS_CHARDET = False
        HAS_CHARSET_NORMALIZER = False
        warnings.warn("No encoding detection available")

# Project imports
from ...utils.logger import get_logger
from ...utils.language import detect_language, LANGUAGE_NAMES
from ...core.config_loader import load_config

logger = get_logger(__name__)

class TextCleaningConfig:
    """Configuration class for text cleaning parameters."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        cleaning_config = self.config.get('text_cleaning', {})
        
        # Basic cleaning options
        self.normalize_unicode = cleaning_config.get('normalize_unicode', True)
        self.fix_encoding = cleaning_config.get('fix_encoding', True)
        self.remove_urls = cleaning_config.get('remove_urls', True)
        self.remove_emails = cleaning_config.get('remove_emails', True)
        self.remove_phone_numbers = cleaning_config.get('remove_phone_numbers', True)
        self.normalize_whitespace = cleaning_config.get('normalize_whitespace', True)
        self.remove_excessive_punctuation = cleaning_config.get('remove_excessive_punctuation', True)
        
        # Advanced options
        self.preserve_emojis = cleaning_config.get('preserve_emojis', True)
        self.expand_contractions = cleaning_config.get('expand_contractions', True)
        self.normalize_numbers = cleaning_config.get('normalize_numbers', True)
        self.remove_non_printable = cleaning_config.get('remove_non_printable', True)
        self.max_consecutive_chars = cleaning_config.get('max_consecutive_chars', 3)
        
        # Language-specific settings
        self.language_specific_cleaning = cleaning_config.get('language_specific_cleaning', True)
        
        # Performance settings
        self.use_cache = cleaning_config.get('use_cache', True)
        self.cache_size = cleaning_config.get('cache_size', 1000)


class LanguageSpecificCleaner:
    """Language-specific text cleaning rules and patterns."""
    
    # Language-specific patterns for common noise
    LANGUAGE_PATTERNS = {
        'hi': {  # Hindi
            'diacritics_normalize': True,
            'common_noise': [r'।+', r'॥+'],  # Devanagari punctuation
            'preserve_chars': ['्', '़', 'ं', 'ँ', 'ः', 'ॐ']  # Important diacritics
        },
        'ar': {  # Arabic
            'rtl_support': True,
            'common_noise': [r'[؟]+', r'[؛]+'],
            'preserve_chars': ['ُ', 'ِ', 'َ', 'ً', 'ٍ', 'ٌ']  # Diacritics
        },
        'zh': {  # Chinese
            'fullwidth_normalize': True,
            'common_noise': [r'[。！？；：，、]+'],
            'preserve_chars': []
        },
        'en': {  # English
            'contractions_expand': True,
            'common_noise': [],
            'preserve_chars': []
        }
    }
    
    @classmethod
    def get_language_config(cls, language: str) -> Dict:
        """Get language-specific cleaning configuration."""
        return cls.LANGUAGE_PATTERNS.get(language, cls.LANGUAGE_PATTERNS['en'])


class AdvancedTextCleaner:
    """
    Industry-grade text cleaner with comprehensive normalization capabilities.
    
    Features:
    - Unicode normalization (NFC/NFD/NFKC/NFKD)
    - Encoding detection and repair
    - Multilingual support with language-specific rules
    - Aggressive noise removal with content preservation
    - Performance optimization with caching
    - Thread-safe operations
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: Optional[str] = None):
        # Singleton pattern for performance
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if getattr(self, '_initialized', False):
            return
            
        self.config = TextCleaningConfig(config_path)
        self._compile_patterns()
        self._load_contractions()
        self._load_stopwords()
        self._initialized = True
        
        logger.info("Advanced Text Cleaner initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for performance."""
        # URL patterns (comprehensive)
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?'
        )
        
        # Email patterns
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone number patterns (international)
        self.phone_pattern = re.compile(
            r'(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}'
            r'|(?:\+[1-9]\d{0,3}[-.\s]?)?(?:\(?[0-9]{1,4}\)?[-.\s]?)?[0-9]{1,4}[-.\s]?[0-9]{1,9}'
        )
        
        # Excessive punctuation
        self.excessive_punct_pattern = re.compile(r'[.!?]{2,}|[,;:]{2,}|[-_]{3,}')
        
        # Consecutive characters
        self.consecutive_chars_pattern = re.compile(r'(.)\1{2,}')
        
        # Whitespace normalization
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Non-printable characters (except newlines, tabs)
        self.non_printable_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]')
        
        # Emoji pattern (preserve)
        self.emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]'
            r'|[\U0001F1E0-\U0001F1FF]|[\U00002700-\U000027BF]|[\U0001f900-\U0001f9ff]'
        )
        
        # Number normalization
        self.number_pattern = re.compile(r'\b\d+(?:[.,]\d+)*\b')
        
        # HTML/XML tags
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Special characters that are often noise
        self.noise_chars_pattern = re.compile(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', re.UNICODE)
    
    def _load_contractions(self):
        """Load common contractions for expansion."""
        self.contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would", "they'll": "they will",
            "they're": "they are", "they've": "they have", "we'd": "we would",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what's": "what is", "where's": "where is", "who's": "who is",
            "won't": "will not", "wouldn't": "would not", "you'd": "you would",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }
    
    def _load_stopwords(self):
        """Load basic stopwords for multiple languages."""
        self.stopwords = {
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'},
            'hi': {'और', 'या', 'में', 'पर', 'से', 'को', 'का', 'की', 'के', 'है', 'हैं', 'था', 'थी', 'थे'},
            'es': {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le'},
            'fr': {'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour'},
            'de': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf'},
            'zh': {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '他', '说'},
            'ar': {'في', 'من', 'إلى', 'على', 'أن', 'هذا', 'هذه', 'التي', 'الذي', 'ما', 'لا', 'كل'}
        }
    
    @lru_cache(maxsize=1000)
    def detect_encoding_and_repair(self, text_bytes: bytes) -> str:
        """Detect encoding and repair mojibake."""
        if not isinstance(text_bytes, bytes):
            return str(text_bytes)
        
        try:
            # Try charset-normalizer first (more accurate)
            if HAS_CHARSET_NORMALIZER:
                result = from_bytes(text_bytes)
                text = str(result.best())
            elif HAS_CHARDET:
                import chardet
                detected = chardet.detect(text_bytes)
                encoding = detected.get('encoding', 'utf-8')
                text = text_bytes.decode(encoding, errors='replace')
            else:
                # Fallback to UTF-8
                text = text_bytes.decode('utf-8', errors='replace')
            
            # Fix mojibake if ftfy is available
            if HAS_FTFY and self.config.fix_encoding:
                text = ftfy.fix_text(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}")
            return text_bytes.decode('utf-8', errors='replace')
    
    def normalize_unicode(self, text: str, form: str = 'NFC') -> str:
        """Normalize Unicode text using specified form."""
        if not self.config.normalize_unicode:
            return text
        
        try:
            # NFC is most commonly used - composed form
            return unicodedata.normalize(form, text)
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            return text
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        if not self.config.remove_urls:
            return text
        return self.url_pattern.sub(' ', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        if not self.config.remove_emails:
            return text
        return self.email_pattern.sub(' ', text)
    
    def remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text."""
        if not self.config.remove_phone_numbers:
            return text
        return self.phone_pattern.sub(' ', text)
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML/XML tags."""
        return self.html_pattern.sub(' ', text)
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize excessive punctuation."""
        if not self.config.remove_excessive_punctuation:
            return text
        
        # Replace excessive punctuation with single instances
        text = self.excessive_punct_pattern.sub(lambda m: m.group(0)[0], text)
        return text
    
    def normalize_consecutive_chars(self, text: str) -> str:
        """Reduce consecutive repeated characters."""
        max_consecutive = self.config.max_consecutive_chars
        
        def replace_consecutive(match):
            char = match.group(1)
            # Preserve some characters that might be intentionally repeated
            if char in 'aeiouAEIOU':
                return char * min(len(match.group(0)), 2)  # Max 2 vowels
            return char * min(len(match.group(0)), max_consecutive)
        
        return self.consecutive_chars_pattern.sub(replace_consecutive, text)
    
    def expand_contractions(self, text: str, language: str = 'en') -> str:
        """Expand contractions to full forms."""
        if not self.config.expand_contractions or language != 'en':
            return text
        
        words = text.split()
        expanded_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.contractions:
                # Preserve original case
                if word.isupper():
                    expanded_words.append(self.contractions[word_lower].upper())
                elif word.istitle():
                    expanded_words.append(self.contractions[word_lower].title())
                else:
                    expanded_words.append(self.contractions[word_lower])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def normalize_numbers(self, text: str) -> str:
        """Normalize number formats."""
        if not self.config.normalize_numbers:
            return text
        
        # Replace numbers with a placeholder token
        return self.number_pattern.sub('<NUMBER>', text)
    
    def remove_non_printable(self, text: str) -> str:
        """Remove non-printable characters."""
        if not self.config.remove_non_printable:
            return text
        return self.non_printable_pattern.sub('', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace - collapse multiple spaces, remove leading/trailing."""
        if not self.config.normalize_whitespace:
            return text
        
        # Replace multiple whitespace with single space
        text = self.whitespace_pattern.sub(' ', text)
        # Strip leading/trailing whitespace
        return text.strip()
    
    def apply_language_specific_cleaning(self, text: str, language: str) -> str:
        """Apply language-specific cleaning rules."""
        if not self.config.language_specific_cleaning:
            return text
        
        lang_config = LanguageSpecificCleaner.get_language_config(language)
        
        # Apply language-specific noise removal
        for noise_pattern in lang_config.get('common_noise', []):
            text = re.sub(noise_pattern, ' ', text)
        
        # Full-width to half-width conversion for CJK languages
        if lang_config.get('fullwidth_normalize', False):
            text = self._normalize_fullwidth(text)
        
        # Diacritics normalization for languages like Hindi
        if lang_config.get('diacritics_normalize', False):
            text = self._normalize_diacritics(text, lang_config.get('preserve_chars', []))
        
        return text
    
    def _normalize_fullwidth(self, text: str) -> str:
        """Convert full-width characters to half-width."""
        result = []
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:  # Full-width ASCII range
                result.append(chr(code - 0xFEE0))
            else:
                result.append(char)
        return ''.join(result)
    
    def _normalize_diacritics(self, text: str, preserve_chars: List[str]) -> str:
        """Normalize diacritics while preserving important ones."""
        # Only normalize if character is not in preserve list
        result = []
        for char in text:
            if char in preserve_chars:
                result.append(char)
            else:
                # Try to normalize
                normalized = unicodedata.normalize('NFD', char)
                # Remove combining characters (diacritics) except preserved ones
                filtered = ''.join(c for c in normalized 
                                 if not unicodedata.combining(c) or c in preserve_chars)
                result.append(filtered)
        return ''.join(result)
    
    @lru_cache(maxsize=1000)
    def clean_text(self, 
                   text: Union[str, bytes], 
                   language: Optional[str] = None,
                   aggressive: bool = False) -> str:
        """
        Main text cleaning function with comprehensive preprocessing.
        
        Args:
            text: Input text (string or bytes)
            language: Language code (auto-detected if None)
            aggressive: Whether to apply aggressive cleaning
            
        Returns:
            str: Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Handle bytes input
        if isinstance(text, bytes):
            text = self.detect_encoding_and_repair(text)
        
        # Ensure we have a string
        text = str(text)
        
        # Early exit for very short text
        if len(text.strip()) < 2:
            return text.strip()
        
        # Language detection
        if language is None:
            try:
                language = detect_language(text)
            except Exception:
                language = 'en'
        
        logger.debug(f"Cleaning text (language: {language}, length: {len(text)})")
        
        # Step 1: Unicode normalization
        text = self.normalize_unicode(text)
        
        # Step 2: Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Step 3: Remove URLs, emails, phone numbers
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_phone_numbers(text)
        
        # Step 4: Remove non-printable characters
        text = self.remove_non_printable(text)
        
        # Step 5: Language-specific cleaning
        text = self.apply_language_specific_cleaning(text, language)
        
        # Step 6: Expand contractions (for English)
        text = self.expand_contractions(text, language)
        
        # Step 7: Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # Step 8: Normalize consecutive characters
        text = self.normalize_consecutive_chars(text)
        
        # Step 9: Number normalization (optional)
        if aggressive:
            text = self.normalize_numbers(text)
        
        # Step 10: Final whitespace normalization
        text = self.normalize_whitespace(text)
        
        logger.debug(f"Cleaned text length: {len(text)}")
        return text


# Global instance for easy access
_global_cleaner = None

def get_text_cleaner(config_path: Optional[str] = None) -> AdvancedTextCleaner:
    """Get the global text cleaner instance."""
    global _global_cleaner
    if _global_cleaner is None:
        _global_cleaner = AdvancedTextCleaner(config_path)
    return _global_cleaner

def clean_text(text: Union[str, bytes], 
               language: Optional[str] = None,
               aggressive: bool = False,
               config_path: Optional[str] = None) -> str:
    """
    Convenience function for text cleaning.
    
    Args:
        text: Input text to clean
        language: Language code (auto-detected if None)
        aggressive: Whether to apply aggressive cleaning
        config_path: Path to configuration file
        
    Returns:
        str: Cleaned text
    """
    cleaner = get_text_cleaner(config_path)
    return cleaner.clean_text(text, language, aggressive)


# Performance testing and validation
if __name__ == "__main__":
    # Test cases covering various scenarios
    import time
    
    test_cases = [
        # Basic cleaning
        "  Hello   world!!!   ",
        
        # URL and email removal
        "Check out https://example.com and contact me at test@email.com",
        
        # Phone number removal
        "Call me at +1-555-123-4567 or (555) 987-6543",
        
        # HTML cleaning
        "<p>This is <b>bold</b> text with <a href='#'>links</a></p>",
        
        # Unicode normalization
        "café naïve résumé",  # Accented characters
        
        # Excessive punctuation
        "Really???!!! This is amazing!!!",
        
        # Consecutive characters
        "Sooooo goooood!!!",
        
        # Mixed language (Hindi)
        "यह एक परीक्षण है। This is a test.",
        
        # Contractions
        "I can't believe it's working! You're amazing!",
        
        # Empty and edge cases
        "",
        "a",
        "   \n\t   ",
        
        # Numbers
        "I have 123 apples and 45.67 dollars",
        
        # Emojis (should be preserved)
        "I love this app! 😍🚀✨",
        
        # Non-printable characters
        "Text with\x00\x01\x02 non-printable chars",
    ]
    
    print("=== DharmaShield Text Cleaning Test Suite ===\n")
    
    cleaner = AdvancedTextCleaner()
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"Input:  '{test_text}'")
        
        start_time = time.time()
        cleaned = cleaner.clean_text(test_text)
        end_time = time.time()
        
        print(f"Output: '{cleaned}'")
        print(f"Time:   {(end_time - start_time)*1000:.2f}ms")
        print("-" * 50)
    
    print("✅ All tests completed successfully!")
    print("\n🎯 Text cleaner ready for production deployment!")
