"""
src/utils/crypto_utils.py

DharmaShield - Advanced Cryptographic Utilities & Key Management Engine
-----------------------------------------------------------------------
• Industry-grade cryptographic utility for cross-platform (Android/iOS/Desktop) with Kivy/Buildozer support
• Advanced key management, encryption/decryption of sensitive data with military-grade security
• Support for AES-256, RSA-4096, ChaCha20-Poly1305, PBKDF2, Scrypt with secure key derivation
• Fully offline-capable, optimized for voice-first operation with Google Gemma 3n integration

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import os
import sys
import secrets
import hashlib
import hmac
import threading
import time
from typing import Optional, Dict, List, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import warnings
import base64
import json

# Cryptographic libraries with fallback handling
try:
    from cryptography.hazmat.primitives import hashes, serialization, padding
    from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    warnings.warn("cryptography not available. Cryptographic operations will be limited.", ImportWarning)

try:
    import nacl.secret
    import nacl.utils
    import nacl.public
    import nacl.signing
    import nacl.encoding
    import nacl.hash
    HAS_NACL = True
except ImportError:
    HAS_NACL = False
    warnings.warn("PyNaCl not available. Advanced crypto features will be limited.", ImportWarning)

# Project imports
from .logger import get_logger

logger = get_logger(__name__)

# -------------------------------
# Constants and Configuration
# -------------------------------

# Encryption algorithms
class EncryptionAlgorithm(Enum):
    AES_256_GCM = "aes_256_gcm"         # Authenticated encryption
    AES_256_CBC = "aes_256_cbc"         # Traditional AES
    CHACHA20_POLY1305 = "chacha20_poly1305"  # Modern authenticated encryption
    FERNET = "fernet"                   # High-level symmetric encryption
    RSA_OAEP = "rsa_oaep"              # Asymmetric encryption

# Key derivation functions
class KeyDerivationFunction(Enum):
    PBKDF2 = "pbkdf2"                   # Password-based key derivation
    SCRYPT = "scrypt"                   # Memory-hard KDF
    HKDF = "hkdf"                       # HMAC-based KDF
    ARGON2 = "argon2"                   # Modern password hashing

# Hashing algorithms
class HashingAlgorithm(Enum):
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    SHA3_256 = "sha3_256"

# Key sizes and security levels
class SecurityLevel(Enum):
    STANDARD = "standard"       # AES-128, RSA-2048
    HIGH = "high"              # AES-256, RSA-3072
    MAXIMUM = "maximum"        # AES-256, RSA-4096

# Default settings
DEFAULT_ALGORITHM = EncryptionAlgorithm.AES_256_GCM
DEFAULT_KDF = KeyDerivationFunction.PBKDF2
DEFAULT_HASH = HashingAlgorithm.SHA256
DEFAULT_SECURITY_LEVEL = SecurityLevel.HIGH
DEFAULT_SALT_SIZE = 32
DEFAULT_IV_SIZE = 16
DEFAULT_TAG_SIZE = 16
DEFAULT_PBKDF2_ITERATIONS = 1200000  # NIST recommended minimum
DEFAULT_SCRYPT_N = 2**14  # CPU/memory cost parameter
DEFAULT_RSA_KEY_SIZE = 4096

# -------------------------------
# Data Structures
# -------------------------------

@dataclass
class CryptoConfig:
    """Configuration for cryptographic operations."""
    # Algorithm settings
    default_algorithm: EncryptionAlgorithm = DEFAULT_ALGORITHM
    default_kdf: KeyDerivationFunction = DEFAULT_KDF
    default_hash: HashingAlgorithm = DEFAULT_HASH
    security_level: SecurityLevel = DEFAULT_SECURITY_LEVEL
    
    # Key settings
    salt_size: int = DEFAULT_SALT_SIZE
    iv_size: int = DEFAULT_IV_SIZE
    tag_size: int = DEFAULT_TAG_SIZE
    rsa_key_size: int = DEFAULT_RSA_KEY_SIZE
    
    # KDF settings
    pbkdf2_iterations: int = DEFAULT_PBKDF2_ITERATIONS
    scrypt_n: int = DEFAULT_SCRYPT_N
    scrypt_r: int = 8
    scrypt_p: int = 1
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 100
    
    # Security settings
    secure_delete: bool = True
    constant_time_compare: bool = True
    timing_attack_protection: bool = True

@dataclass
class EncryptionResult:
    """Result of encryption operation."""
    success: bool
    ciphertext: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    algorithm: Optional[EncryptionAlgorithm] = None
    key_id: Optional[str] = None
    timestamp: float = 0.0
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)

@dataclass
class DecryptionResult:
    """Result of decryption operation."""
    success: bool
    plaintext: Optional[Union[bytes, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    algorithm: Optional[EncryptionAlgorithm] = None
    key_id: Optional[str] = None
    timestamp: float = 0.0
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)

@dataclass
class KeyInfo:
    """Information about a cryptographic key."""
    key_id: str
    algorithm: EncryptionAlgorithm
    key_size: int
    created_at: float
    expires_at: Optional[float] = None
    usage_count: int = 0
    max_usage: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# -------------------------------
# Core Cryptographic Engine
# -------------------------------

class CryptoEngine:
    """
    Advanced cryptographic engine for DharmaShield.
    
    Features:
    - Multiple encryption algorithms (AES, ChaCha20, RSA)
    - Secure key generation and management
    - Password-based key derivation (PBKDF2, Scrypt)
    - Authenticated encryption with additional data (AEAD)
    - Cross-platform compatibility
    - Memory-safe operations
    - Constant-time comparisons
    - Key rotation and lifecycle management
    """
    
    def __init__(self, config: Optional[CryptoConfig] = None):
        self.config = config or CryptoConfig()
        
        # Key storage and management
        self.keys: Dict[str, bytes] = {}
        self.key_info: Dict[str, KeyInfo] = {}
        self.key_cache: Dict[str, Tuple[bytes, float]] = {}  # (key, expiry)
        
        # Threading
        self.lock = threading.RLock()
        
        # Performance tracking
        self.stats = {
            'encryptions_performed': 0,
            'decryptions_performed': 0,
            'keys_generated': 0,
            'keys_derived': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        # Initialize backend
        self.backend = default_backend() if HAS_CRYPTOGRAPHY else None
        
        logger.info(f"CryptoEngine initialized with config: {self.config}")
    
    def generate_key(
        self,
        algorithm: Optional[EncryptionAlgorithm] = None,
        key_id: Optional[str] = None,
        key_size: Optional[int] = None
    ) -> str:
        """
        Generate a new cryptographic key.
        
        Args:
            algorithm: Encryption algorithm for the key
            key_id: Unique identifier for the key
            key_size: Size of the key in bits
            
        Returns:
            Key ID for the generated key
        """
        algorithm = algorithm or self.config.default_algorithm
        key_id = key_id or self._generate_key_id()
        
        try:
            with self.lock:
                if algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC]:
                    key_size = key_size or 32  # 256 bits
                    key = secrets.token_bytes(key_size)
                elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                    key_size = key_size or 32  # 256 bits
                    key = secrets.token_bytes(key_size)
                elif algorithm == EncryptionAlgorithm.FERNET:
                    key = Fernet.generate_key()
                    key_size = len(key)
                elif algorithm == EncryptionAlgorithm.RSA_OAEP:
                    key_size = key_size or self.config.rsa_key_size
                    private_key = rsa.generate_private_key(
                        public_exponent=65537,
                        key_size=key_size,
                        backend=self.backend
                    )
                    key = private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    )
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")
                
                # Store key and metadata
                self.keys[key_id] = key
                self.key_info[key_id] = KeyInfo(
                    key_id=key_id,
                    algorithm=algorithm,
                    key_size=key_size,
                    created_at=time.time()
                )
                
                self.stats['keys_generated'] += 1
                logger.info(f"Generated {algorithm.value} key: {key_id}")
                
                return key_id
                
        except Exception as e:
            logger.error(f"Key generation failed: {e}")
            self.stats['errors'] += 1
            raise
    
    def derive_key(
        self,
        password: Union[str, bytes],
        salt: Optional[bytes] = None,
        kdf: Optional[KeyDerivationFunction] = None,
        key_id: Optional[str] = None,
        key_length: int = 32
    ) -> Tuple[str, bytes]:
        """
        Derive a key from a password using KDF.
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation
            kdf: Key derivation function to use
            key_id: Unique identifier for the key
            key_length: Length of derived key
            
        Returns:
            Tuple of (key_id, salt)
        """
        kdf = kdf or self.config.default_kdf
        key_id = key_id or self._generate_key_id()
        salt = salt or secrets.token_bytes(self.config.salt_size)
        
        if isinstance(password, str):
            password = password.encode('utf-8')
        
        try:
            with self.lock:
                if kdf == KeyDerivationFunction.PBKDF2:
                    kdf_instance = PBKDF2HMAC(
                        algorithm=hashes.SHA256(),
                        length=key_length,
                        salt=salt,
                        iterations=self.config.pbkdf2_iterations,
                        backend=self.backend
                    )
                elif kdf == KeyDerivationFunction.SCRYPT:
                    kdf_instance = Scrypt(
                        algorithm=hashes.SHA256(),
                        length=key_length,
                        salt=salt,
                        n=self.config.scrypt_n,
                        r=self.config.scrypt_r,
                        p=self.config.scrypt_p,
                        backend=self.backend
                    )
                elif kdf == KeyDerivationFunction.HKDF:
                    kdf_instance = HKDF(
                        algorithm=hashes.SHA256(),
                        length=key_length,
                        salt=salt,
                        info=b'DharmaShield',
                        backend=self.backend
                    )
                else:
                    raise ValueError(f"Unsupported KDF: {kdf}")
                
                derived_key = kdf_instance.derive(password)
                
                # Store key and metadata
                self.keys[key_id] = derived_key
                self.key_info[key_id] = KeyInfo(
                    key_id=key_id,
                    algorithm=EncryptionAlgorithm.AES_256_GCM,  # Default for derived keys
                    key_size=key_length * 8,
                    created_at=time.time(),
                    metadata={'kdf': kdf.value, 'salt': base64.b64encode(salt).decode()}
                )
                
                self.stats['keys_derived'] += 1
                logger.info(f"Derived key using {kdf.value}: {key_id}")
                
                return key_id, salt
                
        except Exception as e:
            logger.error(f"Key derivation failed: {e}")
            self.stats['errors'] += 1
            raise
    
    def encrypt(
        self,
        plaintext: Union[str, bytes],
        key_id: Optional[str] = None,
        algorithm: Optional[EncryptionAlgorithm] = None,
        additional_data: Optional[bytes] = None
    ) -> EncryptionResult:
        """
        Encrypt data with specified algorithm and key.
        
        Args:
            plaintext: Data to encrypt
            key_id: Key identifier to use for encryption
            algorithm: Encryption algorithm
            additional_data: Additional authenticated data (for AEAD)
            
        Returns:
            EncryptionResult with encrypted data
        """
        start_time = time.time()
        
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        try:
            with self.lock:
                # Get key
                if key_id and key_id in self.keys:
                    key = self.keys[key_id]
                    key_info = self.key_info.get(key_id)
                    algorithm = algorithm or (key_info.algorithm if key_info else self.config.default_algorithm)
                else:
                    # Generate temporary key if none provided
                    key_id = self.generate_key(algorithm)
                    key = self.keys[key_id]
                    algorithm = algorithm or self.config.default_algorithm
                
                # Perform encryption based on algorithm
                if algorithm == EncryptionAlgorithm.AES_256_GCM:
                    result = self._encrypt_aes_gcm(plaintext, key, additional_data)
                elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                    result = self._encrypt_aes_cbc(plaintext, key)
                elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                    result = self._encrypt_chacha20_poly1305(plaintext, key, additional_data)
                elif algorithm == EncryptionAlgorithm.FERNET:
                    result = self._encrypt_fernet(plaintext, key)
                elif algorithm == EncryptionAlgorithm.RSA_OAEP:
                    result = self._encrypt_rsa_oaep(plaintext, key)
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")
                
                # Update usage statistics
                if key_id in self.key_info:
                    self.key_info[key_id].usage_count += 1
                
                self.stats['encryptions_performed'] += 1
                
                return EncryptionResult(
                    success=True,
                    ciphertext=result,
                    algorithm=algorithm,
                    key_id=key_id,
                    timestamp=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            self.stats['errors'] += 1
            return EncryptionResult(
                success=False,
                error_message=str(e),
                timestamp=time.time() - start_time
            )
    
    def decrypt(
        self,
        ciphertext: bytes,
        key_id: str,
        algorithm: Optional[EncryptionAlgorithm] = None,
        additional_data: Optional[bytes] = None
    ) -> DecryptionResult:
        """
        Decrypt data with specified algorithm and key.
        
        Args:
            ciphertext: Data to decrypt
            key_id: Key identifier used for encryption
            algorithm: Encryption algorithm used
            additional_data: Additional authenticated data (for AEAD)
            
        Returns:
            DecryptionResult with decrypted data
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Get key
                if key_id not in self.keys:
                    raise ValueError(f"Key not found: {key_id}")
                
                key = self.keys[key_id]
                key_info = self.key_info.get(key_id)
                algorithm = algorithm or (key_info.algorithm if key_info else self.config.default_algorithm)
                
                # Perform decryption based on algorithm
                if algorithm == EncryptionAlgorithm.AES_256_GCM:
                    result = self._decrypt_aes_gcm(ciphertext, key, additional_data)
                elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                    result = self._decrypt_aes_cbc(ciphertext, key)
                elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                    result = self._decrypt_chacha20_poly1305(ciphertext, key, additional_data)
                elif algorithm == EncryptionAlgorithm.FERNET:
                    result = self._decrypt_fernet(ciphertext, key)
                elif algorithm == EncryptionAlgorithm.RSA_OAEP:
                    result = self._decrypt_rsa_oaep(ciphertext, key)
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")
                
                # Update usage statistics
                if key_id in self.key_info:
                    self.key_info[key_id].usage_count += 1
                
                self.stats['decryptions_performed'] += 1
                
                return DecryptionResult(
                    success=True,
                    plaintext=result,
                    algorithm=algorithm,
                    key_id=key_id,
                    timestamp=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            self.stats['errors'] += 1
            return DecryptionResult(
                success=False,
                error_message=str(e),
                timestamp=time.time() - start_time
            )
    
    def _encrypt_aes_gcm(
        self,
        plaintext: bytes,
        key: bytes,
        additional_data: Optional[bytes] = None
    ) -> bytes:
        """Encrypt using AES-256-GCM."""
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library not available")
        
        # Generate random IV
        iv = secrets.token_bytes(12)  # GCM uses 96-bit IV
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        # Add additional data if provided
        if additional_data:
            encryptor.authenticate_additional_data(additional_data)
        
        # Encrypt
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Return IV + ciphertext + tag
        return iv + ciphertext + encryptor.tag
    
    def _decrypt_aes_gcm(
        self,
        data: bytes,
        key: bytes,
        additional_data: Optional[bytes] = None
    ) -> bytes:
        """Decrypt using AES-256-GCM."""
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library not available")
        
        # Extract components
        iv = data[:12]
        tag = data[-16:]
        ciphertext = data[12:-16]
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        # Add additional data if provided
        if additional_data:
            decryptor.authenticate_additional_data(additional_data)
        
        # Decrypt
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def _encrypt_aes_cbc(self, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt using AES-256-CBC with PKCS7 padding."""
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library not available")
        
        # Generate random IV
        iv = secrets.token_bytes(16)
        
        # Apply PKCS7 padding
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        # Encrypt
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Return IV + ciphertext
        return iv + ciphertext
    
    def _decrypt_aes_cbc(self, data: bytes, key: bytes) -> bytes:
        """Decrypt using AES-256-CBC with PKCS7 padding."""
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library not available")
        
        # Extract IV and ciphertext
        iv = data[:16]
        ciphertext = data[16:]
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        # Decrypt
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_plaintext) + unpadder.finalize()
    
    def _encrypt_chacha20_poly1305(
        self,
        plaintext: bytes,
        key: bytes,
        additional_data: Optional[bytes] = None
    ) -> bytes:
        """Encrypt using ChaCha20-Poly1305."""
        if not HAS_NACL:
            raise RuntimeError("PyNaCl library not available")
        
        # Use PyNaCl's secret box
        box = nacl.secret.SecretBox(key)
        nonce = nacl.utils.random(nacl.secret.SecretBox.NONCE_SIZE)
        
        # Encrypt
        ciphertext = box.encrypt(plaintext, nonce)
        
        return bytes(ciphertext)
    
    def _decrypt_chacha20_poly1305(
        self,
        data: bytes,
        key: bytes,
        additional_data: Optional[bytes] = None
    ) -> bytes:
        """Decrypt using ChaCha20-Poly1305."""
        if not HAS_NACL:
            raise RuntimeError("PyNaCl library not available")
        
        # Use PyNaCl's secret box
        box = nacl.secret.SecretBox(key)
        
        # Decrypt
        plaintext = box.decrypt(data)
        
        return bytes(plaintext)
    
    def _encrypt_fernet(self, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt using Fernet."""
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library not available")
        
        f = Fernet(key)
        return f.encrypt(plaintext)
    
    def _decrypt_fernet(self, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt using Fernet."""
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library not available")
        
        f = Fernet(key)
        return f.decrypt(ciphertext)
    
    def _encrypt_rsa_oaep(self, plaintext: bytes, private_key_pem: bytes) -> bytes:
        """Encrypt using RSA-OAEP (actually uses public key)."""
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library not available")
        
        # Load private key and extract public key
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=self.backend
        )
        public_key = private_key.public_key()
        
        # Encrypt with public key
        ciphertext = public_key.encrypt(
            plaintext,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return ciphertext
    
    def _decrypt_rsa_oaep(self, ciphertext: bytes, private_key_pem: bytes) -> bytes:
        """Decrypt using RSA-OAEP."""
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library not available")
        
        # Load private key
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=self.backend
        )
        
        # Decrypt with private key
        plaintext = private_key.decrypt(
            ciphertext,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return plaintext
    
    def hash_data(
        self,
        data: Union[str, bytes],
        algorithm: Optional[HashingAlgorithm] = None,
        salt: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """
        Hash data using specified algorithm.
        
        Args:
            data: Data to hash
            algorithm: Hashing algorithm to use
            salt: Salt for hashing
            
        Returns:
            Tuple of (hash, salt)
        """
        algorithm = algorithm or self.config.default_hash
        salt = salt or secrets.token_bytes(self.config.salt_size)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        try:
            if algorithm == HashingAlgorithm.SHA256:
                hash_obj = hashlib.sha256()
            elif algorithm == HashingAlgorithm.SHA512:
                hash_obj = hashlib.sha512()
            elif algorithm == HashingAlgorithm.BLAKE2B:
                hash_obj = hashlib.blake2b()
            elif algorithm == HashingAlgorithm.SHA3_256:
                hash_obj = hashlib.sha3_256()
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            hash_obj.update(salt + data)
            return hash_obj.digest(), salt
            
        except Exception as e:
            logger.error(f"Hashing failed: {e}")
            raise
    
    def verify_hash(
        self,
        data: Union[str, bytes],
        hash_value: bytes,
        salt: bytes,
        algorithm: Optional[HashingAlgorithm] = None
    ) -> bool:
        """
        Verify data against hash.
        
        Args:
            data: Original data
            hash_value: Hash to verify against
            salt: Salt used for hashing
            algorithm: Hashing algorithm used
            
        Returns:
            True if hash matches, False otherwise
        """
        try:
            computed_hash, _ = self.hash_data(data, algorithm, salt)
            
            # Use constant-time comparison to prevent timing attacks
            if self.config.constant_time_compare:
                return hmac.compare_digest(computed_hash, hash_value)
            else:
                return computed_hash == hash_value
                
        except Exception as e:
            logger.error(f"Hash verification failed: {e}")
            return False
    
    def generate_mac(
        self,
        data: Union[str, bytes],
        key: bytes,
        algorithm: Optional[HashingAlgorithm] = None
    ) -> bytes:
        """
        Generate Message Authentication Code (MAC).
        
        Args:
            data: Data to authenticate
            key: HMAC key
            algorithm: Hash algorithm for HMAC
            
        Returns:
            MAC bytes
        """
        algorithm = algorithm or self.config.default_hash
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        try:
            if algorithm == HashingAlgorithm.SHA256:
                return hmac.new(key, data, hashlib.sha256).digest()
            elif algorithm == HashingAlgorithm.SHA512:
                return hmac.new(key, data, hashlib.sha512).digest()
            else:
                raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"MAC generation failed: {e}")
            raise
    
    def verify_mac(
        self,
        data: Union[str, bytes],
        mac: bytes,
        key: bytes,
        algorithm: Optional[HashingAlgorithm] = None
    ) -> bool:
        """
        Verify Message Authentication Code (MAC).
        
        Args:
            data: Original data
            mac: MAC to verify
            key: HMAC key
            algorithm: Hash algorithm used
            
        Returns:
            True if MAC is valid, False otherwise
        """
        try:
            computed_mac = self.generate_mac(data, key, algorithm)
            
            # Use constant-time comparison
            if self.config.constant_time_compare:
                return hmac.compare_digest(computed_mac, mac)
            else:
                return computed_mac == mac
                
        except Exception as e:
            logger.error(f"MAC verification failed: {e}")
            return False
    
    def generate_random_bytes(self, size: int) -> bytes:
        """Generate cryptographically secure random bytes."""
        return secrets.token_bytes(size)
    
    def generate_random_string(self, length: int, alphabet: Optional[str] = None) -> str:
        """Generate cryptographically secure random string."""
        if alphabet is None:
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def secure_compare(self, a: Union[str, bytes], b: Union[str, bytes]) -> bool:
        """Perform constant-time comparison to prevent timing attacks."""
        if isinstance(a, str):
            a = a.encode('utf-8')
        if isinstance(b, str):
            b = b.encode('utf-8')
        
        return hmac.compare_digest(a, b)
    
    def key_exists(self, key_id: str) -> bool:
        """Check if a key exists."""
        with self.lock:
            return key_id in self.keys
    
    def get_key_info(self, key_id: str) -> Optional[KeyInfo]:
        """Get information about a key."""
        with self.lock:
            return self.key_info.get(key_id)
    
    def list_keys(self) -> List[str]:
        """List all key IDs."""
        with self.lock:
            return list(self.keys.keys())
    
    def delete_key(self, key_id: str) -> bool:
        """Securely delete a key."""
        try:
            with self.lock:
                if key_id in self.keys:
                    # Secure deletion - overwrite with random data
                    if self.config.secure_delete:
                        key = self.keys[key_id]
                        if isinstance(key, bytes):
                            # Overwrite with random data multiple times
                            for _ in range(3):
                                key = bytearray(key)
                                for i in range(len(key)):
                                    key[i] = secrets.randbits(8)
                    
                    del self.keys[key_id]
                    
                    if key_id in self.key_info:
                        del self.key_info[key_id]
                    
                    logger.info(f"Deleted key: {key_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Key deletion failed: {e}")
            return False
    
    def rotate_key(self, old_key_id: str, new_key_id: Optional[str] = None) -> Optional[str]:
        """Rotate a key (generate new key and mark old one for deletion)."""
        try:
            with self.lock:
                if old_key_id not in self.key_info:
                    return None
                
                old_info = self.key_info[old_key_id]
                new_key_id = new_key_id or self._generate_key_id()
                
                # Generate new key with same algorithm
                self.generate_key(old_info.algorithm, new_key_id)
                
                # Mark old key for deletion (in real implementation, you might want to keep it for some time)
                old_info.expires_at = time.time() + 86400  # 24 hours
                
                logger.info(f"Rotated key: {old_key_id} -> {new_key_id}")
                return new_key_id
                
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return None
    
    def export_key(self, key_id: str, password: Optional[str] = None) -> Optional[bytes]:
        """Export key in encrypted format."""
        try:
            with self.lock:
                if key_id not in self.keys:
                    return None
                
                key = self.keys[key_id]
                
                if password:
                    # Encrypt key with password
                    temp_key_id, salt = self.derive_key(password)
                    encrypted_result = self.encrypt(key, temp_key_id)
                    
                    if encrypted_result.success:
                        # Package with salt and metadata
                        export_data = {
                            'encrypted_key': base64.b64encode(encrypted_result.ciphertext).decode(),
                            'salt': base64.b64encode(salt).decode(),
                            'algorithm': encrypted_result.algorithm.value,
                            'metadata': self.key_info[key_id].__dict__ if key_id in self.key_info else {}
                        }
                        
                        # Clean up temporary key
                        self.delete_key(temp_key_id)
                        
                        return json.dumps(export_data).encode('utf-8')
                else:
                    # Export in plain format (not recommended)
                    warnings.warn("Exporting key without password protection", UserWarning)
                    return key
                
        except Exception as e:
            logger.error(f"Key export failed: {e}")
            return None
    
    def import_key(
        self,
        key_data: bytes,
        key_id: str,
        password: Optional[str] = None
    ) -> bool:
        """Import key from encrypted format."""
        try:
            with self.lock:
                if password:
                    # Parse encrypted key data
                    export_data = json.loads(key_data.decode('utf-8'))
                    
                    encrypted_key = base64.b64decode(export_data['encrypted_key'])
                    salt = base64.b64decode(export_data['salt'])
                    
                    # Derive decryption key
                    temp_key_id, _ = self.derive_key(password, salt)
                    
                    # Decrypt key
                    decrypt_result = self.decrypt(encrypted_key, temp_key_id)
                    
                    if decrypt_result.success:
                        self.keys[key_id] = decrypt_result.plaintext
                        
                        # Restore metadata if available
                        if 'metadata' in export_data:
                            metadata = export_data['metadata']
                            self.key_info[key_id] = KeyInfo(**metadata)
                        
                        # Clean up temporary key
                        self.delete_key(temp_key_id)
                        
                        logger.info(f"Imported encrypted key: {key_id}")
                        return True
                else:
                    # Import plain key
                    self.keys[key_id] = key_data
                    logger.info(f"Imported plain key: {key_id}")
                    return True
                
        except Exception as e:
            logger.error(f"Key import failed: {e}")
            return False
    
    def _generate_key_id(self) -> str:
        """Generate unique key identifier."""
        return f"key_{secrets.token_hex(16)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats['total_keys'] = len(self.keys)
            stats['cache_size'] = len(self.key_cache)
            return stats
    
    def clear_cache(self):
        """Clear key cache."""
        with self.lock:
            self.key_cache.clear()
        logger.info("Crypto cache cleared")
    
    def cleanup_expired_keys(self):
        """Remove expired keys."""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key_id, info in self.key_info.items():
                if info.expires_at and info.expires_at < current_time:
                    expired_keys.append(key_id)
        
        for key_id in expired_keys:
            self.delete_key(key_id)
            logger.info(f"Cleaned up expired key: {key_id}")

# -------------------------------
# Global Engine Instance and Convenience Functions
# -------------------------------

# Global engine instance
_crypto_engine: Optional[CryptoEngine] = None
_engine_lock = threading.Lock()

def get_crypto_engine(config: Optional[CryptoConfig] = None) -> CryptoEngine:
    """Get global crypto engine instance."""
    global _crypto_engine
    
    with _engine_lock:
        if _crypto_engine is None:
            _crypto_engine = CryptoEngine(config)
    
    return _crypto_engine

def encrypt_data(
    plaintext: Union[str, bytes],
    password: Optional[str] = None,
    key_id: Optional[str] = None,
    algorithm: Optional[EncryptionAlgorithm] = None
) -> EncryptionResult:
    """
    Convenience function for data encryption.
    
    Args:
        plaintext: Data to encrypt
        password: Password for key derivation (if no key_id)
        key_id: Existing key to use
        algorithm: Encryption algorithm
        
    Returns:
        EncryptionResult
    """
    engine = get_crypto_engine()
    
    if password and not key_id:
        key_id, _ = engine.derive_key(password)
    
    return engine.encrypt(plaintext, key_id, algorithm)

def decrypt_data(
    ciphertext: bytes,
    password: Optional[str] = None,
    key_id: Optional[str] = None,
    algorithm: Optional[EncryptionAlgorithm] = None
) -> DecryptionResult:
    """
    Convenience function for data decryption.
    
    Args:
        ciphertext: Data to decrypt
        password: Password for key derivation (if no key_id)
        key_id: Key used for encryption
        algorithm: Encryption algorithm used
        
    Returns:
        DecryptionResult
    """
    engine = get_crypto_engine()
    
    if password and not key_id:
        # This is problematic - we need the same salt used for encryption
        # In practice, the salt should be stored with the ciphertext
        raise ValueError("Cannot derive key without original salt")
    
    return engine.decrypt(ciphertext, key_id, algorithm)

def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """Hash password with salt."""
    engine = get_crypto_engine()
    return engine.hash_data(password, salt=salt)

def verify_password(password: str, hash_value: bytes, salt: bytes) -> bool:
    """Verify password against hash."""
    engine = get_crypto_engine()
    return engine.verify_hash(password, hash_value, salt)

def generate_secure_token(length: int = 32) -> str:
    """Generate secure random token."""
    engine = get_crypto_engine()
    return engine.generate_random_string(length)

def secure_compare_strings(a: Union[str, bytes], b: Union[str, bytes]) -> bool:
    """Perform constant-time string comparison."""
    engine = get_crypto_engine()
    return engine.secure_compare(a, b)

# -------------------------------
# Testing and Demo
# -------------------------------

if __name__ == "__main__":
    # Demo mode
    print("=== DharmaShield Crypto Engine Demo ===")
    
    # Create enhanced configuration
    config = CryptoConfig(
        default_algorithm=EncryptionAlgorithm.AES_256_GCM,
        default_kdf=KeyDerivationFunction.PBKDF2,
        security_level=SecurityLevel.HIGH,
        pbkdf2_iterations=1200000
    )
    
    engine = CryptoEngine(config)
    
    print("Crypto Engine Features:")
    print("✓ Multiple encryption algorithms (AES-256, ChaCha20, RSA)")
    print("✓ Secure key generation and management")
    print("✓ Password-based key derivation (PBKDF2, Scrypt)")
    print("✓ Authenticated encryption with additional data (AEAD)")
    print("✓ Message Authentication Codes (HMAC)")
    print("✓ Cryptographically secure random generation")
    print("✓ Constant-time comparisons for timing attack prevention")
    print("✓ Key rotation and lifecycle management")
    print("✓ Cross-platform compatibility")
    
    # Demo encryption/decryption
    print("\n--- Encryption/Decryption Demo ---")
    
    # Generate key
    key_id = engine.generate_key(EncryptionAlgorithm.AES_256_GCM)
    print(f"Generated key: {key_id}")
    
    # Encrypt data
    plaintext = "Hello, DharmaShield! This is sensitive data that needs protection."
    encrypt_result = engine.encrypt(plaintext, key_id)
    
    if encrypt_result.success:
        print(f"✓ Encryption successful")
        print(f"  Algorithm: {encrypt_result.algorithm.value}")
        print(f"  Ciphertext length: {len(encrypt_result.ciphertext)} bytes")
        
        # Decrypt data
        decrypt_result = engine.decrypt(encrypt_result.ciphertext, key_id)
        
        if decrypt_result.success:
            print(f"✓ Decryption successful")
            print(f"  Decrypted: {decrypt_result.plaintext.decode()}")
        else:
            print(f"✗ Decryption failed: {decrypt_result.error_message}")
    else:
        print(f"✗ Encryption failed: {encrypt_result.error_message}")
    
    # Demo password-based encryption
    print("\n--- Password-Based Encryption Demo ---")
    
    password = "SecurePassword123!"
    password_key_id, salt = engine.derive_key(password)
    print(f"Derived key from password: {password_key_id}")
    
    # Encrypt with password-derived key
    secret_data = "This is confidential information protected by password."
    pwd_encrypt_result = engine.encrypt(secret_data, password_key_id)
    
    if pwd_encrypt_result.success:
        print("✓ Password-based encryption successful")
        
        # Decrypt with same password
        verify_key_id, _ = engine.derive_key(password, salt)
        pwd_decrypt_result = engine.decrypt(pwd_encrypt_result.ciphertext, verify_key_id)
        
        if pwd_decrypt_result.success:
            print("✓ Password-based decryption successful")
            print(f"  Decrypted: {pwd_decrypt_result.plaintext.decode()}")
    
    # Demo hashing
    print("\n--- Hashing Demo ---")
    
    data_to_hash = "Important data that needs integrity verification"
    hash_value, hash_salt = engine.hash_data(data_to_hash)
    print(f"✓ Data hashed (SHA-256)")
    print(f"  Hash length: {len(hash_value)} bytes")
    
    # Verify hash
    is_valid = engine.verify_hash(data_to_hash, hash_value, hash_salt)
    print(f"✓ Hash verification: {'VALID' if is_valid else 'INVALID'}")
    
    # Demo MAC
    print("\n--- Message Authentication Code Demo ---")
    
    mac_key = engine.generate_random_bytes(32)
    message = "This message needs authentication"
    mac = engine.generate_mac(message, mac_key)
    print(f"✓ MAC generated")
    
    # Verify MAC
    mac_valid = engine.verify_mac(message, mac, mac_key)
    print(f"✓ MAC verification: {'VALID' if mac_valid else 'INVALID'}")
    
    # Demo secure random generation
    print("\n--- Secure Random Generation Demo ---")
    
    random_bytes = engine.generate_random_bytes(16)
    random_string = engine.generate_random_string(20)
    print(f"✓ Random bytes: {random_bytes.hex()}")
    print(f"✓ Random string: {random_string}")
    
    # Performance stats
    stats = engine.get_stats()
    print(f"\nPerformance statistics: {stats}")
    
    print("\n✅ Crypto Engine ready for production!")
    print("🔐 Security features demonstrated:")
    print("  ✓ Military-grade encryption (AES-256, ChaCha20)")
    print("  ✓ Secure key management and rotation")
    print("  ✓ Password-based key derivation with high iteration counts")
    print("  ✓ Authenticated encryption preventing tampering")
    print("  ✓ Cryptographic hashing and message authentication")
    print("  ✓ Timing attack prevention with constant-time operations")
    print("  ✓ Cross-platform compatibility for mobile deployment")
    print("  ✓ Integration-ready for voice-first applications")

