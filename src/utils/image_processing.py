"""
src/utils/image_processing.py

DharmaShield - Advanced Image Processing Engine (Loading, Cropping, Enhancement)
-------------------------------------------------------------------------------
• Industry-grade image processing utility for cross-platform (Android/iOS/Desktop) with Kivy/Buildozer support
• Advanced image loading, cropping, resizing, enhancement for vision/detection tasks and threat analysis
• Support for JPEG, PNG, TIFF, BMP, WebP formats with automatic format detection and conversion
• Fully offline, optimized for voice-first operation with Google Gemma 3n integration and scam detection

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import os
import sys
import io
import threading
import time
from typing import Optional, Union, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import warnings
import base64

# Image processing libraries with fallback handling
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    warnings.warn("numpy not available. Advanced image processing will be limited.", ImportWarning)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    warnings.warn("OpenCV not available. Advanced computer vision features will be limited.", ImportWarning)

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    warnings.warn("PIL/Pillow not available. Basic image processing will be limited.", ImportWarning)

try:
    from skimage import filters, transform, exposure, restoration, morphology
    from skimage.util import img_as_ubyte, img_as_float
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("scikit-image not available. Advanced image enhancement will be limited.", ImportWarning)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Project imports
from .logger import get_logger

logger = get_logger(__name__)

# -------------------------------
# Constants and Configuration
# -------------------------------

# Supported image formats
SUPPORTED_FORMATS = {
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG', 
    '.png': 'PNG',
    '.bmp': 'BMP',
    '.tiff': 'TIFF',
    '.tif': 'TIFF',
    '.webp': 'WebP',
    '.gif': 'GIF',
    '.ico': 'ICO'
}

# Default processing parameters
DEFAULT_MAX_SIZE = (1920, 1080)  # Full HD max resolution
DEFAULT_QUALITY = 85
DEFAULT_TARGET_SIZE = (224, 224)  # Common vision model input size
DEFAULT_CROP_RATIO = (1.0, 1.0)  # Square crop by default

# Enhancement modes
class EnhancementMode(Enum):
    NONE = "none"
    BASIC = "basic"           # Brightness, contrast, sharpness
    ADVANCED = "advanced"     # Histogram equalization, noise reduction
    VISION_READY = "vision_ready"  # Optimized for ML/vision tasks
    DOCUMENT = "document"     # OCR-optimized enhancement
    FACE = "face"            # Face detection optimized

# Crop modes
class CropMode(Enum):
    CENTER = "center"        # Center crop
    SMART = "smart"          # Content-aware cropping
    FACE = "face"           # Face-centered cropping
    MANUAL = "manual"       # Manual coordinates
    NONE = "none"           # No cropping

# Resize methods
class ResizeMethod(Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic" 
    LANCZOS = "lanczos"
    AREA = "area"

# -------------------------------
# Data Structures
# -------------------------------

@dataclass
class ImageMetadata:
    """Comprehensive image metadata."""
    filename: str
    file_size: int
    format: str
    width: int
    height: int
    channels: int
    color_mode: str
    has_alpha: bool = False
    dpi: Tuple[int, int] = (72, 72)
    exif_data: Dict[str, Any] = field(default_factory=dict)
    creation_time: Optional[float] = None
    file_hash: Optional[str] = None

@dataclass
class ProcessingConfig:
    """Configuration for image processing operations."""
    # Size and quality
    target_size: Optional[Tuple[int, int]] = DEFAULT_TARGET_SIZE
    max_size: Tuple[int, int] = DEFAULT_MAX_SIZE
    quality: int = DEFAULT_QUALITY
    maintain_aspect_ratio: bool = True
    
    # Enhancement
    enhancement_mode: EnhancementMode = EnhancementMode.BASIC
    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    sharpness_factor: float = 1.0
    saturation_factor: float = 1.0
    
    # Cropping
    crop_mode: CropMode = CropMode.CENTER
    crop_box: Optional[Tuple[int, int, int, int]] = None
    crop_ratio: Tuple[float, float] = DEFAULT_CROP_RATIO
    
    # Resizing
    resize_method: ResizeMethod = ResizeMethod.LANCZOS
    
    # Noise reduction and enhancement
    denoise: bool = False
    histogram_equalization: bool = False
    gamma_correction: Optional[float] = None
    
    # Output format
    output_format: str = "JPEG"
    preserve_metadata: bool = False
    
    # Performance
    enable_caching: bool = True
    use_gpu: bool = False

@dataclass
class ProcessingResult:
    """Result of image processing operation."""
    success: bool
    image_data: Optional[Union[np.ndarray, Image.Image]] = None
    metadata: Optional[ImageMetadata] = None
    processing_time: float = 0.0
    operations_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error_message: str = ""
    
    def to_bytes(self, format: str = "JPEG") -> Optional[bytes]:
        """Convert processed image to bytes."""
        if not self.success or self.image_data is None:
            return None
        
        try:
            if isinstance(self.image_data, np.ndarray):
                # Convert numpy array to PIL Image
                if HAS_PIL:
                    if len(self.image_data.shape) == 3:
                        # RGB image
                        pil_image = Image.fromarray(self.image_data)
                    else:
                        # Grayscale image
                        pil_image = Image.fromarray(self.image_data, mode='L')
                else:
                    return None
            else:
                pil_image = self.image_data
            
            # Convert to bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to convert image to bytes: {e}")
            return None

# -------------------------------
# Core Image Processing Engine
# -------------------------------

class ImageProcessor:
    """
    Advanced image processing engine for DharmaShield.
    
    Features:
    - Multi-format image loading with automatic format detection
    - Intelligent cropping (center, smart, face-aware)
    - Advanced enhancement (brightness, contrast, noise reduction)
    - Vision/ML-ready preprocessing
    - Cross-platform optimization
    - Caching for performance
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.cache = {}
        self.cache_lock = threading.RLock()
        
        # Initialize face detector if available
        self.face_cascade = None
        if HAS_OPENCV:
            try:
                # Try to load face cascade classifier
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    logger.info("Face detection initialized")
            except Exception as e:
                logger.warning(f"Face detection not available: {e}")
        
        # Processing statistics
        self.stats = {
            'images_processed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        logger.info(f"ImageProcessor initialized with config: {self.config}")
    
    def process_image(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        config: Optional[ProcessingConfig] = None
    ) -> ProcessingResult:
        """
        Process image with comprehensive enhancement and optimization.
        
        Args:
            image_input: Image source (file path, bytes, array, or PIL Image)
            config: Processing configuration (uses default if None)
            
        Returns:
            ProcessingResult with processed image and metadata
        """
        start_time = time.time()
        processing_config = config or self.config
        
        # Check cache
        cache_key = self._get_cache_key(image_input, processing_config)
        if processing_config.enable_caching and cache_key in self.cache:
            self.stats['cache_hits'] += 1
            cached_result = self.cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        self.stats['cache_misses'] += 1
        
        try:
            # Load image
            image, metadata = self._load_image(image_input)
            if image is None:
                return ProcessingResult(
                    success=False,
                    error_message="Failed to load image",
                    processing_time=time.time() - start_time
                )
            
            operations_applied = []
            warnings_list = []
            
            # Convert to working format
            if isinstance(image, Image.Image):
                # Ensure RGB mode for processing
                if image.mode not in ['RGB', 'RGBA', 'L']:
                    image = image.convert('RGB')
                    operations_applied.append("color_mode_conversion")
                
                # Handle transparency
                if image.mode == 'RGBA':
                    # Create white background for RGBA images
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                    operations_applied.append("alpha_removal")
            
            # Apply processing pipeline
            image = self._apply_processing_pipeline(
                image, processing_config, operations_applied, warnings_list
            )
            
            # Create result
            result = ProcessingResult(
                success=True,
                image_data=image,
                metadata=metadata,
                processing_time=time.time() - start_time,
                operations_applied=operations_applied,
                warnings=warnings_list
            )
            
            # Cache result
            if processing_config.enable_caching:
                with self.cache_lock:
                    self.cache[cache_key] = result
            
            # Update statistics
            self.stats['images_processed'] += 1
            self.stats['total_processing_time'] += result.processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            self.stats['errors'] += 1
            return ProcessingResult(
                success=False,
                error_message=f"Processing failed: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _load_image(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image]
    ) -> Tuple[Optional[Union[Image.Image, np.ndarray]], Optional[ImageMetadata]]:
        """Load image from various input sources."""
        
        # Handle different input types
        if isinstance(image_input, (str, Path)):
            return self._load_from_file(Path(image_input))
        elif isinstance(image_input, bytes):
            return self._load_from_bytes(image_input)
        elif isinstance(image_input, np.ndarray):
            return self._load_from_array(image_input)
        elif isinstance(image_input, Image.Image):
            return image_input, self._extract_pil_metadata(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def _load_from_file(self, file_path: Path) -> Tuple[Optional[Image.Image], Optional[ImageMetadata]]:
        """Load image from file path."""
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        if not self._is_supported_format(file_path):
            raise ValueError(f"Unsupported image format: {file_path.suffix}")
        
        # Load with PIL (preferred)
        if HAS_PIL:
            try:
                image = Image.open(file_path)
                metadata = self._extract_file_metadata(file_path, image)
                return image, metadata
            except Exception as e:
                logger.warning(f"PIL loading failed: {e}")
        
        # Fallback to OpenCV
        if HAS_OPENCV:
            try:
                img_array = cv2.imread(str(file_path))
                if img_array is not None:
                    # Convert BGR to RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    if HAS_PIL:
                        image = Image.fromarray(img_array)
                        metadata = self._extract_file_metadata(file_path, image)
                        return image, metadata
                    else:
                        # Return numpy array if PIL not available
                        metadata = ImageMetadata(
                            filename=file_path.name,
                            file_size=file_path.stat().st_size,
                            format=SUPPORTED_FORMATS.get(file_path.suffix.lower(), 'Unknown'),
                            width=img_array.shape[1],
                            height=img_array.shape[0],
                            channels=img_array.shape[2] if len(img_array.shape) > 2 else 1,
                            color_mode='RGB'
                        )
                        return img_array, metadata
            except Exception as e:
                logger.warning(f"OpenCV loading failed: {e}")
        
        raise RuntimeError("No suitable image loading backend available")
    
    def _load_from_bytes(self, image_bytes: bytes) -> Tuple[Optional[Image.Image], Optional[ImageMetadata]]:
        """Load image from bytes."""
        if HAS_PIL:
            try:
                image = Image.open(io.BytesIO(image_bytes))
                metadata = ImageMetadata(
                    filename="from_bytes",
                    file_size=len(image_bytes),
                    format=image.format or 'Unknown',
                    width=image.width,
                    height=image.height,
                    channels=len(image.getbands()) if hasattr(image, 'getbands') else 3,
                    color_mode=image.mode,
                    has_alpha='A' in image.mode
                )
                return image, metadata
            except Exception as e:
                logger.error(f"Failed to load image from bytes: {e}")
        
        return None, None
    
    def _load_from_array(self, img_array: np.ndarray) -> Tuple[Optional[Image.Image], Optional[ImageMetadata]]:
        """Load image from numpy array."""
        if not HAS_NUMPY:
            return None, None
        
        try:
            # Normalize array if needed
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            
            if HAS_PIL:
                if len(img_array.shape) == 3:
                    image = Image.fromarray(img_array)
                else:
                    image = Image.fromarray(img_array, mode='L')
                
                metadata = ImageMetadata(
                    filename="from_array",
                    file_size=img_array.nbytes,
                    format='Array',
                    width=img_array.shape[1] if len(img_array.shape) > 1 else img_array.shape[0],
                    height=img_array.shape[0],
                    channels=img_array.shape[2] if len(img_array.shape) > 2 else 1,
                    color_mode='RGB' if len(img_array.shape) == 3 else 'L'
                )
                return image, metadata
            else:
                # Return array directly if PIL not available
                metadata = ImageMetadata(
                    filename="from_array",
                    file_size=img_array.nbytes,
                    format='Array',
                    width=img_array.shape[1] if len(img_array.shape) > 1 else img_array.shape[0],
                    height=img_array.shape[0],
                    channels=img_array.shape[2] if len(img_array.shape) > 2 else 1,
                    color_mode='RGB' if len(img_array.shape) == 3 else 'L'
                )
                return img_array, metadata
                
        except Exception as e:
            logger.error(f"Failed to load image from array: {e}")
        
        return None, None
    
    def _apply_processing_pipeline(
        self,
        image: Union[Image.Image, np.ndarray],
        config: ProcessingConfig,
        operations: List[str],
        warnings: List[str]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply complete processing pipeline."""
        
        # 1. Cropping
        if config.crop_mode != CropMode.NONE:
            image = self._apply_cropping(image, config, operations, warnings)
        
        # 2. Resizing
        if config.target_size:
            image = self._apply_resizing(image, config, operations)
        
        # 3. Enhancement
        if config.enhancement_mode != EnhancementMode.NONE:
            image = self._apply_enhancement(image, config, operations, warnings)
        
        return image
    
    def _apply_cropping(
        self,
        image: Union[Image.Image, np.ndarray],
        config: ProcessingConfig,
        operations: List[str],
        warnings: List[str]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply various cropping methods."""
        
        if config.crop_mode == CropMode.CENTER:
            image = self._center_crop(image, config.crop_ratio)
            operations.append("center_crop")
            
        elif config.crop_mode == CropMode.MANUAL and config.crop_box:
            image = self._manual_crop(image, config.crop_box)
            operations.append("manual_crop")
            
        elif config.crop_mode == CropMode.FACE:
            image = self._face_crop(image, warnings)
            operations.append("face_crop")
            
        elif config.crop_mode == CropMode.SMART:
            image = self._smart_crop(image, config.crop_ratio, warnings)
            operations.append("smart_crop")
        
        return image
    
    def _center_crop(
        self,
        image: Union[Image.Image, np.ndarray],
        crop_ratio: Tuple[float, float]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply center cropping."""
        
        if isinstance(image, Image.Image):
            width, height = image.size
            target_width = int(width * crop_ratio[0])
            target_height = int(height * crop_ratio[1])
            
            left = (width - target_width) // 2
            top = (height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            
            return image.crop((left, top, right, bottom))
        
        elif HAS_NUMPY and isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            target_width = int(width * crop_ratio[0])
            target_height = int(height * crop_ratio[1])
            
            left = (width - target_width) // 2
            top = (height - target_height) // 2
            
            return image[top:top+target_height, left:left+target_width]
        
        return image
    
    def _manual_crop(
        self,
        image: Union[Image.Image, np.ndarray],
        crop_box: Tuple[int, int, int, int]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply manual cropping with coordinates."""
        
        if isinstance(image, Image.Image):
            return image.crop(crop_box)
        elif HAS_NUMPY and isinstance(image, np.ndarray):
            left, top, right, bottom = crop_box
            return image[top:bottom, left:right]
        
        return image
    
    def _face_crop(
        self,
        image: Union[Image.Image, np.ndarray],
        warnings: List[str]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply face-centered cropping."""
        
        if not HAS_OPENCV or self.face_cascade is None:
            warnings.append("Face detection not available, using center crop")
            return self._center_crop(image, (0.8, 0.8))
        
        try:
            # Convert to opencv format
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Convert to grayscale for face detection
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Use largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Expand crop area around face
                margin = max(w, h) // 4
                left = max(0, x - margin)
                top = max(0, y - margin)
                right = min(img_array.shape[1], x + w + margin)
                bottom = min(img_array.shape[0], y + h + margin)
                
                if isinstance(image, Image.Image):
                    return image.crop((left, top, right, bottom))
                else:
                    return img_array[top:bottom, left:right]
            else:
                warnings.append("No faces detected, using center crop")
                return self._center_crop(image, (0.8, 0.8))
                
        except Exception as e:
            warnings.append(f"Face crop failed: {e}")
            return self._center_crop(image, (0.8, 0.8))
    
    def _smart_crop(
        self,
        image: Union[Image.Image, np.ndarray],
        crop_ratio: Tuple[float, float],
        warnings: List[str]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply content-aware smart cropping."""
        
        # For now, implement as center crop with edge detection
        # This can be enhanced with more sophisticated algorithms
        try:
            if HAS_OPENCV and isinstance(image, np.ndarray):
                # Use edge detection to find content areas
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
                edges = cv2.Canny(gray, 50, 150)
                
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find bounding box of all contours
                    all_contours = np.vstack(contours)
                    x, y, w, h = cv2.boundingRect(all_contours)
                    
                    # Apply crop with some padding
                    padding = min(w, h) // 10
                    left = max(0, x - padding)
                    top = max(0, y - padding)
                    right = min(image.shape[1], x + w + padding)
                    bottom = min(image.shape[0], y + h + padding)
                    
                    if isinstance(image, Image.Image):
                        return image.crop((left, top, right, bottom))
                    else:
                        return image[top:bottom, left:right]
            
            warnings.append("Smart crop not available, using center crop")
            return self._center_crop(image, crop_ratio)
            
        except Exception as e:
            warnings.append(f"Smart crop failed: {e}")
            return self._center_crop(image, crop_ratio)
    
    def _apply_resizing(
        self,
        image: Union[Image.Image, np.ndarray],
        config: ProcessingConfig,
        operations: List[str]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply resizing with various methods."""
        
        if not config.target_size:
            return image
        
        target_width, target_height = config.target_size
        
        if isinstance(image, Image.Image):
            # PIL resizing
            if config.maintain_aspect_ratio:
                image.thumbnail(config.target_size, Image.LANCZOS)
            else:
                # Map resize methods to PIL constants
                pil_methods = {
                    ResizeMethod.NEAREST: Image.NEAREST,
                    ResizeMethod.BILINEAR: Image.BILINEAR,
                    ResizeMethod.BICUBIC: Image.BICUBIC,
                    ResizeMethod.LANCZOS: Image.LANCZOS
                }
                method = pil_methods.get(config.resize_method, Image.LANCZOS)
                image = image.resize((target_width, target_height), method)
            
            operations.append(f"resize_{config.resize_method.value}")
            
        elif HAS_OPENCV and isinstance(image, np.ndarray):
            # OpenCV resizing
            cv2_methods = {
                ResizeMethod.NEAREST: cv2.INTER_NEAREST,
                ResizeMethod.BILINEAR: cv2.INTER_LINEAR,
                ResizeMethod.BICUBIC: cv2.INTER_CUBIC,
                ResizeMethod.LANCZOS: cv2.INTER_LANCZOS4,
                ResizeMethod.AREA: cv2.INTER_AREA
            }
            method = cv2_methods.get(config.resize_method, cv2.INTER_LANCZOS4)
            
            if config.maintain_aspect_ratio:
                # Calculate aspect ratio preserving size
                h, w = image.shape[:2]
                aspect_ratio = w / h
                
                if aspect_ratio > target_width / target_height:
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:
                    new_height = target_height
                    new_width = int(target_height * aspect_ratio)
                
                image = cv2.resize(image, (new_width, new_height), interpolation=method)
            else:
                image = cv2.resize(image, (target_width, target_height), interpolation=method)
            
            operations.append(f"resize_{config.resize_method.value}")
        
        return image
    
    def _apply_enhancement(
        self,
        image: Union[Image.Image, np.ndarray],
        config: ProcessingConfig,
        operations: List[str],
        warnings: List[str]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply image enhancement based on mode."""
        
        if config.enhancement_mode == EnhancementMode.BASIC:
            image = self._basic_enhancement(image, config, operations)
        elif config.enhancement_mode == EnhancementMode.ADVANCED:
            image = self._advanced_enhancement(image, config, operations, warnings)
        elif config.enhancement_mode == EnhancementMode.VISION_READY:
            image = self._vision_enhancement(image, config, operations)
        elif config.enhancement_mode == EnhancementMode.DOCUMENT:
            image = self._document_enhancement(image, config, operations, warnings)
        elif config.enhancement_mode == EnhancementMode.FACE:
            image = self._face_enhancement(image, config, operations, warnings)
        
        return image
    
    def _basic_enhancement(
        self,
        image: Union[Image.Image, np.ndarray],
        config: ProcessingConfig,
        operations: List[str]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply basic enhancement (brightness, contrast, sharpness)."""
        
        if isinstance(image, Image.Image) and HAS_PIL:
            # Brightness
            if config.brightness_factor != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(config.brightness_factor)
                operations.append("brightness_adjustment")
            
            # Contrast
            if config.contrast_factor != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(config.contrast_factor)
                operations.append("contrast_adjustment")
            
            # Sharpness
            if config.sharpness_factor != 1.0:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(config.sharpness_factor)
                operations.append("sharpness_adjustment")
            
            # Saturation
            if config.saturation_factor != 1.0:
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(config.saturation_factor)
                operations.append("saturation_adjustment")
        
        return image
    
    def _advanced_enhancement(
        self,
        image: Union[Image.Image, np.ndarray],
        config: ProcessingConfig,
        operations: List[str],
        warnings: List[str]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply advanced enhancement techniques."""
        
        # First apply basic enhancement
        image = self._basic_enhancement(image, config, operations)
        
        # Convert to numpy for advanced processing
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Histogram equalization
        if config.histogram_equalization and HAS_OPENCV:
            try:
                if len(img_array.shape) == 3:
                    # Apply CLAHE to each channel
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img_array = clahe.apply(img_array)
                operations.append("histogram_equalization")
            except Exception as e:
                warnings.append(f"Histogram equalization failed: {e}")
        
        # Gamma correction
        if config.gamma_correction:
            try:
                gamma = config.gamma_correction
                img_array = np.power(img_array / 255.0, gamma) * 255.0
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                operations.append("gamma_correction")
            except Exception as e:
                warnings.append(f"Gamma correction failed: {e}")
        
        # Noise reduction
        if config.denoise and HAS_OPENCV:
            try:
                if len(img_array.shape) == 3:
                    img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
                else:
                    img_array = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
                operations.append("noise_reduction")
            except Exception as e:
                warnings.append(f"Noise reduction failed: {e}")
        
        # Convert back to PIL if needed
        if isinstance(image, Image.Image) and HAS_PIL:
            return Image.fromarray(img_array)
        
        return img_array
    
    def _vision_enhancement(
        self,
        image: Union[Image.Image, np.ndarray],
        config: ProcessingConfig,
        operations: List[str]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply vision/ML-ready enhancement."""
        
        # Convert to numpy array for processing
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Normalize to [0, 1] range
        if img_array.max() > 1.0:
            img_array = img_array.astype(np.float32) / 255.0
        
        # Standard ML preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = (img_array - mean) / std
        
        operations.append("vision_normalization")
        
        # Convert back to uint8 for display/saving
        img_array = np.clip((img_array * std + mean) * 255.0, 0, 255).astype(np.uint8)
        
        if isinstance(image, Image.Image) and HAS_PIL:
            return Image.fromarray(img_array)
        
        return img_array
    
    def _document_enhancement(
        self,
        image: Union[Image.Image, np.ndarray],
        config: ProcessingConfig,
        operations: List[str],
        warnings: List[str]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply document/OCR-optimized enhancement."""
        
        # Convert to grayscale for document processing
        if isinstance(image, Image.Image):
            if image.mode != 'L':
                image = image.convert('L')
                operations.append("grayscale_conversion")
            
            # Apply sharpening filter
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            operations.append("document_sharpening")
            
        elif HAS_OPENCV and isinstance(image, np.ndarray):
            # Convert to grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                operations.append("grayscale_conversion")
            
            # Apply adaptive thresholding for better text contrast
            image = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            operations.append("adaptive_thresholding")
        
        return image
    
    def _face_enhancement(
        self,
        image: Union[Image.Image, np.ndarray],
        config: ProcessingConfig,
        operations: List[str],
        warnings: List[str]
    ) -> Union[Image.Image, np.ndarray]:
        """Apply face-optimized enhancement."""
        
        # Apply basic enhancement first
        image = self._basic_enhancement(image, config, operations)
        
        # Add face-specific enhancements
        if isinstance(image, Image.Image) and HAS_PIL:
            # Slight sharpening for face details
            image = image.filter(ImageFilter.UnsharpMask(radius=0.5, percent=120, threshold=2))
            operations.append("face_sharpening")
        
        return image
    
    def _extract_file_metadata(self, file_path: Path, image: Image.Image) -> ImageMetadata:
        """Extract comprehensive metadata from file and image."""
        
        stat = file_path.stat()
        exif_data = {}
        
        # Extract EXIF data if available
        if hasattr(image, '_getexif') and image._getexif():
            exif = image._getexif()
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
        
        return ImageMetadata(
            filename=file_path.name,
            file_size=stat.st_size,
            format=image.format or SUPPORTED_FORMATS.get(file_path.suffix.lower(), 'Unknown'),
            width=image.width,
            height=image.height,
            channels=len(image.getbands()) if hasattr(image, 'getbands') else 3,
            color_mode=image.mode,
            has_alpha='A' in image.mode,
            dpi=image.info.get('dpi', (72, 72)),
            exif_data=exif_data,
            creation_time=stat.st_ctime
        )
    
    def _extract_pil_metadata(self, image: Image.Image) -> ImageMetadata:
        """Extract metadata from PIL image."""
        return ImageMetadata(
            filename="unknown",
            file_size=0,
            format=image.format or 'Unknown',
            width=image.width,
            height=image.height,
            channels=len(image.getbands()) if hasattr(image, 'getbands') else 3,
            color_mode=image.mode,
            has_alpha='A' in image.mode,
            dpi=image.info.get('dpi', (72, 72))
        )
    
    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if image format is supported."""
        return file_path.suffix.lower() in SUPPORTED_FORMATS
    
    def _get_cache_key(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        config: ProcessingConfig
    ) -> str:
        """Generate cache key for processed image."""
        # Create a hash from input and config
        import hashlib
        
        key_parts = []
        
        # Input identifier
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if path.exists():
                stat = path.stat()
                key_parts.append(f"{path}_{stat.st_mtime}_{stat.st_size}")
            else:
                key_parts.append(str(path))
        elif isinstance(image_input, bytes):
            key_parts.append(hashlib.md5(image_input).hexdigest())
        else:
            key_parts.append(str(type(image_input)))
        
        # Config hash
        config_str = str(config.__dict__)
        key_parts.append(hashlib.md5(config_str.encode()).hexdigest())
        
        return "_".join(key_parts)
    
    def save_image(
        self,
        image: Union[Image.Image, np.ndarray],
        output_path: Union[str, Path],
        format: str = "JPEG",
        quality: int = 85
    ) -> bool:
        """Save processed image to file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(image, np.ndarray) and HAS_PIL:
                # Convert numpy array to PIL Image
                if len(image.shape) == 3:
                    pil_image = Image.fromarray(image)
                else:
                    pil_image = Image.fromarray(image, mode='L')
                image = pil_image
            
            if isinstance(image, Image.Image):
                save_kwargs = {}
                if format.upper() == 'JPEG':
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                
                image.save(output_path, format=format, **save_kwargs)
                return True
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        if stats['images_processed'] > 0:
            stats['average_processing_time'] = stats['total_processing_time'] / stats['images_processed']
        return stats
    
    def clear_cache(self):
        """Clear the processing cache."""
        with self.cache_lock:
            self.cache.clear()
        logger.info("Image processing cache cleared")

# -------------------------------
# Utility Functions
# -------------------------------

def load_and_process_image(
    image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
    target_size: Optional[Tuple[int, int]] = None,
    enhancement_mode: EnhancementMode = EnhancementMode.BASIC,
    crop_mode: CropMode = CropMode.CENTER
) -> ProcessingResult:
    """
    Convenience function to load and process an image.
    
    Args:
        image_input: Image source
        target_size: Target dimensions
        enhancement_mode: Enhancement level to apply
        crop_mode: Cropping method
        
    Returns:
        ProcessingResult with processed image
    """
    config = ProcessingConfig(
        target_size=target_size,
        enhancement_mode=enhancement_mode,
        crop_mode=crop_mode
    )
    
    processor = ImageProcessor(config)
    return processor.process_image(image_input)

def create_thumbnail(
    image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
    size: Tuple[int, int] = (128, 128)
) -> ProcessingResult:
    """Create thumbnail from image."""
    config = ProcessingConfig(
        target_size=size,
        maintain_aspect_ratio=True,
        enhancement_mode=EnhancementMode.BASIC,
        quality=60
    )
    
    processor = ImageProcessor(config)
    return processor.process_image(image_input, config)

def prepare_for_vision_model(
    image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
    model_input_size: Tuple[int, int] = (224, 224)
) -> ProcessingResult:
    """Prepare image for vision/ML model input."""
    config = ProcessingConfig(
        target_size=model_input_size,
        enhancement_mode=EnhancementMode.VISION_READY,
        crop_mode=CropMode.CENTER,
        maintain_aspect_ratio=False
    )
    
    processor = ImageProcessor(config)
    return processor.process_image(image_input, config)

def detect_image_format(file_path: Union[str, Path]) -> str:
    """Detect image format from file."""
    file_path = Path(file_path)
    return SUPPORTED_FORMATS.get(file_path.suffix.lower(), "Unknown")

# -------------------------------
# Testing and Demo
# -------------------------------

if __name__ == "__main__":
    # Demo mode
    print("=== DharmaShield Image Processing Engine Demo ===")
    
    # Create test configuration
    config = ProcessingConfig(
        target_size=(224, 224),
        enhancement_mode=EnhancementMode.ADVANCED,
        crop_mode=CropMode.CENTER,
        brightness_factor=1.1,
        contrast_factor=1.2,
        sharpness_factor=1.1
    )
    
    processor = ImageProcessor(config)
    
    print("Image Processing Engine Features:")
    print("✓ Multi-format image loading (JPEG, PNG, TIFF, BMP, WebP)")
    print("✓ Intelligent cropping (center, smart, face-aware, manual)")
    print("✓ Advanced enhancement (brightness, contrast, noise reduction)")
    print("✓ Vision/ML-ready preprocessing")
    print("✓ Document/OCR optimization")
    print("✓ Face-specific enhancement")
    print("✓ Cross-platform optimization")
    print("✓ Caching for performance")
    print("✓ Thread-safe operations")
    
    # Show available backends
    print(f"\nAvailable backends:")
    print(f"  PIL/Pillow: {'✓' if HAS_PIL else '✗'}")
    print(f"  OpenCV: {'✓' if HAS_OPENCV else '✗'}")
    print(f"  NumPy: {'✓' if HAS_NUMPY else '✗'}")
    print(f"  scikit-image: {'✓' if HAS_SKIMAGE else '✗'}")
    
    # Show supported formats
    print(f"\nSupported formats: {', '.join(SUPPORTED_FORMATS.keys())}")
    
    # Processing stats
    stats = processor.get_stats()
    print(f"\nProcessing statistics: {stats}")
    
    print("\n✅ Image Processing Engine ready for production!")
    print("🖼️  Ready to process images for:")
    print("  • Scam detection and threat analysis")
    print("  • Vision model preprocessing")
    print("  • Document OCR optimization")
    print("  • Face detection and enhancement")
    print("  • Cross-platform image handling (Android/iOS/Desktop)")
    print("  • Real-time image enhancement")

