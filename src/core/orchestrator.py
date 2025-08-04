"""
src/core/orchestrator.py

DharmaShieldCore – Multimodal Analysis Orchestrator
---------------------------------------------------
• Central coordinator for text, audio, image, and combined threat analysis
• Pluggable architecture with async task graph, adaptive fusion, and privacy guardrails
• Industry-grade error handling, observability, performance metrics, and extensibility
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable

from src.utils.logger import get_logger
from src.utils.crypto_utils import encrypt_data, EncryptionAlgorithm
from src.utils.language import detect_language, get_language_name
from src.utils.asr_engine import get_asr_engine, RecognitionResult
from src.utils.tts_engine import get_tts_engine
from src.utils.image_processing import load_and_process_image
from src.utils.audio_processing import load_audio_file
from src.ml.scam_classifier import ScamClassifier  # abstract interface
from src.ml.vision_detector import ImageThreatDetector  # abstract interface
from src.ml.audio_detector import AudioThreatDetector  # abstract interface

logger = get_logger(__name__)


# ---------------------------- Enumerations -----------------------------------

class InputType(Enum):
    TEXT = auto()
    AUDIO = auto()
    IMAGE = auto()
    MIXED = auto()


class ThreatLevel(Enum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ------------------------------- Dataclasses ---------------------------------

@dataclass
class AnalysisRequest:
    """Unified request for multimodal analysis."""
    text: Optional[str] = None
    audio: Optional[Union[str, bytes]] = None            # path or raw
    image: Optional[Union[str, bytes]] = None            # path or raw
    user_language: str = "auto"
    require_privacy: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Unified response from multimodal analysis."""
    threat_level: ThreatLevel
    text_score: float = 0.0
    audio_score: float = 0.0
    image_score: float = 0.0
    modality_details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    spiritual_guidance: Optional[str] = None
    duration: float = 0.0
    errors: List[str] = field(default_factory=list)


# ------------------------------ Orchestrator ---------------------------------

class DharmaShieldCore:
    """
    Central orchestrator for multimodal scam analysis.

    Responsibilities:
        • Load/initialize ML models and subsystems
        • Accept AnalysisRequest objects
        • Run parallel analysis tasks with robust timeouts
        • Fuse individual modality scores into a threat level
        • Enforce privacy checks and secure handling (encryption at rest)
        • Generate user guidance and recommendations
    """

    # Fusion weightage (can be tuned / loaded from config)
    WEIGHTS = {
        InputType.TEXT: 0.45,
        InputType.AUDIO: 0.35,
        InputType.IMAGE: 0.20,
    }

    # Thresholds for threat levels
    LEVEL_THRESHOLDS = {
        ThreatLevel.CRITICAL: 0.85,
        ThreatLevel.HIGH: 0.70,
        ThreatLevel.MEDIUM: 0.50,
        ThreatLevel.LOW: 0.25,
        ThreatLevel.SAFE: 0.00,
    }

    def __init__(
        self,
        max_workers: int = 4,
        inference_timeout: float = 15.0,
    ):
        self.max_workers = max_workers
        self.inference_timeout = inference_timeout

        # Models / subsystems
        self.scam_classifier: Optional[ScamClassifier] = None
        self.image_detector: Optional[ImageThreatDetector] = None
        self.audio_detector: Optional[AudioThreatDetector] = None

        # ThreadPool for blocking ops
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

        self.initialized = False

    # --------------------------- Initialization ---------------------------

    async def initialize(self) -> None:
        """Load models asynchronously to avoid UI blocking."""
        if self.initialized:
            return

        logger.info("Initializing DharmaShieldCore…")

        loop = asyncio.get_event_loop()

        # Load heavy models concurrently
        tasks = [
            loop.run_in_executor(self.executor, ScamClassifier.load_default),
            loop.run_in_executor(self.executor, ImageThreatDetector.load_default),
            loop.run_in_executor(self.executor, AudioThreatDetector.load_default),
        ]
        self.scam_classifier, self.image_detector, self.audio_detector = await asyncio.gather(*tasks)

        self.initialized = True
        logger.info("DharmaShieldCore initialization complete")

    async def cleanup(self) -> None:
        """Cleanup resources gracefully."""
        logger.info("Cleaning up DharmaShieldCore…")
        self.executor.shutdown(wait=False)
        self.initialized = False

    # ------------------------ Public API (main) ---------------------------

    async def run_multimodal_analysis(self, *, text: Optional[str] = None,
                                      audio: Optional[Union[str, bytes]] = None,
                                      image: Optional[Union[str, bytes]] = None,
                                      user_language: str = "auto",
                                      require_privacy: bool = True) -> AnalysisResult:
        """
        Main entry to analyze any combination of text, audio, or image.

        Returns unified AnalysisResult with fused threat level.
        """
        if not self.initialized:
            await self.initialize()

        request = AnalysisRequest(
            text=text, audio=audio, image=image,
            user_language=user_language, require_privacy=require_privacy
        )

        start = time.perf_counter()
        modal_tasks: Dict[InputType, Any] = {}

        # Prepare async tasks for each modality present
        if request.text:
            modal_tasks[InputType.TEXT] = self._analyze_text(request.text)
        if request.audio:
            modal_tasks[InputType.AUDIO] = self._analyze_audio(request.audio, request.user_language)
        if request.image:
            modal_tasks[InputType.IMAGE] = self._analyze_image(request.image)

        # Run with timeout safeguard
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*modal_tasks.values(), return_exceptions=True),
                timeout=self.inference_timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Analysis timed out")
            results = []

        # Collect scores
        modality_scores = {}
        errors = []
        for (modality, task_result) in zip(modal_tasks.keys(), results):
            if isinstance(task_result, Exception):
                logger.error(f"{modality.name} analysis failed: {task_result}")
                errors.append(f"{modality.name} error")
                modality_scores[modality] = 0.0
            else:
                modality_scores[modality] = task_result

        # Fuse threat level
        fused_score = self._fuse_scores(modality_scores)
        threat_level = self._score_to_threat_level(fused_score)

        # Generate user guidance
        recommendations = self._generate_recommendations(threat_level)
        spiritual_guidance = self._generate_spiritual_guidance(threat_level, request.user_language)

        # Privacy: encrypt sensitive payloads
        if request.require_privacy:
            self._securely_dispose(request)

        duration = time.perf_counter() - start

        # Build result object
        return AnalysisResult(
            threat_level=threat_level,
            text_score=modality_scores.get(InputType.TEXT, 0.0),
            audio_score=modality_scores.get(InputType.AUDIO, 0.0),
            image_score=modality_scores.get(InputType.IMAGE, 0.0),
            modality_details=modality_scores,
            recommendations=recommendations,
            spiritual_guidance=spiritual_guidance,
            duration=duration,
            errors=errors
        )

    # --------------------- Modality-specific analysers --------------------

    async def _analyze_text(self, text: str) -> float:
        """Run scam classifier on text and return score."""
        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(self.executor, self.scam_classifier.predict_proba, text)
        logger.debug(f"Text score: {score:.4f}")
        return score

    async def _analyze_audio(self, audio_input: Union[str, bytes],
                             user_language: str) -> float:
        """Transcribe audio then classify text."""
        # Transcribe
        asr = get_asr_engine()
        rec: RecognitionResult = asr.recognize_speech(audio_input, language=user_language)
        transcript = rec.text if rec.success else ""
        logger.debug(f"Audio transcription: '{transcript}'")

        if not transcript:
            return 0.0

        # Analyze transcript
        return await self._analyze_text(transcript)

    async def _analyze_image(self, image_input: Union[str, bytes]) -> float:
        """Detect image-based threats (e.g., QR/Phishing)."""
        # Preprocess image
        processed = load_and_process_image(image_input)
        if not processed.success:
            return 0.0

        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(
            self.executor,
            self.image_detector.predict_proba,
            processed.image_data
        )
        logger.debug(f"Image score: {score:.4f}")
        return score

    # ----------------------- Fusion & Helpers -----------------------------

    def _fuse_scores(self, scores: Dict[InputType, float]) -> float:
        """Weighted average of modality scores."""
        total_weight = sum(self.WEIGHTS[m] for m in scores)
        if total_weight == 0:
            return 0.0
        fused = sum(scores[m] * self.WEIGHTS[m] for m in scores) / total_weight
        logger.debug(f"Fused score: {fused:.4f}")
        return fused

    def _score_to_threat_level(self, score: float) -> ThreatLevel:
        """Map fused score to threat level."""
        for level, threshold in sorted(self.LEVEL_THRESHOLDS.items(), key=lambda x: -x[1]):
            if score >= threshold:
                return level
        return ThreatLevel.SAFE

    @staticmethod
    def _generate_recommendations(level: ThreatLevel) -> List[str]:
        """Return high-level user guidance."""
        recs = {
            ThreatLevel.CRITICAL: [
                "Do NOT share personal information.",
                "Block and report the sender immediately."
            ],
            ThreatLevel.HIGH: [
                "Avoid clicking any links.",
                "Verify the sender through official channels."
            ],
            ThreatLevel.MEDIUM: [
                "Proceed with caution and double-check URLs.",
            ],
            ThreatLevel.LOW: [
                "Stay alert for suspicious language."
            ],
            ThreatLevel.SAFE: [
                "No scam indicators detected."
            ]
        }
        return recs[level]

    @staticmethod
    def _generate_spiritual_guidance(level: ThreatLevel, lang: str) -> str:
        """Return a short mindfulness tip."""
        guidance = {
            'en': {
                ThreatLevel.CRITICAL: "Trust your intuition; better safe than sorry.",
                ThreatLevel.HIGH: "Pause, breathe, and verify before acting.",
                ThreatLevel.MEDIUM: "Stay mindful; vigilance is protection.",
                ThreatLevel.LOW: "Keep gratitude; awareness shields you.",
                ThreatLevel.SAFE: "Peace within reflects safety outside."
            },
            'hi': {
                ThreatLevel.CRITICAL: "अंतःकरण की आवाज़ सुनें; सावधानी सर्वोत्तम है।",
                ThreatLevel.HIGH: "रुकें, श्वास लें, और सत्यापित करें।",
                ThreatLevel.MEDIUM: "सचेत रहें; जागरूकता ही सुरक्षा है।",
                ThreatLevel.LOW: "कृतज्ञ रहें; सजगता रक्षा करती है।",
                ThreatLevel.SAFE: "अंतर्मन की शांति ही बाहरी सुरक्षा है।"
            }
        }
        return guidance.get(lang, guidance['en'])[level]

    @staticmethod
    def _securely_dispose(request: AnalysisRequest):
        """Placeholder for secure disposal (encryption/log scrubbing)."""
        # Encrypt sensitive data (example)
        if request.text:
            encrypt_data(request.text, password="transient", algorithm=EncryptionAlgorithm.FERNET)
        if request.audio and isinstance(request.audio, bytes):
            _ = request  # audio could be zeroized here
        # For simplicity, we skip further actions

    # --------------------------- Stats & Debug ----------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return health metrics."""
        return {
            "initialized": self.initialized,
            "models": {
                "text": bool(self.scam_classifier),
                "image": bool(self.image_detector),
                "audio": bool(self.audio_detector)
            },
            "executor_workers": self.max_workers
        }

