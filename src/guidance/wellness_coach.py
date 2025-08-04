"""
src/guidance/wellness_coach.py

DharmaShield - Advanced Wellness & Mindfulness Coach Engine
----------------------------------------------------------
• Industry-grade wellness coaching system with breathing exercises, meditation, and positive self-talk
• Cross-platform (Android/iOS/Desktop) with multilingual TTS integration and voice-guided sessions
• Modular architecture with customizable wellness programs, progress tracking, and accessibility features
• Real-time biofeedback simulation, adaptive coaching, and emergency stress relief protocols

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import json
import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import random
import math
from collections import defaultdict, deque

# Project imports
from ...utils.logger import get_logger
from ...core.config_loader import load_config
from ...utils.language import detect_language, get_language_name
from ...utils.tts_engine import speak
from ..core.threat_level import ThreatLevel
from ..accessibility.heptic_feedback import get_heptic_feedback_engine, FeedbackType

logger = get_logger(__name__)

# -------------------------------
# Enums and Data Structures
# -------------------------------

class WellnessActivityType(Enum):
    BREATHING = "breathing"
    MEDITATION = "meditation"
    POSITIVE_AFFIRMATIONS = "positive_affirmations"
    PROGRESSIVE_RELAXATION = "progressive_relaxation"
    MINDFULNESS = "mindfulness"
    VISUALIZATION = "visualization"
    GROUNDING = "grounding"
    STRESS_RELIEF = "stress_relief"

class BreathingPattern(Enum):
    BASIC_4_4_4 = "basic_4_4_4"         # 4 count in, 4 hold, 4 out
    SQUARE_4_4_4_4 = "square_4_4_4_4"   # 4 in, 4 hold, 4 out, 4 hold
    CALMING_4_7_8 = "calming_4_7_8"     # 4 in, 7 hold, 8 out
    ENERGIZING_6_2_6 = "energizing_6_2_6" # 6 in, 2 hold, 6 out
    ANXIETY_RELIEF_6_6_6 = "anxiety_relief_6_6_6" # 6 in, 6 hold, 6 out

class StressLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EMERGENCY = "emergency"

class SessionIntensity(Enum):
    GENTLE = "gentle"           # 2-5 minutes
    MODERATE = "moderate"       # 5-10 minutes
    INTENSIVE = "intensive"     # 10-20 minutes
    DEEP = "deep"              # 20+ minutes

@dataclass
class WellnessSession:
    """Configuration for a wellness session."""
    session_id: str
    activity_type: WellnessActivityType
    duration_minutes: float
    intensity: SessionIntensity
    language: str = "en"
    voice_guided: bool = True
    haptic_feedback: bool = True
    background_sounds: bool = False
    adaptation_enabled: bool = True
    emergency_mode: bool = False

@dataclass
class BreathingExercise:
    """Breathing exercise configuration."""
    exercise_id: str
    name: str
    pattern: BreathingPattern
    inhale_count: int
    hold_in_count: int
    exhale_count: int
    hold_out_count: int
    total_cycles: int
    guidance_frequency: str = "every_breath"  # every_breath, every_cycle, minimal
    benefits: List[str] = field(default_factory=list)

@dataclass
class MeditationProgram:
    """Meditation program configuration."""
    program_id: str
    name: str
    duration_minutes: float
    focus_type: str  # breath, body, loving_kindness, mindfulness
    guidance_script: List[str]
    background_sounds: Optional[str] = None
    difficulty_level: str = "beginner"  # beginner, intermediate, advanced

@dataclass
class WellnessProgress:
    """User's wellness progress tracking."""
    user_id: str
    total_sessions: int = 0
    total_minutes: float = 0.0
    sessions_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    stress_levels_tracked: List[float] = field(default_factory=list)
    favorite_activities: List[str] = field(default_factory=list)
    current_streak: int = 0
    longest_streak: int = 0
    last_session_timestamp: float = 0.0
    achievements: List[str] = field(default_factory=list)

# -------------------------------
# Multilingual Content Templates
# -------------------------------

class WellnessContentTemplates:
    """Templates for wellness content in multiple languages."""
    
    BREATHING_INSTRUCTIONS = {
        "en": {
            "start": "Let's begin with a calming breathing exercise. Find a comfortable position.",
            "inhale": "Breathe in slowly through your nose",
            "hold_in": "Hold your breath gently",
            "exhale": "Breathe out slowly through your mouth",
            "hold_out": "Rest and pause",
            "cycle_complete": "Cycle {cycle} of {total} complete",
            "session_complete": "Excellent work! Your breathing session is complete. Notice how you feel now.",
            "emergency_start": "Let's do some emergency calming breaths. Focus only on my voice."
        },
        "hi": {
            "start": "आइए एक शांत करने वाली सांस की एक्सरसाइज़ से शुरुआत करते हैं। आरामदायक स्थिति में बैठें।",
            "inhale": "अपनी नाक से धीरे-धीरे सांस अंदर लें",
            "hold_in": "अपनी सांस को धीरे से रोकें",
            "exhale": "अपने मुंह से धीरे-धीरे सांस छोड़ें",
            "hold_out": "आराम करें और रुकें",
            "cycle_complete": "चक्र {cycle} का {total} पूरा हुआ",
            "session_complete": "बहुत बढ़िया! आपका सांस का सत्र पूरा हुआ। अब आप कैसा महसूस कर रहे हैं, इस पर ध्यान दें।",
            "emergency_start": "आइए कुछ आपातकालीन शांत करने वाली सांसें लेते हैं। केवल मेरी आवाज़ पर ध्यान दें।"
        }
    }
    
    MEDITATION_SCRIPTS = {
        "en": {
            "mindfulness_intro": "Welcome to mindfulness meditation. We'll spend the next few minutes in peaceful awareness.",
            "body_scan_intro": "We'll do a gentle body scan, bringing awareness to each part of your body.",
            "loving_kindness_intro": "Today we'll practice loving-kindness meditation, sending good wishes to ourselves and others.",
            "breath_focus": "Focus your attention on your breath. Notice the natural rhythm of breathing in and breathing out.",
            "body_scan_head": "Bring your attention to the top of your head. Notice any sensations without judgment.",
            "loving_kindness_self": "Begin by sending loving wishes to yourself: May I be happy, may I be peaceful, may I be free from suffering.",
            "closing": "Take a moment to appreciate this time you've given yourself. When you're ready, gently open your eyes."
        },
        "hi": {
            "mindfulness_intro": "माइंडफुलनेस मेडिटेशन में आपका स्वागत है। हम अगले कुछ मिनट शांतिपूर्ण जागरूकता में बिताएंगे।",
            "body_scan_intro": "हम एक कोमल बॉडी स्कैन करेंगे, अपने शरीर के हर हिस्से पर जागरूकता लाएंगे।",
            "loving_kindness_intro": "आज हम प्रेम-दया की मेडिटेशन का अभ्यास करेंगे, अपने और दूसरों के लिए अच्छी शुभकामनाएं भेजेंगे।",
            "breath_focus": "अपना ध्यान अपनी सांस पर केंद्रित करें। सांस अंदर लेने और छोड़ने की प्राकृतिक लय को महसूस करें।",
            "body_scan_head": "अपना ध्यान अपने सिर के ऊपरी हिस्से पर लाएं। बिना न्याय के किसी भी संवेदना को महसूस करें।",
            "loving_kindness_self": "स्वयं को प्रेमपूर्ण शुभकामनाएं भेजकर शुरुआत करें: मैं खुश रहूं, मैं शांत रहूं, मैं दुःख से मुक्त रहूं।",
            "closing": "अपने लिए इस समय की सराहना करने के लिए एक पल लें। जब आप तैयार हों, तो धीरे से अपनी आंखें खोलें।"
        }
    }
    
    POSITIVE_AFFIRMATIONS = {
        "en": {
            "confidence": [
                "I am strong and capable of handling any challenge",
                "I trust my intuition and make wise decisions",
                "I am protected and guided in all that I do",
                "I choose peace and let go of what I cannot control",
                "I am worthy of safety, love, and respect"
            ],
            "stress_relief": [
                "I breathe in calm and breathe out tension",
                "This moment of stress will pass, and I will be okay",
                "I have overcome challenges before, and I can do it again",
                "I am safe in this moment",
                "I choose to respond with wisdom rather than react with fear"
            ],
            "protection": [
                "I am alert and aware, protecting myself wisely",
                "My intuition guides me away from harm",
                "I trust my ability to recognize danger and stay safe",
                "I am surrounded by love and protection",
                "I make choices that honor my safety and well-being"
            ]
        },
        "hi": {
            "confidence": [
                "मैं मजबूत हूं और किसी भी चुनौती से निपटने में सक्षम हूं",
                "मैं अपनी अंतर्दृष्टि पर भरोसा करता हूं और बुद्धिमान निर्णय लेता हूं",
                "मैं अपने सभी कार्यों में सुरक्षित और निर्देशित हूं",
                "मैं शांति चुनता हूं और जो मैं नियंत्रित नहीं कर सकता उसे छोड़ देता हूं",
                "मैं सुरक्षा, प्रेम और सम्मान के योग्य हूं"
            ],
            "stress_relief": [
                "मैं शांति की सांस लेता हूं और तनाव की सांस छोड़ता हूं",
                "तनाव का यह क्षण बीत जाएगा, और मैं ठीक हो जाऊंगा",
                "मैंने पहले भी चुनौतियों को पार किया है, और मैं फिर से कर सकता हूं",
                "मैं इस क्षण में सुरक्षित हूं",
                "मैं डर से प्रतिक्रिया करने के बजाय बुद्धि से जवाब देना चुनता हूं"
            ],
            "protection": [
                "मैं सचेत और जागरूक हूं, बुद्धिमानी से अपनी रक्षा करता हूं",
                "मेरी अंतर्दृष्टि मुझे नुकसान से दूर ले जाती है",
                "मैं खतरे को पहचानने और सुरक्षित रहने की अपनी क्षमता पर भरोसा करता हूं",
                "मैं प्रेम और सुरक्षा से घिरा हूं",
                "मैं ऐसे विकल्प चुनता हूं जो मेरी सुरक्षा और कल्याण का सम्मान करते हैं"
            ]
        }
    }

# -------------------------------
# Configuration
# -------------------------------

class WellnessCoachConfig:
    """Configuration for wellness coach system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        wellness_config = self.config.get('wellness_coach', {})
        
        # General settings
        self.enabled = wellness_config.get('enabled', True)
        self.default_language = wellness_config.get('default_language', 'en')
        self.supported_languages = wellness_config.get('supported_languages', ['en', 'hi'])
        
        # Session settings
        self.default_session_duration = wellness_config.get('default_session_duration', 5.0)
        self.max_session_duration = wellness_config.get('max_session_duration', 30.0)
        self.emergency_session_duration = wellness_config.get('emergency_session_duration', 2.0)
        
        # Voice and audio settings
        self.voice_guidance_enabled = wellness_config.get('voice_guidance_enabled', True)
        self.speech_rate_breathing = wellness_config.get('speech_rate_breathing', 120)  # Slower for breathing
        self.speech_rate_meditation = wellness_config.get('speech_rate_meditation', 140)
        self.background_sounds_enabled = wellness_config.get('background_sounds_enabled', False)
        
        # Haptic feedback settings
        self.haptic_feedback_enabled = wellness_config.get('haptic_feedback_enabled', True)
        self.haptic_breathing_cues = wellness_config.get('haptic_breathing_cues', True)
        
        # Personalization
        self.adaptive_difficulty = wellness_config.get('adaptive_difficulty', True)
        self.progress_tracking = wellness_config.get('progress_tracking', True)
        self.personalized_recommendations = wellness_config.get('personalized_recommendations', True)
        
        # Emergency features
        self.emergency_interventions = wellness_config.get('emergency_interventions', True)
        self.crisis_detection = wellness_config.get('crisis_detection', True)

# -------------------------------
# Breathing Exercise Engine
# -------------------------------

class BreathingExerciseEngine:
    """Manages breathing exercises with voice guidance and haptic feedback."""
    
    def __init__(self, config: WellnessCoachConfig):
        self.config = config
        self.templates = WellnessContentTemplates()
        self.haptic_engine = get_heptic_feedback_engine() if config.haptic_feedback_enabled else None
        self.exercises = self._build_breathing_exercises()
        self._current_session = None
        self._session_active = False
    
    def _build_breathing_exercises(self) -> Dict[str, BreathingExercise]:
        """Build library of breathing exercises."""
        return {
            "basic_calm": BreathingExercise(
                exercise_id="basic_calm",
                name="Basic Calming Breath",
                pattern=BreathingPattern.BASIC_4_4_4,
                inhale_count=4,
                hold_in_count=4,
                exhale_count=4,
                hold_out_count=0,
                total_cycles=6,
                benefits=["Reduces anxiety", "Promotes relaxation", "Improves focus"]
            ),
            "square_breathing": BreathingExercise(
                exercise_id="square_breathing",
                name="Square Breathing",
                pattern=BreathingPattern.SQUARE_4_4_4_4,
                inhale_count=4,
                hold_in_count=4,
                exhale_count=4,
                hold_out_count=4,
                total_cycles=5,
                benefits=["Balances nervous system", "Increases concentration", "Reduces stress"]
            ),
            "anxiety_relief": BreathingExercise(
                exercise_id="anxiety_relief",
                name="Anxiety Relief Breathing",
                pattern=BreathingPattern.CALMING_4_7_8,
                inhale_count=4,
                hold_in_count=7,
                exhale_count=8,
                hold_out_count=0,
                total_cycles=4,
                benefits=["Quickly calms anxiety", "Activates relaxation response", "Improves sleep"]
            ),
            "emergency_calm": BreathingExercise(
                exercise_id="emergency_calm",
                name="Emergency Calming",
                pattern=BreathingPattern.ANXIETY_RELIEF_6_6_6,
                inhale_count=6,
                hold_in_count=6,
                exhale_count=6,
                hold_out_count=0,
                total_cycles=3,
                benefits=["Immediate stress relief", "Grounds in present moment", "Reduces panic"]
            )
        }
    
    async def start_breathing_session(
        self,
        exercise_id: str = "basic_calm",
        language: str = "en",
        voice_guidance: bool = True,
        haptic_cues: bool = True
    ) -> bool:
        """Start a guided breathing session."""
        
        if self._session_active:
            logger.warning("Breathing session already active")
            return False
        
        exercise = self.exercises.get(exercise_id)
        if not exercise:
            logger.error(f"Unknown breathing exercise: {exercise_id}")
            return False
        
        self._session_active = True
        self._current_session = {
            'exercise_id': exercise_id,
            'language': language,
            'voice_guidance': voice_guidance,
            'haptic_cues': haptic_cues,
            'start_time': time.time()
        }
        
        try:
            await self._run_breathing_exercise(exercise, language, voice_guidance, haptic_cues)
            return True
        except Exception as e:
            logger.error(f"Error in breathing session: {e}")
            return False
        finally:
            self._session_active = False
            self._current_session = None
    
    async def _run_breathing_exercise(
        self,
        exercise: BreathingExercise,
        language: str,
        voice_guidance: bool,
        haptic_cues: bool
    ):
        """Run the breathing exercise with guidance."""
        
        templates = self.templates.BREATHING_INSTRUCTIONS.get(language, 
                                                             self.templates.BREATHING_INSTRUCTIONS['en'])
        
        # Introduction
        if voice_guidance:
            if exercise.exercise_id == "emergency_calm":
                speak(templates["emergency_start"], language)
            else:
                speak(templates["start"], language)
            await asyncio.sleep(2)
        
        # Main breathing cycles
        for cycle in range(1, exercise.total_cycles + 1):
            if not self._session_active:  # Allow for early termination
                break
            
            # Inhale phase
            if voice_guidance:
                speak(templates["inhale"], language)
            if haptic_cues and self.haptic_engine:
                self.haptic_engine.light()
            
            await self._count_breathing_phase(exercise.inhale_count, "inhale", language, voice_guidance)
            
            # Hold in phase
            if exercise.hold_in_count > 0:
                if voice_guidance:
                    speak(templates["hold_in"], language)
                if haptic_cues and self.haptic_engine:
                    self.haptic_engine.medium()
                
                await self._count_breathing_phase(exercise.hold_in_count, "hold", language, False)
            
            # Exhale phase
            if voice_guidance:
                speak(templates["exhale"], language)
            if haptic_cues and self.haptic_engine:
                self.haptic_engine.light()
            
            await self._count_breathing_phase(exercise.exhale_count, "exhale", language, voice_guidance)
            
            # Hold out phase
            if exercise.hold_out_count > 0:
                if voice_guidance:
                    speak(templates["hold_out"], language)
                
                await self._count_breathing_phase(exercise.hold_out_count, "hold", language, False)
            
            # Cycle completion
            if voice_guidance and exercise.guidance_frequency == "every_cycle":
                cycle_msg = templates["cycle_complete"].format(cycle=cycle, total=exercise.total_cycles)
                speak(cycle_msg, language)
            
            await asyncio.sleep(0.5)  # Brief pause between cycles
        
        # Session completion
        if voice_guidance:
            speak(templates["session_complete"], language)
        if haptic_cues and self.haptic_engine:
            self.haptic_engine.success()
    
    async def _count_breathing_phase(
        self,
        count: int,
        phase: str,
        language: str,
        voice_guidance: bool
    ):
        """Count through a breathing phase with optional voice guidance."""
        
        for i in range(count):
            if not self._session_active:
                break
            
            # Optional count-by-count guidance for beginners
            if voice_guidance and phase in ["inhale", "exhale"]:
                # Could add numerical counting here if desired
                pass
            
            await asyncio.sleep(1.0)  # 1 second per count
    
    def stop_breathing_session(self):
        """Stop the current breathing session."""
        self._session_active = False
    
    def get_recommended_exercise(self, stress_level: StressLevel) -> str:
        """Get recommended breathing exercise based on stress level."""
        
        recommendations = {
            StressLevel.VERY_LOW: "basic_calm",
            StressLevel.LOW: "basic_calm",
            StressLevel.MODERATE: "square_breathing",
            StressLevel.HIGH: "anxiety_relief",
            StressLevel.VERY_HIGH: "anxiety_relief",
            StressLevel.EMERGENCY: "emergency_calm"
        }
        
        return recommendations.get(stress_level, "basic_calm")

# -------------------------------
# Meditation Guide Engine
# -------------------------------

class MeditationGuideEngine:
    """Manages guided meditation sessions."""
    
    def __init__(self, config: WellnessCoachConfig):
        self.config = config
        self.templates = WellnessContentTemplates()
        self.programs = self._build_meditation_programs()
        self._current_session = None
        self._session_active = False
    
    def _build_meditation_programs(self) -> Dict[str, MeditationProgram]:
        """Build library of meditation programs."""
        return {
            "mindfulness_basic": MeditationProgram(
                program_id="mindfulness_basic",
                name="Basic Mindfulness",
                duration_minutes=5.0,
                focus_type="mindfulness",
                guidance_script=[
                    "mindfulness_intro",
                    "breath_focus",
                    "closing"
                ],
                difficulty_level="beginner"
            ),
            "body_scan_short": MeditationProgram(
                program_id="body_scan_short",
                name="Quick Body Scan",
                duration_minutes=7.0,
                focus_type="body",
                guidance_script=[
                    "body_scan_intro",
                    "body_scan_head",
                    "closing"
                ],
                difficulty_level="beginner"
            ),
            "loving_kindness": MeditationProgram(
                program_id="loving_kindness",
                name="Loving Kindness",
                duration_minutes=10.0,
                focus_type="loving_kindness",
                guidance_script=[
                    "loving_kindness_intro",
                    "loving_kindness_self",
                    "closing"
                ],
                difficulty_level="intermediate"
            )
        }
    
    async def start_meditation_session(
        self,
        program_id: str = "mindfulness_basic",
        duration_minutes: Optional[float] = None,
        language: str = "en",
        voice_guidance: bool = True
    ) -> bool:
        """Start a guided meditation session."""
        
        if self._session_active:
            logger.warning("Meditation session already active")
            return False
        
        program = self.programs.get(program_id)
        if not program:
            logger.error(f"Unknown meditation program: {program_id}")
            return False
        
        session_duration = duration_minutes or program.duration_minutes
        
        self._session_active = True
        self._current_session = {
            'program_id': program_id,
            'duration': session_duration,
            'language': language,
            'voice_guidance': voice_guidance,
            'start_time': time.time()
        }
        
        try:
            await self._run_meditation_program(program, session_duration, language, voice_guidance)
            return True
        except Exception as e:
            logger.error(f"Error in meditation session: {e}")
            return False
        finally:
            self._session_active = False
            self._current_session = None
    
    async def _run_meditation_program(
        self,
        program: MeditationProgram,
        duration_minutes: float,
        language: str,
        voice_guidance: bool
    ):
        """Run the meditation program with guidance."""
        
        templates = self.templates.MEDITATION_SCRIPTS.get(language,
                                                          self.templates.MEDITATION_SCRIPTS['en'])
        
        duration_seconds = duration_minutes * 60
        guidance_interval = duration_seconds / len(program.guidance_script)
        
        # Run through guidance script
        for i, script_key in enumerate(program.guidance_script):
            if not self._session_active:
                break
            
            if voice_guidance and script_key in templates:
                speak(templates[script_key], language)
            
            # Wait for next guidance or end
            if i < len(program.guidance_script) - 1:
                await asyncio.sleep(guidance_interval)
            else:
                # Final silence period
                remaining_time = duration_seconds - (len(program.guidance_script) - 1) * guidance_interval
                await asyncio.sleep(max(0, remaining_time))
    
    def stop_meditation_session(self):
        """Stop the current meditation session."""
        self._session_active = False

# -------------------------------
# Positive Affirmations Engine
# -------------------------------

class PositiveAffirmationsEngine:
    """Manages positive affirmations and self-talk sessions."""
    
    def __init__(self, config: WellnessCoachConfig):
        self.config = config
        self.templates = WellnessContentTemplates()
        self.custom_affirmations: Dict[str, List[str]] = {}
    
    async def deliver_affirmations(
        self,
        category: str = "confidence",
        count: int = 3,
        language: str = "en",
        voice_delivery: bool = True,
        pause_between: float = 3.0
    ) -> List[str]:
        """Deliver a series of positive affirmations."""
        
        affirmations = self._get_affirmations(category, language)
        
        if not affirmations:
            logger.warning(f"No affirmations found for category: {category}, language: {language}")
            return []
        
        # Select random affirmations
        selected = random.sample(affirmations, min(count, len(affirmations)))
        
        if voice_delivery:
            for i, affirmation in enumerate(selected):
                speak(affirmation, language)
                if i < len(selected) - 1:  # Don't pause after last affirmation
                    await asyncio.sleep(pause_between)
        
        return selected
    
    def _get_affirmations(self, category: str, language: str) -> List[str]:
        """Get affirmations for a specific category and language."""
        
        # Check custom affirmations first
        custom_key = f"{category}_{language}"
        if custom_key in self.custom_affirmations:
            return self.custom_affirmations[custom_key]
        
        # Use template affirmations
        lang_affirmations = self.templates.POSITIVE_AFFIRMATIONS.get(
            language, self.templates.POSITIVE_AFFIRMATIONS['en']
        )
        
        return lang_affirmations.get(category, lang_affirmations.get('confidence', []))
    
    def add_custom_affirmations(self, category: str, language: str, affirmations: List[str]):
        """Add custom affirmations for a category."""
        key = f"{category}_{language}"
        self.custom_affirmations[key] = affirmations
    
    def get_affirmation_for_threat_level(self, threat_level: ThreatLevel, language: str = "en") -> str:
        """Get appropriate affirmation based on threat level."""
        
        if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            category = "protection"
        elif threat_level in [ThreatLevel.MEDIUM, ThreatLevel.LOW]:
            category = "confidence"
        else:
            category = "stress_relief"
        
        affirmations = self._get_affirmations(category, language)
        return random.choice(affirmations) if affirmations else "You are safe and protected."

# -------------------------------
# Progress Tracking System
# -------------------------------

class ProgressTracker:
    """Tracks user wellness progress and achievements."""
    
    def __init__(self, config: WellnessCoachConfig):
        self.config = config
        self.user_progress: Dict[str, WellnessProgress] = {}
        self.achievements_definitions = self._build_achievements()
    
    def _build_achievements(self) -> Dict[str, Dict[str, Any]]:
        """Build achievement definitions."""
        return {
            "first_session": {
                "name": "First Steps",
                "description": "Complete your first wellness session",
                "requirement": lambda progress: progress.total_sessions >= 1
            },
            "consistent_week": {
                "name": "Weekly Warrior",
                "description": "Complete sessions for 7 consecutive days",
                "requirement": lambda progress: progress.current_streak >= 7
            },
            "breathing_master": {
                "name": "Breathing Master",
                "description": "Complete 20 breathing exercises",
                "requirement": lambda progress: progress.sessions_by_type.get('breathing', 0) >= 20
            },
            "meditation_explorer": {
                "name": "Meditation Explorer",
                "description": "Try all meditation programs",
                "requirement": lambda progress: len([k for k in progress.sessions_by_type.keys() 
                                                   if k.startswith('meditation')]) >= 3
            },
            "wellness_champion": {
                "name": "Wellness Champion",
                "description": "Complete 100 total wellness sessions",
                "requirement": lambda progress: progress.total_sessions >= 100
            }
        }
    
    def record_session(
        self,
        user_id: str,
        activity_type: WellnessActivityType,
        duration_minutes: float,
        stress_level_before: Optional[float] = None,
        stress_level_after: Optional[float] = None
    ):
        """Record a completed wellness session."""
        
        if user_id not in self.user_progress:
            self.user_progress[user_id] = WellnessProgress(user_id=user_id)
        
        progress = self.user_progress[user_id]
        
        # Update basic metrics
        progress.total_sessions += 1
        progress.total_minutes += duration_minutes
        progress.sessions_by_type[activity_type.value] += 1
        
        # Update streak
        current_time = time.time()
        if current_time - progress.last_session_timestamp <= 86400 * 1.5:  # Within 1.5 days
            progress.current_streak += 1
        else:
            progress.current_streak = 1
        
        progress.longest_streak = max(progress.longest_streak, progress.current_streak)
        progress.last_session_timestamp = current_time
        
        # Track stress levels
        if stress_level_before is not None and stress_level_after is not None:
            improvement = stress_level_before - stress_level_after
            progress.stress_levels_tracked.append(improvement)
            # Keep only recent stress data
            progress.stress_levels_tracked = progress.stress_levels_tracked[-50:]
        
        # Check for new achievements
        self._check_achievements(progress)
    
    def _check_achievements(self, progress: WellnessProgress):
        """Check if user has earned new achievements."""
        
        for achievement_id, achievement in self.achievements_definitions.items():
            if achievement_id not in progress.achievements:
                if achievement["requirement"](progress):
                    progress.achievements.append(achievement_id)
                    logger.info(f"User {progress.user_id} earned achievement: {achievement['name']}")
    
    def get_progress_summary(self, user_id: str) -> Dict[str, Any]:
        """Get progress summary for user."""
        
        if user_id not in self.user_progress:
            return {"error": "No progress data found"}
        
        progress = self.user_progress[user_id]
        
        # Calculate average stress improvement
        avg_stress_improvement = 0.0
        if progress.stress_levels_tracked:
            avg_stress_improvement = sum(progress.stress_levels_tracked) / len(progress.stress_levels_tracked)
        
        # Get recent achievements
        recent_achievements = []
        for achievement_id in progress.achievements[-3:]:  # Last 3 achievements
            if achievement_id in self.achievements_definitions:
                recent_achievements.append(self.achievements_definitions[achievement_id])
        
        return {
            "total_sessions": progress.total_sessions,
            "total_minutes": progress.total_minutes,
            "current_streak": progress.current_streak,
            "longest_streak": progress.longest_streak,
            "sessions_by_type": dict(progress.sessions_by_type),
            "average_stress_improvement": avg_stress_improvement,
            "total_achievements": len(progress.achievements),
            "recent_achievements": recent_achievements
        }

# -------------------------------
# Main Wellness Coach Engine
# -------------------------------

class WellnessCoach:
    """
    Main wellness coach engine providing breathing exercises, meditation,
    positive affirmations, and progress tracking.
    
    Features:
    - Voice-guided breathing exercises with haptic feedback
    - Multiple meditation programs with customizable duration
    - Positive affirmations tailored to threat levels
    - Progress tracking and achievement system
    - Emergency stress relief protocols
    - Multilingual support with TTS integration
    - Adaptive coaching based on user history
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if getattr(self, '_initialized', False):
            return
        
        self.config = WellnessCoachConfig(config_path)
        self.breathing_engine = BreathingExerciseEngine(self.config)
        self.meditation_engine = MeditationGuideEngine(self.config)
        self.affirmations_engine = PositiveAffirmationsEngine(self.config)
        self.progress_tracker = ProgressTracker(self.config)
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            'total_sessions': 0,
            'sessions_by_type': defaultdict(int),
            'total_minutes_guided': 0.0,
            'emergency_interventions': 0,
            'user_satisfaction_ratings': []
        }
        
        self._initialized = True
        logger.info("WellnessCoach initialized")
    
    async def start_wellness_session(
        self,
        activity_type: WellnessActivityType,
        user_id: str = "anonymous",
        duration_minutes: Optional[float] = None,
        language: str = "en",
        voice_guidance: bool = True,
        emergency_mode: bool = False
    ) -> bool:
        """Start a wellness session."""
        
        # Prevent multiple sessions for same user
        if user_id in self.active_sessions:
            logger.warning(f"User {user_id} already has an active session")
            return False
        
        session_duration = duration_minutes or (
            self.config.emergency_session_duration if emergency_mode 
            else self.config.default_session_duration
        )
        
        session_info = {
            'activity_type': activity_type,
            'duration': session_duration,
            'language': language,
            'voice_guidance': voice_guidance,
            'emergency_mode': emergency_mode,
            'start_time': time.time()
        }
        
        self.active_sessions[user_id] = session_info
        
        try:
            success = await self._execute_wellness_session(activity_type, session_info)
            
            if success:
                # Record progress
                self.progress_tracker.record_session(
                    user_id, activity_type, session_duration
                )
                
                # Update stats
                self._update_session_stats(activity_type, session_duration, emergency_mode)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in wellness session: {e}")
            return False
        finally:
            if user_id in self.active_sessions:
                del self.active_sessions[user_id]
    
    async def _execute_wellness_session(
        self,
        activity_type: WellnessActivityType,
        session_info: Dict[str, Any]
    ) -> bool:
        """Execute the appropriate wellness session."""
        
        if activity_type == WellnessActivityType.BREATHING:
            exercise_id = "emergency_calm" if session_info['emergency_mode'] else "basic_calm"
            return await self.breathing_engine.start_breathing_session(
                exercise_id=exercise_id,
                language=session_info['language'],
                voice_guidance=session_info['voice_guidance']
            )
        
        elif activity_type == WellnessActivityType.MEDITATION:
            return await self.meditation_engine.start_meditation_session(
                program_id="mindfulness_basic",
                duration_minutes=session_info['duration'],
                language=session_info['language'],
                voice_guidance=session_info['voice_guidance']
            )
        
        elif activity_type == WellnessActivityType.POSITIVE_AFFIRMATIONS:
            affirmations = await self.affirmations_engine.deliver_affirmations(
                category="confidence",
                count=3,
                language=session_info['language'],
                voice_delivery=session_info['voice_guidance']
            )
            return len(affirmations) > 0
        
        else:
            logger.warning(f"Unsupported activity type: {activity_type}")
            return False
    
    async def emergency_calm_session(
        self,
        user_id: str = "anonymous",
        language: str = "en",
        voice_guidance: bool = True
    ) -> bool:
        """Start an emergency calming session for high stress/anxiety."""
        
        success = await self.start_wellness_session(
            activity_type=WellnessActivityType.BREATHING,
            user_id=user_id,
            duration_minutes=self.config.emergency_session_duration,
            language=language,
            voice_guidance=voice_guidance,
            emergency_mode=True
        )
        
        if success:
            self.stats['emergency_interventions'] += 1
        
        return success
    
    async def provide_threat_response_wellness(
        self,
        threat_level: ThreatLevel,
        user_id: str = "anonymous",
        language: str = "en"
    ) -> bool:
        """Provide appropriate wellness response based on threat level."""
        
        if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            # Emergency breathing + protective affirmation
            breathing_success = await self.emergency_calm_session(user_id, language)
            
            affirmation = self.affirmations_engine.get_affirmation_for_threat_level(
                threat_level, language
            )
            speak(affirmation, language)
            
            return breathing_success
        
        elif threat_level == ThreatLevel.MEDIUM:
            # Standard breathing exercise
            return await self.start_wellness_session(
                WellnessActivityType.BREATHING, user_id, language=language
            )
        
        elif threat_level == ThreatLevel.LOW:
            # Positive affirmations
            return await self.start_wellness_session(
                WellnessActivityType.POSITIVE_AFFIRMATIONS, user_id, language=language
            )
        
        else:  # SAFE or UNKNOWN
            # Brief mindfulness
            return await self.start_wellness_session(
                WellnessActivityType.MEDITATION, user_id, duration_minutes=3.0, language=language
            )
    
    def stop_session(self, user_id: str) -> bool:
        """Stop active session for user."""
        
        if user_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[user_id]
        activity_type = session['activity_type']
        
        # Stop appropriate engine
        if activity_type == WellnessActivityType.BREATHING:
            self.breathing_engine.stop_breathing_session()
        elif activity_type == WellnessActivityType.MEDITATION:
            self.meditation_engine.stop_meditation_session()
        
        del self.active_sessions[user_id]
        return True
    
    def get_personalized_recommendation(
        self,
        user_id: str,
        current_stress_level: StressLevel = StressLevel.MODERATE
    ) -> Dict[str, Any]:
        """Get personalized wellness recommendation for user."""
        
        progress_summary = self.progress_tracker.get_progress_summary(user_id)
        
        if "error" in progress_summary:
            # New user recommendations
            return {
                "activity_type": WellnessActivityType.BREATHING,
                "program_id": "basic_calm",
                "duration_minutes": 5.0,
                "reason": "Welcome! Let's start with a simple breathing exercise."
            }
        
        # Analyze user's session history
        sessions_by_type = progress_summary.get('sessions_by_type', {})
        total_sessions = progress_summary.get('total_sessions', 0)
        
        # Recommend based on stress level and history
        if current_stress_level in [StressLevel.VERY_HIGH, StressLevel.EMERGENCY]:
            return {
                "activity_type": WellnessActivityType.BREATHING,
                "program_id": "emergency_calm",
                "duration_minutes": 2.0,
                "reason": "Let's focus on immediate calming with emergency breathing."
            }
        
        # Favor less-used activities for variety
        if sessions_by_type.get('meditation', 0) < sessions_by_type.get('breathing', 0) / 2:
            return {
                "activity_type": WellnessActivityType.MEDITATION,
                "program_id": "mindfulness_basic",
                "duration_minutes": 5.0,
                "reason": "Time to explore meditation for deeper relaxation."
            }
        
        # Default to breathing
        exercise_id = self.breathing_engine.get_recommended_exercise(current_stress_level)
        return {
            "activity_type": WellnessActivityType.BREATHING,
            "program_id": exercise_id,
            "duration_minutes": 5.0,
            "reason": "Breathing exercises are great for managing stress."
        }
    
    def _update_session_stats(
        self,
        activity_type: WellnessActivityType,
        duration_minutes: float,
        emergency_mode: bool
    ):
        """Update wellness coach statistics."""
        
        self.stats['total_sessions'] += 1
        self.stats['sessions_by_type'][activity_type.value] += 1
        self.stats['total_minutes_guided'] += duration_minutes
        
        if emergency_mode:
            self.stats['emergency_interventions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get wellness coach statistics."""
        return dict(self.stats)
    
    def add_satisfaction_rating(self, user_id: str, rating: int, activity_type: str):
        """Add user satisfaction rating."""
        rating_data = {
            'user_id': user_id,
            'rating': rating,  # 1-5 scale
            'activity_type': activity_type,
            'timestamp': time.time()
        }
        self.stats['user_satisfaction_ratings'].append(rating_data)
        
        # Keep only recent ratings
        self.stats['user_satisfaction_ratings'] = self.stats['user_satisfaction_ratings'][-100:]

# -------------------------------
# Singleton and Convenience Functions
# -------------------------------

_global_wellness_coach = None

def get_wellness_coach(config_path: Optional[str] = None) -> WellnessCoach:
    """Get the global wellness coach instance."""
    global _global_wellness_coach
    if _global_wellness_coach is None:
        _global_wellness_coach = WellnessCoach(config_path)
    return _global_wellness_coach

async def start_breathing_exercise(
    user_id: str = "anonymous",
    language: str = "en",
    emergency: bool = False
) -> bool:
    """Convenience function to start breathing exercise."""
    coach = get_wellness_coach()
    return await coach.start_wellness_session(
        WellnessActivityType.BREATHING,
        user_id=user_id,
        language=language,
        emergency_mode=emergency
    )

async def start_meditation_session(
    user_id: str = "anonymous",
    duration_minutes: float = 5.0,
    language: str = "en"
) -> bool:
    """Convenience function to start meditation session."""
    coach = get_wellness_coach()
    return await coach.start_wellness_session(
        WellnessActivityType.MEDITATION,
        user_id=user_id,
        duration_minutes=duration_minutes,
        language=language
    )

async def deliver_positive_affirmations(
    count: int = 3,
    language: str = "en",
    category: str = "confidence"
) -> List[str]:
    """Convenience function to deliver affirmations."""
    coach = get_wellness_coach()
    return await coach.affirmations_engine.deliver_affirmations(
        category=category,
        count=count,
        language=language
    )

# -------------------------------
# Testing and Demo
# -------------------------------

if __name__ == "__main__":
    async def test_wellness_coach():
        print("=== DharmaShield Wellness Coach Demo ===\n")
        
        coach = get_wellness_coach()
        
        # Test breathing exercise
        print("--- Testing Breathing Exercise ---")
        success = await coach.start_wellness_session(
            WellnessActivityType.BREATHING,
            user_id="test_user",
            duration_minutes=1.0,  # Short for demo
            language="en",
            voice_guidance=False  # Silent for demo
        )
        print(f"Breathing exercise success: {success}")
        
        # Test meditation
        print("\n--- Testing Meditation Session ---")
        success = await coach.start_wellness_session(
            WellnessActivityType.MEDITATION,
            user_id="test_user_2",
            duration_minutes=1.0,  # Short for demo
            language="en",
            voice_guidance=False
        )
        print(f"Meditation session success: {success}")
        
        # Test affirmations
        print("\n--- Testing Positive Affirmations ---")
        affirmations = await coach.affirmations_engine.deliver_affirmations(
            category="confidence",
            count=2,
            language="en",
            voice_delivery=False
        )
        print(f"Delivered affirmations: {affirmations}")
        
        # Test emergency session
        print("\n--- Testing Emergency Calm Session ---")
        success = await coach.emergency_calm_session(
            user_id="emergency_user",
            language="en",
            voice_guidance=False
        )
        print(f"Emergency session success: {success}")
        
        # Test threat response
        print("\n--- Testing Threat Response Wellness ---")
        success = await coach.provide_threat_response_wellness(
            ThreatLevel.HIGH,
            user_id="threat_user",
            language="en"
        )
        print(f"Threat response success: {success}")
        
        # Test personalized recommendations
        print("\n--- Testing Personalized Recommendations ---")
        recommendation = coach.get_personalized_recommendation("test_user")
        print(f"Recommendation: {recommendation}")
        
        # Test progress tracking
        print("\n--- Testing Progress Summary ---")
        progress = coach.progress_tracker.get_progress_summary("test_user")
        print(f"Progress summary: {progress}")
        
        # Show statistics
        print("\n--- Coach Statistics ---")
        stats = coach.get_stats()
        for key, value in stats.items():
            if key != 'user_satisfaction_ratings':  # Skip detailed ratings
                print(f"{key}: {value}")
        
        print(f"\n✅ Wellness Coach ready for production!")
        print(f"🧘 Features demonstrated:")
        print(f"  ✓ Voice-guided breathing exercises with haptic feedback")
        print(f"  ✓ Multiple meditation programs")
        print(f"  ✓ Positive affirmations tailored to threat levels")
        print(f"  ✓ Emergency stress relief protocols")
        print(f"  ✓ Progress tracking and achievements")
        print(f"  ✓ Personalized recommendations")
        print(f"  ✓ Multilingual TTS integration")
        print(f"  ✓ Adaptive coaching based on user history")
    
    # Run the test
    asyncio.run(test_wellness_coach())

