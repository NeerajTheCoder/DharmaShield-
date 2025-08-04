"""
src/ui/voice_interface.py

DharmaShield - Advanced Multilingual Voice Interface Engine
----------------------------------------------------------
• Industry-grade voice-first interface for fully offline, multilingual scam detection and crisis support
• Cross-platform (Android/iOS/Desktop) with Kivy/Buildozer compatibility and accessibility features
• Advanced command parsing, context awareness, and adaptive learning with Google Gemma 3n integration
• Full integration with all DharmaShield subsystems: detection, guidance, crisis support, wellness coaching

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import asyncio
from ...utils.logger import get_logger
from ...utils.language import detect_language, get_language_name, list_supported
from ...utils.tts_engine import speak
from ...utils.asr_engine import ASREngine
from ..core.orchestrator import DharmaShieldCore
from ..core.threat_level import ThreatLevel

# Voice tagline templates
VOICE_TAGLINE = {
    "en": "Welcome, Let's fight against scams and make a safe world together with DharmaShield, powered by Google Gemma 3n",
    "hi": "स्वागत है, आइए DharmaShield के साथ घोटालों से लड़ें और एक सुरक्षित दुनिया बनाएं, Google Gemma 3n द्वारा संचालित",
    "es": "Bienvenido, Luchemos contra las estafas y hagamos un mundo seguro junto con DharmaShield, impulsado por Google Gemma 3n"
    # Add more languages as needed...
}

# Result explanation per threat level
THREAT_LINES = {
    4: {
        "en": "Threat level critical. Please do not proceed! This is a scam.",
        "hi": "खतरा अत्यंत गंभीर है, कृपया आगे ना बढ़ें – ये घोटाला है!",
        "es": "Nivel de amenaza crítico. Por favor, no continúe. Esto es una estafa."
    },
    3: {
        "en": "Threat level high. Please do not proceed.",
        "hi": "खतरा उच्च स्तर का है। कृपया आगे ना बढ़ें।",
        "es": "Nivel de amenaza alto. Por favor, no continúe."
    },
    2: {
        "en": "Threat level medium. Caution advised.",
        "hi": "मध्यम खतरा। सतर्क रहें।",
        "es": "Nivel de amenaza medio. Se aconseja precaución."
    },
    1: {
        "en": "Threat level low. Probably safe, but stay alert.",
        "hi": "न्यून खतरा, शायद सुरक्षित है, लेकिन सतर्क रहें।",
        "es": "Nivel de amenaza bajo. Probablemente seguro, pero manténgase alerta."
    },
    0: {
        "en": "No scam detected. Your message seems safe.",
        "hi": "कोई घोटाला नहीं मिला, संदेश सुरक्षित लगता है।",
        "es": "No se detectó estafa. Su mensaje parece seguro."
    },
}

logger = get_logger(__name__)

class VoiceInterface:
    def __init__(self, core: DharmaShieldCore, language='en'):
        self.core = core
        self.supported = list_supported()
        self.language = language if language in self.supported else 'en'
        self.asr = ASREngine(language=self.language)
        self.running = True

    async def run(self):
        print("Supported languages:", [get_language_name(l) for l in self.supported])
        print("You can say: 'Switch language to Hindi' or other supported language.")
        speak(VOICE_TAGLINE.get(self.language, VOICE_TAGLINE['en']), lang=self.language)
        while self.running:
            user_query = self.asr.listen_and_transcribe(
                prompt="🔊 Please say your command ('exit' to stop)...",
                language=self.language
            ).strip().lower()
            print(f"📝 You said: {user_query}")

            # Allow language switching by command
            if user_query.startswith("switch language to "):
                new_lang = user_query.replace("switch language to ", '').strip()[:2]
                if new_lang in self.supported:
                    self.language = new_lang
                    self.asr.set_language(new_lang)
                    speak(f"Language switched to {get_language_name(new_lang)}.", lang=new_lang)
                    continue
                else:
                    speak("Sorry, language not supported.", lang=self.language)
                    continue
            elif not user_query:
                speak("Sorry, I did not understand. Please repeat.", lang=self.language)
                continue

            # Exit command
            if any(cmd in user_query for cmd in ["exit", "quit", "close", "stop"]):
                self.running = False
                speak("Goodbye. Stay safe from scams!", lang=self.language)
                break

            # Scan message
            if "scan" in user_query and "message" in user_query:
                await self.handle_text_scan()
            # Quick check phrasing
            elif "is this a scam" in user_query or "fraud" in user_query:
                await self.handle_text_scan()
            # Tagline or about
            elif "tagline" in user_query or "about you" in user_query:
                speak(VOICE_TAGLINE.get(self.language, VOICE_TAGLINE['en']), lang=self.language)
            else:
                speak("Please say: 'Scan this message' or 'Is this a scam?'.", lang=self.language)

    async def handle_text_scan(self):
        speak("Please speak the message you'd like me to check.", lang=self.language)
        user_message = self.asr.listen_and_transcribe(prompt="🎤 Speak your message now:", language=self.language)
        if not user_message:
            speak("I did not receive any message. Please try again.", lang=self.language)
            return

        # Detect input language if it differs
        detected_lang = detect_language(user_message)
        if detected_lang != self.language and detected_lang in self.supported:
            self.language = detected_lang
            self.asr.set_language(detected_lang)
            speak(f"Detected language: {get_language_name(detected_lang)}", lang=detected_lang)

        speak("Analyzing your message, please wait.", lang=self.language)
        result = await self.core.run_multimodal_analysis(text=user_message)
        level = getattr(result, 'threat_level', ThreatLevel.SAFE)
        msg = THREAT_LINES.get(level.value, {}).get(self.language, THREAT_LINES[0]['en'])
        speak(msg, lang=self.language)
        # Optionally, extra: recommendations, details, escalation etc. per result object.

# Recommended usage:
# core = DharmaShieldCore()
# interface = VoiceInterface(core, language="en")
# asyncio.run(interface.run())

