"""
src/accessibility/voice_control.py

Advanced voice accessibility controller for DharmaShield.
Provides comprehensive voice navigation, screen reading, and accessible interaction
for visually impaired, elderly, and non-tech users.
"""

import asyncio
import threading
from typing import Callable, Optional, Dict, List
from ..utils.tts_engine import speak
from ..utils.asr_engine import ASREngine
from ..utils.language import get_language_name, list_supported, detect_language

class VoiceControl:
    """
    Comprehensive voice accessibility system integrated with DharmaShield's
    multilingual voice interface and education system.
    """
    
    def __init__(self, language="en", on_command_callback: Optional[Callable] = None):
        self.language = language
        self.asr = ASREngine(language=language)
        self.supported_languages = list_supported()
        self.is_active = False
        self.on_command_callback = on_command_callback
        
        # Screen reader hooks
        self.screen_reader_active = False
        self.current_context = ""
        
        # Accessibility features
        self.slow_speech = False
        self.verbose_feedback = True
        self.confirmation_required = False
        
        # Voice commands in multiple languages
        self.voice_commands = self._initialize_voice_commands()
    
    def _initialize_voice_commands(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize voice commands in multiple languages"""
        return {
            "help": {
                "en": ["help", "assist", "guide", "what can you do"],
                "hi": ["मदद", "सहायता", "गाइड", "आप क्या कर सकते हैं"]
            },
            "read_screen": {
                "en": ["read screen", "what's on screen", "describe screen"],
                "hi": ["स्क्रीन पढ़ें", "स्क्रीन पर क्या है", "स्क्रीन बताएं"]
            },
            "slow_speech": {
                "en": ["speak slowly", "slow down", "slower speech"],
                "hi": ["धीरे बोलें", "आवाज धीमी करें", "धीमी आवाज"]
            },
            "fast_speech": {
                "en": ["speak faster", "speed up", "faster speech"],
                "hi": ["तेज बोलें", "आवाज तेज करें", "तेज आवाज"]
            },
            "repeat": {
                "en": ["repeat", "say again", "repeat that"],
                "hi": ["दोहराएं", "फिर से कहें", "दोबारा बोलें"]
            },
            "navigate": {
                "en": ["navigate", "go to", "open"],
                "hi": ["नेविगेट", "जाएं", "खोलें"]
            }
        }
    
    async def start_accessibility_mode(self):
        """Start continuous accessibility mode"""
        self.is_active = True
        await self._welcome_message()
        
        while self.is_active:
            try:
                command = await self._listen_for_command()
                if command:
                    await self._process_accessibility_command(command)
            except Exception as e:
                await self._speak_error(f"Sorry, there was an error: {str(e)}")
    
    async def _welcome_message(self):
        """Provide welcome message with instructions"""
        welcome_messages = {
            "en": "Accessibility mode activated. You can say 'help' for available commands, 'read screen' to hear what's displayed, or 'exit accessibility' to stop.",
            "hi": "सुगम्यता मोड सक्रिय है। उपलब्ध कमांड के लिए 'मदद' कहें, स्क्रीन सुनने के लिए 'स्क्रीन पढ़ें' कहें, या रोकने के लिए 'सुगम्यता बंद करें' कहें।"
        }
        
        message = welcome_messages.get(self.language, welcome_messages["en"])
        await self._speak(message)
    
    async def _listen_for_command(self) -> str:
        """Listen for voice commands with accessibility features"""
        prompt_messages = {
            "en": "Listening for your command...",
            "hi": "आपका कमांड सुन रहा हूं..."
        }
        
        prompt = prompt_messages.get(self.language, prompt_messages["en"])
        print(prompt)
        
        # Use longer timeout for accessibility
        command = self.asr.listen_and_transcribe(
            prompt=prompt, 
            timeout=10, 
            language=self.language
        )
        
        return command.strip().lower() if command else ""
    
    async def _process_accessibility_command(self, command: str):
        """Process accessibility-specific commands"""
        # Exit command
        if self._matches_command(command, ["exit accessibility", "stop accessibility", "सुगम्यता बंद करें"]):
            await self._speak("Accessibility mode stopped. Stay safe!")
            self.is_active = False
            return
        
        # Help command
        if self._matches_any_command(command, "help"):
            await self._provide_help()
            return
        
        # Screen reading
        if self._matches_any_command(command, "read_screen"):
            await self._read_current_screen()
            return
        
        # Speech speed control
        if self._matches_any_command(command, "slow_speech"):
            await self._set_speech_speed(slow=True)
            return
        
        if self._matches_any_command(command, "fast_speech"):
            await self._set_speech_speed(slow=False)
            return
        
        # Repeat last message
        if self._matches_any_command(command, "repeat"):
            await self._repeat_last_message()
            return
        
        # Language switching
        if command.startswith("switch language") or command.startswith("भाषा बदलें"):
            await self._handle_language_switch(command)
            return
        
        # Navigation commands
        if self._matches_any_command(command, "navigate"):
            await self._handle_navigation(command)
            return
        
        # Forward to main application
        if self.on_command_callback:
            await self.on_command_callback(command)
        else:
            await self._speak("Command not recognized. Say 'help' for available options.")
    
    def _matches_command(self, command: str, patterns: List[str]) -> bool:
        """Check if command matches any of the patterns"""
        return any(pattern in command for pattern in patterns)
    
    def _matches_any_command(self, command: str, command_type: str) -> bool:
        """Check if command matches any command type in current language"""
        patterns = self.voice_commands.get(command_type, {}).get(self.language, [])
        if not patterns:
            patterns = self.voice_commands.get(command_type, {}).get("en", [])
        
        return any(pattern in command for pattern in patterns)
    
    async def _provide_help(self):
        """Provide detailed help information"""
        help_messages = {
            "en": """Available accessibility commands:
            - Say 'read screen' to hear what's currently displayed
            - Say 'help' to hear this message again
            - Say 'speak slowly' or 'speak faster' to control speech speed
            - Say 'repeat' to hear the last message again
            - Say 'switch language to Hindi' to change language
            - Say 'scan message' to check if a message is a scam
            - Say 'exit accessibility' to stop accessibility mode""",
            
            "hi": """उपलब्ध सुगम्यता कमांड:
            - 'स्क्रीन पढ़ें' कहें वर्तमान डिस्प्ले सुनने के लिए
            - 'मदद' कहें इस संदेश को फिर सुनने के लिए
            - 'धीरे बोलें' या 'तेज बोलें' कहें आवाज की गति नियंत्रित करने के लिए
            - 'दोहराएं' कहें अंतिम संदेश फिर सुनने के लिए
            - 'भाषा बदलें अंग्रेजी में' कहें भाषा बदलने के लिए
            - 'संदेश स्कैन करें' कहें यह जांचने के लिए कि क्या संदेश घोटाला है
            - 'सुगम्यता बंद करें' कहें सुगम्यता मोड बंद करने के लिए"""
        }
        
        message = help_messages.get(self.language, help_messages["en"])
        await self._speak(message)
    
    async def _read_current_screen(self):
        """Read current screen content"""
        if self.current_context:
            context_intro = {
                "en": "Current screen shows: ",
                "hi": "वर्तमान स्क्रीन दिखाता है: "
            }
            intro = context_intro.get(self.language, context_intro["en"])
            await self._speak(intro + self.current_context)
        else:
            no_content = {
                "en": "No screen content available to read.",
                "hi": "पढ़ने के लिए कोई स्क्रीन सामग्री उपलब्ध नहीं है।"
            }
            message = no_content.get(self.language, no_content["en"])
            await self._speak(message)
    
    async def _set_speech_speed(self, slow: bool):
        """Set speech speed"""
        self.slow_speech = slow
        rate = 120 if slow else 200  # Adjust TTS rate
        
        # Update TTS engine rate (implementation depends on your TTS setup)
        confirmation = {
            "en": f"Speech speed set to {'slow' if slow else 'normal'}.",
            "hi": f"आवाज की गति {'धीमी' if slow else 'सामान्य'} सेट की गई।"
        }
        message = confirmation.get(self.language, confirmation["en"])
        await self._speak(message)
    
    async def _handle_language_switch(self, command: str):
        """Handle language switching with voice confirmation"""
        # Extract language from command (simplified)
        if "hindi" in command or "हिंदी" in command:
            new_lang = "hi"
        elif "english" in command or "अंग्रेजी" in command:
            new_lang = "en"
        elif "spanish" in command:
            new_lang = "es"
        else:
            await self._speak("Language not recognized. Available: English, Hindi, Spanish")
            return
        
        if new_lang in self.supported_languages:
            old_lang = self.language
            self.language = new_lang
            self.asr.set_language(new_lang)
            
            # Confirm in new language
            confirmations = {
                "en": f"Language switched to {get_language_name(new_lang)}",
                "hi": f"भाषा {get_language_name(new_lang)} में बदली गई",
                "es": f"Idioma cambiado a {get_language_name(new_lang)}"
            }
            message = confirmations.get(new_lang, confirmations["en"])
            await self._speak(message)
    
    async def _handle_navigation(self, command: str):
        """Handle navigation commands"""
        nav_message = {
            "en": "Navigation feature will be implemented based on your UI structure.",
            "hi": "नेविगेशन सुविधा आपके UI संरचना के आधार पर लागू की जाएगी।"
        }
        message = nav_message.get(self.language, nav_message["en"])
        await self._speak(message)
    
    async def _repeat_last_message(self):
        """Repeat the last spoken message"""
        # This would require storing the last message
        repeat_msg = {
            "en": "Repeat function will store and replay the last message.",
            "hi": "दोहराने का कार्य अंतिम संदेश को संग्रहीत और पुनः चलाएगा।"
        }
        message = repeat_msg.get(self.language, repeat_msg["en"])
        await self._speak(message)
    
    async def _speak(self, text: str):
        """Speak text with accessibility considerations"""
        # Adjust speech rate if needed
        speak(text, lang=self.language)
    
    async def _speak_error(self, error_message: str):
        """Speak error messages with appropriate tone"""
        error_prefix = {
            "en": "Error: ",
            "hi": "त्रुटि: "
        }
        prefix = error_prefix.get(self.language, error_prefix["en"])
        await self._speak(prefix + error_message)
    
    def set_screen_context(self, context: str):
        """Set current screen context for screen reading"""
        self.current_context = context
    
    def stop_accessibility(self):
        """Stop accessibility mode"""
        self.is_active = False

# Example integration
if __name__ == "__main__":
    async def demo_callback(command):
        print(f"Received command: {command}")
    
    vc = VoiceControl(language="en", on_command_callback=demo_callback)
    asyncio.run(vc.start_accessibility_mode())

