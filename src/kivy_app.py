"""
src/kivy_app.py

DharmaShield - Root Kivy App Class
----------------------------------
• Cross-platform GUI (Android/iOS/Desktop) built with Kivy/KivyMD
• Multi-language splash screen → main screen with ScreenManager
• Bootstraps core modules and injects them into GUI layers
• Optimized, modular, testable, production-grade code
"""

from __future__ import annotations

import os
import sys
import asyncio
import importlib
from functools import partial
from pathlib import Path
from typing import Optional, Dict, Any

import kivy
kivy.require("2.3.0")                     # minimum supported version

from kivy.clock import Clock
from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition

# --- KivyMD optional support -------------------------------------------------
try:
    from kivymd.app import MDApp           # type: ignore
    from kivymd.uix.label import MDLabel   # type: ignore
    from kivymd.uix.boxlayout import MDBoxLayout  # type: ignore
    MD_AVAILABLE = True
except ImportError:                        # graceful degradation
    MD_AVAILABLE = False
    MDApp = App
    from kivy.uix.label import Label as MDLabel
    from kivy.uix.boxlayout import BoxLayout as MDBoxLayout

# --- Project imports ---------------------------------------------------------
from src.utils.language import get_language_name, list_supported
from src.utils.tts_engine import speak
from src.utils.logger import get_logger
from src.core.orchestrator import DharmaShieldCore

logger = get_logger(__name__)

# -----------------------------------------------------------------------------


KV = r"""
<SplashScreen>:
    name: "splash"
    MDBoxLayout if app.md else MDBoxLayout:
        orientation: "vertical"
        padding: dp(24)
        spacing: dp(16)

        MDLabel if app.md else MDLabel:
            text: "[b]DharmaShield[/b]"
            markup: True
            halign: "center"
            font_style: "H4" if app.md else None
            font_size: "28sp" if not app.md else None

        MDLabel if app.md else MDLabel:
            id: welcome_lbl
            text: ""
            halign: "center"
            font_style: "Subtitle1" if app.md else None
            font_size: "16sp" if not app.md else None

        MDLabel if app.md else MDLabel:
            text: "Loading..."
            halign: "center"
            font_style: "Caption" if app.md else None
            font_size: "14sp" if not app.md else None


<MainScreen>:
    name: "main"
    MDBoxLayout if app.md else MDBoxLayout:
        orientation: "vertical"
        padding: dp(24)
        spacing: dp(16)

        MDLabel if app.md else MDLabel:
            text: "🛡️ DharmaShield is Active"
            halign: "center"
            font_style: "H5" if app.md else None
            font_size: "24sp" if not app.md else None

        MDLabel if app.md else MDLabel:
            text: "Speak or type to analyze a message."
            halign: "center"
            font_style: "Body1" if app.md else None
            font_size: "16sp" if not app.md else None
"""


# -------------------------- Screen Definitions -------------------------------

class SplashScreen(Screen):
    """Initial splash / welcome screen."""
    pass


class MainScreen(Screen):
    """Primary GUI after splash."""
    pass


# -------------------------- Root Application ---------------------------------

class DharmaShieldGUI(MDApp if MD_AVAILABLE else App):
    """
    Root Kivy App: manages screens, initializes Core, injects dependencies.
    """

    WINDOW_SIZE = (480, 800)  # desktop testing convenience

    def __init__(
        self,
        core: DharmaShieldCore,
        language: str = "en",
        splash_duration: float = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.core = core
        self.language = language
        self.splash_duration = splash_duration
        self.sm: Optional[ScreenManager] = None
        self.md: bool = MD_AVAILABLE

        # Ensure consistent desktop sizing (ignored on mobile)
        if sys.platform in ("win32", "linux", "darwin"):
            Window.size = self.WINDOW_SIZE

    # --------------------------------------------------------------------- KV --
    def build(self):
        # Load dynamic KV string
        Builder.load_string(KV)

        # Build screen manager
        self.sm = ScreenManager(transition=FadeTransition(duration=0.4))
        self.sm.add_widget(SplashScreen())
        self.sm.add_widget(MainScreen())

        # Initialize splash text
        welcome_lbl = self.sm.get_screen("splash").ids.welcome_lbl
        welcome_tpl = self._get_welcome_template()
        user = self._get_user_name()
        welcome_lbl.text = welcome_tpl.format(user=user)

        # Schedule splash timeout
        Clock.schedule_once(self._go_to_main_screen, self.splash_duration)

        # Say welcome via TTS asynchronously
        Clock.schedule_once(lambda *_: speak(welcome_lbl.text, lang=self.language, wait=False), 0.1)

        return self.sm

    # ----------------------------------------------------------------- Events --
    def _go_to_main_screen(self, *_):
        logger.debug("Switching to main screen")
        self.sm.current = "main"

    # ----------------------------------------------------------------- Helpers --
    def _get_welcome_template(self) -> str:
        return {
            "en": "Welcome {user}, Let's fight against scam and make safe world together with DharmaShield, powered by Google Gemma 3n",
            "hi": "स्वागत है {user}, चलिए धोखाधड़ी के खिलाफ लड़ते हैं और DharmaShield के साथ एक सुरक्षित दुनिया बनाते हैं, Google Gemma 3n द्वारा संचालित",
            "es": "Bienvenido {user}, ¡Luchemos contra las estafas y hagamos un mundo seguro juntos con DharmaShield, impulsado por Google Gemma 3n!",
            "fr": "Bienvenue {user}, Combattons les arnaques et rendons le monde sûr avec DharmaShield, propulsé par Google Gemma 3n",
            "de": "Willkommen {user}, Lassen Sie uns gemeinsam mit DharmaShield Betrug bekämpfen und eine sichere Welt schaffen, unterstützt von Google Gemma 3n",
            "zh": "欢迎 {user}，让我们携手 DharmaShield 与诈骗作斗争，共同创造安全世界，Powered by Google Gemma 3n",
            "ar": "مرحباً {user}، دعنا نحارب الاحتيال ونبني عالماً آمناً معاً باستخدام DharmaShield، المدعوم من Google Gemma 3n",
            "ru": "Добро пожаловать {user}! Давайте вместе с DharmaShield бороться с мошенничеством и сделаем мир безопасным. Работает на Google Gemma 3n",
        }.get(self.language, 
              "Welcome {user}, Let's fight against scam and make safe world together with DharmaShield, powered by Google Gemma 3n")

    @staticmethod
    def _get_user_name() -> str:
        try:
            import getpass
            return getpass.getuser().capitalize()
        except Exception:
            return "Friend"

    # ---------------------- Public static startup helper -----------------
    @staticmethod
    def run_app(
        language: str = "en",
        splash_duration: float = 3.0,
        core: Optional[DharmaShieldCore] = None,
    ):
        """
        Utility to start the Kivy GUI with given parameters.
        """
        # Lazy import to avoid heavy initialization when unused
        if core is None:
            core = DharmaShieldCore()
            asyncio.run(core.initialize())

        app = DharmaShieldGUI(core=core, language=language, splash_duration=splash_duration)
        app.run()


# ------------------- Direct Execution (for testing) --------------------------

if __name__ == "__main__":
    # Minimal core stub for local GUI testing
    class _DummyCore(DharmaShieldCore):
        async def initialize(self):
            return True

        async def cleanup(self):
            return True

    # Run GUI with default English language
    DharmaShieldGUI.run_app(language="en", core=_DummyCore())

