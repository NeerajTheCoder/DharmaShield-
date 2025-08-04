"""
src/main.py

DharmaShield - Main Application Entry Point
-------------------------------------------
• Industry-grade main application entry point for cross-platform (Android/iOS/Desktop) with Kivy/Buildozer support
• Advanced configuration loading, initialization, interface selection, and multi-language support
• Splash screen with welcome message, CLI/Voice interface selection, and optimized startup flow
• Fully offline-capable, voice-first operation with Google Gemma 3n integration

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import os
import sys
import time
import asyncio
import signal
import logging
import warnings
from typing import Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass
import argparse

# Ensure proper path setup for modular imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configuration and utility imports
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    warnings.warn("PyYAML not available. Configuration loading will be limited.", ImportWarning)

try:
    import colorama
    from colorama import Fore, Back, Style, init as colorama_init
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    Fore = Back = Style = None

# Project imports
from src.utils.logger import get_logger, setup_logging
from src.utils.language import get_language_name, list_supported as list_supported_languages
from src.utils.crypto_utils import get_crypto_engine
from src.core.orchestrator import DharmaShieldCore
from src.ui.cli_interface import CLIInterface
from src.utils.system_info import get_system_info

# Initialize colorama for cross-platform colored output
if HAS_COLORAMA:
    colorama_init(autoreset=True)

logger = get_logger(__name__)

# -------------------------------
# Constants and Configuration
# -------------------------------

APP_NAME = "DharmaShield"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Advanced AI-Powered Scam Detection & Protection System"
AUTHOR = "DharmaShield Expert Team"
LICENSE = "Apache 2.0"

# Configuration paths
DEFAULT_CONFIG_PATH = Path("config/app_config.yaml")
USER_CONFIG_PATH = Path.home() / ".dharmashield" / "config.yaml"
ENV_CONFIG_PATH = Path(os.getenv("DHARMASHIELD_CONFIG", ""))

# Splash screen configuration
SPLASH_DURATION = 3.0  # seconds
WELCOME_MESSAGES = {
    'en': f"Welcome {{user}}, Let's fight against scam and make safe world together with {APP_NAME}, powered by Google Gemma 3n",
    'hi': f"स्वागत है {{user}}, आइए घोटालों के खिलाफ लड़ें और {APP_NAME} के साथ सुरक्षित दुनिया बनाएं, Google Gemma 3n द्वारा संचालित",
    'es': f"Bienvenido {{user}}, Luchemos contra las estafas y hagamos un mundo seguro juntos con {APP_NAME}, impulsado por Google Gemma 3n",
    'fr': f"Bienvenue {{user}}, Luttons contre les arnaques et créons un monde sûr ensemble avec {APP_NAME}, alimenté par Google Gemma 3n",
    'de': f"Willkommen {{user}}, Lasst uns gegen Betrug kämpfen und gemeinsam eine sichere Welt mit {APP_NAME} schaffen, angetrieben von Google Gemma 3n",
    'zh': f"欢迎 {{user}}，让我们与 {APP_NAME} 一起对抗诈骗，共同创造一个安全的世界，由 Google Gemma 3n 提供支持",
    'ar': f"مرحباً {{user}}، دعونا نحارب الاحتيال ونصنع عالماً آمناً معاً مع {APP_NAME}، مدعوم من Google Gemma 3n",
    'ru': f"Добро пожаловать {{user}}, Давайте бороться с мошенничеством и создавать безопасный мир вместе с {APP_NAME}, работающим на Google Gemma 3n"
}

# Interface modes
class InterfaceMode:
    CLI = "cli"
    VOICE = "voice"
    AUTO = "auto"

# -------------------------------
# Configuration Management
# -------------------------------

@dataclass
class AppConfig:
    """Application configuration structure."""
    # Core settings
    app_name: str = APP_NAME
    version: str = APP_VERSION
    debug: bool = False
    verbose: bool = False
    log_level: str = "INFO"
    
    # Interface settings
    default_interface: str = InterfaceMode.AUTO
    default_language: str = "en"
    show_splash: bool = True
    splash_duration: float = SPLASH_DURATION
    
    # Voice UI settings
    voice_ui: Dict[str, Any] = None
    
    # Security settings
    enable_encryption: bool = True
    secure_mode: bool = True
    
    # Performance settings
    max_workers: int = 4
    cache_enabled: bool = True
    gpu_acceleration: bool = False
    
    # Paths
    models_path: str = "models"
    cache_path: str = "cache"
    logs_path: str = "logs"
    
    def __post_init__(self):
        if self.voice_ui is None:
            self.voice_ui = {
                'default_language': self.default_language,
                'supported_languages': ['en', 'hi', 'es', 'fr', 'de', 'zh', 'ar', 'ru', 'bn', 'ur', 'ta', 'te', 'mr'],
                'tts_rate': 170,
                'offline_asr_engine': 'vosk'
            }

class ConfigManager:
    """Advanced configuration manager with multiple sources and validation."""
    
    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.config_sources: list = []
        
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> AppConfig:
        """
        Load configuration from multiple sources with priority order:
        1. Command line arguments
        2. Environment variables
        3. User config file
        4. Project config file
        5. Default values
        """
        config_data = {}
        
        # Load from default project config
        default_config = self._load_yaml_config(DEFAULT_CONFIG_PATH)
        if default_config:
            config_data.update(default_config)
            self.config_sources.append(f"Project config: {DEFAULT_CONFIG_PATH}")
        
        # Load from user config
        if USER_CONFIG_PATH.exists():
            user_config = self._load_yaml_config(USER_CONFIG_PATH)
            if user_config:
                config_data.update(user_config)
                self.config_sources.append(f"User config: {USER_CONFIG_PATH}")
        
        # Load from specified config path
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                specified_config = self._load_yaml_config(config_path)
                if specified_config:
                    config_data.update(specified_config)
                    self.config_sources.append(f"Specified config: {config_path}")
        
        # Load from environment config path
        if ENV_CONFIG_PATH and ENV_CONFIG_PATH.exists():
            env_config = self._load_yaml_config(ENV_CONFIG_PATH)
            if env_config:
                config_data.update(env_config)
                self.config_sources.append(f"Environment config: {ENV_CONFIG_PATH}")
        
        # Override with environment variables
        env_overrides = self._load_env_config()
        if env_overrides:
            config_data.update(env_overrides)
            self.config_sources.append("Environment variables")
        
        # Create and validate config
        try:
            self.config = AppConfig(**config_data)
            logger.info(f"Configuration loaded from: {', '.join(self.config_sources)}")
            return self.config
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            # Fallback to default config
            self.config = AppConfig()
            self.config_sources = ["Default values (fallback)"]
            return self.config
    
    def _load_yaml_config(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML file."""
        if not HAS_YAML or not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                logger.debug(f"Loaded config from: {config_path}")
                return config_data
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return None
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to config keys
        env_mappings = {
            'DHARMASHIELD_DEBUG': ('debug', lambda x: x.lower() in ('true', '1', 'yes')),
            'DHARMASHIELD_VERBOSE': ('verbose', lambda x: x.lower() in ('true', '1', 'yes')),
            'DHARMASHIELD_LOG_LEVEL': ('log_level', str),
            'DHARMASHIELD_LANGUAGE': ('default_language', str),
            'DHARMASHIELD_INTERFACE': ('default_interface', str),
            'DHARMASHIELD_MODELS_PATH': ('models_path', str),
            'DHARMASHIELD_CACHE_PATH': ('cache_path', str),
            'DHARMASHIELD_LOGS_PATH': ('logs_path', str),
        }
        
        for env_var, (config_key, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    env_config[config_key] = converter(value)
                except Exception as e:
                    logger.warning(f"Invalid value for {env_var}: {value} ({e})")
        
        return env_config
    
    def save_config(self, config_path: Optional[Path] = None) -> bool:
        """Save current configuration to file."""
        if not HAS_YAML or not self.config:
            return False
        
        config_path = config_path or USER_CONFIG_PATH
        
        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dict
            config_dict = {
                'app_name': self.config.app_name,
                'version': self.config.version,
                'debug': self.config.debug,
                'verbose': self.config.verbose,
                'log_level': self.config.log_level,
                'default_interface': self.config.default_interface,
                'default_language': self.config.default_language,
                'show_splash': self.config.show_splash,
                'splash_duration': self.config.splash_duration,
                'voice_ui': self.config.voice_ui,
                'enable_encryption': self.config.enable_encryption,
                'secure_mode': self.config.secure_mode,
                'max_workers': self.config.max_workers,
                'cache_enabled': self.config.cache_enabled,
                'gpu_acceleration': self.config.gpu_acceleration,
                'models_path': self.config.models_path,
                'cache_path': self.config.cache_path,
                'logs_path': self.config.logs_path
            }
            
            # Save to file
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            return False

# -------------------------------
# Splash Screen and UI Management
# -------------------------------

class SplashScreen:
    """Advanced splash screen with animations and multi-language support."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.user_name = self._get_user_name()
    
    def _get_user_name(self) -> str:
        """Get user name from system."""
        try:
            import getpass
            return getpass.getuser()
        except:
            return "User"
    
    def _print_colored(self, text: str, color: Optional[str] = None, style: Optional[str] = None):
        """Print colored text if colorama is available."""
        if HAS_COLORAMA and color:
            color_code = getattr(Fore, color.upper(), "")
            style_code = getattr(Style, style.upper(), "") if style else ""
            print(f"{style_code}{color_code}{text}{Style.RESET_ALL}")
        else:
            print(text)
    
    def show(self):
        """Display splash screen with welcome message."""
        if not self.config.show_splash:
            return
        
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # ASCII Art Logo
        logo = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ██████╗ ██╗  ██╗ █████╗ ██████╗ ███╗   ███╗ █████╗ ███████╗██╗  ██╗██╗███████╗ ║
║  ██╔══██╗██║  ██║██╔══██╗██╔══██╗████╗ ████║██╔══██╗██╔════╝██║  ██║██║██╔════╝ ║
║  ██║  ██║███████║███████║██████╔╝██╔████╔██║███████║███████╗███████║██║█████╗   ║
║  ██║  ██║██╔══██║██╔══██║██╔══██╗██║╚██╔╝██║██╔══██║╚════██║██╔══██║██║██╔══╝   ║
║  ██████╔╝██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║███████║██║  ██║██║███████╗ ║
║  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝ ║
║                                                                              ║
║                    🛡️  Advanced AI-Powered Scam Protection  🛡️                  ║
║                         Powered by Google Gemma 3n                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        
        self._print_colored(logo, "cyan", "bright")
        
        # Version and info
        info = f"""
    Version: {self.config.version}
    Mode: {'Debug' if self.config.debug else 'Production'}
    Language: {get_language_name(self.config.default_language)}
    Interface: {self.config.default_interface.upper()}
        """
        
        self._print_colored(info, "white")
        
        # Welcome message
        welcome_template = WELCOME_MESSAGES.get(
            self.config.default_language, 
            WELCOME_MESSAGES['en']
        )
        welcome_msg = welcome_template.format(user=self.user_name)
        
        print("\n" + "="*80)
        self._print_colored(f"    {welcome_msg}", "green", "bright")
        print("="*80 + "\n")
        
        # Loading animation
        self._show_loading_animation()
    
    def _show_loading_animation(self):
        """Show loading animation during splash."""
        loading_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        loading_messages = [
            "Initializing AI Engine...",
            "Loading Google Gemma 3n Model...",
            "Setting up Voice Recognition...",
            "Configuring Security Systems...",
            "Preparing Scam Detection...",
            "Ready to Protect!"
        ]
        
        start_time = time.time()
        msg_index = 0
        char_index = 0
        
        while time.time() - start_time < self.config.splash_duration:
            # Update message periodically
            if int((time.time() - start_time) * 2) % len(loading_messages) != msg_index:
                msg_index = int((time.time() - start_time) * 2) % len(loading_messages)
            
            current_msg = loading_messages[min(msg_index, len(loading_messages) - 1)]
            spinner = loading_chars[char_index % len(loading_chars)]
            
            # Clear line and print loading message
            print(f"\r    {spinner} {current_msg}{'.' * (char_index % 4)}", end='', flush=True)
            
            char_index += 1
            time.sleep(0.1)
        
        print(f"\r    ✅ System Ready!{' ' * 50}")
        time.sleep(0.5)

class InterfaceSelector:
    """Interactive interface and language selector."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def select_interface_and_language(self) -> tuple[str, str]:
        """Interactive selection of interface mode and language."""
        
        # Language selection
        language = self._select_language()
        
        # Interface selection
        interface = self._select_interface(language)
        
        return interface, language
    
    def _select_language(self) -> str:
        """Select language interactively."""
        if self.config.default_language and self.config.default_language != "auto":
            return self.config.default_language
        
        print("\n🌐 Language Selection / भाषा चयन / Selección de idioma")
        print("=" * 60)
        
        supported_langs = list_supported_languages()
        
        print("Available languages:")
        for i, lang_code in enumerate(supported_langs[:10], 1):  # Show first 10
            lang_name = get_language_name(lang_code)
            print(f"  {i}. {lang_name} ({lang_code})")
        
        print(f"  {len(supported_langs[:10]) + 1}. Auto-detect")
        
        while True:
            try:
                choice = input(f"\nSelect language (1-{len(supported_langs[:10]) + 1}) [1]: ").strip()
                
                if not choice:
                    return supported_langs[0]  # Default to first (English)
                
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(supported_langs[:10]):
                    selected_lang = supported_langs[choice_num - 1]
                    print(f"✅ Selected: {get_language_name(selected_lang)}")
                    return selected_lang
                elif choice_num == len(supported_langs[:10]) + 1:
                    print("✅ Auto-detect enabled")
                    return "auto"
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid input. Please enter a number.")
                continue
    
    def _select_interface(self, language: str) -> str:
        """Select interface mode interactively."""
        if self.config.default_interface and self.config.default_interface != InterfaceMode.AUTO:
            return self.config.default_interface
        
        # Multilingual interface selection text
        interface_text = {
            'en': {
                'title': "🎯 Interface Selection",
                'options': [
                    "📝 CLI (Text-based interface)",
                    "🎤 Voice (Voice-first interface)",
                    "🤖 Auto (Detect best interface)"
                ],
                'prompt': "Select interface mode (1-3) [2]: ",
                'selected': "✅ Selected: ",
                'invalid': "❌ Invalid choice. Please try again."
            },
            'hi': {
                'title': "🎯 इंटरफेस चयन",
                'options': [
                    "📝 CLI (टेक्स्ट-आधारित इंटरफेस)",
                    "🎤 Voice (आवाज-प्राथमिक इंटरफेस)", 
                    "🤖 Auto (सर्वोत्तम इंटरफेस का पता लगाएं)"
                ],
                'prompt': "इंटरफेस मोड चुनें (1-3) [2]: ",
                'selected': "✅ चयनित: ",
                'invalid': "❌ अमान्य विकल्प। कृपया पुनः प्रयास करें।"
            }
        }
        
        text = interface_text.get(language, interface_text['en'])
        
        print(f"\n{text['title']}")
        print("=" * 60)
        
        for i, option in enumerate(text['options'], 1):
            print(f"  {i}. {option}")
        
        interface_modes = [InterfaceMode.CLI, InterfaceMode.VOICE, InterfaceMode.AUTO]
        
        while True:
            try:
                choice = input(f"\n{text['prompt']}").strip()
                
                if not choice:
                    choice = "2"  # Default to Voice
                
                choice_num = int(choice)
                
                if 1 <= choice_num <= 3:
                    selected_mode = interface_modes[choice_num - 1]
                    mode_names = ["CLI", "Voice", "Auto"]
                    print(f"{text['selected']}{mode_names[choice_num - 1]}")
                    return selected_mode
                else:
                    print(text['invalid'])
                    
            except (ValueError, KeyboardInterrupt):
                print(text['invalid'])
                continue

# -------------------------------
# Main Application Class
# -------------------------------

class DharmaShieldApp:
    """
    Main DharmaShield application class.
    
    Features:
    - Advanced configuration management
    - Multi-language splash screen
    - Interactive interface selection
    - CLI and Voice interface support
    - Graceful shutdown handling
    - Error recovery and logging
    - Cross-platform compatibility
    """
    
    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.config_manager = ConfigManager()
        self.core: Optional[DharmaShieldCore] = None
        self.interface_selector: Optional[InterfaceSelector] = None
        self.splash_screen: Optional[SplashScreen] = None
        self.running = False
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        sys.exit(0)
    
    async def initialize(self, config_path: Optional[str] = None) -> bool:
        """Initialize the application with all components."""
        try:
            # Load configuration
            self.config = self.config_manager.load_config(config_path)
            
            # Setup logging
            setup_logging(
                level=self.config.log_level,
                log_file=Path(self.config.logs_path) / "dharmashield.log",
                enable_file_logging=True
            )
            
            logger.info("="*80)
            logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
            logger.info(f"Python: {sys.version}")
            logger.info(f"Platform: {sys.platform}")
            logger.info("="*80)
            
            # Initialize components
            self.interface_selector = InterfaceSelector(self.config)
            self.splash_screen = SplashScreen(self.config)
            
            # Initialize crypto engine
            if self.config.enable_encryption:
                crypto_engine = get_crypto_engine()
                logger.info("Cryptographic engine initialized")
            
            # Initialize core system
            self.core = DharmaShieldCore()
            await self.core.initialize()
            
            # Log system information
            system_info = get_system_info()
            logger.info(f"System Info: {system_info}")
            
            self.running = True
            logger.info("Application initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Application initialization failed: {e}")
            return False
    
    async def run(self, config_path: Optional[str] = None, interface: Optional[str] = None, language: Optional[str] = None) -> int:
        """
        Main application entry point.
        
        Args:
            config_path: Path to configuration file
            interface: Force specific interface mode
            language: Force specific language
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Initialize application
            if not await self.initialize(config_path):
                return 1
            
            # Show splash screen
            if self.config.show_splash:
                self.splash_screen.show()
            
            # Determine interface and language
            if interface and language:
                selected_interface = interface
                selected_language = language
            else:
                selected_interface, selected_language = self.interface_selector.select_interface_and_language()
            
            # Update configuration
            self.config.default_interface = selected_interface
            self.config.default_language = selected_language
            
            # Start appropriate interface
            if selected_interface == InterfaceMode.VOICE:
                await self._run_voice_interface(selected_language)
            elif selected_interface == InterfaceMode.CLI:
                await self._run_cli_interface(selected_language)
            elif selected_interface == InterfaceMode.AUTO:
                # Auto-detect best interface
                if self._has_audio_support():
                    await self._run_voice_interface(selected_language)
                else:
                    await self._run_cli_interface(selected_language)
            
            logger.info("Application shutdown completed successfully")
            return 0
            
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
            return 130  # Standard exit code for Ctrl+C
        except Exception as e:
            logger.error(f"Application error: {e}")
            return 1
        finally:
            await self._cleanup()
    
    async def _run_voice_interface(self, language: str):
        """Run voice-first interface."""
        try:
            from src.ui.voice_interface import VoiceInterface
            
            logger.info(f"Starting voice interface in {get_language_name(language)}")
            
            voice_ui = VoiceInterface(self.core, language=language)
            await voice_ui.run()
            
        except ImportError as e:
            logger.error(f"Voice interface dependencies not available: {e}")
            logger.info("Falling back to CLI interface")
            await self._run_cli_interface(language)
        except Exception as e:
            logger.error(f"Voice interface error: {e}")
            raise
    
    async def _run_cli_interface(self, language: str):
        """Run CLI interface."""
        try:
            logger.info(f"Starting CLI interface in {get_language_name(language)}")
            
            cli = CLIInterface(self.core, language=language)
            await cli.run()
            
        except Exception as e:
            logger.error(f"CLI interface error: {e}")
            raise
    
    def _has_audio_support(self) -> bool:
        """Check if system has audio input/output support."""
        try:
            # Check for microphone
            import pyaudio
            p = pyaudio.PyAudio()
            
            # Check for input devices
            has_input = False
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    has_input = True
                    break
            
            p.terminate()
            return has_input
            
        except ImportError:
            logger.debug("PyAudio not available for audio detection")
            return False
        except Exception as e:
            logger.debug(f"Audio support detection failed: {e}")
            return False
    
    async def _cleanup(self):
        """Cleanup resources on shutdown."""
        try:
            if self.core:
                await self.core.cleanup()
            
            # Save configuration if modified
            if self.config_manager and self.config:
                self.config_manager.save_config()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# -------------------------------
# Command Line Interface
# -------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog=APP_NAME.lower(),
        description=APP_DESCRIPTION,
        epilog=f"Version {APP_VERSION} - {AUTHOR}"
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--interface', '-i',
        choices=[InterfaceMode.CLI, InterfaceMode.VOICE, InterfaceMode.AUTO],
        help='Force specific interface mode'
    )
    
    parser.add_argument(
        '--language', '-l',
        type=str,
        help='Force specific language (e.g., en, hi, es)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-splash',
        action='store_true',
        help='Skip splash screen'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'{APP_NAME} {APP_VERSION}'
    )
    
    # Development options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--models-path',
        type=str,
        help='Path to models directory'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    return parser.parse_args()

def setup_environment():
    """Setup environment variables and paths."""
    # Ensure required directories exist
    required_dirs = ['models', 'cache', 'logs', 'config']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ.setdefault('PYTHONPATH', os.getcwd())
    os.environ.setdefault('DHARMASHIELD_HOME', os.getcwd())

# -------------------------------
# Main Entry Point
# -------------------------------

async def main() -> int:
    """
    Main entry point for DharmaShield application.
    
    Returns:
        Exit code
    """
    # Setup environment
    setup_environment()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run application
    app = DharmaShieldApp()
    
    # Override config with command line arguments
    if hasattr(app, 'config') and app.config:
        if args.debug:
            app.config.debug = True
            app.config.log_level = 'DEBUG'
        if args.verbose:
            app.config.verbose = True
        if args.no_splash:
            app.config.show_splash = False
        if args.log_level:
            app.config.log_level = args.log_level
        if args.models_path:
            app.config.models_path = args.models_path
    
    # Run with profiling if requested
    if args.profile:
        import cProfile
        import pstats
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        exit_code = await app.run(
            config_path=args.config,
            interface=args.interface,
            language=args.language
        )
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return exit_code
    else:
        return await app.run(
            config_path=args.config,
            interface=args.interface,
            language=args.language
        )

def run_sync():
    """Synchronous wrapper for main function."""
    try:
        if sys.version_info >= (3, 7):
            return asyncio.run(main())
        else:
            # Fallback for Python < 3.7
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Stay safe from scams!")
        return 130
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1

# -------------------------------
# Entry Point Guard
# -------------------------------

if __name__ == "__main__":
    # Welcome message for direct execution
    print(f"🚀 Starting {APP_NAME} v{APP_VERSION}")
    print(f"🔗 {APP_DESCRIPTION}")
    print(f"👨‍💻 {AUTHOR}")
    print(f"📜 License: {LICENSE}")
    print("-" * 80)
    
    # Run application
    exit_code = run_sync()
    
    # Exit with appropriate code
    sys.exit(exit_code)

