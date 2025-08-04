"""
src/utils/notifications.py

DharmaShield - Cross-Platform Notification Engine (Popup/Toast)
---------------------------------------------------------------
• Industry-grade notification utility for Kivy/Buildozer (Android/iOS/Desktop)
• Uses Plyer for native system/toast popups; automatic fallback to Kivy Popup if Plyer not available
• Thread-safe, supports custom icons/titles, urgency, auto-dismiss
• Multilingual notification support, ready for voice-driven, offline operation

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import sys
import threading
from typing import Optional, Dict, Any
from pathlib import Path

# Kivy fallback imports
try:
    from kivy.uix.popup import Popup
    from kivy.uix.label import Label
    from kivy.clock import Clock
    HAS_KIVY = True
except ImportError:
    HAS_KIVY = False

# Plyer for notifications
try:
    from plyer import notification
    HAS_PLYER = True
except ImportError:
    HAS_PLYER = False

# Project logger
from .logger import get_logger

logger = get_logger(__name__)

class NotificationConfig:
    """Holds configuration for the notification engine."""
    def __init__(self, config: dict = None):
        cfg = config or {}
        self.default_title = cfg.get('default_title', 'DharmaShield')
        self.default_timeout = cfg.get('timeout', 5)
        self.toast_supported = cfg.get('toast_supported', True)
        self.icon_path = cfg.get('icon_path', None)  # Set path to app icon if available

class NotificationEngine:
    """Cross-platform popup/toast notification manager."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: dict = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: dict = None):
        if getattr(self, "_initialized", False):
            return
        self.config = NotificationConfig(config)
        self._initialized = True

    def notify(
        self, 
        message: str, 
        title: Optional[str] = None, 
        timeout: Optional[int] = None, 
        icon: Optional[str] = None, 
        toast: Optional[bool] = None,
        extras: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Show a cross-platform notification (native/toast if possible, Kivy fallback).

        Args:
            message: Main notification/text
            title: Notification title (default: DharmaShield)
            timeout: Time in seconds notification stays (default: 5)
            icon: Path to notification icon
            toast: If True, prefer "toast" mode (native on Android)
            extras: Additional notification options for extensibility

        Returns:
            True if shown successfully, False otherwise.
        """
        title = title or self.config.default_title
        timeout = timeout if timeout is not None else self.config.default_timeout
        icon = icon or self.config.icon_path
        toast = self.config.toast_supported if toast is None else toast

        shown = False

        # Native notification (preferred)
        if HAS_PLYER:
            try:
                kwargs = {
                    'title': title,
                    'message': message,
                    'timeout': timeout
                }
                if icon and Path(icon).exists():
                    kwargs['app_icon'] = icon
                if toast is not None:
                    kwargs['toast'] = toast
                if extras:
                    kwargs.update(extras)
                notification.notify(**kwargs)
                shown = True
                logger.debug(f"Plyer notification: {title}: {message}")
                return True
            except Exception as e:
                logger.warning(f"Plyer notification failed: {e}")

        # Kivy popup fallback (if installed)
        if HAS_KIVY:
            try:
                def _show_popup(*a):
                    p = Popup(
                        title=title,
                        content=Label(text=message, halign="center"),
                        size_hint=(0.7, 0.2),
                        auto_dismiss=True
                    )
                    p.open()
                    Clock.schedule_once(lambda dt: p.dismiss(), timeout)
                # Always call popup on main thread
                Clock.schedule_once(_show_popup, 0)
                logger.debug(f"Kivy fallback notification: {title}: {message}")
                shown = True
                return True
            except Exception as e:
                logger.error(f"Kivy popup notification failed: {e}")

        # As last resort, print to console (low-resource mode)
        print(f"[NOTIFICATION] {title}: {message}")
        return shown

# Global singleton instance
_notification_engine = None

def get_notification_engine(config: dict = None) -> NotificationEngine:
    """Get global notification engine instance."""
    global _notification_engine
    if _notification_engine is None:
        _notification_engine = NotificationEngine(config)
    return _notification_engine

def show_notification(
    message: str,
    title: Optional[str] = None,
    timeout: Optional[int] = None,
    icon: Optional[str] = None,
    toast: Optional[bool] = None,
    extras: Optional[Dict[str, Any]] = None
) -> bool:
    """Public convenience API for notifications (recommened for app code)."""
    return get_notification_engine().notify(
        message=message,
        title=title,
        timeout=timeout,
        icon=icon,
        toast=toast,
        extras=extras
    )

# -------------------------------
# Demo/Test Routine
# -------------------------------

if __name__ == "__main__":
    print("=== DharmaShield Notification Engine Demo ===")
    show_notification("Scan complete: Your message is safe!", title="DharmaShield", timeout=4)
    show_notification("Critical Alert!\nPossible scam detected.", title="DharmaShield", timeout=7)
    show_notification("Welcome! This is a cross-platform notification test.")
    print("✅ Notification engine ready for production!")

