"""
src/utils/logger.py

DharmaShield - Modular Logging Utility (File, Console, Rotation, Verbosity)
----------------------------------------------------------------------------
• Industry-grade logging for file/console, log rotation, verbosity, error trace support
• Cross-platform (Android/iOS/Desktop) with Kivy/Buildozer support and thread safety
• Timestamp, module, level, message format; failsafe to console on disk error

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock
from pathlib import Path

DEFAULT_LOG_FILE = "dharmashield.log"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_MAX_BYTES = 2 * 1024 * 1024  # 2 MB before rotating
DEFAULT_BACKUP_COUNT = 5  # Keep 5 old logs

_LOGGERS = {}
_LOG_INIT_LOCK = Lock()

class LoggerConfig:
    """Load and store logger config, integrates with YAML config if available."""
    def __init__(self, config_dict: dict = None):
        # Sensible defaults for production & offline/low-resource operation
        cfg = config_dict.get('logger', {}) if config_dict else {}
        self.log_to_file = cfg.get('log_to_file', True)
        self.log_to_console = cfg.get('log_to_console', True)
        self.log_file = cfg.get('log_file', DEFAULT_LOG_FILE)
        self.log_level = getattr(logging, str(cfg.get('level', 'INFO')).upper(), DEFAULT_LOG_LEVEL)
        self.max_bytes = cfg.get('max_bytes', DEFAULT_MAX_BYTES)
        self.backup_count = cfg.get('backup_count', DEFAULT_BACKUP_COUNT)
        self.include_traceback = cfg.get('include_traceback', True)

def get_logger(name: str = "DharmaShield", config_dict: dict = None) -> logging.Logger:
    """Get a logger. Respect singleton/init-once per logger name."""
    global _LOGGERS
    if name in _LOGGERS:
        return _LOGGERS[name]

    with _LOG_INIT_LOCK:
        if name in _LOGGERS:
            return _LOGGERS[name]

        # Parse config
        cfg = LoggerConfig(config_dict)
        logger = logging.getLogger(name)
        logger.setLevel(cfg.log_level)
        logger.propagate = False  # Only log to our own handlers

        fmt = logging.Formatter(
            '[%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler with rotation
        if cfg.log_to_file:
            log_file = Path(cfg.log_file)
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=cfg.max_bytes, backupCount=cfg.backup_count, encoding='utf-8')
                file_handler.setFormatter(fmt)
                file_handler.setLevel(cfg.log_level)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Logger: File handler setup failed: {e}", file=sys.stderr)

        # Console handler
        if cfg.log_to_console or not logger.hasHandlers():
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(fmt)
            console_handler.setLevel(cfg.log_level)
            logger.addHandler(console_handler)

        _LOGGERS[name] = logger
        return logger

def set_verbosity(level: str = "INFO"):
    """Change verbosity for all loggers."""
    lvl = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)
    for logger in _LOGGERS.values():
        logger.setLevel(lvl)
        for handler in logger.handlers:
            handler.setLevel(lvl)

def log_traceback(logger: logging.Logger = None, exc: Exception = None, msg: str = "Unhandled Exception"):
    """Log traceback with optional message."""
    import traceback as tb
    logger = logger or get_logger()
    exc_info = sys.exc_info() if exc is None else (type(exc), exc, exc.__traceback__)
    logger.error(f"{msg}\n{''.join(tb.format_exception(*exc_info))}")

def clear_log_file(log_file: str = DEFAULT_LOG_FILE):
    """Clear log file content."""
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("")
    except Exception as e:
        print(f"Failed to clear log file: {e}", file=sys.stderr)

# -------------------------------
# Demo / Test Routine
# -------------------------------

if __name__ == "__main__":
    print("=== DharmaShield Logger Test ===")
    log = get_logger("Test")
    log.debug("Debug log")
    log.info("Info log")
    log.warning("Warning")
    log.error("Error")
    try:
        1/0
    except Exception as ex:
        log_traceback(log, ex)
    print("Switching to DEBUG verbosity")
    set_verbosity("DEBUG")
    log.debug("Now in DEBUG mode")
    print("Clearing log file...")
    clear_log_file()
    print("✅ Logger ready for production!")

