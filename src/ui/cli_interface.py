"""
src/ui/cli_interface.py

DharmaShield - Advanced CLI Interface Engine
--------------------------------------------
• Industry-grade command-line interface for dev/test/low-resource environments with full feature parity
• Cross-platform (Android/iOS/Desktop) with Kivy/Buildozer compatibility and multilingual support
• Modular architecture with interactive commands, batch processing, and comprehensive testing modes
• Full integration with all DharmaShield subsystems: detection, guidance, crisis support, wellness coaching

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import os
import sys
import time
import json
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import traceback
import argparse
from collections import defaultdict

# Project imports
from ...utils.logger import get_logger
from ...core.config_loader import load_config
from ...utils.language import detect_language, get_language

