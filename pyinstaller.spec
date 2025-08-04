# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, 'src')

block_cipher = None

# Platform-specific settings
if sys.platform == 'win32':
    icon_file = 'assets/icons/icon.ico'
    console = False
    separator = ';'
elif sys.platform == 'darwin':
    icon_file = 'assets/icons/icon.icns'
    console = False
    separator = ':'
else:  # Linux
    icon_file = None
    console = False
    separator = ':'

# Data files to include
datas = [
    ('config/', 'config/'),
    ('assets/', 'assets/'),
    ('models/', 'models/'),
    ('src/ui/templates/', 'ui/templates/'),
    ('src/core/data/', 'core/data/'),
]

# Hidden imports for bundling
hiddenimports = [
    # Kivy core
    'kivy.deps.sdl2',
    'kivy.deps.glew', 
    'kivy.deps.gstreamer',
    'kivymd.icon_definitions',
    'kivymd.material_resources',
    
    # ML/AI
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'transformers',
    'transformers.models.gemma',
    'numpy',
    'numpy.core',
    'numpy.core._multiarray_umath',
    
    # Voice processing
    'speechrecognition',
    'pyttsx3',
    'pyttsx3.drivers',
    'pyttsx3.drivers.sapi5',
    'pyttsx3.drivers.nsss',
    'pyttsx3.drivers.espeak',
    'vosk',
    'pyaudio',
    'wave',
    'audioop',
    
    # Language detection
    'langdetect',
    'langdetect.lang_detect_exception',
    
    # Security & Privacy
    'cryptography',
    'cryptography.hazmat',
    'cryptography.hazmat.primitives',
    'cryptography.hazmat.backends',
    'nacl',
    'nacl.secret',
    'nacl.utils',
    
    # Image processing
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'cv2',
    'qrcode',
    'pyzbar',
    'pyzbar.pyzbar',
    
    # System utilities
    'psutil',
    'platform',
    'pathlib',
    'asyncio',
    'concurrent.futures',
    'multiprocessing',
    'threading',
    'sqlite3',
    'json',
    'yaml',
    'requests',
    'aiohttp',
    
    # Pydantic
    'pydantic',
    'pydantic.dataclasses',
    'pydantic.json',
]

# Binary files to include
binaries = []

# Add PyTorch libraries
if sys.platform == 'win32':
    torch_libs = [
        ('venv/Lib/site-packages/torch/lib/*.dll', 'torch/lib/'),
        ('venv/Lib/site-packages/torch/bin/*.dll', 'torch/bin/'),
    ]
elif sys.platform == 'darwin':
    torch_libs = [
        ('venv/lib/python*/site-packages/torch/lib/*.dylib', 'torch/lib/'),
    ]
else:  # Linux
    torch_libs = [
        ('venv/lib/python*/site-packages/torch/lib/*.so*', 'torch/lib/'),
    ]

binaries.extend(torch_libs)

# Add audio libraries
if sys.platform == 'win32':
    audio_libs = [
        ('C:/Windows/System32/winmm.dll', '.'),
    ]
    binaries.extend(audio_libs)

# Exclude unnecessary modules
excludes = [
    'tkinter',
    'matplotlib',
    'scipy',
    'pandas',
    'jupyter',
    'notebook',
    'IPython',
    'setuptools',
    'distutils',
    'test',
    'unittest',
    'doctest',
    'pydoc',
    'xml',
    'email',
    'http',
    'urllib',
    'html',
    'bz2',
    'lzma',
    'zipfile',
    'tarfile',
]

a = Analysis(
    ['src/main.py'],
    pathex=['.', 'src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate binaries
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DharmaShield',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=console,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
    version='version_info.txt'
)

# macOS app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='DharmaShield.app',
        icon=icon_file,
        bundle_identifier='org.dharmashield.app',
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSAppleScriptEnabled': False,
            'CFBundleShortVersionString': '2.0.0',
            'CFBundleVersion': '2.0.0',
            'NSMicrophoneUsageDescription': 'DharmaShield needs microphone access for voice commands and scam detection.',
            'NSCameraUsageDescription': 'DharmaShield needs camera access for QR code and document scanning.',
            'NSPhotoLibraryUsageDescription': 'DharmaShield needs photo access to analyze images for scams.',
            'LSMinimumSystemVersion': '10.13.0',
            'NSHighResolutionCapable': True,
        },
    )

# Linux .desktop file
if sys.platform.startswith('linux'):
    # Create .desktop file for Linux
    desktop_content = """[Desktop Entry]
Version=1.0
Type=Application
Name=DharmaShield
Comment=Advanced Scam Detection with Voice AI
Exec=./DharmaShield
Icon=dharmashield
Terminal=false
Categories=Utility;Security;
Keywords=scam;detection;voice;AI;security;
"""
    
    with open('dist/dharmashield.desktop', 'w') as f:
        f.write(desktop_content)
        
