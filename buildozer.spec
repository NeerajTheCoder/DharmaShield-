[app]
title = DharmaShield
package.name = dharmashield  
package.domain = org.dharmashield.app

source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,wav,mp3,ogg,ttf,otf,yaml,json,pth,onnx,txt,md
source.include_patterns = assets/*,models/*,config/*,src/*,requirements.txt
source.exclude_dirs = tests,docs,.git,.buildozer,venv,__pycache__,.pytest_cache,.coverage,htmlcov,build,dist
source.exclude_patterns = *.pyc,*.pyo,*.spec,*.log,*.bak,*~

version = 2.0.0
version.regex = __version__ = ['"]([^'"]*?)['"]
version.filename = %(source.dir)s/src/__init__.py

requirements = python3==3.9.18,hostpython3==3.9.18,kivy==2.1.0,kivymd,numpy,torch==1.13.1,transformers==4.25.1,speechrecognition==3.8.1,pyttsx3==2.90,pyaudio==0.2.11,vosk==0.3.45,langdetect==1.0.9,pydantic==1.10.4,cryptography==38.0.4,psutil==5.9.4,pillow==9.4.0,opencv-python==4.7.0.68,qrcode==7.3.1,pyzbar==0.1.9,requests==2.28.2,aiohttp==3.8.3,pyyaml==6.0,cffi==1.15.1,cython==0.29.32

[buildozer]
log_level = 2
warn_on_root = 1

[android]
android.permissions = INTERNET,RECORD_AUDIO,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,CAMERA,VIBRATE,WAKE_LOCK,ACCESS_NETWORK_STATE,ACCESS_WIFI_STATE,MODIFY_AUDIO_SETTINGS,BLUETOOTH,BLUETOOTH_ADMIN
android.api = 31
android.minapi = 23  
android.ndk = 25c
android.sdk = 33
android.accept_sdk_license = True

android.arch = arm64-v8a
android.allow_backup = True
android.backup_rules = android/backup_rules.xml

icon.filename = assets/icons/icon.png
icon.adaptive_foreground = assets/icons/adaptive_foreground.png  
icon.adaptive_background = assets/icons/adaptive_background.png

presplash.filename = assets/splash/presplash.png
presplash.fill_color = #1a1a2e

orientation = portrait

android.add_src = android/src
android.gradle_dependencies = androidx.core:core:1.8.0,androidx.appcompat:appcompat:1.5.0
android.add_compile_options = sourceCompatibility JavaVersion.VERSION_1_8, targetCompatibility JavaVersion.VERSION_1_8

android.release_artifact = aab
android.debug_artifact = apk

android.enable_androidx = True
android.use_legacy_support = False

android.add_jars = 
android.add_aars = 

android.add_java_dir = android/src/main/java

android.gradle_template = android/build.tmpl.gradle
android.manifest_template = android/AndroidManifest.tmpl.xml

android.release_keystore = %(source.dir)s/android/release.keystore
android.release_keystore_passwd = 
android.release_key_alias = 
android.release_key_passwd =

[buildozer:linux]
docker_image = kivy/buildozer

[buildozer:osx] 
codesign.mode = adhoc

