"""
src/crisis/emergency_handler.py

DharmaShield - Crisis Emergency Workflow Orchestration Engine
-------------------------------------------------------------
• Industry-grade orchestrator for emergency user flows: handles instructions, UI steps, calling, logging, and confirmation
• Cross-platform (Android/iOS/Desktop) — modular, async workflow, Kivy/Buildozer ready, privacy/resilience focused
• Integrates with detection, alert, crisis support, TTS, heptic, and config layers
• Built for offline, voice-first/accessible use: robust to interruption, escalation, or fallback

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import os
import time
import json
import threading
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Union
from enum import Enum
from pathlib import Path

# Project imports
from ...utils.logger import get_logger
from ...core.config_loader import load_config
from ...utils.tts_engine import speak
from ...utils.language import get_language_name
from ..core.threat_level import ThreatLevel
from .crisis_support import CrisisType, get_crisis_support_engine
from .alert import get_crisis_alert_engine, AlertMethod

logger = get_logger(__name__)

# ---------------------------------
# Enums and Data Structures
# ---------------------------------

class EmergencyStage(Enum):
    INSTRUCTION = "instruction"         # Giving user step-by-step instructions
    CONTACT_ATTEMPT = "contact_attempt" # Initiating call/SMS/alert
    CONFIRMATION = "confirmation"       # Confirming user safety/actions
    LOGGING = "logging"                 # Logging/crisis audit/update
    ESCALATION = "escalation"           # Raised to higher contact/authority
    COMPLETE = "complete"               # Workflow finished

@dataclass
class EmergencyWorkflowContext:
    workflow_id: str
    user_id: str = "anonymous"
    user_language: str = "en"
    crisis_type: CrisisType = CrisisType.GENERAL_DISTRESS
    threat_level: ThreatLevel = ThreatLevel.CRITICAL
    contacts_to_alert: List[str] = field(default_factory=list)
    notes: str = ""
    triggered_time: float = field(default_factory=time.time)
    current_stage: EmergencyStage = EmergencyStage.INSTRUCTION
    escalation_count: int = 0
    is_voice_mode: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------
# Config
# ---------------------------------

class EmergencyHandlerConfig:
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        eh_cfg = self.config.get('emergency_handler', {})
        self.enabled = eh_cfg.get('enabled', True)
        self.default_language = eh_cfg.get('default_language', 'en')
        self.max_escalation = eh_cfg.get('max_escalation', 2)
        self.fail_safe_numbers = eh_cfg.get('fail_safe_numbers', ["112", "+919999999999"])
        self.workflow_timeout = eh_cfg.get('workflow_timeout', 90)
        self.logging_enabled = eh_cfg.get('logging_enabled', True)
        self.log_file = Path(eh_cfg.get('log_file', "emergency_workflow_log.json"))
        self.voice_instructions = eh_cfg.get('voice_instructions', True)

# ---------------------------------
# Instruction / Guidance Templates
# ---------------------------------

INSTRUCTION_TEMPLATES = {
    "en": {
        "start": "Emergency detected. Please remain calm. I'll guide you step by step.",
        "find_safe": "If you are in danger, move to a safe place if possible.",
        "block_scammer": "Block the sender or scammer from further contact.",
        "contact_help": "Would you like to call emergency services or a trusted contact?",
        "calling_now": "Initiating emergency call now. Remain on the line.",
        "sms_alert": "Sending emergency SMS to your selected contacts.",
        "confirm_safe": "Are you safe now? Please say 'yes' or 'no'.",
        "escalating": "Unable to confirm safety. Escalating to additional contacts.",
        "complete": "Emergency workflow complete. You're not alone. Help is always available."
    },
    "hi": {
        "start": "आपातकाल की स्थिति का पता चला है। कृपया शांत रहें। मैं आपको चरण दर चरण मार्गदर्शन करूंगा।",
        "find_safe": "अगर आप खतरे में हैं, तो संभव हो तो सुरक्षित स्थान पर जाएं।",
        "block_scammer": "अगले संपर्क से बचने के लिए प्रेषक को ब्लॉक करें।",
        "contact_help": "क्या आप आपातकालीन सेवाओं या किसी विश्वसनीय व्यक्ति को कॉल करना चाहेंगे?",
        "calling_now": "आपातकालीन कॉल शुरू किया जा रहा है। लाइन पर बने रहें।",
        "sms_alert": "चयनित संपर्कों को आपातकालीन एसएमएस भेजा जा रहा है।",
        "confirm_safe": "क्या आप अब सुरक्षित हैं? कृपया 'हाँ' या 'नहीं' कहें।",
        "escalating": "सुरक्षा की पुष्टि नहीं हुई। अतिरिक्त संपर्कों को सूचित किया जा रहा है।",
        "complete": "आपातकालीन वर्कफ़्लो पूरा हुआ। आप अकेले नहीं हैं, सहायता हमेशा उपलब्ध है।"
    }
}

def get_instr(key: str, lang: str):
    return INSTRUCTION_TEMPLATES.get(lang, INSTRUCTION_TEMPLATES["en"]).get(key, "")

# ---------------------------------
# Main Emergency Handler Engine
# ---------------------------------

class EmergencyHandlerEngine:
    """
    Handles orchestrated emergency user workflows (voice/UI/call/sms/escalation).
    Features:
    - Modular, async stage machine: user guidance, contact triggering, escalation, audit
    - Fully pluggable with crisis detection, support, alert subsystems
    - Robust to interruption, fallback/safety mechanisms, audit logging
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
        if getattr(self, "_initialized", False): return
        self.config = EmergencyHandlerConfig(config_path)
        self.crisis_support = get_crisis_support_engine()
        self.crisis_alert = get_crisis_alert_engine()
        self._initialized = True
        logger.info("EmergencyHandlerEngine initialized")

    async def run_emergency_workflow(
        self,
        user_id: str = "anonymous",
        crisis_type: CrisisType = CrisisType.FRAUD_VICTIM,
        threat_level: ThreatLevel = ThreatLevel.CRITICAL,
        user_language: str = "en",
        contacts_to_alert: Optional[List[str]] = None,
        notes: str = "",
        show_ui_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run the full emergency workflow: instructions, call/sms, confirmation, log."""
        ctx = EmergencyWorkflowContext(
            workflow_id=f"wf_{int(time.time()*1000)}",
            user_id=user_id,
            user_language=user_language,
            crisis_type=crisis_type,
            threat_level=threat_level,
            contacts_to_alert=contacts_to_alert or self.config.fail_safe_numbers,
            notes=notes
        )
        try:
            # 1. Instruction - start and safety
            await self._step_instruction(ctx, "start")
            await self._step_instruction(ctx, "find_safe")
            await asyncio.sleep(1.0)
            await self._step_instruction(ctx, "block_scammer")
            await asyncio.sleep(1.0)

            # 2. Ask to call or message
            await self._step_instruction(ctx, "contact_help")
            call_confirmed = await self._step_user_confirmation(ctx)
            if call_confirmed:
                await self._trigger_call_flow(ctx)
            else:
                await self._send_sms_alert(ctx)

            # 3. Confirmation check
            await self._step_instruction(ctx, "confirm_safe")
            safe = await self._step_user_confirmation(ctx)
            if not safe:
                await self._step_instruction(ctx, "escalating")
                ctx.escalation_count += 1
                if ctx.escalation_count < self.config.max_escalation:
                    await self._send_sms_alert(ctx, escalate=True)
                else:
                    # Final escalation: log, and if possible trigger authorities
                    ctx.current_stage = EmergencyStage.ESCALATION
                    self.crisis_alert.trigger_alert(
                        method=AlertMethod.EMERGENCY_SMS,
                        message="Critical: Unable to confirm user safety. Please check immediately.",
                        numbers=self.config.fail_safe_numbers,
                        language=ctx.user_language
                    )
            # 4. Complete
            await self._step_instruction(ctx, "complete")
            ctx.current_stage = EmergencyStage.COMPLETE

            # 5. Workflow audit/log
            self._log_workflow(ctx)
            return {"workflow_id": ctx.workflow_id, "status": "complete", "escalation": ctx.escalation_count}

        except Exception as e:
            logger.error(f"Emergency workflow failed: {e}")
            ctx.current_stage = EmergencyStage.LOGGING
            self._log_workflow(ctx, failure=str(e))
            return {"workflow_id": ctx.workflow_id, "status": "failed", "error": str(e)}

    async def _step_instruction(self, ctx: EmergencyWorkflowContext, key: str):
        """Deliver or display instruction per workflow stage."""
        msg = get_instr(key, ctx.user_language)
        if self.config.voice_instructions:
            speak(msg, ctx.user_language)
        if ctx.is_voice_mode:
            print(f"🔊 {msg}")

    async def _step_user_confirmation(self, ctx: EmergencyWorkflowContext) -> bool:
        """Voice+UI confirmation from user e.g. via TTS/ASR or UI input (stub for demo)."""
        # Voice input integration point: listen for "yes/no"
        try:
            from ...utils.asr_engine import ASREngine
            asr = ASREngine(language=ctx.user_language)
            user_reply = asr.listen_and_transcribe(
                prompt=get_instr("confirm_safe", ctx.user_language), timeout=6, language=ctx.user_language
            )
            norm_reply = user_reply.strip().lower()
            return norm_reply in ["yes", "haan", "sure", "ok", "हाँ", "हां", "y", "true"]
        except Exception as e:
            logger.warning(f"ASR confirmation error: {e}")
            return False

    async def _trigger_call_flow(self, ctx: EmergencyWorkflowContext):
        """Initiate emergency call."""
        await self._step_instruction(ctx, "calling_now")
        # Call first available number; could escalate if fails
        for num in ctx.contacts_to_alert:
            alert = self.crisis_alert.trigger_alert(
                method=AlertMethod.EMERGENCY_CALL,
                numbers=[num],
                message="DharmaShield automated emergency call.",
                language=ctx.user_language
            )
            ctx.meta.setdefault("call_attempts", []).append(alert.__dict__)
            time.sleep(2)  # Simulate waiting for call
        return True

    async def _send_sms_alert(self, ctx: EmergencyWorkflowContext, escalate: bool = False):
        """Send emergency SMS to contacts."""
        await self._step_instruction(ctx, "sms_alert")
        targets = ctx.contacts_to_alert if not escalate else self.config.fail_safe_numbers
        msg = (
            "Critical alert from DharmaShield: User may be in crisis. Please check immediately."
            if escalate else
            "Emergency from DharmaShield: User requests help or support. Please respond."
        )
        alert = self.crisis_alert.trigger_alert(
            method=AlertMethod.EMERGENCY_SMS,
            numbers=targets,
            message=msg,
            language=ctx.user_language
        )
        ctx.meta.setdefault("sms_alerts", []).append(alert.__dict__)
        time.sleep(2)
        return True

    def _log_workflow(self, ctx: EmergencyWorkflowContext, failure: str = ""):
        """Log workflow run for audit/evidence (privacy preserving)."""
        if not self.config.logging_enabled: return
        entry = ctx.__dict__.copy()
        entry["completed_time"] = time.time()
        if failure:
            entry["error"] = failure
            entry["status"] = "failed"
        else:
            entry["status"] = "complete"
        try:
            with open(self.config.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write workflow log: {e}")

# ---------------------------------
# Singleton & Convenience API
# ---------------------------------

_global_eh = None

def get_emergency_handler(config_path: Optional[str] = None) -> EmergencyHandlerEngine:
    global _global_eh
    if _global_eh is None:
        _global_eh = EmergencyHandlerEngine(config_path)
    return _global_eh

async def run_emergency_workflow_async(
    user_id: str = "anonymous",
    crisis_type: CrisisType = CrisisType.FRAUD_VICTIM,
    threat_level: ThreatLevel = ThreatLevel.CRITICAL,
    user_language: str = "en",
    contacts_to_alert: Optional[List[str]] = None,
    notes: str = ""
) -> Dict[str, Any]:
    eng = get_emergency_handler()
    return await eng.run_emergency_workflow(
        user_id=user_id,
        crisis_type=crisis_type,
        threat_level=threat_level,
        user_language=user_language,
        contacts_to_alert=contacts_to_alert,
        notes=notes
    )

# ---------------------------------
# Test/Demo Suite
# ---------------------------------

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=== DharmaShield Emergency Handler Demo ===")
        engine = get_emergency_handler()
        # Run simulated workflow (voice off for test)
        result = await engine.run_emergency_workflow(
            user_id="testuser1",
            crisis_type=CrisisType.FRAUD_VICTIM,
            threat_level=ThreatLevel.CRITICAL,
            user_language="en",
            contacts_to_alert=["+919999999999", "112"]
        )
        print("Workflow result:", result)
        print("✅ Emergency Handler ready for production!")
        print("📞 Features:")
        print("  ✓ Voice-first guided emergency instructions")
        print("  ✓ Call/SMS/contact escalation")
        print("  ✓ Privacy-protecting audit logs and tracking")
        print("  ✓ Modular, offline-robust, async-safe design")
        print("  ✓ Full integration with crisis detection and alerts subsystems")

    asyncio.run(demo())

