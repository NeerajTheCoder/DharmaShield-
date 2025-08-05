"""
src/education/knowledge_base.py

Educational knowledge base for scam awareness, crisis support, and user empowerment.
Integrates with voice interface and supports all languages from your system.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from ..utils.language import get_language_name, list_supported, detect_language

class KnowledgeBase:
    """
    Comprehensive scam education system with multilingual support.
    Provides facts, crisis guidance, FAQ responses, and educational content.
    """
    
    def __init__(self, language="en", knowledge_file="config/knowledge_base.json"):
        self.language = language
        self.knowledge_file = knowledge_file
        self.supported_languages = list_supported()
        self.knowledge_data = {}
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load educational content from JSON file or create default content"""
        kb_path = Path(self.knowledge_file)
        
        if kb_path.exists():
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    self.knowledge_data = json.load(f)
            except Exception:
                self._create_default_knowledge()
        else:
            self._create_default_knowledge()
    
    def _create_default_knowledge(self):
        """Create comprehensive default knowledge base"""
        self.knowledge_data = {
            "scam_facts": {
                "en": [
                    "Never share OTP, PIN, or passwords with anyone over phone or SMS",
                    "Banks never ask for passwords or PINs through calls or messages",
                    "If something sounds too good to be true, it probably is a scam",
                    "Always verify URLs before clicking - look for 'https' and correct spelling",
                    "Scammers create urgency - take time to think and verify",
                    "Government agencies don't demand immediate payment via gift cards",
                    "Be suspicious of unsolicited calls asking for personal information"
                ],
                "hi": [
                    "OTP, PIN या पासवर्ड कभी किसी को फोन या SMS पर न बताएं",
                    "बैंक कभी भी कॉल या मैसेज के जरिए पासवर्ड नहीं मांगता",
                    "अगर कोई ऑफर बहुत अच्छा लगे तो समझें यह धोखा हो सकता है",
                    "लिंक पर क्लिक करने से पहले URL की जांच करें",
                    "ठग जल्दबाजी कराते हैं - सोचने का समय लें",
                    "सरकारी एजेंसियां गिफ्ट कार्ड से पेमेंट नहीं मांगती",
                    "अनजान कॉल से सावधान रहें"
                ],
                "es": [
                    "Nunca compartas OTP, PIN o contraseñas por teléfono o SMS",
                    "Los bancos nunca piden contraseñas por llamadas o mensajes",
                    "Si algo suena demasiado bueno para ser verdad, probablemente es una estafa"
                ]
            },
            "crisis_guidelines": {
                "en": [
                    "If you've shared sensitive info, contact your bank immediately",
                    "Don't panic - document everything and report to authorities",
                    "Change all passwords and PINs right away",
                    "Monitor your accounts for unauthorized transactions",
                    "Report the scam to local police and cybercrime cells"
                ],
                "hi": [
                    "अगर आपने जानकारी दे दी है तो तुरंत बैंक से संपर्क करें",
                    "घबराएं नहीं - सब कुछ लिख कर रखें और शिकायत करें",
                    "सभी पासवर्ड और PIN तुरंत बदलें",
                    "अपने खातों की निगरानी करें",
                    "पुलिस और साइबर क्राइम सेल में शिकायत दर्ज कराएं"
                ]
            },
            "prevention_tips": {
                "en": [
                    "Enable two-factor authentication on all accounts",
                    "Regularly check bank statements and credit reports",
                    "Use official apps downloaded from verified app stores",
                    "Keep your software and apps updated",
                    "Be cautious with public Wi-Fi for financial transactions"
                ],
                "hi": [
                    "सभी खातों पर दो-चरणीय सत्यापन चालू करें",
                    "बैंक स्टेटमेंट और क्रेडिट रिपोर्ट नियमित चेक करें",
                    "केवल वेरिफाइड ऐप स्टोर से ऐप डाउनलोड करें",
                    "अपने सॉफ्टवेयर को अपडेट रखें",
                    "पब्लिक Wi-Fi पर वित्तीय लेनदेन से बचें"
                ]
            },
            "faq_responses": {
                "phishing": {
                    "en": "Phishing is when scammers impersonate trusted organizations to steal your personal information through fake emails, messages, or websites.",
                    "hi": "फिशिंग में ठग भरोसेमंद संस्थाओं का रूप धारण कर नकली ईमेल, मैसेज या वेबसाइट के जरिए आपकी निजी जानकारी चुराते हैं।"
                },
                "upi": {
                    "en": "Never approve UPI requests you didn't initiate. Always verify the merchant and amount before confirming any payment.",
                    "hi": "UPI रिक्वेस्ट को बिना जांचे कभी approve न करें। पेमेंट करने से पहले मर्चेंट और रकम की जांच करें।"
                },
                "identity_theft": {
                    "en": "Identity theft occurs when criminals use your personal information without permission. Protect your Aadhaar, PAN, and other documents.",
                    "hi": "पहचान चोरी में अपराधी आपकी निजी जानकारी का गलत इस्तेमाल करते हैं। अपने आधार, PAN और अन्य दस्तावेजों की सुरक्षा करें।"
                }
            }
        }
    
    def get_random_fact(self) -> str:
        """Get a random educational fact in user's language"""
        facts = self.knowledge_data.get("scam_facts", {}).get(self.language, [])
        if not facts:
            facts = self.knowledge_data.get("scam_facts", {}).get("en", [])
        
        if facts:
            return random.choice(facts)
        return "Stay alert and verify before sharing any personal information."
    
    def get_crisis_guidance(self) -> str:
        """Get crisis management guidance"""
        guidelines = self.knowledge_data.get("crisis_guidelines", {}).get(self.language, [])
        if not guidelines:
            guidelines = self.knowledge_data.get("crisis_guidelines", {}).get("en", [])
        
        if guidelines:
            return random.choice(guidelines)
        return "If you suspect fraud, immediately contact your bank and change your passwords."
    
    def get_prevention_tip(self) -> str:
        """Get a prevention tip"""
        tips = self.knowledge_data.get("prevention_tips", {}).get(self.language, [])
        if not tips:
            tips = self.knowledge_data.get("prevention_tips", {}).get("en", [])
        
        if tips:
            return random.choice(tips)
        return "Always verify before you trust any financial communication."
    
    def answer_faq(self, query: str) -> str:
        """Answer frequently asked questions"""
        query_lower = query.lower()
        
        # Check for specific topics
        if any(keyword in query_lower for keyword in ["phishing", "fishing", "फिशिंग"]):
            return self._get_faq_response("phishing")
        elif any(keyword in query_lower for keyword in ["upi", "payment", "पेमेंट"]):
            return self._get_faq_response("upi")
        elif any(keyword in query_lower for keyword in ["identity", "theft", "पहचान"]):
            return self._get_faq_response("identity_theft")
        else:
            # Return a general fact if no specific topic matched
            return self.get_random_fact()
    
    def _get_faq_response(self, topic: str) -> str:
        """Get FAQ response for specific topic"""
        responses = self.knowledge_data.get("faq_responses", {}).get(topic, {})
        response = responses.get(self.language) or responses.get("en", "")
        return response or self.get_random_fact()
    
    def set_language(self, language: str):
        """Change the language for responses"""
        if language in self.supported_languages:
            self.language = language
    
    def get_educational_content_by_threat_level(self, threat_level: int) -> str:
        """Get educational content based on threat level"""
        education_by_level = {
            4: {
                "en": "CRITICAL: This appears to be a dangerous scam. Never share personal information through unsolicited communications.",
                "hi": "गंभीर: यह खतरनाक घोटाला लगता है। बिना मांगी गई बातचीत में निजी जानकारी कभी न दें।"
            },
            3: {
                "en": "HIGH RISK: Be very cautious. Verify the source independently before taking any action.",
                "hi": "उच्च जोखिम: बहुत सावधान रहें। कोई भी कार्रवाई से पहले स्रोत की स्वतंत्र जांच करें।"
            },
            2: {
                "en": "MEDIUM RISK: This could be suspicious. Take time to verify before proceeding.",
                "hi": "मध्यम जोखिम: यह संदिग्ध हो सकता है। आगे बढ़ने से पहले सत्यापन करें।"
            },
            1: {
                "en": "LOW RISK: Appears relatively safe, but always stay vigilant.",
                "hi": "कम जोखिम: अपेक्षाकृत सुरक्षित लगता है, लेकिन हमेशा सतर्क रहें।"
            },
            0: {
                "en": "SAFE: No immediate threats detected, but continue practicing good security habits.",
                "hi": "सुरक्षित: कोई तत्काल खतरा नहीं मिला, लेकिन अच्छी सुरक्षा आदतें जारी रखें।"
            }
        }
        
        content = education_by_level.get(threat_level, {})
        return content.get(self.language) or content.get("en", "Stay alert and verify information.")

# Example usage
if __name__ == "__main__":
    kb = KnowledgeBase(language="hi")
    print("Random Fact:", kb.get_random_fact())
    print("Crisis Guide:", kb.get_crisis_guidance())
    print("FAQ Response:", kb.answer_faq("What is phishing?"))

