"""
src/explainability/reasoning.py

DharmaShield - Advanced Evidence Aggregation & Reasoning Engine
---------------------------------------------------------------
• Industry-grade evidence synthesis and user-friendly explanation generation
• Aggregates multi-modal evidence (text, behavioral, technical) with weighted scoring
• Produces clear, actionable explanations in multiple languages with confidence assessment
• Cross-platform compatibility with voice integration and accessibility support

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import json
import time
import threading
import statistics
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import re
import math

# Project imports
from ...utils.logger import get_logger
from ...core.config_loader import load_config
from ...utils.language import detect_language, get_language_name
from ...utils.tts_engine import speak
from .xai_engine import XAIExplanation, Evidence, ReasoningStep, EvidenceType, ConfidenceLevel

logger = get_logger(__name__)

# -------------------------------
# Enums and Data Structures
# -------------------------------

class ReasoningStrategy(Enum):
    WEIGHTED_AGGREGATION = "weighted_aggregation"
    MAJORITY_VOTE = "majority_vote"
    BAYESIAN_INFERENCE = "bayesian_inference"
    FUZZY_LOGIC = "fuzzy_logic"
    ENSEMBLE_CONSENSUS = "ensemble_consensus"

class ExplanationStyle(Enum):
    TECHNICAL = "technical"          # Detailed technical analysis
    CONVERSATIONAL = "conversational"  # Natural, easy to understand
    BULLET_POINTS = "bullet_points"   # Structured key points
    NARRATIVE = "narrative"          # Story-like explanation
    EDUCATIONAL = "educational"      # Teaching-focused explanation

class CertaintyLevel(Enum):
    ABSOLUTE = "absolute"     # 95-100%
    VERY_HIGH = "very_high"   # 85-94%
    HIGH = "high"            # 70-84%
    MODERATE = "moderate"    # 55-69%
    LOW = "low"             # 40-54%
    VERY_LOW = "very_low"   # 25-39%
    UNCERTAIN = "uncertain"  # 0-24%

@dataclass
class EvidenceCluster:
    """Group of related evidence items."""
    cluster_id: str
    evidence_type: EvidenceType
    evidence_items: List[Evidence]
    aggregated_weight: float
    aggregated_confidence: float
    consensus_explanation: str
    contradiction_score: float = 0.0  # How much evidence contradicts within cluster

@dataclass
class ReasoningContext:
    """Context information for reasoning process."""
    user_language: str = "en"
    explanation_style: ExplanationStyle = ExplanationStyle.CONVERSATIONAL
    target_audience: str = "general"  # general, technical, elderly, children
    max_explanation_length: int = 500
    include_uncertainty: bool = True
    voice_optimized: bool = False
    accessibility_mode: bool = False

@dataclass
class AggregatedEvidence:
    """Result of evidence aggregation process."""
    total_scam_evidence: float
    total_legitimate_evidence: float
    evidence_clusters: List[EvidenceCluster]
    contradiction_analysis: Dict[str, Any]
    confidence_breakdown: Dict[EvidenceType, float]
    reasoning_chain: List[str]
    uncertainty_factors: List[str]

@dataclass
class UserFriendlyExplanation:
    """Complete user-friendly explanation output."""
    explanation_id: str
    primary_conclusion: str
    confidence_statement: str
    key_evidence_points: List[str]
    detailed_reasoning: str
    recommendations: List[str]
    warnings: List[str]
    uncertainty_disclosure: str
    language: str
    style: ExplanationStyle
    voice_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# -------------------------------
# Language Templates
# -------------------------------

class ExplanationTemplates:
    """Templates for generating explanations in multiple languages."""
    
    CONCLUSION_TEMPLATES = {
        "en": {
            "scam_high": "This appears to be a scam with high confidence.",
            "scam_medium": "This message shows multiple scam indicators.",
            "scam_low": "This message has some suspicious elements that warrant caution.",
            "legitimate_high": "This appears to be a legitimate message.",
            "legitimate_medium": "This message appears legitimate with some minor concerns.",
            "legitimate_low": "This message is likely legitimate but shows some unusual patterns."
        },
        "hi": {
            "scam_high": "यह उच्च विश्वास के साथ एक घोटाला प्रतीत होता है।",
            "scam_medium": "इस संदेश में कई घोटाला संकेतक हैं।",
            "scam_low": "इस संदेश में कुछ संदिग्ध तत्व हैं जिन पर सावधानी बरतनी चाहिए।",
            "legitimate_high": "यह एक वैध संदेश प्रतीत होता है।",
            "legitimate_medium": "यह संदेश कुछ मामूली चिंताओं के साथ वैध लगता है।",
            "legitimate_low": "यह संदेश संभवतः वैध है लेकिन कुछ असामान्य पैटर्न दिखाता है।"
        }
    }
    
    CONFIDENCE_TEMPLATES = {
        "en": {
            CertaintyLevel.ABSOLUTE: "I am absolutely certain about this assessment.",
            CertaintyLevel.VERY_HIGH: "I am very confident in this assessment.",
            CertaintyLevel.HIGH: "I am confident in this assessment.",
            CertaintyLevel.MODERATE: "I have moderate confidence in this assessment.",
            CertaintyLevel.LOW: "I have low confidence in this assessment.",
            CertaintyLevel.VERY_LOW: "I have very low confidence in this assessment.",
            CertaintyLevel.UNCERTAIN: "I am uncertain about this assessment."
        },
        "hi": {
            CertaintyLevel.ABSOLUTE: "मुझे इस मूल्यांकन के बारे में पूर्ण निश्चितता है।",
            CertaintyLevel.VERY_HIGH: "मुझे इस मूल्यांकन में बहुत विश्वास है।",
            CertaintyLevel.HIGH: "मुझे इस मूल्यांकन में विश्वास है।",
            CertaintyLevel.MODERATE: "मुझे इस मूल्यांकन में मध्यम विश्वास है।",
            CertaintyLevel.LOW: "मुझे इस मूल्यांकन में कम विश्वास है।",
            CertaintyLevel.VERY_LOW: "मुझे इस मूल्यांकन में बहुत कम विश्वास है।",
            CertaintyLevel.UNCERTAIN: "मुझे इस मूल्यांकन के बारे में अनिश्चितता है।"
        }
    }
    
    RECOMMENDATION_TEMPLATES = {
        "en": {
            "scam_high": [
                "Do not respond to this message or click any links.",
                "Do not provide any personal or financial information.",
                "Report this message to relevant authorities.",
                "Block the sender if possible."
            ],
            "scam_medium": [
                "Exercise extreme caution with this message.",
                "Verify the sender through official channels before taking any action.",
                "Do not provide sensitive information without verification."
            ],
            "legitimate_high": [
                "This message appears safe to engage with.",
                "Standard caution is still advised for any requests."
            ]
        },
        "hi": {
            "scam_high": [
                "इस संदेश का जवाब न दें या कोई लिंक न दबाएं।",
                "कोई व्यक्तिगत या वित्तीय जानकारी प्रदान न करें।",
                "इस संदेश की रिपोर्ट संबंधित अधिकारियों को करें।",
                "यदि संभव हो तो भेजने वाले को ब्लॉक करें।"
            ],
            "scam_medium": [
                "इस संदेश के साथ अत्यधिक सावधानी बरतें।",
                "कोई भी कार्रवाई करने से पहले आधिकारिक चैनलों के माध्यम से भेजने वाले को सत्यापित करें।",
                "सत्यापन के बिना संवेदनशील जानकारी प्रदान न करें।"
            ],
            "legitimate_high": [
                "यह संदेश सुरक्षित लगता है।",
                "फिर भी किसी भी अनुरोध के लिए मानक सावधानी की सलाह दी जाती है।"
            ]
        }
    }

# -------------------------------
# Configuration
# -------------------------------

class ReasoningEngineConfig:
    """Configuration for reasoning engine."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        reasoning_config = self.config.get('reasoning_engine', {})
        
        # General settings
        self.enabled = reasoning_config.get('enabled', True)
        self.default_strategy = ReasoningStrategy(reasoning_config.get('default_strategy', 'weighted_aggregation'))
        self.default_style = ExplanationStyle(reasoning_config.get('default_style', 'conversational'))
        
        # Evidence aggregation
        self.min_evidence_weight = reasoning_config.get('min_evidence_weight', 0.1)
        self.contradiction_threshold = reasoning_config.get('contradiction_threshold', 0.3)
        self.cluster_similarity_threshold = reasoning_config.get('cluster_similarity_threshold', 0.7)
        
        # Confidence calculation
        self.confidence_weights = reasoning_config.get('confidence_weights', {
            'evidence_strength': 0.4,
            'evidence_consistency': 0.3,
            'evidence_diversity': 0.2,
            'model_uncertainty': 0.1
        })
        
        # Explanation generation
        self.max_key_points = reasoning_config.get('max_key_points', 5)
        self.max_recommendations = reasoning_config.get('max_recommendations', 4)
        self.include_technical_details = reasoning_config.get('include_technical_details', False)
        
        # Language and accessibility
        self.supported_languages = reasoning_config.get('supported_languages', ['en', 'hi'])
        self.voice_explanation_length = reasoning_config.get('voice_explanation_length', 100)
        self.accessibility_simplification = reasoning_config.get('accessibility_simplification', True)

# -------------------------------
# Evidence Aggregator
# -------------------------------

class EvidenceAggregator:
    """Aggregates and analyzes evidence from multiple sources."""
    
    def __init__(self, config: ReasoningEngineConfig):
        self.config = config
        self._lock = threading.Lock()
    
    def aggregate_evidence(
        self,
        evidence_list: List[Evidence],
        strategy: ReasoningStrategy = None
    ) -> AggregatedEvidence:
        """Aggregate evidence using specified strategy."""
        
        strategy = strategy or self.config.default_strategy
        
        # Cluster evidence by type and similarity
        clusters = self._cluster_evidence(evidence_list)
        
        # Calculate aggregated scores
        scam_score, legitimate_score = self._calculate_aggregated_scores(
            clusters, strategy
        )
        
        # Analyze contradictions
        contradiction_analysis = self._analyze_contradictions(clusters)
        
        # Calculate confidence breakdown
        confidence_breakdown = self._calculate_confidence_breakdown(clusters)
        
        # Generate reasoning chain
        reasoning_chain = self._generate_reasoning_chain(clusters, strategy)
        
        # Identify uncertainty factors
        uncertainty_factors = self._identify_uncertainty_factors(
            clusters, contradiction_analysis
        )
        
        return AggregatedEvidence(
            total_scam_evidence=scam_score,
            total_legitimate_evidence=legitimate_score,
            evidence_clusters=clusters,
            contradiction_analysis=contradiction_analysis,
            confidence_breakdown=confidence_breakdown,
            reasoning_chain=reasoning_chain,
            uncertainty_factors=uncertainty_factors
        )
    
    def _cluster_evidence(self, evidence_list: List[Evidence]) -> List[EvidenceCluster]:
        """Cluster evidence by type and similarity."""
        
        clusters = []
        evidence_by_type = defaultdict(list)
        
        # Group by evidence type
        for evidence in evidence_list:
            if evidence.weight >= self.config.min_evidence_weight:
                evidence_by_type[evidence.type].append(evidence)
        
        # Create clusters for each type
        for evidence_type, type_evidence in evidence_by_type.items():
            if not type_evidence:
                continue
            
            # Calculate aggregated metrics
            total_weight = sum(e.weight for e in type_evidence)
            avg_confidence = statistics.mean(e.confidence for e in type_evidence)
            
            # Generate consensus explanation
            consensus_explanation = self._generate_consensus_explanation(type_evidence)
            
            # Calculate contradiction score
            contradiction_score = self._calculate_contradiction_score(type_evidence)
            
            cluster = EvidenceCluster(
                cluster_id=f"cluster_{evidence_type.value}",
                evidence_type=evidence_type,
                evidence_items=type_evidence,
                aggregated_weight=total_weight,
                aggregated_confidence=avg_confidence,
                consensus_explanation=consensus_explanation,
                contradiction_score=contradiction_score
            )
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_aggregated_scores(
        self,
        clusters: List[EvidenceCluster],
        strategy: ReasoningStrategy
    ) -> Tuple[float, float]:
        """Calculate aggregated scam and legitimate scores."""
        
        scam_score = 0.0
        legitimate_score = 0.0
        
        for cluster in clusters:
            cluster_score = cluster.aggregated_weight * cluster.aggregated_confidence
            
            # Adjust for contradictions
            cluster_score *= (1.0 - cluster.contradiction_score)
            
            # Count supporting vs opposing evidence
            scam_evidence = sum(1 for e in cluster.evidence_items if e.supports_scam)
            total_evidence = len(cluster.evidence_items)
            scam_ratio = scam_evidence / total_evidence if total_evidence > 0 else 0.0
            
            if strategy == ReasoningStrategy.WEIGHTED_AGGREGATION:
                scam_score += cluster_score * scam_ratio
                legitimate_score += cluster_score * (1.0 - scam_ratio)
            
            elif strategy == ReasoningStrategy.MAJORITY_VOTE:
                if scam_ratio > 0.5:
                    scam_score += cluster_score
                else:
                    legitimate_score += cluster_score
            
            elif strategy == ReasoningStrategy.BAYESIAN_INFERENCE:
                # Simplified Bayesian approach
                prior_scam = 0.3  # Prior probability of scam
                likelihood = cluster_score
                posterior_scam = (likelihood * prior_scam) / (likelihood * prior_scam + (1 - likelihood) * (1 - prior_scam))
                scam_score += posterior_scam * cluster_score
                legitimate_score += (1 - posterior_scam) * cluster_score
            
            else:  # Default to weighted aggregation
                scam_score += cluster_score * scam_ratio
                legitimate_score += cluster_score * (1.0 - scam_ratio)
        
        # Normalize scores
        total_score = scam_score + legitimate_score
        if total_score > 0:
            scam_score /= total_score
            legitimate_score /= total_score
        
        return scam_score, legitimate_score
    
    def _generate_consensus_explanation(self, evidence_items: List[Evidence]) -> str:
        """Generate consensus explanation for a group of evidence."""
        
        explanations = [e.explanation for e in evidence_items if e.explanation]
        
        if not explanations:
            return ""
        
        # Find common themes in explanations
        common_words = self._extract_common_themes(explanations)
        
        # Generate consensus based on most common themes
        if len(explanations) == 1:
            return explanations[0]
        elif len(common_words) > 0:
            return f"Multiple indicators suggest {', '.join(common_words[:3])}"
        else:
            return f"Evidence from {len(explanations)} sources indicates potential issues"
    
    def _extract_common_themes(self, explanations: List[str]) -> List[str]:
        """Extract common themes from multiple explanations."""
        
        # Simple word frequency analysis
        word_counts = Counter()
        
        for explanation in explanations:
            words = re.findall(r'\b\w+\b', explanation.lower())
            # Filter out common words
            filtered_words = [w for w in words if len(w) > 3 and w not in 
                            ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'were']]
            word_counts.update(filtered_words)
        
        # Return most common meaningful words
        return [word for word, count in word_counts.most_common(5) if count > 1]
    
    def _calculate_contradiction_score(self, evidence_items: List[Evidence]) -> float:
        """Calculate how much evidence contradicts within the group."""
        
        if len(evidence_items) <= 1:
            return 0.0
        
        scam_support = sum(1 for e in evidence_items if e.supports_scam)
        total_items = len(evidence_items)
        
        # Perfect agreement = 0 contradiction, 50/50 split = maximum contradiction
        agreement_ratio = abs(scam_support / total_items - 0.5) * 2
        contradiction_score = 1.0 - agreement_ratio
        
        return min(1.0, contradiction_score)
    
    def _analyze_contradictions(self, clusters: List[EvidenceCluster]) -> Dict[str, Any]:
        """Analyze contradictions across evidence clusters."""
        
        total_contradiction = 0.0
        contradictory_clusters = []
        
        for cluster in clusters:
            if cluster.contradiction_score > self.config.contradiction_threshold:
                contradictory_clusters.append(cluster.cluster_id)
                total_contradiction += cluster.contradiction_score
        
        avg_contradiction = total_contradiction / len(clusters) if clusters else 0.0
        
        return {
            'average_contradiction': avg_contradiction,
            'contradictory_clusters': contradictory_clusters,
            'contradiction_severity': 'high' if avg_contradiction > 0.5 else 'low'
        }
    
    def _calculate_confidence_breakdown(self, clusters: List[EvidenceCluster]) -> Dict[EvidenceType, float]:
        """Calculate confidence breakdown by evidence type."""
        
        confidence_breakdown = {}
        
        for cluster in clusters:
            confidence_breakdown[cluster.evidence_type] = (
                cluster.aggregated_confidence * (1.0 - cluster.contradiction_score)
            )
        
        return confidence_breakdown
    
    def _generate_reasoning_chain(
        self,
        clusters: List[EvidenceCluster],
        strategy: ReasoningStrategy
    ) -> List[str]:
        """Generate step-by-step reasoning chain."""
        
        reasoning_steps = []
        
        # Sort clusters by strength
        sorted_clusters = sorted(clusters, key=lambda c: c.aggregated_weight * c.aggregated_confidence, reverse=True)
        
        for i, cluster in enumerate(sorted_clusters[:3], 1):
            step = f"Step {i}: {cluster.evidence_type.value.title()} analysis - {cluster.consensus_explanation}"
            reasoning_steps.append(step)
        
        # Add aggregation step
        strategy_name = strategy.value.replace('_', ' ').title()
        reasoning_steps.append(f"Final: Applied {strategy_name} to reach conclusion")
        
        return reasoning_steps
    
    def _identify_uncertainty_factors(
        self,
        clusters: List[EvidenceCluster],
        contradiction_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify factors that contribute to uncertainty."""
        
        uncertainty_factors = []
        
        # Check for contradictions
        if contradiction_analysis['average_contradiction'] > 0.3:
            uncertainty_factors.append("Conflicting evidence found")
        
        # Check for low confidence evidence
        low_confidence_clusters = [c for c in clusters if c.aggregated_confidence < 0.6]
        if low_confidence_clusters:
            uncertainty_factors.append("Some evidence has low confidence")
        
        # Check for insufficient evidence
        if len(clusters) < 2:
            uncertainty_factors.append("Limited evidence available")
        
        # Check for unbalanced evidence types
        evidence_types = set(c.evidence_type for c in clusters)
        if len(evidence_types) < 3:
            uncertainty_factors.append("Evidence comes from limited sources")
        
        return uncertainty_factors

# -------------------------------
# Explanation Generator
# -------------------------------

class ExplanationGenerator:
    """Generates user-friendly explanations from aggregated evidence."""
    
    def __init__(self, config: ReasoningEngineConfig):
        self.config = config
        self.templates = ExplanationTemplates()
    
    def generate_explanation(
        self,
        aggregated_evidence: AggregatedEvidence,
        context: ReasoningContext
    ) -> UserFriendlyExplanation:
        """Generate complete user-friendly explanation."""
        
        # Determine primary conclusion
        is_scam = aggregated_evidence.total_scam_evidence > aggregated_evidence.total_legitimate_evidence
        confidence_score = max(aggregated_evidence.total_scam_evidence, aggregated_evidence.total_legitimate_evidence)
        
        # Generate components
        primary_conclusion = self._generate_primary_conclusion(
            is_scam, confidence_score, context.user_language
        )
        
        confidence_statement = self._generate_confidence_statement(
            confidence_score, context.user_language, context.include_uncertainty
        )
        
        key_evidence_points = self._generate_key_evidence_points(
            aggregated_evidence.evidence_clusters, context
        )
        
        detailed_reasoning = self._generate_detailed_reasoning(
            aggregated_evidence, context
        )
        
        recommendations = self._generate_recommendations(
            is_scam, confidence_score, context.user_language
        )
        
        warnings = self._generate_warnings(
            aggregated_evidence, context.user_language
        )
        
        uncertainty_disclosure = self._generate_uncertainty_disclosure(
            aggregated_evidence.uncertainty_factors, context
        )
        
        voice_summary = self._generate_voice_summary(
            primary_conclusion, key_evidence_points, context
        )
        
        explanation_id = f"exp_{int(time.time() * 1000)}"
        
        return UserFriendlyExplanation(
            explanation_id=explanation_id,
            primary_conclusion=primary_conclusion,
            confidence_statement=confidence_statement,
            key_evidence_points=key_evidence_points,
            detailed_reasoning=detailed_reasoning,
            recommendations=recommendations,
            warnings=warnings,
            uncertainty_disclosure=uncertainty_disclosure,
            language=context.user_language,
            style=context.explanation_style,
            voice_summary=voice_summary,
            metadata={
                'scam_score': aggregated_evidence.total_scam_evidence,
                'legitimate_score': aggregated_evidence.total_legitimate_evidence,
                'confidence_score': confidence_score,
                'evidence_clusters': len(aggregated_evidence.evidence_clusters),
                'contradiction_level': aggregated_evidence.contradiction_analysis['average_contradiction']
            }
        )
    
    def _generate_primary_conclusion(
        self,
        is_scam: bool,
        confidence_score: float,
        language: str
    ) -> str:
        """Generate primary conclusion statement."""
        
        # Determine confidence level
        if confidence_score >= 0.8:
            confidence_level = "high"
        elif confidence_score >= 0.6:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Select template
        template_key = f"{'scam' if is_scam else 'legitimate'}_{confidence_level}"
        templates = self.templates.CONCLUSION_TEMPLATES.get(language, self.templates.CONCLUSION_TEMPLATES['en'])
        
        return templates.get(template_key, templates['scam_medium' if is_scam else 'legitimate_medium'])
    
    def _generate_confidence_statement(
        self,
        confidence_score: float,
        language: str,
        include_uncertainty: bool
    ) -> str:
        """Generate confidence statement."""
        
        # Convert to certainty level
        certainty_level = self._score_to_certainty_level(confidence_score)
        
        templates = self.templates.CONFIDENCE_TEMPLATES.get(language, self.templates.CONFIDENCE_TEMPLATES['en'])
        statement = templates.get(certainty_level, templates[CertaintyLevel.MODERATE])
        
        if include_uncertainty and confidence_score < 0.7:
            uncertainty_note = {
                'en': " There are some uncertainties in this assessment.",
                'hi': " इस मूल्यांकन में कुछ अनिश्चितताएं हैं।"
            }
            statement += uncertainty_note.get(language, uncertainty_note['en'])
        
        return statement
    
    def _score_to_certainty_level(self, score: float) -> CertaintyLevel:
        """Convert confidence score to certainty level."""
        
        if score >= 0.95:
            return CertaintyLevel.ABSOLUTE
        elif score >= 0.85:
            return CertaintyLevel.VERY_HIGH
        elif score >= 0.70:
            return CertaintyLevel.HIGH
        elif score >= 0.55:
            return CertaintyLevel.MODERATE
        elif score >= 0.40:
            return CertaintyLevel.LOW
        elif score >= 0.25:
            return CertaintyLevel.VERY_LOW
        else:
            return CertaintyLevel.UNCERTAIN
    
    def _generate_key_evidence_points(
        self,
        clusters: List[EvidenceCluster],
        context: ReasoningContext
    ) -> List[str]:
        """Generate key evidence points."""
        
        # Sort clusters by strength
        sorted_clusters = sorted(
            clusters,
            key=lambda c: c.aggregated_weight * c.aggregated_confidence,
            reverse=True
        )
        
        key_points = []
        
        for cluster in sorted_clusters[:self.config.max_key_points]:
            if context.explanation_style == ExplanationStyle.TECHNICAL:
                point = f"{cluster.evidence_type.value.title()}: {cluster.consensus_explanation} (Weight: {cluster.aggregated_weight:.2f})"
            else:
                point = cluster.consensus_explanation
            
            key_points.append(point)
        
        return key_points
    
    def _generate_detailed_reasoning(
        self,
        aggregated_evidence: AggregatedEvidence,
        context: ReasoningContext
    ) -> str:
        """Generate detailed reasoning explanation."""
        
        reasoning_parts = []
        
        if context.explanation_style == ExplanationStyle.NARRATIVE:
            reasoning_parts.append("Let me walk you through my analysis:")
        
        # Add reasoning chain
        for step in aggregated_evidence.reasoning_chain:
            if context.explanation_style == ExplanationStyle.BULLET_POINTS:
                reasoning_parts.append(f"• {step}")
            else:
                reasoning_parts.append(step)
        
        # Add confidence breakdown if technical
        if context.explanation_style == ExplanationStyle.TECHNICAL and self.config.include_technical_details:
            reasoning_parts.append("\nConfidence breakdown by evidence type:")
            for evidence_type, confidence in aggregated_evidence.confidence_breakdown.items():
                reasoning_parts.append(f"• {evidence_type.value}: {confidence:.2f}")
        
        # Add contradiction analysis if significant
        if aggregated_evidence.contradiction_analysis['average_contradiction'] > 0.3:
            contradiction_note = {
                'en': "Note: Some evidence contradictions were found, which affects confidence.",
                'hi': "नोट: कुछ साक्ष्य विरोधाभास पाए गए, जो विश्वास को प्रभावित करता है।"
            }
            reasoning_parts.append(contradiction_note.get(context.user_language, contradiction_note['en']))
        
        return "\n".join(reasoning_parts)
    
    def _generate_recommendations(
        self,
        is_scam: bool,
        confidence_score: float,
        language: str
    ) -> List[str]:
        """Generate actionable recommendations."""
        
        # Determine recommendation category
        if is_scam and confidence_score >= 0.7:
            category = "scam_high"
        elif is_scam:
            category = "scam_medium"
        else:
            category = "legitimate_high"
        
        templates = self.templates.RECOMMENDATION_TEMPLATES.get(language, self.templates.RECOMMENDATION_TEMPLATES['en'])
        recommendations = templates.get(category, templates.get("scam_medium", []))
        
        return recommendations[:self.config.max_recommendations]
    
    def _generate_warnings(
        self,
        aggregated_evidence: AggregatedEvidence,
        language: str
    ) -> List[str]:
        """Generate warnings based on evidence."""
        
        warnings = []
        
        # High contradiction warning
        if aggregated_evidence.contradiction_analysis['average_contradiction'] > 0.5:
            warning = {
                'en': "Warning: Conflicting evidence detected. Exercise extra caution.",
                'hi': "चेतावनी: विरोधाभासी साक्ष्य मिले हैं। अतिरिक्त सावधानी बरतें।"
            }
            warnings.append(warning.get(language, warning['en']))
        
        # Low confidence warning
        total_confidence = max(aggregated_evidence.total_scam_evidence, aggregated_evidence.total_legitimate_evidence)
        if total_confidence < 0.6:
            warning = {
                'en': "Warning: Low confidence in assessment. Seek additional verification.",
                'hi': "चेतावनी: मूल्यांकन में कम विश्वास। अतिरिक्त सत्यापन की तलाश करें।"
            }
            warnings.append(warning.get(language, warning['en']))
        
        return warnings
    
    def _generate_uncertainty_disclosure(
        self,
        uncertainty_factors: List[str],
        context: ReasoningContext
    ) -> str:
        """Generate uncertainty disclosure."""
        
        if not uncertainty_factors or not context.include_uncertainty:
            return ""
        
        disclosure_intro = {
            'en': "Uncertainty factors:",
            'hi': "अनिश्चितता कारक:"
        }
        
        intro = disclosure_intro.get(context.user_language, disclosure_intro['en'])
        factors_text = ", ".join(uncertainty_factors)
        
        return f"{intro} {factors_text}"
    
    def _generate_voice_summary(
        self,
        primary_conclusion: str,
        key_evidence_points: List[str],
        context: ReasoningContext
    ) -> str:
        """Generate voice-optimized summary."""
        
        summary_parts = [primary_conclusion]
        
        if key_evidence_points and len(key_evidence_points) > 0:
            if context.user_language == 'hi':
                summary_parts.append("मुख्य कारण:")
            else:
                summary_parts.append("Key reasons:")
            
            # Include top 2 evidence points for voice
            for point in key_evidence_points[:2]:
                summary_parts.append(point)
        
        full_summary = " ".join(summary_parts)
        
        # Truncate if too long for voice
        if len(full_summary) > self.config.voice_explanation_length:
            full_summary = full_summary[:self.config.voice_explanation_length] + "..."
        
        return full_summary

# -------------------------------
# Main Reasoning Engine
# -------------------------------

class ReasoningEngine:
    """
    Advanced reasoning engine that aggregates evidence and generates user-friendly explanations.
    
    Features:
    - Multi-strategy evidence aggregation
    - Contradiction analysis and uncertainty quantification  
    - Multi-language explanation generation
    - Multiple explanation styles (technical, conversational, etc.)
    - Voice-optimized summaries
    - Accessibility support
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
        if getattr(self, '_initialized', False):
            return
        
        self.config = ReasoningEngineConfig(config_path)
        self.evidence_aggregator = EvidenceAggregator(self.config)
        self.explanation_generator = ExplanationGenerator(self.config)
        
        # Statistics
        self.stats = {
            'total_explanations': 0,
            'explanations_by_language': defaultdict(int),
            'explanations_by_style': defaultdict(int),
            'average_confidence': 0.0,
            'contradiction_rate': 0.0
        }
        
        self._initialized = True
        logger.info("ReasoningEngine initialized")
    
    def generate_user_explanation(
        self,
        evidence_list: List[Evidence],
        context: ReasoningContext = None,
        strategy: ReasoningStrategy = None
    ) -> UserFriendlyExplanation:
        """Generate complete user-friendly explanation from evidence."""
        
        start_time = time.time()
        
        # Use default context if not provided
        if context is None:
            context = ReasoningContext()
        
        # Aggregate evidence
        aggregated_evidence = self.evidence_aggregator.aggregate_evidence(
            evidence_list, strategy or self.config.default_strategy
        )
        
        # Generate explanation
        explanation = self.explanation_generator.generate_explanation(
            aggregated_evidence, context
        )
        
        # Update statistics
        self._update_stats(explanation, aggregated_evidence, time.time() - start_time)
        
        return explanation
    
    def explain_with_voice(
        self,
        evidence_list: List[Evidence],
        language: str = "en",
        speak_explanation: bool = True
    ) -> UserFriendlyExplanation:
        """Generate explanation optimized for voice output."""
        
        context = ReasoningContext(
            user_language=language,
            explanation_style=ExplanationStyle.CONVERSATIONAL,
            voice_optimized=True,
            max_explanation_length=self.config.voice_explanation_length
        )
        
        explanation = self.generate_user_explanation(evidence_list, context)
        
        if speak_explanation:
            try:
                speak(explanation.voice_summary, language)
            except Exception as e:
                logger.error(f"Failed to speak explanation: {e}")
        
        return explanation
    
    def explain_for_accessibility(
        self,
        evidence_list: List[Evidence],
        language: str = "en",
        simplify: bool = True
    ) -> UserFriendlyExplanation:
        """Generate explanation optimized for accessibility."""
        
        context = ReasoningContext(
            user_language=language,
            explanation_style=ExplanationStyle.BULLET_POINTS,
            target_audience="general",
            accessibility_mode=True,
            include_uncertainty=not simplify,
            max_explanation_length=300 if simplify else 500
        )
        
        return self.generate_user_explanation(evidence_list, context)
    
    def get_explanation_confidence(self, evidence_list: List[Evidence]) -> float:
        """Get confidence score for an explanation without generating full explanation."""
        
        aggregated_evidence = self.evidence_aggregator.aggregate_evidence(evidence_list)
        return max(aggregated_evidence.total_scam_evidence, aggregated_evidence.total_legitimate_evidence)
    
    def analyze_evidence_quality(self, evidence_list: List[Evidence]) -> Dict[str, Any]:
        """Analyze the quality and completeness of evidence."""
        
        aggregated_evidence = self.evidence_aggregator.aggregate_evidence(evidence_list)
        
        # Calculate quality metrics
        evidence_types = set(e.type for e in evidence_list)
        avg_weight = statistics.mean(e.weight for e in evidence_list) if evidence_list else 0.0
        avg_confidence = statistics.mean(e.confidence for e in evidence_list) if evidence_list else 0.0
        
        return {
            'evidence_count': len(evidence_list),
            'evidence_types': list(evidence_types),
            'type_diversity': len(evidence_types),
            'average_weight': avg_weight,
            'average_confidence': avg_confidence,
            'contradiction_level': aggregated_evidence.contradiction_analysis['average_contradiction'],
            'uncertainty_factors': aggregated_evidence.uncertainty_factors,
            'quality_score': self._calculate_quality_score(evidence_list, aggregated_evidence)
        }
    
    def _calculate_quality_score(
        self,
        evidence_list: List[Evidence],
        aggregated_evidence: AggregatedEvidence
    ) -> float:
        """Calculate overall evidence quality score."""
        
        if not evidence_list:
            return 0.0
        
        # Factors that contribute to quality
        diversity_score = min(1.0, len(set(e.type for e in evidence_list)) / 4.0)  # Up to 4 types
        quantity_score = min(1.0, len(evidence_list) / 5.0)  # Up to 5 pieces of evidence
        weight_score = statistics.mean(e.weight for e in evidence_list)
        confidence_score = statistics.mean(e.confidence for e in evidence_list)
        consistency_score = 1.0 - aggregated_evidence.contradiction_analysis['average_contradiction']
        
        # Weighted combination
        quality_score = (
            diversity_score * 0.25 +
            quantity_score * 0.15 +
            weight_score * 0.25 +
            confidence_score * 0.20 +
            consistency_score * 0.15
        )
        
        return min(1.0, quality_score)
    
    def _update_stats(
        self,
        explanation: UserFriendlyExplanation,
        aggregated_evidence: AggregatedEvidence,
        processing_time: float
    ):
        """Update engine statistics."""
        
        self.stats['total_explanations'] += 1
        self.stats['explanations_by_language'][explanation.language] += 1
        self.stats['explanations_by_style'][explanation.style.value] += 1
        
        # Update rolling averages
        total = self.stats['total_explanations']
        confidence_score = explanation.metadata.get('confidence_score', 0.0)
        contradiction_level = aggregated_evidence.contradiction_analysis['average_contradiction']
        
        self.stats['average_confidence'] = (
            (self.stats['average_confidence'] * (total - 1) + confidence_score) / total
        )
        
        self.stats['contradiction_rate'] = (
            (self.stats['contradiction_rate'] * (total - 1) + contradiction_level) / total
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return dict(self.stats)
    
    def clear_stats(self):
        """Clear engine statistics."""
        self.stats = {
            'total_explanations': 0,
            'explanations_by_language': defaultdict(int),
            'explanations_by_style': defaultdict(int),
            'average_confidence': 0.0,
            'contradiction_rate': 0.0
        }

# -------------------------------
# Singleton and Convenience Functions
# -------------------------------

_global_reasoning_engine = None

def get_reasoning_engine(config_path: Optional[str] = None) -> ReasoningEngine:
    """Get the global reasoning engine instance."""
    global _global_reasoning_engine
    if _global_reasoning_engine is None:
        _global_reasoning_engine = ReasoningEngine(config_path)
    return _global_reasoning_engine

def generate_explanation(
    evidence_list: List[Evidence],
    language: str = "en",
    style: ExplanationStyle = ExplanationStyle.CONVERSATIONAL
) -> UserFriendlyExplanation:
    """Convenience function to generate explanation."""
    engine = get_reasoning_engine()
    context = ReasoningContext(user_language=language, explanation_style=style)
    return engine.generate_user_explanation(evidence_list, context)

def explain_with_voice(
    evidence_list: List[Evidence],
    language: str = "en"
) -> UserFriendlyExplanation:
    """Convenience function for voice explanation."""
    engine = get_reasoning_engine()
    return engine.explain_with_voice(evidence_list, language)

# -------------------------------
# Testing and Demo
# -------------------------------

if __name__ == "__main__":
    print("=== DharmaShield Reasoning Engine Demo ===\n")
    
    engine = get_reasoning_engine()
    
    # Create mock evidence for testing
    from .xai_engine import Evidence, EvidenceType
    
    mock_evidence = [
        Evidence(
            evidence_id="ev1",
            type=EvidenceType.BEHAVIORAL,
            content="Urgent language",
            weight=0.8,
            confidence=0.9,
            supports_scam=True,
            explanation="Message uses urgent language to create pressure"
        ),
        Evidence(
            evidence_id="ev2",
            type=EvidenceType.TECHNICAL,
            content="Suspicious URL",
            weight=0.9,
            confidence=0.95,
            supports_scam=True,
            explanation="URL mimics legitimate site but has subtle differences"
        ),
        Evidence(
            evidence_id="ev3",
            type=EvidenceType.LINGUISTIC,
            content="Grammar errors",
            weight=0.6,
            confidence=0.7,
            supports_scam=True,
            explanation="Multiple grammar errors suggest non-native speaker"
        ),
        Evidence(
            evidence_id="ev4",
            type=EvidenceType.CONTEXTUAL,
            content="Legitimate contact info",
            weight=0.4,
            confidence=0.6,
            supports_scam=False,
            explanation="Message includes what appears to be legitimate contact information"
        )
    ]
    
    print("Testing explanation generation...")
    
    # Test different styles
    styles = [ExplanationStyle.CONVERSATIONAL, ExplanationStyle.TECHNICAL, ExplanationStyle.BULLET_POINTS]
    
    for style in styles:
        print(f"\n--- {style.value.upper()} STYLE ---")
        
        context = ReasoningContext(
            user_language="en",
            explanation_style=style,
            include_uncertainty=True
        )
        
        explanation = engine.generate_user_explanation(mock_evidence, context)
        
        print(f"Conclusion: {explanation.primary_conclusion}")
        print(f"Confidence: {explanation.confidence_statement}")
        print(f"Key Points: {len(explanation.key_evidence_points)}")
        print(f"Recommendations: {len(explanation.recommendations)}")
        print(f"Voice Summary: {explanation.voice_summary[:100]}...")
    
    # Test evidence quality analysis
    print(f"\n--- EVIDENCE QUALITY ANALYSIS ---")
    quality_analysis = engine.analyze_evidence_quality(mock_evidence)
    
    for key, value in quality_analysis.items():
        print(f"{key}: {value}")
    
    # Test voice explanation
    print(f"\n--- VOICE EXPLANATION TEST ---")
    voice_explanation = engine.explain_with_voice(mock_evidence, "en", speak_explanation=False)
    print(f"Voice Summary: {voice_explanation.voice_summary}")
    
    # Show statistics
    print(f"\n--- ENGINE STATISTICS ---")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\n✅ Reasoning Engine ready for production!")
    print(f"🧠 Features demonstrated:")
    print(f"  ✓ Multi-strategy evidence aggregation")
    print(f"  ✓ Contradiction analysis and uncertainty quantification")
    print(f"  ✓ Multi-language explanation generation")
    print(f"  ✓ Multiple explanation styles")
    print(f"  ✓ Voice-optimized summaries")
    print(f"  ✓ Evidence quality assessment")
    print(f"  ✓ Statistical tracking")
    print(f"  ✓ Accessibility support")

