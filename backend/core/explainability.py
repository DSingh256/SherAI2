"""
VanRakshak AI - Explainability Engine
Generates human-readable explanations for every AI decision.

Every important decision includes:
    - Signal-by-signal assessment (✓/⚠/✗)
    - Confidence breakdown
    - Reasoning chain
    - Recommendation

This is NOT exposing internal model weights — it exposes the
orchestration-level signals and reasoning.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class SignalStatus(str, Enum):
    """Status indicator for each signal"""
    PASS = "pass"       # ✓
    WARNING = "warning"  # ⚠
    FAIL = "fail"       # ✗


@dataclass
class SignalAssessment:
    """Assessment of a single signal"""
    signal_name: str
    status: SignalStatus
    value: float
    threshold: float
    description: str
    icon: str = ""

    def __post_init__(self):
        if not self.icon:
            self.icon = {"pass": "✓", "warning": "⚠", "fail": "✗"}[self.status.value]

    def to_dict(self) -> dict:
        return {
            "signal_name": self.signal_name,
            "status": self.status.value,
            "icon": self.icon,
            "value": round(self.value, 4),
            "threshold": round(self.threshold, 4),
            "description": self.description,
        }


@dataclass
class ExplanationReport:
    """Complete explanation for a decision"""
    image_id: str
    decision: str
    species: Optional[str]
    confidence: float
    summary: str
    signal_assessments: List[SignalAssessment] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    recommendation: str = ""
    is_tiger: bool = False

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "decision": self.decision,
            "species": self.species,
            "confidence": round(self.confidence, 4),
            "summary": self.summary,
            "signal_assessments": [s.to_dict() for s in self.signal_assessments],
            "reasoning_chain": self.reasoning_chain,
            "recommendation": self.recommendation,
            "is_tiger": self.is_tiger,
        }

    @property
    def formatted_report(self) -> str:
        """Generate a formatted text report"""
        lines = []
        lines.append(f"{'='*50}")
        lines.append(f"DECISION: {self.decision.upper()}")
        lines.append(f"{'='*50}")

        if self.species:
            lines.append(f"Species: {self.species}")
        lines.append(f"Confidence: {self.confidence:.1%}")
        lines.append("")

        lines.append("Signal Assessment:")
        lines.append("-" * 40)
        for sa in self.signal_assessments:
            lines.append(f"  {sa.icon} {sa.description}")
        lines.append("")

        lines.append(f"Final confidence: {self.confidence:.1%}")
        lines.append("")

        if self.recommendation:
            lines.append(f"Recommendation: {self.recommendation}")

        return "\n".join(lines)


class ExplainabilityService:
    """
    Generates human-readable explanations for AI decisions.

    Takes decision engine signals and produces structured
    explanations with signal assessments and reasoning chains.
    """

    @staticmethod
    def explain(
        image_id: str,
        decision: str,
        species: Optional[str],
        confidence: float,
        megadetector_confidence: float = 0.0,
        megadetector_type: str = "",
        speciesnet_confidence: float = 0.0,
        speciesnet_species: str = "",
        openclip_agrees: bool = False,
        openclip_similarity: float = 0.0,
        openclip_prediction: str = "",
        image_quality: float = 1.0,
        model_agreement: float = 0.0,
        is_tiger: bool = False,
        is_known_habitat: bool = True,
        reasoning: List[str] = None,
    ) -> ExplanationReport:
        """
        Generate an explanation for a decision.

        Args:
            All signal values from the decision engine

        Returns:
            ExplanationReport with full explanation
        """
        signal_assessments = []
        reasoning_chain = reasoning or []

        # ============ ASSESS EACH SIGNAL ============

        # 1. Detection
        signal_assessments.append(SignalAssessment(
            signal_name="detection",
            status=ExplainabilityService._assess_status(megadetector_confidence, 0.7, 0.5),
            value=megadetector_confidence,
            threshold=0.5,
            description=f"{'Animal' if megadetector_type == 'animal' else megadetector_type.capitalize()} detected with {megadetector_confidence:.1%} confidence",
        ))

        # 2. Species classification
        if megadetector_type == "animal" and speciesnet_species:
            signal_assessments.append(SignalAssessment(
                signal_name="species_classification",
                status=ExplainabilityService._assess_status(speciesnet_confidence, 0.8, 0.5),
                value=speciesnet_confidence,
                threshold=0.5,
                description=f"SpeciesNet predicts {speciesnet_species} at {speciesnet_confidence:.1%}",
            ))

        # 3. Semantic verification
        if openclip_similarity > 0:
            signal_assessments.append(SignalAssessment(
                signal_name="semantic_verification",
                status=SignalStatus.PASS if openclip_agrees else SignalStatus.WARNING,
                value=openclip_similarity,
                threshold=0.5,
                description=f"OpenCLIP {'agrees' if openclip_agrees else 'disagrees'} with classification (similarity: {openclip_similarity:.1%})",
            ))

        # 4. Image quality
        signal_assessments.append(SignalAssessment(
            signal_name="image_quality",
            status=ExplainabilityService._assess_status(image_quality, 0.8, 0.5),
            value=image_quality,
            threshold=0.5,
            description=f"Image quality is {'good' if image_quality >= 0.8 else 'acceptable' if image_quality >= 0.5 else 'poor'}",
        ))

        # 5. Model agreement
        if model_agreement > 0:
            signal_assessments.append(SignalAssessment(
                signal_name="model_agreement",
                status=ExplainabilityService._assess_status(model_agreement, 0.7, 0.4),
                value=model_agreement,
                threshold=0.5,
                description=f"Models show {'high' if model_agreement >= 0.7 else 'moderate' if model_agreement >= 0.4 else 'low'} agreement",
            ))

        # 6. Habitat context
        if is_known_habitat:
            signal_assessments.append(SignalAssessment(
                signal_name="habitat",
                status=SignalStatus.PASS,
                value=1.0,
                threshold=0.0,
                description="Location is within known wildlife habitat",
            ))

        # ============ GENERATE SUMMARY ============
        pass_count = sum(1 for s in signal_assessments if s.status == SignalStatus.PASS)
        warn_count = sum(1 for s in signal_assessments if s.status == SignalStatus.WARNING)
        fail_count = sum(1 for s in signal_assessments if s.status == SignalStatus.FAIL)

        if decision == "auto_accept":
            summary = f"{species or 'Detection'} classified with {confidence:.1%} confidence. All signals support this classification."
            recommendation = "Classification accepted automatically. No action required."
        elif decision == "human_review":
            summary = f"{species or 'Detection'} classified with {confidence:.1%} confidence. Some signals require verification."
            recommendation = "Human reviewer should verify this classification before acceptance."
        else:
            summary = f"Uncertain classification ({confidence:.1%} confidence). Multiple signals show low confidence."
            recommendation = "This image requires human expert review. Consider additional context or higher-quality image."

        if is_tiger:
            summary = f"🐅 TIGER DETECTION: {summary}"
            recommendation += " Tiger detection triggers special tracking and conservation alerts."

        return ExplanationReport(
            image_id=image_id,
            decision=decision,
            species=species,
            confidence=confidence,
            summary=summary,
            signal_assessments=signal_assessments,
            reasoning_chain=reasoning_chain,
            recommendation=recommendation,
            is_tiger=is_tiger,
        )

    @staticmethod
    def _assess_status(value: float, pass_threshold: float, warn_threshold: float) -> SignalStatus:
        """Determine signal status based on thresholds"""
        if value >= pass_threshold:
            return SignalStatus.PASS
        elif value >= warn_threshold:
            return SignalStatus.WARNING
        else:
            return SignalStatus.FAIL
