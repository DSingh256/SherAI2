"""
VanRakshak AI - Decision Engine
Central decision engine that combines all model signals into a final classification.

Combines:
    - MegaDetector confidence
    - SpeciesNet confidence
    - OpenCLIP agreement score
    - Image quality score
    - Model agreement
    - Geographic context
    - Temporal context

Produces: FINAL_CLASSIFICATION, FINAL_CONFIDENCE, DECISION, REASONING
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

from config import settings


class DecisionType(str, Enum):
    """Routing decisions"""
    AUTO_ACCEPT = "auto_accept"
    HUMAN_REVIEW = "human_review"
    UNCERTAIN = "uncertain"


class ConfidenceLevel(str, Enum):
    """Confidence level classification"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DecisionSignal:
    """A single signal contributing to the decision"""
    name: str
    value: float
    weight: float
    passed: bool
    description: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "weight": round(self.weight, 2),
            "passed": self.passed,
            "description": self.description,
        }


@dataclass
class DecisionEngineResult:
    """Complete decision engine output"""
    image_id: str
    species: Optional[str] = None
    confidence: float = 0.0
    decision: DecisionType = DecisionType.UNCERTAIN
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    reasoning: List[str] = field(default_factory=list)
    signals: List[DecisionSignal] = field(default_factory=list)
    is_tiger: bool = False
    raw_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "species": self.species,
            "confidence": round(self.confidence, 4),
            "decision": self.decision.value,
            "confidence_level": self.confidence_level.value,
            "reasoning": self.reasoning,
            "signals": [s.to_dict() for s in self.signals],
            "is_tiger": self.is_tiger,
            "raw_scores": {k: round(v, 4) for k, v in self.raw_scores.items()},
        }


class DecisionEngineService:
    """
    Central decision engine.

    Combines all upstream model signals with configurable weights
    to produce a final classification with confidence-based routing.

    Thresholds are configurable via config.py:
        HIGH_CONFIDENCE_THRESHOLD = 0.90 → AUTO_ACCEPT
        MEDIUM_CONFIDENCE_THRESHOLD = 0.60 → HUMAN_REVIEW
        Below 0.60 → UNCERTAIN
    """

    # Signal weights (must sum to ~1.0)
    SIGNAL_WEIGHTS = {
        "megadetector": 0.25,
        "speciesnet": 0.30,
        "openclip": 0.20,
        "image_quality": 0.10,
        "model_agreement": 0.15,
    }

    @staticmethod
    def decide(
        image_id: str,
        megadetector_confidence: float = 0.0,
        megadetector_type: str = "",
        speciesnet_species: str = "",
        speciesnet_confidence: float = 0.0,
        openclip_prediction: str = "",
        openclip_similarity: float = 0.0,
        openclip_agrees: bool = False,
        image_quality_score: float = 1.0,
        is_known_habitat: bool = True,
        time_of_day_hour: int = -1,
    ) -> DecisionEngineResult:
        """
        Make a classification decision combining all model signals.

        Args:
            image_id: Image identifier
            megadetector_confidence: MegaDetector detection confidence
            megadetector_type: Detection type (animal/human/vehicle)
            speciesnet_species: SpeciesNet primary species prediction
            speciesnet_confidence: SpeciesNet prediction confidence
            openclip_prediction: OpenCLIP top prediction
            openclip_similarity: OpenCLIP similarity score
            openclip_agrees: Whether OpenCLIP agrees with SpeciesNet
            image_quality_score: Image quality (0-1)
            is_known_habitat: Whether location is known habitat
            time_of_day_hour: Hour of day (0-23), -1 if unknown

        Returns:
            DecisionEngineResult with final classification and routing
        """
        signals = []
        reasoning = []

        # ============ SIGNAL 1: MegaDetector ============
        md_signal = DecisionSignal(
            name="megadetector",
            value=megadetector_confidence,
            weight=DecisionEngineService.SIGNAL_WEIGHTS["megadetector"],
            passed=megadetector_confidence >= settings.MEGADETECTOR_CONFIDENCE_THRESHOLD,
            description=f"MegaDetector detected {megadetector_type} with {megadetector_confidence:.1%} confidence"
        )
        signals.append(md_signal)

        if md_signal.passed:
            reasoning.append(f"✓ {megadetector_type.capitalize()} detected with {megadetector_confidence:.1%} confidence")
        else:
            reasoning.append(f"⚠ Low detection confidence: {megadetector_confidence:.1%}")

        # ============ SIGNAL 2: SpeciesNet ============
        sn_signal = DecisionSignal(
            name="speciesnet",
            value=speciesnet_confidence,
            weight=DecisionEngineService.SIGNAL_WEIGHTS["speciesnet"],
            passed=speciesnet_confidence >= settings.SPECIESNET_CONFIDENCE_THRESHOLD,
            description=f"SpeciesNet predicts {speciesnet_species} at {speciesnet_confidence:.1%}"
        )
        signals.append(sn_signal)

        if sn_signal.passed:
            reasoning.append(f"✓ SpeciesNet predicts {speciesnet_species} at {speciesnet_confidence:.1%}")
        else:
            reasoning.append(f"⚠ Species confidence only {speciesnet_confidence:.1%}")

        # ============ SIGNAL 3: OpenCLIP ============
        oc_signal = DecisionSignal(
            name="openclip",
            value=openclip_similarity,
            weight=DecisionEngineService.SIGNAL_WEIGHTS["openclip"],
            passed=openclip_agrees,
            description=f"OpenCLIP {'agrees' if openclip_agrees else 'disagrees'} (similarity: {openclip_similarity:.1%})"
        )
        signals.append(oc_signal)

        if openclip_agrees:
            reasoning.append(f"✓ OpenCLIP agrees with {speciesnet_species} classification (similarity: {openclip_similarity:.1%})")
        else:
            reasoning.append(f"⚠ OpenCLIP disagrees — predicts {openclip_prediction} instead")

        # ============ SIGNAL 4: Image Quality ============
        iq_signal = DecisionSignal(
            name="image_quality",
            value=image_quality_score,
            weight=DecisionEngineService.SIGNAL_WEIGHTS["image_quality"],
            passed=image_quality_score >= 0.6,
            description=f"Image quality score: {image_quality_score:.1%}"
        )
        signals.append(iq_signal)

        if iq_signal.passed:
            reasoning.append(f"✓ Image quality is {'good' if image_quality_score >= 0.8 else 'acceptable'}")
        else:
            reasoning.append(f"⚠ Poor image quality ({image_quality_score:.1%})")

        # ============ SIGNAL 5: Model Agreement ============
        agreement_score = DecisionEngineService._calculate_agreement(
            speciesnet_species, openclip_prediction, openclip_agrees,
            speciesnet_confidence, openclip_similarity
        )

        ma_signal = DecisionSignal(
            name="model_agreement",
            value=agreement_score,
            weight=DecisionEngineService.SIGNAL_WEIGHTS["model_agreement"],
            passed=agreement_score >= 0.6,
            description=f"Model agreement score: {agreement_score:.1%}"
        )
        signals.append(ma_signal)

        if ma_signal.passed:
            reasoning.append("✓ Models show high agreement")
        else:
            reasoning.append("⚠ Models show low agreement — verification recommended")

        # ============ CONTEXT SIGNALS (bonus) ============
        if is_known_habitat:
            reasoning.append("✓ Location is within known wildlife habitat")

        if time_of_day_hour >= 0:
            if DecisionEngineService._is_typical_activity_time(speciesnet_species, time_of_day_hour):
                reasoning.append(f"✓ Activity time ({time_of_day_hour:02d}:00) is typical for {speciesnet_species}")
            else:
                reasoning.append(f"⚠ Unusual activity time ({time_of_day_hour:02d}:00) for {speciesnet_species}")

        # ============ CALCULATE FINAL CONFIDENCE ============
        final_confidence = sum(
            s.value * s.weight for s in signals
        )

        # Normalize
        total_weight = sum(s.weight for s in signals)
        if total_weight > 0:
            final_confidence = final_confidence / total_weight

        # Apply bonuses/penalties
        if openclip_agrees and agreement_score > 0.7:
            final_confidence = min(1.0, final_confidence * 1.05)

        if not openclip_agrees:
            final_confidence *= 0.90

        final_confidence = max(0.0, min(1.0, final_confidence))

        # ============ MAKE DECISION ============
        if final_confidence >= settings.HIGH_CONFIDENCE_THRESHOLD:
            decision = DecisionType.AUTO_ACCEPT
            confidence_level = ConfidenceLevel.HIGH
            reasoning.append(f"\n✅ DECISION: AUTO ACCEPT (confidence: {final_confidence:.1%})")
        elif final_confidence >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
            decision = DecisionType.HUMAN_REVIEW
            confidence_level = ConfidenceLevel.MEDIUM
            reasoning.append(f"\n🔍 DECISION: HUMAN REVIEW REQUIRED (confidence: {final_confidence:.1%})")
        else:
            decision = DecisionType.UNCERTAIN
            confidence_level = ConfidenceLevel.LOW
            reasoning.append(f"\n❓ DECISION: UNCERTAIN — needs more data (confidence: {final_confidence:.1%})")

        # Determine species
        final_species = speciesnet_species if megadetector_type == "animal" else megadetector_type
        is_tiger = final_species == "Bengal Tiger"

        if is_tiger:
            reasoning.append("🐅 TIGER DETECTION — special tracking activated")

        return DecisionEngineResult(
            image_id=image_id,
            species=final_species,
            confidence=final_confidence,
            decision=decision,
            confidence_level=confidence_level,
            reasoning=reasoning,
            signals=signals,
            is_tiger=is_tiger,
            raw_scores={
                "megadetector": megadetector_confidence,
                "speciesnet": speciesnet_confidence,
                "openclip": openclip_similarity,
                "image_quality": image_quality_score,
                "model_agreement": agreement_score,
            },
        )

    @staticmethod
    def _calculate_agreement(
        speciesnet_species: str,
        openclip_prediction: str,
        openclip_agrees: bool,
        speciesnet_conf: float,
        openclip_similarity: float,
    ) -> float:
        """Calculate model agreement score"""
        if openclip_agrees:
            # Both agree: average their confidences
            return (speciesnet_conf + openclip_similarity) / 2
        else:
            # Disagreement: penalize based on confidence gap
            gap = abs(speciesnet_conf - openclip_similarity)
            return max(0.1, min(speciesnet_conf, openclip_similarity) - gap * 0.5)

    @staticmethod
    def _is_typical_activity_time(species: str, hour: int) -> bool:
        """Check if species is typically active at this time"""
        nocturnal = ["Bengal Tiger", "Indian Leopard", "Sloth Bear", "Indian Porcupine", "Jungle Cat"]
        diurnal = ["Sambar Deer", "Spotted Deer (Chital)", "Peafowl", "Langur", "Rhesus Macaque"]
        crepuscular = ["Wild Boar", "Indian Gaur", "Indian Muntjac (Barking Deer)"]

        if species in nocturnal:
            # Active 18:00 - 07:00
            return hour >= 18 or hour <= 7
        elif species in diurnal:
            # Active 06:00 - 18:00
            return 6 <= hour <= 18
        elif species in crepuscular:
            # Active dawn/dusk: 04:00-08:00 and 16:00-20:00
            return (4 <= hour <= 8) or (16 <= hour <= 20)
        else:
            return True  # Unknown species, any time OK
