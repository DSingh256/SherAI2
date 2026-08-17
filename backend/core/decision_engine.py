"""
VanRakshak AI - Decision Engine & Intelligent Routing
Central decision engine that combines all model signals into a final classification and intelligent routing.

Combines:
    - MegaDetector confidence & object type
    - SpeciesNet confidence & species classification
    - OpenCLIP semantic similarity & model agreement
    - Image quality score & quality gate decision
    - Geographic & temporal habitat context

Produces: FINAL_CLASSIFICATION, FINAL_CONFIDENCE, DECISION, ROUTING_DESTINATION, REASONING, EXPLAINABILITY
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

from config import settings


class DecisionType(str, Enum):
    """Routing decisions"""
    AUTO_ACCEPT = "auto_accept"
    HUMAN_REVIEW = "human_review"
    UNCERTAIN = "uncertain"
    REJECT = "reject"
    NO_ANIMAL = "no_animal"


class RoutingDestination(str, Enum):
    """Intelligent Routing Destination Pipelines"""
    ACCEPTED = "accepted"       # Direct auto-acceptance to processed storage
    REVIEW = "review"           # Human verification queue
    REJECTED = "rejected"       # Rejected quality pipeline
    QUARANTINE = "quarantine"   # Corrupted / invalid image quarantine
    NO_ANIMAL = "no_animal"     # Clean empty frame archive
    ALERT = "alert"             # High-priority wildlife alert notification pipeline


class ConfidenceLevel(str, Enum):
    """Confidence level classification"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


PRIORITY_SPECIES = ["Bengal Tiger", "Indian Leopard", "Asian Elephant"]


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
    """Complete decision engine output with intelligent routing"""
    image_id: str
    species: Optional[str] = None
    confidence: float = 0.0
    decision: DecisionType = DecisionType.UNCERTAIN
    routing_destination: RoutingDestination = RoutingDestination.REVIEW
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    reasoning: List[str] = field(default_factory=list)
    signals: List[DecisionSignal] = field(default_factory=list)
    is_tiger: bool = False
    is_priority_species: bool = False
    is_escalated: bool = False
    processing_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    raw_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "processing_id": self.processing_id,
            "species": self.species,
            "confidence": round(self.confidence, 4),
            "decision": self.decision.value,
            "routing_destination": self.routing_destination.value,
            "confidence_level": self.confidence_level.value,
            "reasoning": self.reasoning,
            "signals": [s.to_dict() for s in self.signals],
            "is_tiger": self.is_tiger,
            "is_priority_species": self.is_priority_species,
            "is_escalated": self.is_escalated,
            "raw_scores": {k: round(v, 4) for k, v in self.raw_scores.items()},
        }


class DecisionEngineService:
    """
    Central decision engine & intelligent routing service.

    Combines all upstream model signals with configurable weights
    to produce a final classification with multi-stage routing.
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
        quality_passed: bool = True,
        no_detections: bool = False,
        is_known_habitat: bool = True,
        time_of_day_hour: int = -1,
        processing_id: str = "",
    ) -> DecisionEngineResult:
        """
        Make an intelligent classification and routing decision combining all model signals.
        """
        proc_id = processing_id or str(uuid.uuid4())
        signals = []
        reasoning = []
        is_escalated = False

        # ============ SPECIAL ROUTING 1: QUALITY GATE FAILURE ============
        if not quality_passed or image_quality_score < 0.5:
            reasoning.append(f"❌ Image failed quality gate (score: {image_quality_score:.1%}). Routing to quarantine.")
            return DecisionEngineResult(
                image_id=image_id,
                processing_id=proc_id,
                species=None,
                confidence=0.0,
                decision=DecisionType.REJECT,
                routing_destination=RoutingDestination.QUARANTINE,
                confidence_level=ConfidenceLevel.LOW,
                reasoning=reasoning,
                signals=[],
                is_tiger=False,
                is_priority_species=False,
                is_escalated=False,
                raw_scores={"image_quality": image_quality_score}
            )

        # ============ SPECIAL ROUTING 2: EMPTY FRAME (NO ANIMAL/OBJECT) ============
        if no_detections or not megadetector_type:
            reasoning.append("✓ No objects detected by MegaDetector (empty camera trap frame).")
            reasoning.append("✅ ROUTED TO NO_ANIMAL ARCHIVE.")
            return DecisionEngineResult(
                image_id=image_id,
                processing_id=proc_id,
                species=None,
                confidence=1.0,
                decision=DecisionType.NO_ANIMAL,
                routing_destination=RoutingDestination.NO_ANIMAL,
                confidence_level=ConfidenceLevel.HIGH,
                reasoning=reasoning,
                signals=[],
                is_tiger=False,
                is_priority_species=False,
                is_escalated=False,
                raw_scores={"megadetector": 0.0}
            )

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
            description=f"SpeciesNet predicts {speciesnet_species or 'unknown'} at {speciesnet_confidence:.1%}"
        )
        signals.append(sn_signal)

        if sn_signal.passed:
            reasoning.append(f"✓ SpeciesNet predicts {speciesnet_species} at {speciesnet_confidence:.1%}")
        else:
            reasoning.append(f"⚠ SpeciesNet confidence only {speciesnet_confidence:.1%}")

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
            # Escalate if both models are moderately confident but conflict
            if speciesnet_confidence >= 0.65 and openclip_similarity >= 0.65:
                is_escalated = True
                reasoning.append(f"🚨 MODEL CONFLICT ESCALATION: SpeciesNet ({speciesnet_species}) vs OpenCLIP ({openclip_prediction})")

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

        # ============ CONTEXT SIGNALS ============
        if is_known_habitat:
            reasoning.append("✓ Location is within known wildlife habitat")

        if time_of_day_hour >= 0:
            if DecisionEngineService._is_typical_activity_time(speciesnet_species, time_of_day_hour):
                reasoning.append(f"✓ Activity time ({time_of_day_hour:02d}:00) is typical for {speciesnet_species}")
            else:
                reasoning.append(f"⚠ Unusual activity time ({time_of_day_hour:02d}:00) for {speciesnet_species}")

        # ============ CALCULATE FINAL CONFIDENCE ============
        final_confidence = sum(s.value * s.weight for s in signals)
        total_weight = sum(s.weight for s in signals)
        if total_weight > 0:
            final_confidence = final_confidence / total_weight

        # Bonuses and Penalties
        if openclip_agrees and agreement_score > 0.7:
            final_confidence = min(1.0, final_confidence * 1.05)

        if not openclip_agrees:
            final_confidence *= 0.90

        final_confidence = max(0.0, min(1.0, final_confidence))

        # ============ SPECIES & PRIORITY CHECK ============
        final_species = speciesnet_species if megadetector_type == "animal" else megadetector_type
        is_tiger = final_species == "Bengal Tiger"
        is_priority_species = final_species in PRIORITY_SPECIES or is_tiger

        if is_tiger:
            reasoning.append("🐅 TIGER DETECTION — Priority 1 tracking and alert activated")
        elif is_priority_species:
            reasoning.append(f"🚨 PRIORITY SPECIES DETECTED: {final_species} — Conservation alert triggered")

        # ============ MAKE ROUTING DECISION ============
        if is_escalated:
            decision = DecisionType.HUMAN_REVIEW
            routing_destination = RoutingDestination.REVIEW
            confidence_level = ConfidenceLevel.MEDIUM
            reasoning.append(f"\n🔍 DECISION: ESCALATED TO HUMAN REVIEW (conflicting model predictions)")
        elif final_confidence >= settings.HIGH_CONFIDENCE_THRESHOLD and openclip_agrees:
            decision = DecisionType.AUTO_ACCEPT
            routing_destination = RoutingDestination.ALERT if is_priority_species else RoutingDestination.ACCEPTED
            confidence_level = ConfidenceLevel.HIGH
            reasoning.append(f"\n✅ DECISION: AUTO ACCEPT (confidence: {final_confidence:.1%}) -> Route: {routing_destination.value.upper()}")
        elif final_confidence >= settings.MEDIUM_CONFIDENCE_THRESHOLD or not openclip_agrees:
            decision = DecisionType.HUMAN_REVIEW
            routing_destination = RoutingDestination.REVIEW
            confidence_level = ConfidenceLevel.MEDIUM
            reasoning.append(f"\n🔍 DECISION: HUMAN REVIEW REQUIRED (confidence: {final_confidence:.1%}) -> Route: REVIEW")
        else:
            decision = DecisionType.UNCERTAIN
            routing_destination = RoutingDestination.REVIEW
            confidence_level = ConfidenceLevel.LOW
            reasoning.append(f"\n❓ DECISION: UNCERTAIN (confidence: {final_confidence:.1%}) -> Route: REVIEW")

        return DecisionEngineResult(
            image_id=image_id,
            processing_id=proc_id,
            species=final_species,
            confidence=final_confidence,
            decision=decision,
            routing_destination=routing_destination,
            confidence_level=confidence_level,
            reasoning=reasoning,
            signals=signals,
            is_tiger=is_tiger,
            is_priority_species=is_priority_species,
            is_escalated=is_escalated,
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
        if openclip_agrees:
            return (speciesnet_conf + openclip_similarity) / 2
        else:
            gap = abs(speciesnet_conf - openclip_similarity)
            return max(0.1, min(speciesnet_conf, openclip_similarity) - gap * 0.5)

    @staticmethod
    def _is_typical_activity_time(species: str, hour: int) -> bool:
        nocturnal = ["Bengal Tiger", "Indian Leopard", "Sloth Bear", "Indian Porcupine", "Jungle Cat"]
        diurnal = ["Sambar Deer", "Spotted Deer", "Spotted Deer (Chital)", "Peafowl", "Common Langur", "Rhesus Macaque"]
        crepuscular = ["Wild Boar", "Indian Gaur"]

        if species in nocturnal:
            return hour >= 18 or hour <= 7
        elif species in diurnal:
            return 6 <= hour <= 18
        elif species in crepuscular:
            return (4 <= hour <= 8) or (16 <= hour <= 20)
        else:
            return True
