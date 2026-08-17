"""
VanRakshak AI - Threat / Poaching Risk Analyzer
Multi-signal threat risk assessment.

IMPORTANT ETHICAL DESIGN:
    - NEVER accuses anyone of poaching
    - NEVER makes criminal assertions
    - Only flags "potential unusual activity" or "elevated threat risk"
    - All threat assessments require ranger/human verification

Combines signals:
    - Human + vehicle co-occurrence
    - Unusual time (2-5 AM in restricted zones)
    - Restricted zone activity
    - Repeated unusual sightings
    - Historical baseline deviation
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ThreatLevel(str, Enum):
    """Threat risk levels — advisory only, not accusations"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ThreatSignal:
    """A single signal contributing to threat assessment"""
    name: str
    triggered: bool
    weight: float
    description: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "triggered": self.triggered,
            "weight": round(self.weight, 2),
            "description": self.description,
        }


@dataclass
class ThreatAnalysisResult:
    """
    Complete threat risk analysis.

    Note: This is an ADVISORY assessment only.
    It identifies potential unusual activity patterns.
    It does NOT make accusations or definitive determinations.
    """
    camera_id: str
    timestamp: datetime
    threat_level: ThreatLevel
    risk_score: float  # 0.0 - 1.0
    signals: List[ThreatSignal] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    recommendation: str = ""
    requires_verification: bool = True

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "threat_level": self.threat_level.value,
            "risk_score": round(self.risk_score, 4),
            "signals": [s.to_dict() for s in self.signals],
            "reasoning": self.reasoning,
            "recommendation": self.recommendation,
            "requires_verification": self.requires_verification,
            "disclaimer": "This is an advisory assessment based on activity patterns. "
                         "It does NOT constitute an accusation. Ranger verification required.",
        }


# Restricted zones and high-risk areas (configurable per reserve)
RESTRICTED_ZONES = ["Zone A", "Zone D", "Core Area", "Buffer North"]
HIGH_SENSITIVITY_CAMERAS = ["CAM007", "CAM011", "CAM012"]


class ThreatAnalyzerService:
    """
    Multi-signal threat risk analyzer.

    Combines human detection, vehicle detection, temporal patterns,
    and location context to produce threat risk assessments.

    All assessments are advisory only and require human verification.
    """

    @staticmethod
    def analyze(
        camera_id: str,
        timestamp: datetime,
        humans_detected: int = 0,
        vehicles_detected: int = 0,
        location: str = "",
        zone: str = "",
        historical_human_avg: float = 2.0,
        historical_vehicle_avg: float = 1.0,
        is_restricted_zone: bool = False,
        recent_unusual_events: int = 0,
    ) -> ThreatAnalysisResult:
        """
        Analyze threat risk based on multiple signals.

        Args:
            camera_id: Camera identifier
            timestamp: Event timestamp
            humans_detected: Number of humans detected
            vehicles_detected: Number of vehicles detected
            location: Location description
            zone: Zone identifier
            historical_human_avg: Average human detections for this camera
            historical_vehicle_avg: Average vehicle detections for this camera
            is_restricted_zone: Whether camera is in restricted zone
            recent_unusual_events: Number of recent unusual events at this camera

        Returns:
            ThreatAnalysisResult with risk assessment
        """
        signals = []
        reasoning = []
        risk_score = 0.0

        # Determine if zone is restricted
        if not is_restricted_zone:
            is_restricted_zone = zone in RESTRICTED_ZONES

        # ============ SIGNAL 1: Human Activity ============
        human_unusual = humans_detected > historical_human_avg * 2
        signals.append(ThreatSignal(
            name="human_activity",
            triggered=human_unusual,
            weight=0.25,
            description=f"{'Elevated' if human_unusual else 'Normal'} human activity "
                       f"({humans_detected} detected, avg: {historical_human_avg:.1f})"
        ))
        if human_unusual:
            risk_score += 0.25
            reasoning.append(f"⚠ Elevated human activity detected ({humans_detected} vs avg {historical_human_avg:.1f})")

        # ============ SIGNAL 2: Vehicle Activity ============
        vehicle_unusual = vehicles_detected > historical_vehicle_avg * 2
        signals.append(ThreatSignal(
            name="vehicle_activity",
            triggered=vehicle_unusual,
            weight=0.20,
            description=f"{'Elevated' if vehicle_unusual else 'Normal'} vehicle activity "
                       f"({vehicles_detected} detected, avg: {historical_vehicle_avg:.1f})"
        ))
        if vehicle_unusual:
            risk_score += 0.20
            reasoning.append(f"⚠ Elevated vehicle activity ({vehicles_detected} vs avg {historical_vehicle_avg:.1f})")

        # ============ SIGNAL 3: Unusual Time ============
        hour = timestamp.hour if timestamp else -1
        unusual_time = 2 <= hour <= 5  # 2 AM - 5 AM
        signals.append(ThreatSignal(
            name="unusual_time",
            triggered=unusual_time,
            weight=0.20,
            description=f"{'Unusual' if unusual_time else 'Normal'} time of activity ({hour:02d}:00)"
        ))
        if unusual_time:
            risk_score += 0.20
            reasoning.append(f"⚠ Activity detected at unusual hour ({hour:02d}:00)")

        # ============ SIGNAL 4: Restricted Zone ============
        signals.append(ThreatSignal(
            name="restricted_zone",
            triggered=is_restricted_zone and (humans_detected > 0 or vehicles_detected > 0),
            weight=0.20,
            description=f"{'Restricted zone activity detected' if is_restricted_zone else 'Normal zone'}"
        ))
        if is_restricted_zone and (humans_detected > 0 or vehicles_detected > 0):
            risk_score += 0.20
            reasoning.append(f"⚠ Human/vehicle activity in restricted zone ({zone or location})")

        # ============ SIGNAL 5: Human + Vehicle Co-occurrence ============
        cooccurrence = humans_detected > 0 and vehicles_detected > 0
        signals.append(ThreatSignal(
            name="cooccurrence",
            triggered=cooccurrence,
            weight=0.15,
            description=f"{'Human and vehicle detected together' if cooccurrence else 'No co-occurrence'}"
        ))
        if cooccurrence:
            risk_score += 0.15
            reasoning.append("⚠ Human and vehicle detected together")

        # ============ SIGNAL 6: Repeated Events ============
        repeated = recent_unusual_events >= 3
        signals.append(ThreatSignal(
            name="repeated_events",
            triggered=repeated,
            weight=0.10,
            description=f"{'Repeated unusual events' if repeated else 'No pattern'} "
                       f"({recent_unusual_events} recent events)"
        ))
        if repeated:
            risk_score += 0.10
            reasoning.append(f"⚠ Repeated unusual activity pattern ({recent_unusual_events} events)")

        # High-sensitivity camera bonus
        if camera_id in HIGH_SENSITIVITY_CAMERAS:
            risk_score *= 1.15

        # Clamp risk score
        risk_score = min(1.0, max(0.0, risk_score))

        # Determine threat level
        if risk_score >= 0.65:
            threat_level = ThreatLevel.HIGH
            recommendation = ("URGENT: Ranger verification required immediately. "
                            "Multiple signals indicate unusual activity pattern.")
        elif risk_score >= 0.35:
            threat_level = ThreatLevel.MEDIUM
            recommendation = ("Elevated activity detected. Ranger patrol recommended. "
                            "Monitor camera for continued unusual patterns.")
        elif risk_score > 0.10:
            threat_level = ThreatLevel.LOW
            recommendation = ("Minor anomaly detected. Log for future reference. "
                            "No immediate action required.")
        else:
            threat_level = ThreatLevel.NONE
            recommendation = "No unusual activity. Normal operations."

        if not reasoning:
            reasoning.append("✓ All activity within normal parameters")

        return ThreatAnalysisResult(
            camera_id=camera_id,
            timestamp=timestamp,
            threat_level=threat_level,
            risk_score=risk_score,
            signals=signals,
            reasoning=reasoning,
            recommendation=recommendation,
        )
