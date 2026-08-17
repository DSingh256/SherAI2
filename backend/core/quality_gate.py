"""
VanRakshak AI - Quality Gate Service
Evaluates image quality and routes to appropriate action
"""

from enum import Enum
from sqlalchemy.orm import Session
from db.models import Image, ImageQuality, AuditTrail
from config import settings
from utils.image_utils import ImageUtils
from datetime import datetime
from typing import Optional, Dict, List


class QualityDecision(str, Enum):
    """Quality assessment decision"""
    ACCEPT = "accept"
    BLUR_REJECT = "blur_reject"
    DARKNESS_REJECT = "darkness_reject"
    OVEREXPOSED_REJECT = "overexposed_reject"
    CORRUPTED_REJECT = "corrupted_reject"
    DUPLICATE_REJECT = "duplicate_reject"


class QualityGateResult:
    """Result of quality gate assessment"""
    
    def __init__(
        self,
        image_id: str,
        decision: QualityDecision,
        quality_score: float,
        reasons: List[str],
        details: Dict
    ):
        self.image_id = image_id
        self.decision = decision
        self.quality_score = quality_score
        self.reasons = reasons
        self.details = details
        self.passed = decision == QualityDecision.ACCEPT


class QualityGateService:
    """Service for image quality assessment and gating"""

    @staticmethod
    def assess_quality(image_id: str, db: Session) -> QualityGateResult:
        """
        Assess image quality using multiple criteria
        
        Checks:
        1. Corruption - is file readable?
        2. Darkness - average brightness too low?
        3. Overexposure - average brightness too high?
        4. Blur - Laplacian variance too low?
        
        Args:
            image_id: ID of image to assess
            db: Database session
        
        Returns:
            QualityGateResult with decision and reasoning
        """
        
        image = db.query(Image).filter(Image.id == image_id).first()
        if not image:
            return QualityGateResult(
                image_id=image_id,
                decision=QualityDecision.CORRUPTED_REJECT,
                quality_score=0.0,
                reasons=["Image not found in database"],
                details={}
            )
        
        # Start with full score
        quality_score = 1.0
        reasons = []
        details = {}
        
        # Check 1: Corruption
        if ImageUtils.is_corrupted(image.image_path):
            details["corrupted"] = True
            reasons.append("Image file is corrupted or unreadable")
            QualityGateService._record_audit(
                image_id, "corruption_check", "fail", db
            )
            return QualityGateResult(
                image_id=image_id,
                decision=QualityDecision.CORRUPTED_REJECT,
                quality_score=0.0,
                reasons=reasons,
                details=details
            )
        
        details["corrupted"] = False
        
        # Check 2: Darkness (brightness too low)
        brightness = image.brightness
        if brightness is None:
            brightness = ImageUtils.get_brightness(image.image_path)
            image.brightness = brightness
        
        if brightness < settings.MIN_BRIGHTNESS:
            quality_score -= 0.3
            reasons.append(
                f"Image too dark (brightness {brightness:.1f} < {settings.MIN_BRIGHTNESS})"
            )
            details["darkness"] = {
                "brightness": brightness,
                "threshold": settings.MIN_BRIGHTNESS
            }
        else:
            details["darkness"] = False
        
        # Check 3: Overexposure (brightness too high)
        if brightness > settings.MAX_BRIGHTNESS:
            quality_score -= 0.3
            reasons.append(
                f"Image overexposed (brightness {brightness:.1f} > {settings.MAX_BRIGHTNESS})"
            )
            details["overexposed"] = {
                "brightness": brightness,
                "threshold": settings.MAX_BRIGHTNESS
            }
        else:
            details["overexposed"] = False
        
        # Check 4: Blur (Laplacian variance too low)
        blur_score = image.blur_score
        if blur_score is None:
            blur_score = ImageUtils.get_blur_score(image.image_path)
            image.blur_score = blur_score
        
        if blur_score < settings.BLUR_THRESHOLD:
            quality_score -= 0.3
            reasons.append(
                f"Image too blurry (blur score {blur_score:.1f} < {settings.BLUR_THRESHOLD})"
            )
            details["blur"] = {
                "blur_score": blur_score,
                "threshold": settings.BLUR_THRESHOLD
            }
        else:
            details["blur"] = False
        
        # Ensure score is in valid range
        quality_score = max(0.0, min(1.0, quality_score))
        details["overall_quality_score"] = quality_score
        
        # Make decision based on quality score
        decision = QualityDecision.ACCEPT
        
        if brightness < settings.MIN_BRIGHTNESS and quality_score < 0.5:
            decision = QualityDecision.DARKNESS_REJECT
        elif brightness > settings.MAX_BRIGHTNESS and quality_score < 0.5:
            decision = QualityDecision.OVEREXPOSED_REJECT
        elif blur_score < settings.BLUR_THRESHOLD and quality_score < 0.5:
            decision = QualityDecision.BLUR_REJECT
        
        # Record audit trail
        QualityGateService._record_audit(
            image_id,
            "quality_assessment",
            "pass" if decision == QualityDecision.ACCEPT else "fail",
            db,
            details
        )
        
        return QualityGateResult(
            image_id=image_id,
            decision=decision,
            quality_score=quality_score,
            reasons=reasons,
            details=details
        )

    @staticmethod
    def apply_quality_gate(image_id: str, db: Session) -> bool:
        """
        Apply quality gate and update image status
        
        Updates image with:
        - quality_status
        - quality_score
        
        Args:
            image_id: ID of image to gate
            db: Database session
        
        Returns:
            True if image passed quality gate, False if rejected
        """
        
        image = db.query(Image).filter(Image.id == image_id).first()
        if not image:
            return False
        
        result = QualityGateService.assess_quality(image_id, db)
        
        # Update image record
        if result.decision == QualityDecision.ACCEPT:
            image.quality_status = ImageQuality.GOOD.value
        elif result.decision == QualityDecision.BLUR_REJECT:
            image.quality_status = ImageQuality.BLURRY.value
        elif result.decision == QualityDecision.DARKNESS_REJECT:
            image.quality_status = ImageQuality.TOO_DARK.value
        elif result.decision == QualityDecision.OVEREXPOSED_REJECT:
            image.quality_status = ImageQuality.OVEREXPOSED.value
        elif result.decision == QualityDecision.CORRUPTED_REJECT:
            image.quality_status = ImageQuality.CORRUPTED.value
        
        image.quality_score = result.quality_score
        db.commit()
        
        return result.passed

    @staticmethod
    def batch_quality_gate(image_ids: List[str], db: Session) -> Dict[str, QualityGateResult]:
        """
        Apply quality gate to multiple images
        
        Args:
            image_ids: List of image IDs
            db: Database session
        
        Returns:
            Dictionary mapping image_id to QualityGateResult
        """
        
        results = {}
        for image_id in image_ids:
            results[image_id] = QualityGateService.assess_quality(image_id, db)
            # Apply changes
            QualityGateService.apply_quality_gate(image_id, db)
        
        return results

    @staticmethod
    def get_quality_breakdown(camera_id: Optional[str] = None, db: Session = None) -> Dict:
        """
        Get quality breakdown statistics
        
        Args:
            camera_id: Filter by camera (optional)
            db: Database session
        
        Returns:
            Dictionary with quality statistics
        """
        
        query = db.query(Image)
        if camera_id:
            query = query.filter(Image.camera_id == camera_id)
        
        images = query.all()
        total = len(images)
        
        if total == 0:
            return {
                "total": 0,
                "breakdown": {}
            }
        
        # Count by quality status
        breakdown = {}
        for status in ImageQuality:
            count = sum(1 for img in images if img.quality_status == status.value)
            breakdown[status.value] = {
                "count": count,
                "percentage": (count / total * 100) if total > 0 else 0
            }
        
        return {
            "total": total,
            "camera_id": camera_id,
            "breakdown": breakdown,
            "good_percentage": breakdown[ImageQuality.GOOD.value]["percentage"]
        }

    @staticmethod
    def get_rejection_reasons(camera_id: Optional[str] = None, db: Session = None) -> Dict[str, int]:
        """
        Get rejection reason statistics
        
        Args:
            camera_id: Filter by camera (optional)
            db: Database session
        
        Returns:
            Dictionary with rejection reasons and counts
        """
        
        query = db.query(Image)
        if camera_id:
            query = query.filter(Image.camera_id == camera_id)
        
        images = query.filter(
            Image.quality_status != ImageQuality.GOOD.value
        ).all()
        
        rejection_reasons = {
            "blurry": 0,
            "too_dark": 0,
            "overexposed": 0,
            "corrupted": 0,
            "duplicate": 0
        }
        
        for img in images:
            if img.quality_status == ImageQuality.BLURRY.value:
                rejection_reasons["blurry"] += 1
            elif img.quality_status == ImageQuality.TOO_DARK.value:
                rejection_reasons["too_dark"] += 1
            elif img.quality_status == ImageQuality.OVEREXPOSED.value:
                rejection_reasons["overexposed"] += 1
            elif img.quality_status == ImageQuality.CORRUPTED.value:
                rejection_reasons["corrupted"] += 1
            elif img.quality_status == ImageQuality.DUPLICATE.value:
                rejection_reasons["duplicate"] += 1
        
        return rejection_reasons

    @staticmethod
    def _record_audit(
        image_id: str,
        event_type: str,
        event_status: str,
        db: Session,
        details: Dict = None
    ):
        """Record audit trail event"""
        
        audit = AuditTrail(
            image_id=image_id,
            event_type=event_type,
            event_status=event_status,
            details=details or {}
        )
        db.add(audit)
        db.commit()
