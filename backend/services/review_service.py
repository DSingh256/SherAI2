"""
VanRakshak AI - Review Service
Handles human-in-the-loop verification, claiming/locking, decision persistence, and audit logging.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import uuid

from db.models import (
    Image, Decision, HumanReview, AuditTrail,
    Detection, Classification, Verification, Segmentation
)


class ReviewService:
    """
    Service for managing the human review process.
    Allows forest rangers and wildlife experts to verify uncertain AI classifications.
    """

    # In-memory item locks: {image_id: {"reviewer_id": str, "claimed_at": datetime}}
    _claimed_items: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def claim_item(cls, image_id: str, reviewer_id: str) -> Dict[str, Any]:
        """Claim an item for review to prevent conflicting concurrent edits"""
        existing = cls._claimed_items.get(image_id)
        if existing and existing["reviewer_id"] != reviewer_id:
            return {
                "success": False,
                "message": f"Item currently claimed by reviewer {existing['reviewer_id']}",
                "claimed_by": existing["reviewer_id"]
            }
        
        cls._claimed_items[image_id] = {
            "reviewer_id": reviewer_id,
            "claimed_at": datetime.utcnow()
        }
        return {
            "success": True,
            "message": "Item claimed successfully",
            "reviewer_id": reviewer_id
        }

    @classmethod
    def release_item(cls, image_id: str, reviewer_id: str = "") -> bool:
        """Release a claimed item back to the queue"""
        if image_id in cls._claimed_items:
            del cls._claimed_items[image_id]
            return True
        return False

    @staticmethod
    def get_item_evidence(image_id: str, db: Session) -> Optional[Dict[str, Any]]:
        """Retrieve complete multi-model evidence for an image"""
        image = db.query(Image).filter(Image.id == image_id).first()
        if not image:
            return None

        decision = db.query(Decision).filter(Decision.image_id == image_id).first()
        detections = db.query(Detection).filter(Detection.image_id == image_id).all()
        classifications = db.query(Classification).filter(Classification.image_id == image_id).all()
        verification = db.query(Verification).filter(Verification.image_id == image_id).first()
        segmentations = db.query(Segmentation).filter(Segmentation.image_id == image_id).all()

        det_list = [
            {
                "id": d.id,
                "object_type": d.object_type,
                "confidence": d.confidence,
                "bbox": {
                    "x_min": d.bbox_x_min,
                    "y_min": d.bbox_y_min,
                    "x_max": d.bbox_x_max,
                    "y_max": d.bbox_y_max,
                },
                "crop_path": d.crop_path,
            }
            for d in detections
        ]

        class_list = [
            {
                "detection_id": c.detection_id,
                "species": c.species,
                "confidence": c.confidence,
                "alternatives": c.alternative_predictions or [],
                "model_name": c.model_name,
            }
            for c in classifications
        ]

        seg_list = [
            {
                "detection_id": s.detection_id,
                "mask_path": s.mask_path,
                "segmented_crop_path": s.segmented_crop_path,
                "model_name": s.model_name,
            }
            for s in segmentations
        ]

        return {
            "image_id": image.id,
            "image_path": image.image_path,
            "camera_id": image.camera_id,
            "timestamp": image.timestamp.isoformat() if image.timestamp else None,
            "quality_status": image.quality_status,
            "decision": {
                "species": decision.species if decision else None,
                "confidence": decision.confidence if decision else 0.0,
                "decision": decision.decision if decision else "pending",
                "confidence_level": decision.confidence_level if decision else "low",
                "reasoning": decision.reasoning if decision else [],
                "signals": decision.signals if decision else {},
                "is_tiger": decision.is_tiger if decision else False,
            } if decision else None,
            "detections": det_list,
            "classifications": class_list,
            "verification": {
                "primary_prediction": verification.primary_prediction,
                "confidence": verification.confidence,
                "scores": verification.semantic_scores,
                "model_name": verification.model_name,
            } if verification else None,
            "segmentations": seg_list,
        }

    @classmethod
    def submit_review(
        cls,
        image_id: str,
        reviewer_id: str,
        action: str,  # ACCEPT, REJECT, CORRECT, ESCALATE, REPROCESS
        human_prediction: Optional[str] = None,
        is_tiger: bool = False,
        notes: str = "",
        db: Session = None,
    ) -> bool:
        """
        Submit a human review decision.
        Preserves original AI prediction while updating the decision routing.
        """
        if db is None:
            return False

        decision = db.query(Decision).filter(Decision.image_id == image_id).first()
        if not decision:
            return False

        orig_ai_prediction = decision.species or "None"
        orig_ai_confidence = decision.confidence

        final_species = human_prediction if human_prediction else orig_ai_prediction
        if action.upper() == "ACCEPT":
            review_decision_status = "verified_accepted"
            final_conf = 1.0
            final_level = "high"
        elif action.upper() == "CORRECT":
            review_decision_status = "human_corrected"
            final_conf = 1.0
            final_level = "high"
        elif action.upper() == "REJECT":
            review_decision_status = "human_rejected"
            final_conf = 0.0
            final_level = "low"
            final_species = "None"
        elif action.upper() == "ESCALATE":
            review_decision_status = "escalated_to_expert"
            final_conf = decision.confidence
            final_level = "medium"
        else:
            review_decision_status = "human_reviewed"
            final_conf = 1.0
            final_level = "high"

        # Record human review entry
        review_record = HumanReview(
            id=str(uuid.uuid4()),
            image_id=image_id,
            ai_prediction=orig_ai_prediction,
            ai_confidence=orig_ai_confidence,
            human_prediction=final_species,
            human_is_tiger=is_tiger or (final_species == "Bengal Tiger"),
            reviewer_id=reviewer_id,
            notes=f"Action: {action.upper()}. {notes}".strip()
        )
        db.add(review_record)

        # Update decision model
        decision.decision = review_decision_status
        decision.species = final_species
        decision.is_tiger = is_tiger or (final_species == "Bengal Tiger")
        decision.confidence = final_conf
        decision.confidence_level = final_level

        if decision.reasoning is None:
            decision.reasoning = []
        decision.reasoning.append(
            f"👤 HUMAN REVIEW ({action.upper()}): Reviewer '{reviewer_id}' set species to '{final_species}'"
        )

        # Record audit trail
        audit = AuditTrail(
            id=str(uuid.uuid4()),
            image_id=image_id,
            event_type="human_review",
            event_status="pass",
            details={
                "reviewer": reviewer_id,
                "action": action.upper(),
                "ai_prediction": orig_ai_prediction,
                "ai_confidence": orig_ai_confidence,
                "human_prediction": final_species,
                "is_correction": orig_ai_prediction != final_species,
                "notes": notes
            }
        )
        db.add(audit)
        db.commit()

        # Release lock
        cls.release_item(image_id, reviewer_id)
        return True

    @staticmethod
    def get_review_stats(db: Session) -> Dict[str, Any]:
        """Calculate statistics on AI vs Human review performance"""
        total_reviews = db.query(HumanReview).count()
        pending_count = db.query(Decision).filter(
            Decision.decision.in_(["human_review", "uncertain"])
        ).count()

        if total_reviews == 0:
            return {
                "total_reviews": 0,
                "pending_reviews": pending_count,
                "agreement_rate": 0.0,
                "corrections": 0,
                "correction_rate": 0.0,
                "ai_accepted": 0,
                "human_rejected": 0,
            }

        agreements = db.query(HumanReview).filter(
            HumanReview.ai_prediction == HumanReview.human_prediction
        ).count()

        corrections = total_reviews - agreements

        return {
            "total_reviews": total_reviews,
            "pending_reviews": pending_count,
            "agreement_rate": round((agreements / total_reviews) * 100, 2),
            "corrections": corrections,
            "correction_rate": round((corrections / total_reviews) * 100, 2),
            "ai_accepted": agreements,
            "human_rejected": db.query(Decision).filter(Decision.decision == "human_rejected").count(),
        }
