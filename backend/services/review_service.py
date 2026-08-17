"""
VanRakshak AI - Review Service
Handles human-in-the-loop verification of AI decisions.
"""

from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from db.models import Image, Decision, HumanReview, AuditTrail


class ReviewService:
    """
    Service for managing the human review process.
    Allows rangers/experts to verify uncertain AI classifications.
    """

    @staticmethod
    def submit_review(
        image_id: str,
        reviewer_id: str,
        human_prediction: str,
        is_tiger: bool,
        notes: str,
        db: Session
    ) -> bool:
        """
        Submit a human review for an image.
        
        Args:
            image_id: Image being reviewed
            reviewer_id: ID of reviewer (ranger/expert)
            human_prediction: Human's classification
            is_tiger: Boolean indicating if it's a tiger
            notes: Optional notes
            db: Database session
            
        Returns:
            True if successful
        """
        # Get existing decision
        decision = db.query(Decision).filter(Decision.image_id == image_id).first()
        if not decision:
            return False
            
        # Record review
        review = HumanReview(
            image_id=image_id,
            ai_prediction=decision.species or "None",
            ai_confidence=decision.confidence,
            human_prediction=human_prediction,
            human_is_tiger=is_tiger,
            reviewer_id=reviewer_id,
            notes=notes
        )
        db.add(review)
        
        # Update decision status to show it was reviewed
        decision.decision = "human_reviewed"
        decision.species = human_prediction
        decision.is_tiger = is_tiger
        decision.confidence = 1.0  # Human confidence is 100%
        decision.confidence_level = "high"
        
        if decision.reasoning is None:
            decision.reasoning = []
            
        decision.reasoning.append(
            f"✅ HUMAN REVIEW: {reviewer_id} classified as {human_prediction}"
        )
        
        # Audit trail
        audit = AuditTrail(
            image_id=image_id,
            event_type="human_review",
            event_status="pass",
            details={
                "reviewer": reviewer_id,
                "ai_prediction": review.ai_prediction,
                "human_prediction": human_prediction,
                "is_correction": review.ai_prediction != human_prediction
            }
        )
        db.add(audit)
        
        db.commit()
        return True
        
    @staticmethod
    def get_review_stats(db: Session) -> Dict:
        """
        Calculate statistics on AI vs Human agreement.
        Used for model performance monitoring.
        """
        total_reviews = db.query(HumanReview).count()
        
        if total_reviews == 0:
            return {
                "total_reviews": 0,
                "agreement_rate": 0,
                "corrections": 0,
            }
            
        # Count where human agreed with AI
        agreements = db.query(HumanReview).filter(
            HumanReview.ai_prediction == HumanReview.human_prediction
        ).count()
        
        # Count corrections (disagreements)
        corrections = total_reviews - agreements
        
        return {
            "total_reviews": total_reviews,
            "agreement_rate": (agreements / total_reviews) * 100,
            "corrections": corrections,
            "correction_rate": (corrections / total_reviews) * 100
        }
