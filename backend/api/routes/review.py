"""
VanRakshak AI - Review Routes
API endpoints for human-in-the-loop verification, claiming, and auditing.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List

from db.database import get_db
from db.schemas import APIResponse
from services.review_service import ReviewService
from services.image_service import ImageService
from db.models import Decision, Image, HumanReview

router = APIRouter(prefix="/api/review", tags=["review"])


class SubmitReviewActionRequest(BaseModel):
    image_id: str
    reviewer_id: str = "Field Ranger"
    action: str = Field(default="ACCEPT", description="ACCEPT, REJECT, CORRECT, ESCALATE, REPROCESS")
    human_prediction: Optional[str] = None
    human_is_tiger: Optional[bool] = False
    notes: Optional[str] = ""


class ClaimItemRequest(BaseModel):
    reviewer_id: str = "Field Ranger"


@router.get("/queue")
def get_review_queue(
    species: Optional[str] = None,
    camera_id: Optional[str] = None,
    priority_only: bool = False,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Get list of images requiring human verification with optional filtering.
    """
    query = db.query(Image, Decision).join(
        Decision, Image.id == Decision.image_id
    ).filter(
        Decision.decision.in_(["human_review", "uncertain", "escalated_to_expert"])
    )

    if species:
        query = query.filter(Decision.species == species)
    if camera_id:
        query = query.filter(Image.camera_id == camera_id)
    if priority_only:
        query = query.filter(Decision.is_tiger == True)

    total_count = query.count()
    records = query.order_by(Image.timestamp.desc()).offset(offset).limit(limit).all()

    queue_items = []
    for img, dec in records:
        queue_items.append({
            "image_id": img.id,
            "camera_id": img.camera_id,
            "timestamp": img.timestamp.isoformat() if img.timestamp else None,
            "image_path": img.image_path,
            "ai_prediction": dec.species,
            "confidence": dec.confidence,
            "decision": dec.decision,
            "confidence_level": dec.confidence_level,
            "reasoning": dec.reasoning,
            "is_tiger": dec.is_tiger,
            "quality_status": img.quality_status
        })

    return APIResponse(
        success=True,
        message=f"Retrieved {len(queue_items)} images for review",
        data={
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "items": queue_items
        }
    ).model_dump()


@router.get("/item/{image_id}")
def get_review_item(
    image_id: str,
    db: Session = Depends(get_db)
):
    """Retrieve complete multi-model evidence for a review item"""
    evidence = ReviewService.get_item_evidence(image_id, db)
    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review item {image_id} not found"
        )

    return APIResponse(
        success=True,
        message="Review item evidence retrieved",
        data=evidence
    ).model_dump()


@router.post("/claim/{image_id}")
def claim_review_item(
    image_id: str,
    req: ClaimItemRequest,
    db: Session = Depends(get_db)
):
    """Claim and lock an item for review"""
    res = ReviewService.claim_item(image_id, req.reviewer_id)
    if not res["success"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=res["message"]
        )

    return APIResponse(
        success=True,
        message=res["message"],
        data=res
    ).model_dump()


@router.post("/release/{image_id}")
def release_review_item(
    image_id: str,
    reviewer_id: Optional[str] = "",
    db: Session = Depends(get_db)
):
    """Release a claimed item back to the queue"""
    released = ReviewService.release_item(image_id, reviewer_id)
    return APIResponse(
        success=True,
        message="Item released" if released else "Item was not claimed",
        data={"released": released}
    ).model_dump()


@router.post("/submit")
def submit_review_decision(
    req: SubmitReviewActionRequest,
    db: Session = Depends(get_db)
):
    """Submit a human verification decision (ACCEPT, REJECT, CORRECT, ESCALATE)"""
    success = ReviewService.submit_review(
        image_id=req.image_id,
        reviewer_id=req.reviewer_id,
        action=req.action,
        human_prediction=req.human_prediction,
        is_tiger=req.human_is_tiger or (req.human_prediction == "Bengal Tiger"),
        notes=req.notes or "",
        db=db
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Decision record not found or review failed"
        )

    return APIResponse(
        success=True,
        message=f"Review action '{req.action.upper()}' submitted successfully",
        data={"image_id": req.image_id, "action": req.action.upper()}
    ).model_dump()


@router.get("/stats")
def get_review_stats(db: Session = Depends(get_db)):
    """Get AI vs Human agreement statistics and queue metrics"""
    stats = ReviewService.get_review_stats(db)
    return APIResponse(
        success=True,
        message="Review statistics retrieved",
        data=stats
    ).model_dump()
