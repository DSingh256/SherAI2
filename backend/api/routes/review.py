"""
VanRakshak AI - Review Routes
API endpoints for human-in-the-loop review.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from db.database import get_db
from db.schemas import APIResponse, HumanReviewRequest
from services.review_service import ReviewService
from services.image_service import ImageService
from db.models import Decision, Image

router = APIRouter(prefix="/api/review", tags=["review"])


@router.get("/queue")
async def get_review_queue(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get list of images requiring human verification"""
    
    images = ImageService.get_images_requiring_review(db, limit=limit)
    
    queue_items = []
    for img in images:
        decision = db.query(Decision).filter(Decision.image_id == img.id).first()
        if decision:
            queue_items.append({
                "image_id": img.id,
                "camera_id": img.camera_id,
                "timestamp": img.timestamp,
                "image_path": img.image_path,
                "ai_prediction": decision.species,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "is_tiger": decision.is_tiger
            })
            
    return APIResponse(
        success=True,
        message=f"Retrieved {len(queue_items)} images for review",
        data={"items": queue_items}
    ).model_dump()


@router.post("/submit")
async def submit_review(
    request: HumanReviewRequest,
    db: Session = Depends(get_db)
):
    """Submit a human verification decision"""
    
    success = ReviewService.submit_review(
        image_id=request.image_id,
        reviewer_id=request.reviewer_id or "Anonymous Ranger",
        human_prediction=request.human_prediction,
        is_tiger=request.human_is_tiger or (request.human_prediction == "Bengal Tiger"),
        notes=request.notes or "",
        db=db
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Decision record not found for this image")
        
    return APIResponse(
        success=True,
        message="Review submitted successfully",
        data={"image_id": request.image_id}
    ).model_dump()


@router.get("/stats")
async def get_review_stats(db: Session = Depends(get_db)):
    """Get AI vs Human agreement statistics"""
    
    stats = ReviewService.get_review_stats(db)
    
    return APIResponse(
        success=True,
        message="Review stats retrieved",
        data=stats
    ).model_dump()
