"""
VanRakshak AI - Quality Gate Routes
API endpoints for image quality assessment
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from db.schemas import APIResponse, ImageQualityResponse
from core.quality_gate import QualityGateService, QualityDecision
from typing import List, Optional

router = APIRouter(prefix="/api/quality", tags=["quality-gate"])


@router.post("/assess/{image_id}")
async def assess_image_quality(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Assess quality of a single image
    
    Evaluates:
    - Corruption
    - Blur (Laplacian variance)
    - Darkness (average brightness)
    - Overexposure (average brightness)
    
    Returns:
    - Quality score (0-1)
    - Decision (accept/reject)
    - Detailed reasons
    """
    
    result = QualityGateService.assess_quality(image_id, db)
    
    return APIResponse(
        success=True,
        message=f"Image assessment complete: {result.decision.value}",
        data={
            "image_id": image_id,
            "decision": result.decision.value,
            "quality_score": result.quality_score,
            "passed": result.passed,
            "reasons": result.reasons,
            "details": result.details
        }
    ).model_dump()


@router.post("/gate/{image_id}")
async def apply_quality_gate(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Apply quality gate to image and update status
    
    Updates image.quality_status and image.quality_score
    
    Returns:
    - Whether image passed quality gate
    - New quality status
    """
    
    from db.models import Image
    
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    passed = QualityGateService.apply_quality_gate(image_id, db)
    
    # Refresh to get updated values
    db.refresh(image)
    
    return APIResponse(
        success=True,
        message="Quality gate applied",
        data={
            "image_id": image_id,
            "passed": passed,
            "quality_status": image.quality_status,
            "quality_score": image.quality_score
        }
    ).model_dump()


@router.post("/gate-batch")
async def batch_apply_quality_gate(
    image_ids: List[str],
    db: Session = Depends(get_db)
):
    """
    Apply quality gate to multiple images
    
    Args:
    - image_ids: List of image IDs to assess
    
    Returns:
    - Results for each image
    """
    
    if not image_ids:
        raise HTTPException(status_code=400, detail="No image IDs provided")
    
    if len(image_ids) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 images per request")
    
    results = QualityGateService.batch_quality_gate(image_ids, db)
    
    # Summary stats
    passed_count = sum(1 for r in results.values() if r.passed)
    rejected_count = len(results) - passed_count
    
    return APIResponse(
        success=True,
        message=f"Quality gate applied to {len(results)} images",
        data={
            "total": len(results),
            "passed": passed_count,
            "rejected": rejected_count,
            "results": {
                img_id: {
                    "decision": result.decision.value,
                    "quality_score": result.quality_score,
                    "passed": result.passed,
                    "reasons": result.reasons
                }
                for img_id, result in results.items()
            }
        }
    ).model_dump()


@router.get("/breakdown")
async def get_quality_breakdown(
    camera_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get quality breakdown statistics
    
    Shows percentage of images in each quality category
    (GOOD, BLURRY, TOO_DARK, OVEREXPOSED, CORRUPTED, DUPLICATE)
    
    Args:
    - camera_id: Filter by camera (optional)
    
    Returns:
    - Breakdown by quality status with counts and percentages
    """
    
    breakdown = QualityGateService.get_quality_breakdown(camera_id, db)
    
    return APIResponse(
        success=True,
        message="Quality breakdown retrieved",
        data=breakdown
    ).model_dump()


@router.get("/rejection-reasons")
async def get_rejection_reasons(
    camera_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get rejection reason statistics
    
    Shows why images were rejected (blur, darkness, etc.)
    
    Args:
    - camera_id: Filter by camera (optional)
    
    Returns:
    - Count of images rejected for each reason
    """
    
    reasons = QualityGateService.get_rejection_reasons(camera_id, db)
    
    return APIResponse(
        success=True,
        message="Rejection reasons retrieved",
        data={
            "camera_id": camera_id,
            "reasons": reasons,
            "total_rejected": sum(reasons.values())
        }
    ).model_dump()


@router.get("/report")
async def get_quality_report(
    camera_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive quality report
    
    Combines breakdown and rejection reasons
    
    Args:
    - camera_id: Filter by camera (optional)
    
    Returns:
    - Complete quality assessment report
    """
    
    breakdown = QualityGateService.get_quality_breakdown(camera_id, db)
    reasons = QualityGateService.get_rejection_reasons(camera_id, db)
    
    return APIResponse(
        success=True,
        message="Quality report retrieved",
        data={
            "camera_id": camera_id,
            "summary": {
                "total_images": breakdown["total"],
                "good_percentage": breakdown["good_percentage"],
                "rejected_percentage": 100 - breakdown["good_percentage"]
            },
            "breakdown": breakdown["breakdown"],
            "rejection_reasons": reasons
        }
    ).model_dump()


@router.get("/quality-status/{image_id}")
async def get_image_quality_status(
    image_id: str,
    db: Session = Depends(get_db)
):
    """Get quality status of a specific image"""
    
    from db.models import Image
    
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return APIResponse(
        success=True,
        message="Quality status retrieved",
        data={
            "image_id": image_id,
            "quality_status": image.quality_status,
            "quality_score": image.quality_score,
            "blur_score": image.blur_score,
            "brightness": image.brightness,
            "contrast": image.contrast
        }
    ).model_dump()
