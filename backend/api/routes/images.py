"""
VanRakshak AI - Image Upload Routes
API endpoints for camera-trap image ingestion
"""

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
from db.database import get_db
from db.schemas import ImageUploadRequest, APIResponse
from db.models import Image, Detection, Classification, Verification, Segmentation, Decision
from services.image_service import ImageService
from core.pipeline import ProcessingPipeline
from config import settings

router = APIRouter(prefix="/api/images", tags=["images"])


@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    camera_id: str = Form("USER_UPLOAD"),
    location: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload a camera-trap image and run the full AI pipeline.

    Returns comprehensive analysis results including:
    - Quality assessment metrics
    - Object detections (animal/human/vehicle)
    - Species classification with alternatives
    - OpenCLIP semantic verification
    - Decision engine routing
    - Explainability reasoning
    """

    # Validate file type
    allowed = ["image/jpeg", "image/png", "image/gif", "image/tiff", "image/webp"]
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Supported: JPG, PNG, GIF, TIFF, WEBP"
        )

    # Read file bytes
    image_bytes = await file.read()
    file_size = len(image_bytes)

    max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size} bytes (max {max_size})"
        )

    ts = datetime.utcnow()

    try:
        # Step 1: Ingest the image (or find existing duplicate)
        import hashlib
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        # Check if this image already exists
        existing = db.query(Image).filter(Image.image_hash == image_hash).first()

        if existing:
            image_id = existing.id
            # Clear old pipeline results so we can re-run fresh
            from db.models import AuditTrail, Segmentation, Verification as VerifModel, TigerReidentification, Alert
            db.query(Detection).filter(Detection.image_id == image_id).delete()
            db.query(Classification).filter(Classification.image_id == image_id).delete()
            db.query(Verification).filter(Verification.image_id == image_id).delete()
            db.query(Segmentation).filter(Segmentation.image_id == image_id).delete()
            db.query(Decision).filter(Decision.image_id == image_id).delete()
            db.query(AuditTrail).filter(AuditTrail.image_id == image_id).delete()
            db.commit()
        else:
            image_id, metadata = ImageService.ingest_image(
                image_bytes=image_bytes,
                camera_id=camera_id,
                timestamp=ts,
                location=location,
                db=db
            )

        # Step 2: Run the full AI pipeline
        pipeline_result = ProcessingPipeline.process_image(image_id, db)

        # Step 3: Query all results from DB to build the response
        image = db.query(Image).filter(Image.id == image_id).first()
        detections = db.query(Detection).filter(Detection.image_id == image_id).all()
        classifications = db.query(Classification).filter(Classification.image_id == image_id).all()
        verifications = db.query(Verification).filter(Verification.image_id == image_id).all()
        decision = db.query(Decision).filter(Decision.image_id == image_id).first()

        # Build detections list
        detection_results = []
        for det in detections:
            det_classifs = [c for c in classifications if c.detection_id == det.id]
            detection_results.append({
                "id": det.id,
                "object_type": det.object_type,
                "confidence": det.confidence,
                "bbox": {
                    "x_min": det.bbox_x_min,
                    "y_min": det.bbox_y_min,
                    "x_max": det.bbox_x_max,
                    "y_max": det.bbox_y_max
                },
                "crop_path": det.crop_path,
                "classifications": [
                    {
                        "species": c.species,
                        "confidence": c.confidence,
                        "alternatives": c.alternative_predictions or [],
                        "model_name": c.model_name
                    }
                    for c in det_classifs
                ]
            })

        # Build verification result
        verification_data = None
        if verifications:
            v = verifications[0]
            verification_data = {
                "primary_prediction": v.primary_prediction,
                "confidence": v.confidence,
                "semantic_scores": v.semantic_scores,
                "model_name": v.model_name
            }

        # Build decision result
        decision_data = None
        if decision:
            decision_data = {
                "species": decision.species,
                "confidence": decision.confidence,
                "decision": decision.decision,
                "confidence_level": decision.confidence_level,
                "reasoning": decision.reasoning,
                "signals": decision.signals,
                "is_tiger": decision.is_tiger
            }

        # Final response
        return APIResponse(
            success=True,
            message="Image analyzed successfully",
            data={
                "image_id": image_id,
                "pipeline_success": pipeline_result.success,
                "pipeline_time_ms": round(pipeline_result.total_time_ms, 2),
                "image": {
                    "camera_id": image.camera_id if image else camera_id,
                    "timestamp": str(ts),
                    "location": location,
                    "width": image.image_width if image else None,
                    "height": image.image_height if image else None,
                    "file_size": file_size,
                    "image_path": image.image_path if image else None,
                },
                "quality": {
                    "status": image.quality_status if image else "unknown",
                    "score": image.quality_score if image else None,
                    "blur_score": image.blur_score if image else None,
                    "brightness": image.brightness if image else None,
                    "contrast": image.contrast if image else None,
                    "passed": pipeline_result.quality_passed
                },
                "detections": detection_results,
                "verification": verification_data,
                "decision": decision_data,
            }
        ).model_dump()

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    timestamp: str = Form(...),
    gps_latitude: Optional[float] = Form(None),
    gps_longitude: Optional[float] = Form(None),
    location: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload a camera-trap image
    
    Accepts multipart/form-data with:
    - file: Image file (JPG, PNG, GIF, TIFF)
    - camera_id: Camera identifier
    - timestamp: ISO format timestamp
    - gps_latitude: GPS latitude (optional)
    - gps_longitude: GPS longitude (optional)
    - location: Location name (optional)
    
    Returns:
    - Image ID
    - Image metadata
    - Quality assessment
    """
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/gif", "image/tiff"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}"
        )
    
    # Validate file size
    file_size = 0
    image_bytes = b""
    async for chunk in file.file:
        image_bytes += chunk
        file_size += len(chunk)
    
    max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size} bytes (max {max_size})"
        )
    
    # Parse timestamp
    try:
        ts = datetime.fromisoformat(timestamp)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timestamp format: {timestamp} (use ISO format)"
        )
    
    try:
        # Ingest image
        image_id, metadata = ImageService.ingest_image(
            image_bytes=image_bytes,
            camera_id=camera_id,
            timestamp=ts,
            gps_latitude=gps_latitude,
            gps_longitude=gps_longitude,
            location=location,
            db=db
        )
        
        return APIResponse(
            success=True,
            message="Image uploaded successfully",
            data={
                "image_id": image_id,
                "metadata": metadata
            }
        ).model_dump()
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload image: {str(e)}"
        )


@router.get("/image/{image_id}")
async def get_image_info(
    image_id: str,
    db: Session = Depends(get_db)
):
    """Get image metadata and current processing status"""
    
    image = ImageService.get_image(image_id, db)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return APIResponse(
        success=True,
        message="Image found",
        data={
            "image_id": image.id,
            "camera_id": image.camera_id,
            "timestamp": image.timestamp,
            "location": image.location,
            "gps": {
                "latitude": image.gps_latitude,
                "longitude": image.gps_longitude
            },
            "image_path": image.image_path,
            "quality_status": image.quality_status,
            "quality_score": image.quality_score,
            "blur_score": image.blur_score,
            "brightness": image.brightness,
            "contrast": image.contrast,
            "dimensions": {
                "width": image.image_width,
                "height": image.image_height
            },
            "file_size": image.file_size,
            "created_at": image.created_at
        }
    ).model_dump()


@router.get("/camera/{camera_id}")
async def get_camera_images(
    camera_id: str,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get recent images from a specific camera"""
    
    images = ImageService.get_images_by_camera(camera_id, db, limit=limit)
    
    return APIResponse(
        success=True,
        message=f"Retrieved {len(images)} images",
        data={
            "camera_id": camera_id,
            "count": len(images),
            "images": [
                {
                    "image_id": img.id,
                    "timestamp": img.timestamp,
                    "quality_status": img.quality_status,
                    "quality_score": img.quality_score
                }
                for img in images
            ]
        }
    ).model_dump()


@router.get("/review-queue")
async def get_review_queue(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get images pending human review"""
    
    images = ImageService.get_images_requiring_review(db, limit=limit)
    
    return APIResponse(
        success=True,
        message=f"Retrieved {len(images)} images for review",
        data={
            "count": len(images),
            "images": [
                {
                    "image_id": img.id,
                    "camera_id": img.camera_id,
                    "timestamp": img.timestamp,
                    "location": img.location,
                    "image_path": img.image_path
                }
                for img in images
            ]
        }
    ).model_dump()


@router.get("/stats")
async def get_image_stats(db: Session = Depends(get_db)):
    """Get image processing statistics"""
    
    total = db.query(Image).count()
    
    from db.models import Decision
    auto_accepted = db.query(Decision).filter(
        Decision.decision == "auto_accept"
    ).count()
    
    human_reviewed = db.query(Decision).filter(
        Decision.decision == "human_review"
    ).count()
    
    pending = db.query(Decision).filter(
        Decision.decision == "uncertain"
    ).count()
    
    return APIResponse(
        success=True,
        message="Image statistics",
        data={
            "total_images": total,
            "auto_accepted": auto_accepted,
            "human_reviewed": human_reviewed,
            "pending_review": pending,
            "processed_percentage": (auto_accepted / total * 100) if total > 0 else 0
        }
    ).model_dump()
