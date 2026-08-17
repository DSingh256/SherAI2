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
from services.image_service import ImageService
from config import settings

router = APIRouter(prefix="/api/images", tags=["images"])


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
