"""
VanRakshak AI - MegaDetector Routes
API endpoints for running MegaDetector and retrieving object detections.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import os
import uuid
import logging
from typing import List

from db.database import get_db
from db.models import Image, Detection, ImageQuality, AuditTrail
from db.schemas import APIResponse, DetectionsResponse, DetectionResult, BoundingBox
from core.megadetector import MegaDetectorService, DetectionCategory
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detections", tags=["detections"])


@router.post("/detect/{image_id}")
def run_detection(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Run MegaDetector on an image
    
    Requirements:
    - Image must exist in the database.
    - Image quality status must be 'good' (verified by Quality Gate).
    
    Performs object detection, crops detected regions, saves detection records,
    and logs the event in the audit trail.
    """
    print(f"DEBUG API: db id = {id(db)}")
    print(f"DEBUG API: db bind = {db.bind}")
    print(f"DEBUG API: db bind URL = {db.bind.url if db.bind else 'None'}")
    print(f"DEBUG API: db in_transaction = {db.in_transaction()}")
    print(f"DEBUG API: db is_active = {db.is_active}")
    if hasattr(db, 'connection'):
        try:
            conn = db.connection()
            print(f"DEBUG API: connection = {conn}")
            print(f"DEBUG API: connection engine = {conn.engine}")
        except Exception as e:
            print(f"DEBUG API: connection error = {e}")
    # 1. Fetch image record
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )
        
    # 2. Check quality gating (must be GOOD to proceed)
    if image.quality_status != ImageQuality.GOOD.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot run detection: Image quality status is '{image.quality_status}' (required: 'good')"
        )
        
    # 3. Clean up existing detections for idempotency
    existing_detections = db.query(Detection).filter(Detection.image_id == image_id).all()
    for det in existing_detections:
        # Delete cropped file if it exists
        if det.crop_path and os.path.exists(det.crop_path):
            try:
                os.remove(det.crop_path)
            except Exception as e:
                logger.warning(f"Failed to delete crop file: {e}")
        db.delete(det)
    db.commit()
    
    try:
        # 4. Execute MegaDetector V6
        md_output = MegaDetectorService.detect(image.image_path, image_id)
        
        # 5. Process and store detections
        stored_detections = []
        for d in md_output.detections:
            # Generate unique ID for crop filename
            det_id = str(uuid.uuid4())
            crop_path = MegaDetectorService.crop_detection(
                image.image_path, d.bbox, detection_id=det_id
            )
            
            det_record = Detection(
                id=det_id,
                image_id=image_id,
                object_type=d.object_type.value,
                confidence=d.confidence,
                bbox_x_min=d.bbox.x_min,
                bbox_y_min=d.bbox.y_min,
                bbox_x_max=d.bbox.x_max,
                bbox_y_max=d.bbox.y_max,
                crop_path=crop_path
            )
            db.add(det_record)
            db.flush() # Populate ID
            
            stored_detections.append(DetectionResult(
                object_type=d.object_type.value,
                confidence=d.confidence,
                bbox=BoundingBox(
                    x_min=d.bbox.x_min,
                    y_min=d.bbox.y_min,
                    x_max=d.bbox.x_max,
                    y_max=d.bbox.y_max
                ),
                crop_path=crop_path
            ))
            
        # 6. Record in Audit Trail
        audit_details = {
            "no_detections": md_output.no_detections,
            "detection_count": len(stored_detections),
            "detections": [
                {"category": d.object_type.value, "confidence": d.confidence}
                for d in md_output.detections
            ],
            "processing_time_ms": md_output.processing_time_ms
        }
        audit = AuditTrail(
            image_id=image_id,
            event_type="detection",
            event_status="pass" if len(stored_detections) > 0 else "pending",
            details=audit_details
        )
        db.add(audit)
        db.commit()
        
        response_data = DetectionsResponse(
            image_id=image_id,
            detections=stored_detections,
            no_detections=len(stored_detections) == 0,
            processing_time_ms=md_output.processing_time_ms
        )
        
        return APIResponse(
            success=True,
            message="MegaDetector run completed successfully",
            data=response_data
        ).model_dump()
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed during detection execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection execution failed: {str(e)}"
        )


@router.get("/image/{image_id}")
def get_image_detections(
    image_id: str,
    db: Session = Depends(get_db)
):
    """Retrieve all detections associated with a specific image"""
    detections = db.query(Detection).filter(Detection.image_id == image_id).all()
    
    results = []
    for d in detections:
        results.append(DetectionResult(
            object_type=d.object_type,
            confidence=d.confidence,
            bbox=BoundingBox(
                x_min=d.bbox_x_min,
                y_min=d.bbox_y_min,
                x_max=d.bbox_x_max,
                y_max=d.bbox_y_max
            ),
            crop_path=d.crop_path
        ))
        
    return APIResponse(
        success=True,
        message=f"Retrieved {len(results)} detections",
        data={
            "image_id": image_id,
            "detections": results
        }
    ).model_dump()


@router.get("/stats")
def get_detection_stats(
    db: Session = Depends(get_db)
):
    """Get statistics summary of detections across all processed images"""
    total_detections = db.query(Detection).count()
    animals = db.query(Detection).filter(Detection.object_type == "animal").count()
    humans = db.query(Detection).filter(Detection.object_type == "human").count()
    vehicles = db.query(Detection).filter(Detection.object_type == "vehicle").count()
    
    return APIResponse(
        success=True,
        message="Detection stats retrieved",
        data={
            "total_detections": total_detections,
            "breakdown": {
                "animals": animals,
                "humans": humans,
                "vehicles": vehicles
            }
        }
    ).model_dump()
