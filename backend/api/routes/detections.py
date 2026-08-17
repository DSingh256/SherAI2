"""
VanRakshak AI - MegaDetector, SpeciesNet & OpenCLIP Routes
API endpoints for running object detection, species classification, and semantic verification.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import os
import uuid
import logging
from typing import List

from db.database import get_db
from db.models import Image, Detection, Classification, Verification, ImageQuality, AuditTrail
from db.schemas import (
    APIResponse, DetectionsResponse, DetectionResult, BoundingBox,
    ClassificationResult, ClassificationsResponse, AlternativePrediction,
    VerificationResult
)
from core.megadetector import MegaDetectorService, DetectionCategory
from core.species_classifier import SpeciesClassifierService
from core.semantic_verifier import SemanticVerifierService
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
        if det.crop_path and os.path.exists(det.crop_path):
            try:
                os.remove(det.crop_path)
            except Exception as e:
                logger.warning(f"Failed to delete crop file: {e}")
        db.query(Classification).filter(Classification.detection_id == det.id).delete()
        db.delete(det)
    db.commit()
    
    try:
        # 4. Execute MegaDetector V6
        md_output = MegaDetectorService.detect(image.image_path, image_id)
        
        # 5. Process and store detections
        stored_detections = []
        for d in md_output.detections:
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
            db.flush()
            
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


@router.post("/classify/{detection_id}")
def classify_detection(
    detection_id: str,
    top_k: int = 5,
    db: Session = Depends(get_db)
):
    """
    Run SpeciesNet classification on a cropped animal detection.
    """
    det = db.query(Detection).filter(Detection.id == detection_id).first()
    if not det:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection {detection_id} not found"
        )
        
    if not det.crop_path or not os.path.exists(det.crop_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Crop file for detection {detection_id} does not exist"
        )
        
    try:
        res = SpeciesClassifierService.classify(det.crop_path, detection_id, top_k=top_k)
        
        existing_class = db.query(Classification).filter(Classification.detection_id == detection_id).first()
        if existing_class:
            existing_class.species = res.primary_species
            existing_class.confidence = res.primary_confidence
            existing_class.alternative_predictions = [a.to_dict() for a in res.alternatives]
            existing_class.model_name = res.model_name
        else:
            new_class = Classification(
                image_id=det.image_id,
                detection_id=detection_id,
                species=res.primary_species,
                confidence=res.primary_confidence,
                alternative_predictions=[a.to_dict() for a in res.alternatives],
                model_name=res.model_name
            )
            db.add(new_class)
            
        audit = AuditTrail(
            image_id=det.image_id,
            event_type="classification",
            event_status="pass",
            details={
                "detection_id": detection_id,
                "primary_species": res.primary_species,
                "confidence": res.primary_confidence,
                "is_tiger": res.is_tiger,
                "passes_threshold": res.passes_threshold,
                "confidence_level": res.confidence_level
            }
        )
        db.add(audit)
        db.commit()
        
        return APIResponse(
            success=True,
            message="Classification completed",
            data=res.to_dict()
        ).model_dump()
        
    except Exception as e:
        db.rollback()
        logger.error(f"Classification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


@router.get("/classifications/{image_id}")
def get_image_classifications(
    image_id: str,
    db: Session = Depends(get_db)
):
    """Retrieve all SpeciesNet classifications for an image"""
    classes = db.query(Classification).filter(Classification.image_id == image_id).all()
    
    results = []
    for c in classes:
        alts = [
            AlternativePrediction(species=a["species"], confidence=a["confidence"])
            for a in (c.alternative_predictions or [])
        ]
        results.append(ClassificationResult(
            species=c.species,
            confidence=c.confidence,
            alternatives=alts,
            model_name=c.model_name
        ))
        
    return APIResponse(
        success=True,
        message=f"Retrieved {len(results)} classifications",
        data={
            "image_id": image_id,
            "classifications": results
        }
    ).model_dump()


@router.post("/verify/{image_id}")
def verify_image_semantics(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Run OpenCLIP semantic verification on an image.
    Cross-checks SpeciesNet prediction against vision-language prompt concepts.
    """
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )
        
    # Get latest classification if available
    latest_class = db.query(Classification).filter(Classification.image_id == image_id).first()
    predicted_species = latest_class.species if latest_class else ""
    predicted_conf = latest_class.confidence if latest_class else 0.0
    
    try:
        ver_res = SemanticVerifierService.verify(
            image.image_path,
            image_id=image_id,
            speciesnet_prediction=predicted_species,
            speciesnet_confidence=predicted_conf
        )
        
        # Save or update Verification in DB
        existing_ver = db.query(Verification).filter(Verification.image_id == image_id).first()
        if existing_ver:
            existing_ver.primary_prediction = ver_res.primary_prediction
            existing_ver.confidence = ver_res.primary_similarity
            existing_ver.semantic_scores = ver_res.scores
            existing_ver.model_name = ver_res.model_name
        else:
            new_ver = Verification(
                image_id=image_id,
                primary_prediction=ver_res.primary_prediction,
                confidence=ver_res.primary_similarity,
                semantic_scores=ver_res.scores,
                model_name=ver_res.model_name
            )
            db.add(new_ver)
            
        audit = AuditTrail(
            image_id=image_id,
            event_type="verification",
            event_status="pass" if ver_res.agrees_with_speciesnet else "review",
            details={
                "primary_prediction": ver_res.primary_prediction,
                "primary_similarity": ver_res.primary_similarity,
                "agrees_with_speciesnet": ver_res.agrees_with_speciesnet,
                "agreement_score": ver_res.agreement_score
            }
        )
        db.add(audit)
        db.commit()
        
        return APIResponse(
            success=True,
            message="Semantic verification completed",
            data=ver_res.to_dict()
        ).model_dump()
        
    except Exception as e:
        db.rollback()
        logger.error(f"Verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}"
        )


@router.get("/verifications/{image_id}")
def get_image_verification(
    image_id: str,
    db: Session = Depends(get_db)
):
    """Retrieve OpenCLIP semantic verification result for an image"""
    ver = db.query(Verification).filter(Verification.image_id == image_id).first()
    if not ver:
        return APIResponse(
            success=True,
            message="No verification record found",
            data=None
        ).model_dump()
        
    return APIResponse(
        success=True,
        message="Verification record retrieved",
        data={
            "image_id": image_id,
            "primary_prediction": ver.primary_prediction,
            "primary_similarity": ver.confidence,
            "semantic_scores": ver.semantic_scores,
            "model_name": ver.model_name
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
