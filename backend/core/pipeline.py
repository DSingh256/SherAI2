"""
VanRakshak AI - Master Orchestration Pipeline
Coordinates all AI models, quality gates, and decision engines.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import time
import uuid
import logging

from db.models import Image, Detection, Classification, Decision, AuditTrail, TigerReidentification, Verification, Segmentation, Alert
from db.schemas import DecisionSignals
from config import settings

from core.quality_gate import QualityGateService, QualityDecision
from core.megadetector import MegaDetectorService
from core.species_classifier import SpeciesClassifierService
from core.semantic_verifier import SemanticVerifierService
from core.segmentation import SegmentationService
from core.decision_engine import DecisionEngineService
from core.explainability import ExplainabilityService
from core.privacy_protection import PrivacyProtectionService
from core.threat_analyzer import ThreatAnalyzerService
from core.reidentification import ReIdentificationService

logger = logging.getLogger(__name__)


class PipelineResult:
    """Complete result of processing an image through the full pipeline"""
    def __init__(self, image_id: str):
        self.image_id = image_id
        self.success = True
        self.error_message = None
        self.total_time_ms = 0.0
        
        # Stage results
        self.quality_passed = False
        self.megadetector_output = None
        self.classifications = {}  # detection_id -> ClassificationResult
        self.semantic_verification = None
        self.segmentations = {}    # detection_id -> SegmentationResult
        self.decision = None       # DecisionEngineResult
        self.explanation = None    # ExplanationReport
        self.privacy_applied = False
        self.threat_analysis = None
        self.reid_result = None
        
    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "success": self.success,
            "error_message": self.error_message,
            "total_time_ms": round(self.total_time_ms, 2),
            "quality_passed": self.quality_passed,
            "has_detections": self.megadetector_output is not None and not self.megadetector_output.no_detections,
            "decision": self.decision.to_dict() if self.decision else None,
            "is_tiger": self.decision.is_tiger if self.decision else False,
            "requires_review": self.decision.decision.value != "auto_accept" if self.decision else False
        }


class ProcessingPipeline:
    """
    Master processing pipeline that orchestrates all AI models
    and business logic in the correct sequence.
    """

    @staticmethod
    def process_image(image_id: str, db: Session) -> PipelineResult:
        """
        Run the complete pipeline on an image.
        
        Pipeline stages:
        1. Quality Gate
        2. MegaDetector
        3. SpeciesNet (for animals)
        4. OpenCLIP Verification
        5. SAM Segmentation
        6. Decision Engine
        7. Explainability
        8. Privacy Protection (for humans)
        9. Threat Analysis
        10. Tiger Re-ID (if tiger)
        11. Database Commit
        """
        start_time = time.time()
        result = PipelineResult(image_id)
        
        try:
            # Fetch image record
            image = db.query(Image).filter(Image.id == image_id).first()
            if not image:
                result.success = False
                result.error_message = "Image not found in database"
                return result
                
            # ==========================================
            # STAGE 1: QUALITY GATE
            # ==========================================
            logger.info(f"[{image_id}] Stage 1: Quality Gate")
            quality_result = QualityGateService.assess_quality(image_id, db)
            QualityGateService.apply_quality_gate(image_id, db)
            result.quality_passed = quality_result.passed
            
            if not quality_result.passed:
                logger.info(f"[{image_id}] Quality gate failed: {quality_result.decision.value}")
                result.total_time_ms = (time.time() - start_time) * 1000
                return result
                
            # ==========================================
            # STAGE 2: MEGADETECTOR
            # ==========================================
            if not settings.ENABLE_MEGADETECTOR:
                logger.info(f"[{image_id}] MegaDetector disabled")
                return result
                
            logger.info(f"[{image_id}] Stage 2: MegaDetector")
            md_output = MegaDetectorService.detect(image.image_path, image_id)
            result.megadetector_output = md_output
            
            # Save detections to DB
            primary_detection = None
            max_conf = 0.0
            
            for d in md_output.detections:
                # Crop for downstream tasks
                crop_path = MegaDetectorService.crop_detection(
                    image.image_path, d.bbox, detection_id=str(uuid.uuid4())
                )
                
                det_record = Detection(
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
                db.flush()  # Get ID without committing
                
                # Keep track of primary (highest confidence) detection for decision engine
                if d.confidence > max_conf:
                    max_conf = d.confidence
                    primary_detection = {"record": det_record, "data": d}
                    
            db.commit()
            
            if md_output.no_detections:
                logger.info(f"[{image_id}] No detections found")
                # Fast track decision for empty images
                decision_record = Decision(
                    image_id=image_id,
                    confidence=1.0,
                    decision="auto_accept",
                    confidence_level="high",
                    reasoning=["No objects detected by MegaDetector (empty frame)"],
                    signals={},
                    is_tiger=False
                )
                db.add(decision_record)
                db.commit()
                result.total_time_ms = (time.time() - start_time) * 1000
                return result
                
            # ==========================================
            # STAGE 3, 4, 5: SPECIES CLASSIFICATION, VERIFICATION, SEGMENTATION
            # ==========================================
            speciesnet_prediction = ""
            speciesnet_conf = 0.0
            openclip_similarity = 0.0
            openclip_agrees = False
            openclip_prediction = ""
            
            # Process the primary detection
            if primary_detection:
                det = primary_detection["data"]
                det_record = primary_detection["record"]
                
                # Privacy protection for humans
                if det.object_type.value == "human":
                    logger.info(f"[{image_id}] Stage 8: Privacy Protection")
                    privacy = PrivacyProtectionService.protect(
                        image.image_path, image_id,
                        det.bbox.x_min, det.bbox.y_min, det.bbox.x_max, det.bbox.y_max
                    )
                    result.privacy_applied = privacy.applied
                    
                    # Create audit trail
                    if privacy.applied:
                        audit = AuditTrail(
                            image_id=image_id,
                            event_type="privacy_protection",
                            event_status="pass",
                            details={"faces_blurred": privacy.faces_blurred}
                        )
                        db.add(audit)
                
                # Classification for animals
                if det.object_type.value == "animal" and det_record.crop_path and settings.ENABLE_SPECIESNET:
                    logger.info(f"[{image_id}] Stage 3: SpeciesNet")
                    class_result = SpeciesClassifierService.classify(
                        det_record.crop_path, det_record.id
                    )
                    result.classifications[det_record.id] = class_result
                    
                    speciesnet_prediction = class_result.primary_species
                    speciesnet_conf = class_result.primary_confidence
                    
                    # Save classification
                    class_record = Classification(
                        image_id=image_id,
                        detection_id=det_record.id,
                        species=class_result.primary_species,
                        confidence=class_result.primary_confidence,
                        alternative_predictions=[a.to_dict() for a in class_result.alternatives],
                        model_name=class_result.model_name
                    )
                    db.add(class_record)
                    
                    # OpenCLIP Verification
                    if settings.ENABLE_OPENCLIP:
                        logger.info(f"[{image_id}] Stage 4: OpenCLIP")
                        verify_result = SemanticVerifierService.verify(
                            image.image_path, image_id,
                            speciesnet_prediction, speciesnet_conf
                        )
                        result.semantic_verification = verify_result
                        
                        openclip_prediction = verify_result.primary_prediction
                        openclip_similarity = verify_result.primary_similarity
                        openclip_agrees = verify_result.agrees_with_speciesnet
                        
                        # Save verification
                        verify_record = Verification(
                            image_id=image_id,
                            semantic_scores=verify_result.scores,
                            primary_prediction=verify_result.primary_prediction,
                            confidence=verify_result.primary_similarity,
                            model_name=verify_result.model_name
                        )
                        db.add(verify_record)
                        
                    # SAM Segmentation
                    if settings.ENABLE_SAM_SEGMENTATION:
                        logger.info(f"[{image_id}] Stage 5: SAM Segmentation")
                        seg_result = SegmentationService.segment(
                            image.image_path, image_id, det_record.id,
                            det.bbox.x_min, det.bbox.y_min, det.bbox.x_max, det.bbox.y_max,
                            species=speciesnet_prediction
                        )
                        
                        if seg_result:
                            result.segmentations[det_record.id] = seg_result
                            
                            seg_record = Segmentation(
                                image_id=image_id,
                                detection_id=det_record.id,
                                mask_path=seg_result.mask_path,
                                segmented_crop_path=seg_result.segmented_crop_path,
                                model_name=seg_result.model_name
                            )
                            db.add(seg_record)
            
            # ==========================================
            # STAGE 6: DECISION ENGINE
            # ==========================================
            logger.info(f"[{image_id}] Stage 6: Decision Engine")
            
            # Extract data for decision engine
            primary_det_type = primary_detection["data"].object_type.value if primary_detection else "none"
            primary_det_conf = primary_detection["data"].confidence if primary_detection else 0.0
            
            # Basic context 
            is_known_habitat = True
            time_of_day = image.timestamp.hour if image.timestamp else -1
            
            decision_result = DecisionEngineService.decide(
                image_id=image_id,
                megadetector_confidence=primary_det_conf,
                megadetector_type=primary_det_type,
                speciesnet_species=speciesnet_prediction,
                speciesnet_confidence=speciesnet_conf,
                openclip_prediction=openclip_prediction,
                openclip_similarity=openclip_similarity,
                openclip_agrees=openclip_agrees,
                image_quality_score=quality_result.quality_score,
                is_known_habitat=is_known_habitat,
                time_of_day_hour=time_of_day
            )
            result.decision = decision_result
            
            # Save decision
            signals_schema = DecisionSignals(
                megadetector_confidence=primary_det_conf,
                speciesnet_confidence=speciesnet_conf,
                openclip_confidence=openclip_similarity,
                image_quality=quality_result.quality_score,
                model_agreement=decision_result.raw_scores.get("model_agreement"),
                is_tiger=decision_result.is_tiger
            )
            
            decision_record = Decision(
                image_id=image_id,
                species=decision_result.species,
                confidence=decision_result.confidence,
                decision=decision_result.decision.value,
                confidence_level=decision_result.confidence_level.value,
                reasoning=decision_result.reasoning,
                signals=signals_schema.model_dump(),
                is_tiger=decision_result.is_tiger
            )
            db.add(decision_record)
            
            # Create high-priority alert if priority species (tiger/leopard/elephant)
            if decision_result.is_priority_species:
                alert_type = "tiger_sighting" if decision_result.is_tiger else "priority_species_sighting"
                alert_severity = "high" if decision_result.is_tiger else "medium"
                alert = Alert(
                    id=str(uuid.uuid4()),
                    alert_type=alert_type,
                    severity=alert_severity,
                    title=f"Priority Wildlife Sighting: {decision_result.species}",
                    message=f"Detected {decision_result.species} with {decision_result.confidence:.1%} confidence on camera {image.camera_id}",
                    camera_id=image.camera_id,
                    image_id=image_id,
                    details={
                        "species": decision_result.species,
                        "confidence": decision_result.confidence,
                        "routing": decision_result.routing_destination.value,
                        "processing_id": decision_result.processing_id
                    }
                )
                db.add(alert)
            
            # ==========================================
            # STAGE 7: EXPLAINABILITY
            # ==========================================
            logger.info(f"[{image_id}] Stage 7: Explainability")
            explanation = ExplainabilityService.explain(
                image_id=image_id,
                decision=decision_result.decision.value,
                species=decision_result.species,
                confidence=decision_result.confidence,
                megadetector_confidence=primary_det_conf,
                megadetector_type=primary_det_type,
                speciesnet_confidence=speciesnet_conf,
                speciesnet_species=speciesnet_prediction,
                openclip_agrees=openclip_agrees,
                openclip_similarity=openclip_similarity,
                openclip_prediction=openclip_prediction,
                image_quality=quality_result.quality_score,
                model_agreement=decision_result.raw_scores.get("model_agreement", 0.0),
                is_tiger=decision_result.is_tiger,
                is_known_habitat=is_known_habitat,
                reasoning=decision_result.reasoning
            )
            result.explanation = explanation
            
            # Save explanation to audit trail
            audit = AuditTrail(
                image_id=image_id,
                event_type="decision_engine",
                event_status="pass",
                details=explanation.to_dict()
            )
            db.add(audit)
            
            # ==========================================
            # STAGE 9: THREAT ANALYSIS
            # ==========================================
            humans = len(md_output.human_detections)
            vehicles = len([d for d in md_output.detections if d.object_type.value == "vehicle"])
            
            if humans > 0 or vehicles > 0:
                logger.info(f"[{image_id}] Stage 9: Threat Analysis")
                threat_result = ThreatAnalyzerService.analyze(
                    camera_id=image.camera_id,
                    timestamp=image.timestamp,
                    humans_detected=humans,
                    vehicles_detected=vehicles,
                    location=image.location or "Unknown",
                    zone="Unknown"
                )
                result.threat_analysis = threat_result
                
                # High threats should force human review if auto-accepted
                if threat_result.threat_level.value in ["high", "medium"] and decision_record.decision == "auto_accept":
                    decision_record.decision = "human_review"
                    decision_record.reasoning.append(f"⚠ Overridden by Threat Analyzer: {threat_result.recommendation}")
            
            # ==========================================
            # STAGE 10: TIGER RE-ID
            # ==========================================
            if decision_result.is_tiger:
                logger.info(f"[{image_id}] Stage 10: Tiger Re-identification")
                # Simulate embedding
                embedding = ReIdentificationService.extract_embedding(image.image_path)
                
                reid_result = ReIdentificationService.search_similar(
                    image_id=image_id,
                    embedding=embedding,
                    camera_id=image.camera_id,
                    timestamp=image.timestamp
                )
                result.reid_result = reid_result
                
                # Save re-id matches
                for match in reid_result.matches:
                    reid_record = TigerReidentification(
                        image_id_1=image_id,
                        image_id_2=match.match_image_id,
                        similarity=match.similarity_score
                    )
                    db.add(reid_record)
            
            # Commit all changes
            db.commit()
            
            result.total_time_ms = (time.time() - start_time) * 1000
            logger.info(f"[{image_id}] Pipeline complete in {result.total_time_ms:.1f}ms")
            return result
            
        except Exception as e:
            db.rollback()
            logger.error(f"[{image_id}] Pipeline error: {e}")
            result.success = False
            result.error_message = str(e)
            result.total_time_ms = (time.time() - start_time) * 1000
            return result
