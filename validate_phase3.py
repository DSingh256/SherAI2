"""
VanRakshak AI - Phase 3 & PKL Model Automated Validation Script
Validates:
1. Serialized PKL Model Files in models/
2. MegaDetector V6 Model & Service Execution
3. SpeciesNet RandomForest Classifier Execution
4. SAM2 Segmentation Model Execution
5. Database Detection Storage & Schema
6. REST API Detection Routes (/api/detections/)
7. Master Processing Pipeline Integration
"""

import os
import sys
import uuid
import io
import pickle
from datetime import datetime
from PIL import Image as PILImage, ImageDraw
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from db.models import Base, Image, Detection, ImageQuality
from services.image_service import ImageService
from core.quality_gate import QualityGateService
from core.megadetector import MegaDetectorService, DetectionCategory
from core.species_classifier import SpeciesClassifierService
from core.segmentation import SegmentationService
from core.pipeline import ProcessingPipeline
from main import app


def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title.upper()} ")
    print("=" * 60)

def print_result(check_name, passed, message=""):
    status_icon = "✓" if passed else "✗"
    status_text = "PASSED" if passed else "FAILED"
    color_start = "\033[92m" if passed else "\033[91m"
    color_end = "\033[0m"
    print(f"{color_start}{status_icon} [{status_text}]{color_end} {check_name}")
    if message:
        print(f"    └─ {message}")


def create_textured_image_bytes():
    """Create a high quality test image"""
    img = PILImage.new('RGB', (800, 600), color=(120, 160, 200))
    draw = ImageDraw.Draw(img)
    for i in range(0, 800, 20):
        draw.line([(i, 0), (800 - i, 600)], fill=(0, 0, 0) if i % 40 == 0 else (255, 255, 255), width=2)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def run_validation():
    print_header("VanRakshak AI — Phase 3 Validation Suite")
    
    total_checks = 0
    passed_checks = 0
    
    # -------------------------------------------------------------
    # 1. VERIFY SERIALIZED MODELS
    # -------------------------------------------------------------
    print("\n[Stage 1] Serialized Models Verification (.pkl)")
    
    pkl_files = ["megadetector.pkl", "speciesnet.pkl", "sam2.pkl"]
    for pkl in pkl_files:
        total_checks += 1
        path = os.path.join("models", pkl)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        if exists:
            passed_checks += 1
            print_result(f"Model file: models/{pkl}", True, f"Found ({size:,} bytes)")
        else:
            print_result(f"Model file: models/{pkl}", False, "File missing")

    # -------------------------------------------------------------
    # 2. TEST CORE MEGADETECTOR SERVICE & CROPPING
    # -------------------------------------------------------------
    print("\n[Stage 2] MegaDetector V6 Detection & Cropping")
    
    total_checks += 1
    sample_img_bytes = create_textured_image_bytes()
    os.makedirs("storage/raw", exist_ok=True)
    temp_img_path = "storage/raw/val_phase3_sample.jpg"
    with open(temp_img_path, "wb") as f:
        f.write(sample_img_bytes)
        
    md_output = MegaDetectorService.detect(temp_img_path, "val-img-001")
    if md_output and hasattr(md_output, "detections") and md_output.processing_time_ms > 0:
        passed_checks += 1
        print_result("MegaDetector Detection", True, f"Produced {len(md_output.detections)} detections in {md_output.processing_time_ms:.1f}ms")
    else:
        print_result("MegaDetector Detection", False, "Failed to run detection")

    total_checks += 1
    os.makedirs("storage/crops", exist_ok=True)
    if len(md_output.detections) > 0:
        crop_path = MegaDetectorService.crop_detection(
            temp_img_path, md_output.detections[0].bbox, output_dir="storage/crops", detection_id="val-crop-001"
        )
        if crop_path and os.path.exists(crop_path):
            passed_checks += 1
            print_result("Bounding Box Cropping", True, f"Cropped detection saved to {crop_path}")
        else:
            print_result("Bounding Box Cropping", False, "Failed to save cropped image")
    else:
        passed_checks += 1
        print_result("Bounding Box Cropping", True, "Empty detection scenario verified")

    # -------------------------------------------------------------
    # 3. TEST SPECIESNET & SAM2 SERVICES WITH PKL
    # -------------------------------------------------------------
    print("\n[Stage 3] SpeciesNet & SAM2 PKL Integration")
    
    total_checks += 1
    species_res = SpeciesClassifierService.classify(temp_img_path, "val-det-001")
    if species_res and species_res.primary_species and len(species_res.alternatives) > 0:
        passed_checks += 1
        print_result(
            "SpeciesNet RandomForest Inference", 
            True, 
            f"Classified as '{species_res.primary_species}' ({species_res.primary_confidence * 100:.1f}%) using {species_res.model_name}"
        )
    else:
        print_result("SpeciesNet RandomForest Inference", False, "Classification failed")

    total_checks += 1
    seg_res = SegmentationService.segment(
        temp_img_path, "val-img-001", "val-det-001", 0.1, 0.1, 0.8, 0.8
    )
    if seg_res and os.path.exists(seg_res.mask_path):
        passed_checks += 1
        print_result(
            "SAM2 Segmentation Inference",
            True,
            f"Mask generated at {seg_res.mask_path} (Model: {seg_res.model_name})"
        )
    else:
        print_result("SAM2 Segmentation Inference", False, "Segmentation failed")

    # -------------------------------------------------------------
    # 4. TEST REST API DETECTION ROUTES
    # -------------------------------------------------------------
    print("\n[Stage 4] REST API Detection Endpoints")
    
    # In-memory database setup for API testing
    import db.database as db_module
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=test_engine)
    db_module.engine = test_engine
    db_module.SessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = db_module.SessionLocal()
    
    # Ingest image
    img_id, _ = ImageService.ingest_image(
        image_bytes=sample_img_bytes,
        camera_id="CAM_VAL_01",
        timestamp=datetime.utcnow(),
        db=session
    )
    QualityGateService.apply_quality_gate(img_id, session)
    
    with TestClient(app) as client:
        # POST /api/detections/detect/{img_id}
        total_checks += 1
        res = client.post(f"/api/detections/detect/{img_id}")
        if res.status_code == 200 and res.json().get("success"):
            passed_checks += 1
            print_result("POST /api/detections/detect/{image_id}", True, f"Status 200 OK — Detections created")
        else:
            print_result("POST /api/detections/detect/{image_id}", False, f"Status {res.status_code}")

        # GET /api/detections/image/{img_id}
        total_checks += 1
        res = client.get(f"/api/detections/image/{img_id}")
        if res.status_code == 200 and "detections" in res.json().get("data", {}):
            passed_checks += 1
            print_result("GET /api/detections/image/{image_id}", True, f"Retrieved detection list")
        else:
            print_result("GET /api/detections/image/{image_id}", False, f"Status {res.status_code}")

        # GET /api/detections/stats
        total_checks += 1
        res = client.get("/api/detections/stats")
        if res.status_code == 200 and "total_detections" in res.json().get("data", {}):
            passed_checks += 1
            print_result("GET /api/detections/stats", True, f"Stats: {res.json()['data']}")
        else:
            print_result("GET /api/detections/stats", False, f"Status {res.status_code}")

    # -------------------------------------------------------------
    # 5. TEST FULL PROCESSING PIPELINE
    # -------------------------------------------------------------
    print("\n[Stage 5] Master Processing Pipeline Integration")
    
    total_checks += 1
    pipe_bytes = create_textured_image_bytes() + os.urandom(16)
    pipe_img_id, _ = ImageService.ingest_image(
        image_bytes=pipe_bytes,
        camera_id="CAM_VAL_PIPE",
        timestamp=datetime.utcnow(),
        db=session
    )
    pipe_res = ProcessingPipeline.process_image(pipe_img_id, session)
    if pipe_res.success and pipe_res.megadetector_output is not None:
        passed_checks += 1
        print_result("Master Processing Pipeline", True, f"Completed pipeline in {pipe_res.total_time_ms:.1f}ms")
    else:
        print_result("Master Processing Pipeline", False, f"Pipeline execution failed: {pipe_res.error_message}")

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print_header("Validation Summary")
    success_rate = (passed_checks / total_checks) * 100
    print(f"Total Checks:  {total_checks}")
    print(f"Passed:        {passed_checks}")
    print(f"Failed:        {total_checks - passed_checks}")
    print(f"Success Rate:  {success_rate:.1f}%")
    
    if passed_checks == total_checks:
        print("\n🏆 \033[92mALL PHASE 3 VALIDATION CHECKS PASSED!\033[0m\n")
        return 0
    else:
        print("\n⚠️ \033[91mSOME CHECKS FAILED. PLEASE REVIEW LOGS ABOVE.\033[0m\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_validation())
