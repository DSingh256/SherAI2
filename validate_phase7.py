"""
VanRakshak AI - Phase 7 Automated Validation Script
SAM/SAM2 Wildlife Segmentation Validation
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

from db.models import Base, Image, Detection, Classification, Verification, Segmentation, Decision, ImageQuality
from services.image_service import ImageService
from core.quality_gate import QualityGateService
from core.megadetector import MegaDetectorService
from core.species_classifier import SpeciesClassifierService
from core.semantic_verifier import SemanticVerifierService
from core.segmentation import SegmentationService, SegmentationResult, SafeUnpickler
from core.decision_engine import DecisionEngineService
from core.pipeline import ProcessingPipeline
from main import app
from config import settings


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


def create_tiger_image_bytes():
    """Create a simulated tiger-like image"""
    img = PILImage.new('RGB', (640, 480), color=(185, 110, 45))
    draw = ImageDraw.Draw(img)
    for x in range(0, 640, 30):
        draw.line([(x, 0), (x + 20, 480)], fill=(15, 15, 15), width=6)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def run_validation():
    print_header("VanRakshak AI — Phase 7 Validation Suite")
    
    total_checks = 0
    passed_checks = 0
    
    # -------------------------------------------------------------
    # 1. SAM2 SERIALIZED MODEL VERIFICATION
    # -------------------------------------------------------------
    print("\n[Stage 1] SAM2 Serialized Model Check (.pkl)")
    
    total_checks += 1
    model_path = os.path.join("models", "sam2.pkl")
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        with open(model_path, "rb") as f:
            model = SafeUnpickler(f).load()
        has_segment = hasattr(model, "segment")
        if has_segment:
            passed_checks += 1
            print_result("SAM2 Model Check", True, f"Loaded SAM2 model ({size:,} bytes)")
        else:
            print_result("SAM2 Model Check", False, "Missing segment method")
    else:
        print_result("SAM2 Model Check", False, "models/sam2.pkl not found")

    # -------------------------------------------------------------
    # 2. BOX PROMPT SEGMENTATION & CROP GENERATION
    # -------------------------------------------------------------
    print("\n[Stage 2] Box Prompt Segmentation & Flank Isolation")
    
    os.makedirs("storage/raw", exist_ok=True)
    tiger_path = "storage/raw/val_p7_tiger.jpg"
    with open(tiger_path, "wb") as f:
        f.write(create_tiger_image_bytes())

    total_checks += 1
    det_id = str(uuid.uuid4())
    seg_res = SegmentationService.segment(
        image_path=tiger_path,
        image_id="val-p7-01",
        detection_id=det_id,
        bbox_x_min=0.15,
        bbox_y_min=0.20,
        bbox_x_max=0.85,
        bbox_y_max=0.80,
        species="Bengal Tiger"
    )
    if seg_res and os.path.exists(seg_res.mask_path) and os.path.exists(seg_res.segmented_crop_path):
        passed_checks += 1
        print_result(
            "Box-Prompt Mask & Transparent Crop",
            True,
            f"Mask Quality: {seg_res.mask_quality:.1%}, Saved: {os.path.basename(seg_res.segmented_crop_path)}"
        )
    else:
        print_result("Box-Prompt Mask & Transparent Crop", False, "Segmentation failed")

    total_checks += 1
    if seg_res and seg_res.flank_crop_path and os.path.exists(seg_res.flank_crop_path):
        passed_checks += 1
        print_result(
            "Flank/Body Region Extraction for Re-ID",
            True,
            f"Extracted Tiger Flank: {os.path.basename(seg_res.flank_crop_path)}"
        )
    else:
        print_result("Flank/Body Region Extraction for Re-ID", False, "Flank extraction failed")

    # -------------------------------------------------------------
    # 3. MULTI-ANIMAL SEGMENTATION
    # -------------------------------------------------------------
    print("\n[Stage 3] Multi-Animal Camera-Trap Isolation")
    
    total_checks += 1
    detections = [
        {"id": str(uuid.uuid4()), "bbox": {"x_min": 0.05, "y_min": 0.1, "x_max": 0.45, "y_max": 0.8}, "species": "Spotted Deer"},
        {"id": str(uuid.uuid4()), "bbox": {"x_min": 0.55, "y_min": 0.1, "x_max": 0.95, "y_max": 0.8}, "species": "Sambar Deer"}
    ]
    multi_res = SegmentationService.segment_all_detections(
        image_path=tiger_path,
        image_id="val-p7-multi",
        detections=detections
    )
    if len(multi_res) == 2 and multi_res[0].mask_path != multi_res[1].mask_path:
        passed_checks += 1
        print_result(
            "Multi-Animal Segmentation",
            True,
            f"Generated {len(multi_res)} distinct masks and crops"
        )
    else:
        print_result("Multi-Animal Segmentation", False, "Multi-animal segmentation failed")

    # -------------------------------------------------------------
    # 4. REST API SEGMENTATION ENDPOINTS
    # -------------------------------------------------------------
    print("\n[Stage 4] REST API Segmentation & Re-ID Crop Endpoints")
    
    import db.database as db_module
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=test_engine)
    db_module.engine = test_engine
    db_module.SessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = db_module.SessionLocal()
    
    img_id, _ = ImageService.ingest_image(
        image_bytes=create_tiger_image_bytes(),
        camera_id="CAM_VAL_P7",
        timestamp=datetime.utcnow(),
        db=session
    )
    
    api_det_id = str(uuid.uuid4())
    det_db = Detection(
        id=api_det_id,
        image_id=img_id,
        object_type="animal",
        confidence=0.96,
        bbox_x_min=0.1,
        bbox_y_min=0.1,
        bbox_x_max=0.9,
        bbox_y_max=0.9
    )
    session.add(det_db)
    session.commit()
    
    with TestClient(app) as client:
        # POST /api/detections/segment/{detection_id}
        total_checks += 1
        res = client.post(f"/api/detections/segment/{api_det_id}")
        if res.status_code == 200 and res.json().get("success"):
            passed_checks += 1
            print_result("POST /api/detections/segment/{detection_id}", True, f"Mask quality: {res.json()['data']['mask_quality']:.1%}")
        else:
            print_result("POST /api/detections/segment/{detection_id}", False, f"Status: {res.status_code}")

        # GET /api/detections/segmentations/{image_id}
        total_checks += 1
        res = client.get(f"/api/detections/segmentations/{img_id}")
        if res.status_code == 200 and len(res.json().get("data", {}).get("segmentations", [])) > 0:
            passed_checks += 1
            print_result("GET /api/detections/segmentations/{image_id}", True, "Retrieved segmentations")
        else:
            print_result("GET /api/detections/segmentations/{image_id}", False, f"Status: {res.status_code}")

        # GET /api/reidentification/crops/{image_id}
        total_checks += 1
        res = client.get(f"/api/reidentification/crops/{img_id}")
        if res.status_code == 200 and len(res.json().get("data", {}).get("crops", [])) > 0:
            passed_checks += 1
            print_result("GET /api/reidentification/crops/{image_id}", True, "Retrieved Re-ID prepared crops")
        else:
            print_result("GET /api/reidentification/crops/{image_id}", False, f"Status: {res.status_code}")

    # -------------------------------------------------------------
    # 5. FULL PIPELINE WITH SAM2 INTEGRATION
    # -------------------------------------------------------------
    print("\n[Stage 5] Complete Master Pipeline Integration")
    
    total_checks += 1
    pipe_img_id, _ = ImageService.ingest_image(
        image_bytes=create_tiger_image_bytes() + b"_P7_TIGER_STABLE_TEST_",
        camera_id="CAM_TEST_0",
        timestamp=datetime.utcnow(),
        db=session
    )
    
    pipe_res = ProcessingPipeline.process_image(pipe_img_id, session)
    if pipe_res.success and len(pipe_res.segmentations) > 0:
        passed_checks += 1
        print_result(
            "Full Pipeline (Gate → MegaDetector → SpeciesNet → OpenCLIP → SAM2 → Decision → Re-ID)",
            True,
            f"Segmented {len(pipe_res.segmentations)} animals in {pipe_res.total_time_ms:.1f}ms"
        )
    else:
        print_result("Full Pipeline Integration", False, f"Pipeline failed: {pipe_res.error_message}")

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
        print("\n🏆 \033[92mALL PHASE 7 VALIDATION CHECKS PASSED!\033[0m\n")
        return 0
    else:
        print("\n⚠️ \033[91mSOME CHECKS FAILED. PLEASE REVIEW LOGS ABOVE.\033[0m\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_validation())
