"""
VanRakshak AI - Phase 4 Automated Validation Script
SpeciesNet Integration & Species Classification Validation
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

from db.models import Base, Image, Detection, Classification, Decision, ImageQuality
from services.image_service import ImageService
from core.quality_gate import QualityGateService
from core.megadetector import MegaDetectorService
from core.species_classifier import SpeciesClassifierService, SpeciesClassificationResult
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


def create_tiger_like_crop_bytes():
    """Create a simulated tiger-like crop with orange hue and stripes"""
    img = PILImage.new('RGB', (320, 280), color=(185, 110, 45))
    draw = ImageDraw.Draw(img)
    for x in range(0, 320, 30):
        draw.line([(x, 0), (x + 15, 280)], fill=(15, 15, 15), width=5)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def run_validation():
    print_header("VanRakshak AI — Phase 4 Validation Suite")
    
    total_checks = 0
    passed_checks = 0
    
    # -------------------------------------------------------------
    # 1. VERIFY SPECIESNET MODEL
    # -------------------------------------------------------------
    print("\n[Stage 1] SpeciesNet Model File Verification (.pkl)")
    
    total_checks += 1
    model_path = os.path.join("models", "speciesnet.pkl")
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        has_clf = "classifier" in data and "species_list" in data
        if has_clf:
            passed_checks += 1
            print_result("SpeciesNet Model Check", True, f"Loaded RandomForest with {len(data['species_list'])} species classes ({size:,} bytes)")
        else:
            print_result("SpeciesNet Model Check", False, "Missing classifier dictionary structure")
    else:
        print_result("SpeciesNet Model Check", False, "models/speciesnet.pkl not found")

    # -------------------------------------------------------------
    # 2. TEST CROP PREPROCESSING & VALIDATION
    # -------------------------------------------------------------
    print("\n[Stage 2] Crop Preprocessing & Input Validation")
    
    os.makedirs("storage/crops", exist_ok=True)
    valid_crop_path = "storage/crops/val_phase4_crop.jpg"
    with open(valid_crop_path, "wb") as f:
        f.write(create_tiger_like_crop_bytes())
        
    total_checks += 1
    features = SpeciesClassifierService.preprocess_image(valid_crop_path)
    if features is not None and len(features) == 11:
        passed_checks += 1
        print_result("Valid Crop Preprocessing", True, f"Extracted 11 normalized feature metrics")
    else:
        print_result("Valid Crop Preprocessing", False, "Feature extraction failed")

    total_checks += 1
    corrupted_features = SpeciesClassifierService.preprocess_image("/nonexistent/file.jpg")
    if corrupted_features is None:
        passed_checks += 1
        print_result("Invalid Crop Graceful Rejection", True, "Missing/corrupted file safely returns None")
    else:
        print_result("Invalid Crop Graceful Rejection", False, "Failed to reject invalid input")

    # -------------------------------------------------------------
    # 3. SPECIES INFERENCE & TOP-K PROBABILITY
    # -------------------------------------------------------------
    print("\n[Stage 3] Species Inference & Top-K Scoring")
    
    total_checks += 1
    top_k = 5
    res = SpeciesClassifierService.classify(valid_crop_path, "val-det-001", top_k=top_k)
    if res and res.primary_species and len(res.alternatives) == top_k - 1:
        passed_checks += 1
        print_result(
            "Top-K Species Prediction",
            True,
            f"Primary: '{res.primary_species}' ({res.primary_confidence * 100:.1f}%), {len(res.alternatives)} alternatives"
        )
    else:
        print_result("Top-K Species Prediction", False, "Inference failed")

    total_checks += 1
    if hasattr(res, "confidence_level") and hasattr(res, "passes_threshold"):
        passed_checks += 1
        print_result(
            "Confidence Thresholding & Review Signals",
            True,
            f"Confidence Level: {res.confidence_level}, Passes Threshold: {res.passes_threshold}, Review Req: {res.requires_human_review}"
        )
    else:
        print_result("Confidence Thresholding & Review Signals", False, "Missing threshold attributes")

    # -------------------------------------------------------------
    # 4. REST API SPECIESNET ENDPOINTS
    # -------------------------------------------------------------
    print("\n[Stage 4] REST API SpeciesNet Endpoints")
    
    # Shared in-memory test database
    import db.database as db_module
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=test_engine)
    db_module.engine = test_engine
    db_module.SessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = db_module.SessionLocal()
    
    # Ingest image
    img_id, _ = ImageService.ingest_image(
        image_bytes=create_tiger_like_crop_bytes(),
        camera_id="CAM_VAL_P4",
        timestamp=datetime.utcnow(),
        db=session
    )
    
    # Create detection
    det_id = str(uuid.uuid4())
    det = Detection(
        id=det_id,
        image_id=img_id,
        object_type="animal",
        confidence=0.95,
        bbox_x_min=0.1,
        bbox_y_min=0.1,
        bbox_x_max=0.8,
        bbox_y_max=0.8,
        crop_path=valid_crop_path
    )
    session.add(det)
    session.commit()
    
    with TestClient(app) as client:
        # POST /api/detections/classify/{detection_id}
        total_checks += 1
        res = client.post(f"/api/detections/classify/{det_id}?top_k=4")
        if res.status_code == 200 and res.json().get("success"):
            passed_checks += 1
            print_result("POST /api/detections/classify/{det_id}", True, f"Classification created: {res.json()['data']['primary_species']}")
        else:
            print_result("POST /api/detections/classify/{det_id}", False, f"Status {res.status_code}")

        # GET /api/detections/classifications/{image_id}
        total_checks += 1
        res = client.get(f"/api/detections/classifications/{img_id}")
        if res.status_code == 200 and len(res.json().get("data", {}).get("classifications", [])) > 0:
            passed_checks += 1
            print_result("GET /api/detections/classifications/{image_id}", True, f"Retrieved stored classification list")
        else:
            print_result("GET /api/detections/classifications/{image_id}", False, f"Status {res.status_code}")

    # -------------------------------------------------------------
    # 5. FULL PIPELINE INTEGRATION
    # -------------------------------------------------------------
    print("\n[Stage 5] Complete Pipeline Orchestration")
    
    total_checks += 1
    pipe_img_id, _ = ImageService.ingest_image(
        image_bytes=create_tiger_like_crop_bytes() + os.urandom(16),
        camera_id="CAM_VAL_P4_PIPE",
        timestamp=datetime.utcnow(),
        db=session
    )
    
    pipe_res = ProcessingPipeline.process_image(pipe_img_id, session)
    if pipe_res.success and pipe_res.quality_passed:
        passed_checks += 1
        print_result("Full Pipeline (Gate → MegaDetector → SpeciesNet → Decision)", True, f"Pipeline processed in {pipe_res.total_time_ms:.1f}ms")
    else:
        print_result("Full Pipeline (Gate → MegaDetector → SpeciesNet → Decision)", False, "Pipeline failed")

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
        print("\n🏆 \033[92mALL PHASE 4 VALIDATION CHECKS PASSED!\033[0m\n")
        return 0
    else:
        print("\n⚠️ \033[91mSOME CHECKS FAILED. PLEASE REVIEW LOGS ABOVE.\033[0m\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_validation())
