"""
VanRakshak AI - Phase 5 Automated Validation Script
OpenCLIP Semantic Verification & Multi-Model Evidence Fusion Validation
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

from db.models import Base, Image, Detection, Classification, Verification, Decision, ImageQuality
from services.image_service import ImageService
from core.quality_gate import QualityGateService
from core.megadetector import MegaDetectorService
from core.species_classifier import SpeciesClassifierService
from core.semantic_verifier import SemanticVerifierService, SemanticVerificationResult
from core.decision_engine import DecisionEngineService, DecisionType
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
    print_header("VanRakshak AI — Phase 5 Validation Suite")
    
    total_checks = 0
    passed_checks = 0
    
    # -------------------------------------------------------------
    # 1. VERIFY OPENCLIP SERIALIZED MODEL
    # -------------------------------------------------------------
    print("\n[Stage 1] OpenCLIP Model File Verification (.pkl)")
    
    total_checks += 1
    model_path = os.path.join("models", "openclip.pkl")
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        from core.semantic_verifier import SafeUnpickler
        with open(model_path, "rb") as f:
            data = SafeUnpickler(f).load()
        has_concepts = hasattr(data, "concepts") and hasattr(data, "predict_similarities")
        if has_concepts:
            passed_checks += 1
            print_result("OpenCLIP Model Check", True, f"Loaded OpenCLIP model with {len(data.concepts)} concept prompts ({size:,} bytes)")
        else:
            print_result("OpenCLIP Model Check", False, "Missing OpenCLIP model structure")
    else:
        print_result("OpenCLIP Model Check", False, "models/openclip.pkl not found")

    # -------------------------------------------------------------
    # 2. IMAGE PREPROCESSING & TEXT PROMPT SIMILARITY
    # -------------------------------------------------------------
    print("\n[Stage 2] Vision-Language Semantic Scoring")
    
    os.makedirs("storage/raw", exist_ok=True)
    tiger_path = "storage/raw/val_phase5_tiger.jpg"
    with open(tiger_path, "wb") as f:
        f.write(create_tiger_image_bytes())
        
    total_checks += 1
    features = SemanticVerifierService.preprocess_image(tiger_path)
    if features is not None and len(features) == 11:
        passed_checks += 1
        print_result("OpenCLIP Visual Preprocessing", True, f"Computed 11-dimension visual feature vector")
    else:
        print_result("OpenCLIP Visual Preprocessing", False, "Visual feature preprocessing failed")

    total_checks += 1
    ver_res = SemanticVerifierService.verify(
        tiger_path,
        image_id="val-p5-01",
        speciesnet_prediction="Bengal Tiger",
        speciesnet_confidence=0.94
    )
    if ver_res and len(ver_res.scores) >= 10 and ver_res.primary_prediction != "":
        passed_checks += 1
        print_result(
            "Semantic Similarity Scoring",
            True,
            f"Top Match: '{ver_res.primary_prediction}' ({ver_res.primary_similarity * 100:.1f}%), {len(ver_res.scores)} concept scores"
        )
    else:
        print_result("Semantic Similarity Scoring", False, "Semantic scoring failed")

    # -------------------------------------------------------------
    # 3. SPECIESNET + OPENCLIP EVIDENCE FUSION
    # -------------------------------------------------------------
    print("\n[Stage 3] Agreement & Disagreement Fusion")
    
    total_checks += 1
    agree_decision = DecisionEngineService.decide(
        image_id="val-agree-img",
        megadetector_confidence=0.95,
        megadetector_type="animal",
        speciesnet_species="Bengal Tiger",
        speciesnet_confidence=0.96,
        openclip_prediction="Bengal Tiger",
        openclip_similarity=0.94,
        openclip_agrees=True,
        image_quality_score=0.95
    )
    if agree_decision.decision == DecisionType.AUTO_ACCEPT:
        passed_checks += 1
        print_result(
            "Model Agreement Boost (AUTO_ACCEPT)",
            True,
            f"Combined Confidence: {agree_decision.confidence:.1%}, Decision: {agree_decision.decision.value}"
        )
    else:
        print_result("Model Agreement Boost (AUTO_ACCEPT)", False, f"Unexpected decision: {agree_decision.decision}")

    total_checks += 1
    disagree_decision = DecisionEngineService.decide(
        image_id="val-disagree-img",
        megadetector_confidence=0.85,
        megadetector_type="animal",
        speciesnet_species="Bengal Tiger",
        speciesnet_confidence=0.70,
        openclip_prediction="Indian Leopard",
        openclip_similarity=0.68,
        openclip_agrees=False,
        image_quality_score=0.90
    )
    if disagree_decision.decision in [DecisionType.HUMAN_REVIEW, DecisionType.UNCERTAIN]:
        passed_checks += 1
        print_result(
            "Model Disagreement Routing (HUMAN_REVIEW)",
            True,
            f"Penalty applied, Decision: {disagree_decision.decision.value}"
        )
    else:
        print_result("Model Disagreement Routing (HUMAN_REVIEW)", False, f"Unexpected decision: {disagree_decision.decision}")

    # -------------------------------------------------------------
    # 4. REST API OPENCLIP ENDPOINTS
    # -------------------------------------------------------------
    print("\n[Stage 4] REST API OpenCLIP Endpoints")
    
    import db.database as db_module
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=test_engine)
    db_module.engine = test_engine
    db_module.SessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = db_module.SessionLocal()
    
    img_id, _ = ImageService.ingest_image(
        image_bytes=create_tiger_image_bytes(),
        camera_id="CAM_VAL_P5",
        timestamp=datetime.utcnow(),
        db=session
    )
    
    with TestClient(app) as client:
        total_checks += 1
        res = client.post(f"/api/detections/verify/{img_id}")
        if res.status_code == 200 and res.json().get("success"):
            passed_checks += 1
            print_result("POST /api/detections/verify/{image_id}", True, f"Verification produced: {res.json()['data']['primary_prediction']}")
        else:
            print_result("POST /api/detections/verify/{image_id}", False, f"Status {res.status_code}")

        total_checks += 1
        res = client.get(f"/api/detections/verifications/{img_id}")
        if res.status_code == 200 and res.json().get("data") is not None:
            passed_checks += 1
            print_result("GET /api/detections/verifications/{image_id}", True, "Retrieved verification record")
        else:
            print_result("GET /api/detections/verifications/{image_id}", False, f"Status {res.status_code}")

    # -------------------------------------------------------------
    # 5. COMPLETE PIPELINE WITH EVIDENCE FUSION
    # -------------------------------------------------------------
    print("\n[Stage 5] Complete Master Pipeline Orchestration")
    
    total_checks += 1
    pipe_img_id, _ = ImageService.ingest_image(
        image_bytes=create_tiger_image_bytes() + os.urandom(16),
        camera_id="CAM_VAL_P5_PIPE",
        timestamp=datetime.utcnow(),
        db=session
    )
    
    pipe_res = ProcessingPipeline.process_image(pipe_img_id, session)
    if pipe_res.success and pipe_res.semantic_verification is not None:
        passed_checks += 1
        print_result(
            "Full Pipeline (Gate → MegaDetector → SpeciesNet → OpenCLIP → Decision)",
            True,
            f"Pipeline executed in {pipe_res.total_time_ms:.1f}ms, Decision: {pipe_res.decision.decision.value if pipe_res.decision else 'N/A'}"
        )
    else:
        print_result("Full Pipeline", False, f"Pipeline failed: {pipe_res.error_message}")

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
        print("\n🏆 \033[92mALL PHASE 5 VALIDATION CHECKS PASSED!\033[0m\n")
        return 0
    else:
        print("\n⚠️ \033[91mSOME CHECKS FAILED. PLEASE REVIEW LOGS ABOVE.\033[0m\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_validation())
