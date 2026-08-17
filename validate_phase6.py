"""
VanRakshak AI - Phase 6 Automated Validation Script
Decision Engine & Intelligent Routing Validation
"""

import os
import sys
import uuid
import io
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

from db.models import Base, Image, Detection, Classification, Verification, Decision, Alert, ImageQuality
from services.image_service import ImageService
from core.quality_gate import QualityGateService
from core.megadetector import MegaDetectorService
from core.species_classifier import SpeciesClassifierService
from core.semantic_verifier import SemanticVerifierService
from core.decision_engine import DecisionEngineService, DecisionType, RoutingDestination, ConfidenceLevel
from core.explainability import ExplainabilityService
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
    print_header("VanRakshak AI — Phase 6 Validation Suite")
    
    total_checks = 0
    passed_checks = 0
    
    # -------------------------------------------------------------
    # 1. TEST ROUTING DESTINATIONS
    # -------------------------------------------------------------
    print("\n[Stage 1] Intelligent Routing Decisions & Thresholds")
    
    # Check 1: High confidence + Agreement -> ACCEPTED
    total_checks += 1
    res_accept = DecisionEngineService.decide(
        image_id="val-img-acc",
        megadetector_confidence=0.96,
        megadetector_type="animal",
        speciesnet_species="Sambar Deer",
        speciesnet_confidence=0.95,
        openclip_prediction="Sambar Deer",
        openclip_similarity=0.94,
        openclip_agrees=True,
        image_quality_score=0.92
    )
    if res_accept.decision == DecisionType.AUTO_ACCEPT and res_accept.routing_destination == RoutingDestination.ACCEPTED:
        passed_checks += 1
        print_result("High Confidence Agreement -> ACCEPTED", True, f"Confidence: {res_accept.confidence:.1%}, Route: {res_accept.routing_destination.value}")
    else:
        print_result("High Confidence Agreement -> ACCEPTED", False, f"Decision: {res_accept.decision}")

    # Check 2: Low confidence -> REVIEW
    total_checks += 1
    res_review = DecisionEngineService.decide(
        image_id="val-img-rev",
        megadetector_confidence=0.65,
        megadetector_type="animal",
        speciesnet_species="Jungle Cat",
        speciesnet_confidence=0.55,
        openclip_prediction="Jungle Cat",
        openclip_similarity=0.58,
        openclip_agrees=True,
        image_quality_score=0.75
    )
    if res_review.routing_destination == RoutingDestination.REVIEW:
        passed_checks += 1
        print_result("Low Confidence -> REVIEW Queue", True, f"Confidence: {res_review.confidence:.1%}, Route: {res_review.routing_destination.value}")
    else:
        print_result("Low Confidence -> REVIEW Queue", False, f"Route: {res_review.routing_destination}")

    # Check 3: Disagreement Escalation -> REVIEW
    total_checks += 1
    res_conflict = DecisionEngineService.decide(
        image_id="val-img-esc",
        megadetector_confidence=0.88,
        megadetector_type="animal",
        speciesnet_species="Bengal Tiger",
        speciesnet_confidence=0.75,
        openclip_prediction="Indian Leopard",
        openclip_similarity=0.78,
        openclip_agrees=False,
        image_quality_score=0.85
    )
    if res_conflict.is_escalated and res_conflict.routing_destination == RoutingDestination.REVIEW:
        passed_checks += 1
        print_result("Model Disagreement Escalation -> REVIEW Queue", True, "Conflict detected & escalated")
    else:
        print_result("Model Disagreement Escalation -> REVIEW Queue", False, "Escalation failed")

    # Check 4: Quality Gate Failure -> QUARANTINE
    total_checks += 1
    res_quarantine = DecisionEngineService.decide(
        image_id="val-img-quar",
        image_quality_score=0.20,
        quality_passed=False
    )
    if res_quarantine.routing_destination == RoutingDestination.QUARANTINE:
        passed_checks += 1
        print_result("Quality Failure -> QUARANTINE Isolation", True, f"Route: {res_quarantine.routing_destination.value}")
    else:
        print_result("Quality Failure -> QUARANTINE Isolation", False, f"Route: {res_quarantine.routing_destination}")

    # Check 5: Clean Empty Frame -> NO_ANIMAL
    total_checks += 1
    res_empty = DecisionEngineService.decide(
        image_id="val-img-empty",
        no_detections=True,
        image_quality_score=0.90
    )
    if res_empty.routing_destination == RoutingDestination.NO_ANIMAL:
        passed_checks += 1
        print_result("Clean Empty Frame -> NO_ANIMAL Archive", True, f"Route: {res_empty.routing_destination.value}")
    else:
        print_result("Clean Empty Frame -> NO_ANIMAL Archive", False, f"Route: {res_empty.routing_destination}")

    # Check 6: Priority Species -> ALERT
    total_checks += 1
    res_alert = DecisionEngineService.decide(
        image_id="val-img-alert",
        megadetector_confidence=0.97,
        megadetector_type="animal",
        speciesnet_species="Bengal Tiger",
        speciesnet_confidence=0.96,
        openclip_prediction="Bengal Tiger",
        openclip_similarity=0.94,
        openclip_agrees=True,
        image_quality_score=0.95
    )
    if res_alert.is_priority_species and res_alert.routing_destination == RoutingDestination.ALERT:
        passed_checks += 1
        print_result("Priority Wildlife (Tiger) -> ALERT Pipeline", True, f"Priority Species: {res_alert.species}, Route: {res_alert.routing_destination.value}")
    else:
        print_result("Priority Wildlife (Tiger) -> ALERT Pipeline", False, f"Route: {res_alert.routing_destination}")

    # -------------------------------------------------------------
    # 2. STRUCTURED EXPLAINABILITY
    # -------------------------------------------------------------
    print("\n[Stage 2] Structured Explainability Output")
    
    total_checks += 1
    exp = ExplainabilityService.explain(
        image_id="val-exp-01",
        decision=res_alert.decision.value,
        species=res_alert.species,
        confidence=res_alert.confidence,
        megadetector_confidence=0.97,
        megadetector_type="animal",
        speciesnet_confidence=0.96,
        speciesnet_species="Bengal Tiger",
        openclip_agrees=True,
        openclip_similarity=0.94,
        openclip_prediction="Bengal Tiger",
        image_quality=0.95,
        model_agreement=0.95,
        is_tiger=True,
        reasoning=res_alert.reasoning
    )
    if exp and len(exp.signal_assessments) >= 4 and exp.recommendation:
        passed_checks += 1
        print_result("Structured Explainability Report", True, f"Generated {len(exp.signal_assessments)} signal checks with recommendation")
    else:
        print_result("Structured Explainability Report", False, "Explainability report incomplete")

    # -------------------------------------------------------------
    # 3. REST API ENDPOINTS & ALERTS
    # -------------------------------------------------------------
    print("\n[Stage 3] REST API Review & Alert Endpoints")
    
    import db.database as db_module
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=test_engine)
    db_module.engine = test_engine
    db_module.SessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = db_module.SessionLocal()
    
    # Ingest image
    img_id, _ = ImageService.ingest_image(
        image_bytes=create_tiger_image_bytes(),
        camera_id="CAM_VAL_P6",
        timestamp=datetime.utcnow(),
        db=session
    )
    
    # Trigger full pipeline
    pipe_res = ProcessingPipeline.process_image(img_id, session)
    
    with TestClient(app) as client:
        # GET /api/alerts/
        total_checks += 1
        res = client.get("/api/alerts/")
        if res.status_code == 200 and "alerts" in res.json().get("data", {}):
            passed_checks += 1
            print_result("GET /api/alerts/", True, f"Alerts retrieved: {len(res.json()['data']['alerts'])}")
        else:
            print_result("GET /api/alerts/", False, f"Status: {res.status_code}")

        # GET /api/review/queue
        total_checks += 1
        res = client.get("/api/review/queue")
        if res.status_code == 200 and "items" in res.json().get("data", {}):
            passed_checks += 1
            print_result("GET /api/review/queue", True, f"Review items retrieved: {len(res.json()['data']['items'])}")
        else:
            print_result("GET /api/review/queue", False, f"Status: {res.status_code}")

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
        print("\n🏆 \033[92mALL PHASE 6 VALIDATION CHECKS PASSED!\033[0m\n")
        return 0
    else:
        print("\n⚠️ \033[91mSOME CHECKS FAILED. PLEASE REVIEW LOGS ABOVE.\033[0m\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_validation())
