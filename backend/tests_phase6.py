"""
VanRakshak AI - Phase 6 Test Suite
Decision Engine & Intelligent Routing Tests
"""

import pytest
import os
import uuid
import io
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from PIL import Image as PILImage, ImageDraw
import numpy as np
from fastapi.testclient import TestClient

from db.models import Base, Image, Detection, Classification, Verification, Decision, Alert, ImageQuality
from services.image_service import ImageService
from core.quality_gate import QualityGateService
from core.megadetector import MegaDetectorService
from core.species_classifier import SpeciesClassifierService
from core.semantic_verifier import SemanticVerifierService
from core.decision_engine import DecisionEngineService, DecisionType, RoutingDestination, ConfidenceLevel
from core.explainability import ExplainabilityService, SignalStatus
from core.pipeline import ProcessingPipeline
from main import app
from config import settings


# ============ FIXTURES ============

@pytest.fixture(scope="function")
def test_db():
    """Create a clean in-memory test database with StaticPool"""
    import db.database as db_module
    
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=test_engine)
    
    db_module.engine = test_engine
    db_module.SessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    
    session = db_module.SessionLocal()
    yield session
    
    session.close()
    test_engine.dispose()


@pytest.fixture
def client(test_db):
    """FastAPI TestClient"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def tiger_image_bytes():
    """Create a valid tiger image"""
    img = PILImage.new('RGB', (600, 450), color=(185, 110, 45))
    draw = ImageDraw.Draw(img)
    for x in range(0, 600, 30):
        draw.line([(x, 0), (x + 20, 450)], fill=(15, 15, 15), width=6)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


@pytest.fixture
def empty_image_bytes():
    """Create an empty landscape image"""
    img = PILImage.new('RGB', (600, 450), color=(40, 100, 40))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


# ============ DECISION ENGINE LOGIC TESTS ============

class TestDecisionEngineRouting:
    """Test decision engine evidence fusion and intelligent routing destinations"""

    def test_high_confidence_agreement_routing(self):
        """Test high confidence + model agreement routes to ACCEPTED or ALERT"""
        res = DecisionEngineService.decide(
            image_id="test-img-01",
            megadetector_confidence=0.96,
            megadetector_type="animal",
            speciesnet_species="Sambar Deer",
            speciesnet_confidence=0.95,
            openclip_prediction="Sambar Deer",
            openclip_similarity=0.93,
            openclip_agrees=True,
            image_quality_score=0.92
        )
        
        assert res.decision == DecisionType.AUTO_ACCEPT
        assert res.routing_destination == RoutingDestination.ACCEPTED
        assert res.confidence_level == ConfidenceLevel.HIGH
        assert res.confidence >= settings.HIGH_CONFIDENCE_THRESHOLD
        assert len(res.reasoning) > 0
        assert res.processing_id != ""

    def test_priority_wildlife_alert_routing(self):
        """Test Bengal Tiger detection triggers ALERT routing destination"""
        res = DecisionEngineService.decide(
            image_id="tiger-alert-01",
            megadetector_confidence=0.98,
            megadetector_type="animal",
            speciesnet_species="Bengal Tiger",
            speciesnet_confidence=0.97,
            openclip_prediction="Bengal Tiger",
            openclip_similarity=0.95,
            openclip_agrees=True,
            image_quality_score=0.95
        )
        
        assert res.is_tiger is True
        assert res.is_priority_species is True
        assert res.routing_destination == RoutingDestination.ALERT
        assert res.decision == DecisionType.AUTO_ACCEPT
        assert any("TIGER DETECTION" in r for r in res.reasoning)

    def test_low_confidence_review_routing(self):
        """Test low confidence prediction routes to REVIEW queue"""
        res = DecisionEngineService.decide(
            image_id="low-conf-01",
            megadetector_confidence=0.62,
            megadetector_type="animal",
            speciesnet_species="Jungle Cat",
            speciesnet_confidence=0.55,
            openclip_prediction="Jungle Cat",
            openclip_similarity=0.58,
            openclip_agrees=True,
            image_quality_score=0.75
        )
        
        assert res.decision in [DecisionType.HUMAN_REVIEW, DecisionType.UNCERTAIN]
        assert res.routing_destination == RoutingDestination.REVIEW
        assert any("HUMAN REVIEW" in r or "UNCERTAIN" in r for r in res.reasoning)

    def test_model_disagreement_escalation(self):
        """Test conflict escalation when SpeciesNet and OpenCLIP disagree"""
        res = DecisionEngineService.decide(
            image_id="conflict-01",
            megadetector_confidence=0.88,
            megadetector_type="animal",
            speciesnet_species="Bengal Tiger",
            speciesnet_confidence=0.75,
            openclip_prediction="Indian Leopard",
            openclip_similarity=0.78,
            openclip_agrees=False,
            image_quality_score=0.85
        )
        
        assert res.is_escalated is True
        assert res.decision == DecisionType.HUMAN_REVIEW
        assert res.routing_destination == RoutingDestination.REVIEW
        assert any("MODEL CONFLICT ESCALATION" in r for r in res.reasoning)

    def test_quality_gate_failure_quarantine_routing(self):
        """Test image that fails quality gate routes to QUARANTINE"""
        res = DecisionEngineService.decide(
            image_id="corrupted-01",
            image_quality_score=0.20,
            quality_passed=False
        )
        
        assert res.decision == DecisionType.REJECT
        assert res.routing_destination == RoutingDestination.QUARANTINE
        assert any("quarantine" in r.lower() for r in res.reasoning)

    def test_no_animal_empty_frame_routing(self):
        """Test clean empty frame routes to NO_ANIMAL"""
        res = DecisionEngineService.decide(
            image_id="empty-01",
            no_detections=True,
            image_quality_score=0.90
        )
        
        assert res.decision == DecisionType.NO_ANIMAL
        assert res.routing_destination == RoutingDestination.NO_ANIMAL


# ============ EXPLAINABILITY TESTS ============

class TestExplainability:
    """Test structured explainability generation"""

    def test_structured_explainability_report(self):
        """Test explainability report contains signal assessments and recommendations"""
        exp = ExplainabilityService.explain(
            image_id="exp-01",
            decision="auto_accept",
            species="Bengal Tiger",
            confidence=0.94,
            megadetector_confidence=0.96,
            megadetector_type="animal",
            speciesnet_confidence=0.95,
            speciesnet_species="Bengal Tiger",
            openclip_agrees=True,
            openclip_similarity=0.93,
            openclip_prediction="Bengal Tiger",
            image_quality=0.92,
            model_agreement=0.94,
            is_tiger=True
        )
        
        assert exp.image_id == "exp-01"
        assert exp.decision == "auto_accept"
        assert exp.is_tiger is True
        assert len(exp.signal_assessments) >= 4
        assert exp.recommendation != ""
        assert exp.formatted_report != ""
        
        d = exp.to_dict()
        assert "signal_assessments" in d
        assert "recommendation" in d


# ============ FULL PIPELINE & ALERT TESTS ============

class TestPhase6PipelineAndAlerts:
    """Test full pipeline execution, routing metadata, and alert generation"""

    def test_pipeline_generates_alert_for_priority_wildlife(self, test_db, tiger_image_bytes):
        """Verify priority wildlife produces Decision and Alert in DB"""
        now = datetime.utcnow()
        img_id, _ = ImageService.ingest_image(
            image_bytes=tiger_image_bytes,
            camera_id="CAM_PRIORITY",
            timestamp=now,
            db=test_db
        )
        
        result = ProcessingPipeline.process_image(img_id, test_db)
        assert result.success is True
        assert result.decision is not None
        
        # Check Decision in DB
        db_decision = test_db.query(Decision).filter(Decision.image_id == img_id).first()
        assert db_decision is not None
        assert db_decision.decision in ["auto_accept", "human_review"]
        
        # Check Alert in DB
        if result.decision.is_priority_species:
            alerts = test_db.query(Alert).filter(Alert.image_id == img_id).all()
            assert len(alerts) > 0
            assert alerts[0].severity in ["high", "medium"]

    def test_review_routes_integration(self, client, test_db, tiger_image_bytes):
        """Test /api/review/queue endpoint returns items requiring review"""
        now = datetime.utcnow()
        img_id, _ = ImageService.ingest_image(
            image_bytes=tiger_image_bytes,
            camera_id="CAM_REV_TEST",
            timestamp=now,
            db=test_db
        )
        
        # Add a decision requiring human review
        dec = Decision(
            id=str(uuid.uuid4()),
            image_id=img_id,
            species="Indian Leopard",
            confidence=0.68,
            decision="human_review",
            confidence_level="medium",
            reasoning=["Species confusion requires expert human review"],
            signals={"speciesnet": 0.68, "openclip": 0.65},
            is_tiger=False
        )
        test_db.add(dec)
        test_db.commit()
        
        response = client.get("/api/review/queue")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["items"]) >= 1

    def test_alerts_routes_integration(self, client, test_db, tiger_image_bytes):
        """Test /api/alerts/ endpoint exposes priority conservation alerts"""
        now = datetime.utcnow()
        img_id, _ = ImageService.ingest_image(
            image_bytes=tiger_image_bytes,
            camera_id="CAM_ALT_TEST",
            timestamp=now,
            db=test_db
        )
        
        alt = Alert(
            id=str(uuid.uuid4()),
            alert_type="tiger_sighting",
            severity="high",
            title="Bengal Tiger Sighting",
            message="Bengal Tiger spotted in Corridor Alpha",
            camera_id="CAM_ALT_TEST",
            image_id=img_id,
            details={"species": "Bengal Tiger", "confidence": 0.96}
        )
        test_db.add(alt)
        test_db.commit()
        
        response = client.get("/api/alerts/")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["alerts"]) >= 1
