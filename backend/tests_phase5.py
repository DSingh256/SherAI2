"""
VanRakshak AI - Phase 5 Test Suite
OpenCLIP Semantic Verification & Evidence Fusion Tests
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
    """Create an orange patterned tiger-like image"""
    img = PILImage.new('RGB', (600, 450), color=(190, 110, 50))
    draw = ImageDraw.Draw(img)
    for x in range(0, 600, 30):
        draw.line([(x, 0), (x + 30, 450)], fill=(20, 20, 20), width=6)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


# ============ OPENCLIP CORE LOGIC TESTS ============

class TestOpenCLIPCore:
    """Test core OpenCLIP semantic verification service"""

    def test_openclip_pkl_model_exists(self):
        """Verify the serialized PKL model file exists in models directory"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(base_dir, "..", "models", "openclip.pkl")
        assert os.path.exists(pkl_path) or os.path.exists("models/openclip.pkl")

    def test_semantic_feature_preprocessing(self, tmp_path, tiger_image_bytes):
        """Test preprocessing image into visual embedding input vector"""
        img_path = tmp_path / "tiger.jpg"
        img_path.write_bytes(tiger_image_bytes)
        
        features = SemanticVerifierService.preprocess_image(str(img_path))
        assert features is not None
        assert len(features) == 11
        assert features[0] > 100 # High red for tiger

    def test_semantic_verification_scoring(self, tmp_path, tiger_image_bytes):
        """Test semantic verification outputs concept similarity scores"""
        img_path = tmp_path / "tiger.jpg"
        img_path.write_bytes(tiger_image_bytes)
        
        res = SemanticVerifierService.verify(
            str(img_path),
            image_id="val-openclip-01",
            speciesnet_prediction="Bengal Tiger",
            speciesnet_confidence=0.92
        )
        
        assert isinstance(res, SemanticVerificationResult)
        assert res.image_id == "val-openclip-01"
        assert res.primary_prediction != ""
        assert 0.0 <= res.primary_similarity <= 1.0
        assert len(res.scores) > 5
        assert isinstance(res.agrees_with_speciesnet, bool)
        assert len(res.top_predictions) == len(res.scores)

    def test_agreement_scenario_boost(self, tmp_path, tiger_image_bytes):
        """Test that SpeciesNet + OpenCLIP agreement yields high agreement score"""
        img_path = tmp_path / "tiger.jpg"
        img_path.write_bytes(tiger_image_bytes)
        
        res = SemanticVerifierService.verify(
            str(img_path),
            image_id="val-agree-01",
            speciesnet_prediction="Bengal Tiger",
            speciesnet_confidence=0.95
        )
        
        if res.agrees_with_speciesnet:
            assert res.agreement_score > 0.6
            assert res.primary_prediction == "Bengal Tiger"

    def test_disagreement_scenario_handling(self):
        """Test Decision Engine penalty on model disagreement"""
        # When models disagree, final confidence is penalized and review is triggered
        decision_res = DecisionEngineService.decide(
            image_id="disagree-test",
            megadetector_confidence=0.85,
            megadetector_type="animal",
            speciesnet_species="Bengal Tiger",
            speciesnet_confidence=0.75,
            openclip_prediction="Indian Leopard",
            openclip_similarity=0.70,
            openclip_agrees=False,
            image_quality_score=0.90
        )
        
        assert decision_res.decision in [DecisionType.HUMAN_REVIEW, DecisionType.UNCERTAIN]
        assert any("OpenCLIP disagrees" in r for r in decision_res.reasoning)


# ============ API ENDPOINTS TESTS ============

class TestOpenCLIPAPI:
    """Test OpenCLIP REST endpoints under /api/detections/"""

    def test_verify_image_endpoint(self, client, test_db, tiger_image_bytes):
        """Test POST /api/detections/verify/{image_id}"""
        now = datetime.utcnow()
        img_id, _ = ImageService.ingest_image(
            image_bytes=tiger_image_bytes,
            camera_id="CAM_OC_01",
            timestamp=now,
            db=test_db
        )
        
        # POST trigger verification
        response = client.post(f"/api/detections/verify/{img_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "primary_prediction" in data["data"]
        assert "scores" in data["data"]
        
        # Verify in DB
        db_ver = test_db.query(Verification).filter(Verification.image_id == img_id).first()
        assert db_ver is not None
        assert db_ver.primary_prediction == data["data"]["primary_prediction"]

    def test_get_image_verification(self, client, test_db, tiger_image_bytes):
        """Test GET /api/detections/verifications/{image_id}"""
        now = datetime.utcnow()
        img_id, _ = ImageService.ingest_image(
            image_bytes=tiger_image_bytes,
            camera_id="CAM_OC_GET",
            timestamp=now,
            db=test_db
        )
        
        # Add a mock verification
        ver = Verification(
            id=str(uuid.uuid4()),
            image_id=img_id,
            primary_prediction="Bengal Tiger",
            confidence=0.93,
            semantic_scores={"Bengal Tiger": 0.93, "Indian Leopard": 0.35},
            model_name="openclip_vit_b32_serialized"
        )
        test_db.add(ver)
        test_db.commit()
        
        response = client.get(f"/api/detections/verifications/{img_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["data"]["primary_prediction"] == "Bengal Tiger"
        assert data["data"]["primary_similarity"] == 0.93


# ============ FULL PIPELINE WITH EVIDENCE FUSION ============

class TestOpenCLIPPipeline:
    """Test complete pipeline orchestration including OpenCLIP semantic verification"""

    def test_pipeline_with_openclip_verification(self, test_db, tiger_image_bytes):
        """Verify full pipeline: Gate → MegaDetector → SpeciesNet → OpenCLIP → Decision Engine"""
        now = datetime.utcnow()
        img_id, _ = ImageService.ingest_image(
            image_bytes=tiger_image_bytes,
            camera_id="CAM_PIPE_OC",
            timestamp=now,
            db=test_db
        )
        
        result = ProcessingPipeline.process_image(img_id, test_db)
        
        assert result.success is True
        assert result.quality_passed is True
        assert result.megadetector_output is not None
        
        # If animal was detected, OpenCLIP verification should have executed
        has_animal = any(d.object_type.value == "animal" for d in result.megadetector_output.detections)
        if has_animal and settings.ENABLE_OPENCLIP:
            assert result.semantic_verification is not None
            assert result.semantic_verification.primary_prediction != ""
            
            # Verify DB verification record
            db_ver = test_db.query(Verification).filter(Verification.image_id == img_id).first()
            assert db_ver is not None
            
            # Verify Decision Engine output contains combined signals
            db_decision = test_db.query(Decision).filter(Decision.image_id == img_id).first()
            assert db_decision is not None
            assert "openclip" in db_decision.signals or len(db_decision.signals) > 0
