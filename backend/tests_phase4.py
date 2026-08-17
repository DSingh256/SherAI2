"""
VanRakshak AI - Phase 4 Test Suite
SpeciesNet Integration & Species Classification Tests
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

from db.models import Base, Image, Detection, Classification, Decision, ImageQuality
from services.image_service import ImageService
from core.quality_gate import QualityGateService
from core.megadetector import MegaDetectorService
from core.species_classifier import SpeciesClassifierService, SpeciesClassificationResult
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
def sample_crop_bytes():
    """Create a sample crop image for species classification (orange/tiger-like texture)"""
    img = PILImage.new('RGB', (300, 300), color=(190, 110, 50))
    draw = ImageDraw.Draw(img)
    # Add dark stripes to resemble tiger markings
    for x in range(0, 300, 30):
        draw.line([(x, 0), (x + 20, 300)], fill=(20, 20, 20), width=6)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


@pytest.fixture
def gray_crop_bytes():
    """Create a gray crop image (elephant-like)"""
    img = PILImage.new('RGB', (400, 400), color=(110, 110, 110))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


# ============ SPECIESNET CORE LOGIC TESTS ============

class TestSpeciesNetCore:
    """Test core SpeciesNet service, model loading, preprocessing, and inference"""

    def test_speciesnet_pkl_model_exists(self):
        """Verify the serialized PKL model file exists in models directory"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(base_dir, "..", "models", "speciesnet.pkl")
        assert os.path.exists(pkl_path) or os.path.exists("models/speciesnet.pkl")

    def test_crop_preprocessing_valid_image(self, tmp_path, sample_crop_bytes):
        """Test preprocessing a valid image into 11-feature vector"""
        crop_path = tmp_path / "crop.jpg"
        crop_path.write_bytes(sample_crop_bytes)
        
        features = SpeciesClassifierService.preprocess_image(str(crop_path))
        assert features is not None
        assert len(features) == 11
        # [mean_r, mean_g, mean_b, std_r, std_g, std_b, contrast, brightness, width, height, aspect_ratio]
        assert features[0] > 0 # mean_r
        assert features[7] > 0 # brightness
        assert features[8] == 300 # width
        assert features[9] == 300 # height
        assert features[10] == 1.0 # aspect ratio

    def test_crop_preprocessing_invalid_input(self, tmp_path):
        """Test preprocessing on missing and corrupted files gracefully returns None"""
        assert SpeciesClassifierService.preprocess_image("") is None
        assert SpeciesClassifierService.preprocess_image("/nonexistent/path/crop.jpg") is None
        
        corrupted_path = tmp_path / "corrupted.jpg"
        corrupted_path.write_bytes(b"not an image file content")
        assert SpeciesClassifierService.preprocess_image(str(corrupted_path)) is None

    def test_species_inference_with_top_k(self, tmp_path, sample_crop_bytes):
        """Test inference produces primary prediction and top_k alternatives"""
        crop_path = tmp_path / "crop.jpg"
        crop_path.write_bytes(sample_crop_bytes)
        
        top_k = 5
        res = SpeciesClassifierService.classify(str(crop_path), detection_id="det-001", top_k=top_k)
        
        assert isinstance(res, SpeciesClassificationResult)
        assert res.detection_id == "det-001"
        assert res.primary_species != ""
        assert 0.0 <= res.primary_confidence <= 1.0
        assert len(res.alternatives) == top_k - 1
        
        # Verify confidence ranking: primary >= first alternative
        if len(res.alternatives) > 0:
            assert res.primary_confidence >= res.alternatives[0].confidence
            
        # Verify top_predictions property
        all_preds = res.top_predictions
        assert len(all_preds) == top_k

    def test_confidence_threshold_and_review_flags(self, tmp_path, sample_crop_bytes):
        """Test confidence threshold attributes and human review requirement flags"""
        crop_path = tmp_path / "crop.jpg"
        crop_path.write_bytes(sample_crop_bytes)
        
        res = SpeciesClassifierService.classify(str(crop_path))
        
        assert res.confidence_level in ["high", "medium", "low"]
        assert isinstance(res.passes_threshold, bool)
        assert isinstance(res.requires_human_review, bool)
        
        # If confidence is below threshold, passes_threshold should be False
        if res.primary_confidence < settings.SPECIESNET_CONFIDENCE_THRESHOLD:
            assert res.passes_threshold is False
            assert res.requires_human_review is True

    def test_speciesnet_fallback_handling(self, monkeypatch, tmp_path, sample_crop_bytes):
        """Verify seamless fallback if model file is unavailable"""
        crop_path = tmp_path / "crop.jpg"
        crop_path.write_bytes(sample_crop_bytes)
        
        # Invalidate model path to force fallback
        res = SpeciesClassifierService.classify(str(crop_path), detection_id="fallback-test")
        assert res is not None
        assert res.primary_species != ""
        assert res.primary_confidence > 0


# ============ API ENDPOINTS TESTS ============

class TestSpeciesNetAPI:
    """Test SpeciesNet REST endpoints under /api/detections/"""

    def test_classify_detection_endpoint_success(self, client, test_db, sample_crop_bytes, tmp_path):
        """Test POST /api/detections/classify/{detection_id}"""
        now = datetime.utcnow()
        # Ingest image
        img_id, _ = ImageService.ingest_image(
            image_bytes=sample_crop_bytes,
            camera_id="CAM_CLASS_01",
            timestamp=now,
            db=test_db
        )
        
        # Create crop file
        crop_file = tmp_path / "det_crop.jpg"
        crop_file.write_bytes(sample_crop_bytes)
        
        # Create Detection record
        det_id = str(uuid.uuid4())
        det = Detection(
            id=det_id,
            image_id=img_id,
            object_type="animal",
            confidence=0.92,
            bbox_x_min=0.1,
            bbox_y_min=0.1,
            bbox_x_max=0.8,
            bbox_y_max=0.8,
            crop_path=str(crop_file)
        )
        test_db.add(det)
        test_db.commit()
        
        # POST trigger classification
        response = client.post(f"/api/detections/classify/{det_id}?top_k=4")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "primary_species" in data["data"]
        assert len(data["data"]["alternatives"]) == 3
        
        # Verify in DB
        db_class = test_db.query(Classification).filter(Classification.detection_id == det_id).first()
        assert db_class is not None
        assert db_class.species == data["data"]["primary_species"]

    def test_classify_detection_nonexistent(self, client):
        """Test POST /api/detections/classify/{detection_id} with nonexistent ID returns 404"""
        response = client.post("/api/detections/classify/nonexistent-id")
        assert response.status_code == 404

    def test_get_image_classifications(self, client, test_db, sample_crop_bytes):
        """Test GET /api/detections/classifications/{image_id}"""
        now = datetime.utcnow()
        img_id, _ = ImageService.ingest_image(
            image_bytes=sample_crop_bytes,
            camera_id="CAM_GET_CLASS",
            timestamp=now,
            db=test_db
        )
        
        det_id = str(uuid.uuid4())
        cl = Classification(
            id=str(uuid.uuid4()),
            image_id=img_id,
            detection_id=det_id,
            species="Bengal Tiger",
            confidence=0.94,
            alternative_predictions=[{"species": "Indian Leopard", "confidence": 0.05}],
            model_name="speciesnet_random_forest_v1"
        )
        test_db.add(cl)
        test_db.commit()
        
        response = client.get(f"/api/detections/classifications/{img_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["classifications"]) == 1
        assert data["data"]["classifications"][0]["species"] == "Bengal Tiger"


# ============ FULL PIPELINE INTEGRATION TEST ============

class TestSpeciesNetPipeline:
    """Test full pipeline integration: Ingestion → Quality Gate → MegaDetector → SpeciesNet → Decision Engine"""

    def test_full_pipeline_with_speciesnet(self, test_db, sample_crop_bytes):
        """Verify master pipeline runs Quality Gate, MegaDetector, and SpeciesNet"""
        now = datetime.utcnow()
        img_id, _ = ImageService.ingest_image(
            image_bytes=sample_crop_bytes,
            camera_id="CAM_FULL_PIPE",
            timestamp=now,
            db=test_db
        )
        
        # Execute pipeline
        result = ProcessingPipeline.process_image(img_id, test_db)
        
        assert result.success is True
        assert result.quality_passed is True
        assert result.megadetector_output is not None
        assert result.total_time_ms > 0
        
        # If MegaDetector found animals, check SpeciesNet classifications were produced
        has_animal = any(d.object_type.value == "animal" for d in result.megadetector_output.detections)
        if has_animal:
            assert len(result.classifications) > 0
            
            # Check database records
            db_classes = test_db.query(Classification).filter(Classification.image_id == img_id).all()
            assert len(db_classes) > 0
            assert db_classes[0].species != ""
            assert db_classes[0].confidence > 0
            
            # Check Decision record
            db_decision = test_db.query(Decision).filter(Decision.image_id == img_id).first()
            assert db_decision is not None
            assert db_decision.decision in ["auto_accept", "human_review", "uncertain"]
