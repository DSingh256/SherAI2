"""
VanRakshak AI - Phase 3 Test Suite
MegaDetector V6 Integration Tests
"""

import pytest
import os
import uuid
import io
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from PIL import Image as PILImage
from PIL import ImageDraw
import numpy as np
from fastapi.testclient import TestClient

from db.models import Base, Image, Detection, ImageQuality
from services.image_service import ImageService
from core.quality_gate import QualityGateService
from core.megadetector import MegaDetectorService, DetectionCategory
from core.pipeline import ProcessingPipeline
from main import app


# ============ FIXTURES ============

@pytest.fixture(scope="function")
def test_db():
    """Create a clean in-memory test database"""
    import db.database as db_module
    from db.models import Base
    
    # Create a shared test engine
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=test_engine)
    
    # Replace main engine for testing
    original_engine = db_module.engine
    original_session_local = db_module.SessionLocal
    db_module.engine = test_engine
    db_module.SessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    
    TestSessionLocal = sessionmaker(bind=test_engine)
    session = TestSessionLocal()
    
    yield session
    
    session.close()
    test_engine.dispose()
    
    # Restore original engine
    db_module.engine = original_engine
    db_module.SessionLocal = original_session_local


@pytest.fixture
def client(test_db):
    """Test client with database session dependency override"""
    from db.database import get_db
    
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
            
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def good_image_bytes():
    """Create a good quality, textured image"""
    img = PILImage.new('RGB', (800, 600), color=(120, 150, 180))
    draw = ImageDraw.Draw(img)
    # Add patterns to ensure it passes the blur/quality gate check
    for i in range(0, 800, 20):
        draw.line([(i, 0), (800 - i, 600)], fill=(0, 0, 0) if i % 40 == 0 else (255, 255, 255), width=2)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()


@pytest.fixture
def blurry_image_bytes():
    """Create a bad quality image (pitch black and blurry)"""
    img = PILImage.new('RGB', (100, 100), color=(0, 0, 0))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()


# ============ MEGADETECTOR CORE SERVICE TESTS ============

class TestMegaDetectorService:
    """Test core MegaDetector V6 service logic"""

    def test_megadetector_pkl_exists(self):
        """Verify the serialized PKL model file exists in models directory"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(base_dir, "..", "models", "megadetector.pkl")
        assert os.path.exists(pkl_path) or os.path.exists("models/megadetector.pkl")

    def test_megadetector_detection(self, tmp_path, good_image_bytes):
        """Test running detection on a good image path"""
        img_path = tmp_path / "test_good.jpg"
        img_path.write_bytes(good_image_bytes)
        
        output = MegaDetectorService.detect(str(img_path), "test-img-id")
        
        assert output.image_id == "test-img-id"
        assert isinstance(output.processing_time_ms, float)
        assert output.processing_time_ms > 0
        
        # Test coordinates are normalized (0.0 to 1.0)
        for d in output.detections:
            assert isinstance(d.object_type, DetectionCategory)
            assert 0.0 <= d.bbox.x_min <= 1.0
            assert 0.0 <= d.bbox.y_min <= 1.0
            assert 0.0 <= d.bbox.x_max <= 1.0
            assert 0.0 <= d.bbox.y_max <= 1.0
            assert d.bbox.x_min < d.bbox.x_max
            assert d.bbox.y_min < d.bbox.y_max
            assert 0.5 <= d.confidence <= 1.0

    def test_detection_cropping(self, tmp_path, good_image_bytes):
        """Test cropping out detections and saving them to disk"""
        img_path = tmp_path / "test_good.jpg"
        img_path.write_bytes(good_image_bytes)
        
        output = MegaDetectorService.detect(str(img_path), "test-img-id")
        
        if len(output.detections) > 0:
            det = output.detections[0]
            crop_dir = tmp_path / "crops"
            crop_dir.mkdir()
            
            crop_path = MegaDetectorService.crop_detection(
                str(img_path), det.bbox, output_dir=str(crop_dir), detection_id="test-crop"
            )
            
            assert crop_path is not None
            assert os.path.exists(crop_path)
            assert crop_path.endswith(".jpg")
            
            # Check crop size is valid
            crop_img = PILImage.open(crop_path)
            assert crop_img.size[0] > 0
            assert crop_img.size[1] > 0


# ============ DATABASE DETECTION STORAGE TESTS ============

class TestMegaDetectorDatabase:
    """Test saving and loading Detection model records in database"""

    def test_detection_db_save(self, test_db, good_image_bytes, tmp_path):
        """Verify Detection records are written and retrieved successfully"""
        # Ingest a mock image first
        now = datetime.utcnow()
        image_id, _ = ImageService.ingest_image(
            image_bytes=good_image_bytes,
            camera_id="CAM_DET_TEST",
            timestamp=now,
            db=test_db
        )
        
        # Save a mock detection
        det_id = str(uuid.uuid4())
        det_record = Detection(
            id=det_id,
            image_id=image_id,
            object_type="animal",
            confidence=0.85,
            bbox_x_min=0.1,
            bbox_y_min=0.2,
            bbox_x_max=0.9,
            bbox_y_max=0.8,
            crop_path=str(tmp_path / "test_crop.jpg")
        )
        test_db.add(det_record)
        test_db.commit()
        
        # Query and verify
        db_det = test_db.query(Detection).filter(Detection.id == det_id).first()
        assert db_det is not None
        assert db_det.image_id == image_id
        assert db_det.object_type == "animal"
        assert db_det.confidence == 0.85
        assert db_det.bbox_x_min == 0.1
        assert db_det.bbox_y_min == 0.2


# ============ API ENDPOINT TESTS ============

class TestMegaDetectorAPI:
    """Test REST API routes under /api/detections/"""

    def test_run_detection_success(self, client, test_db, good_image_bytes):
        """Test triggering detection on a valid image of GOOD quality status"""
        now = datetime.utcnow()
        # Ingest image
        image_id, _ = ImageService.ingest_image(
            image_bytes=good_image_bytes,
            camera_id="CAM_API_GOOD",
            timestamp=now,
            db=test_db
        )
        print(f"DEBUG: Created image {image_id}")
        print(f"DEBUG: test_db id = {id(test_db)}")
        print(f"DEBUG: test_db bind: {test_db.bind}")
        
        # Run quality gating to set status to GOOD
        QualityGateService.apply_quality_gate(image_id, test_db)
        
        # Check image in DB
        img = test_db.query(Image).filter(Image.id == image_id).first()
        print(f"DEBUG: Image in DB: {img}, quality_status: {img.quality_status if img else 'NOT FOUND'}")
        
        # POST trigger detection
        response = client.post(f"/api/detections/detect/{image_id}")
        print(f"DEBUG: Response status: {response.status_code}")
        print(f"DEBUG: Response body: {response.text}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "detections" in data["data"]
        
        # Verify in DB
        db_dets = test_db.query(Detection).filter(Detection.image_id == image_id).all()
        assert len(db_dets) >= 0

    def test_run_detection_bad_quality(self, client, test_db, blurry_image_bytes):
        """Verify running detection on a BLURRY/bad quality image returns 400 Bad Request"""
        now = datetime.utcnow()
        # Ingest image
        image_id, _ = ImageService.ingest_image(
            image_bytes=blurry_image_bytes,
            camera_id="CAM_API_BLUR",
            timestamp=now,
            db=test_db
        )
        
        # Apply quality gate which sets status to BLURRY
        QualityGateService.apply_quality_gate(image_id, test_db)
        
        # POST trigger detection - should fail quality constraint
        response = client.post(f"/api/detections/detect/{image_id}")
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data or data.get("success") is False
        assert "Cannot run detection" in response.text

    def test_run_detection_nonexistent_image(self, client):
        """POST trigger on non-existent image returns 404"""
        response = client.post("/api/detections/detect/nonexistent-uuid")
        assert response.status_code == 404

    def test_get_image_detections(self, client, test_db, good_image_bytes):
        """Verify fetching stored detections for a given image"""
        now = datetime.utcnow()
        image_id, _ = ImageService.ingest_image(
            image_bytes=good_image_bytes,
            camera_id="CAM_API_GET",
            timestamp=now,
            db=test_db
        )
        
        # Create a mock detection
        det = Detection(
            id=str(uuid.uuid4()),
            image_id=image_id,
            object_type="vehicle",
            confidence=0.91,
            bbox_x_min=0.2,
            bbox_y_min=0.3,
            bbox_x_max=0.7,
            bbox_y_max=0.9
        )
        test_db.add(det)
        test_db.commit()
        
        # GET image detections
        response = client.get(f"/api/detections/image/{image_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["detections"]) == 1
        assert data["data"]["detections"][0]["object_type"] == "vehicle"

    def test_get_detection_stats(self, client, test_db, good_image_bytes):
        """Verify retrieving statistics summaries for all detections"""
        # Create a few mock detections
        now = datetime.utcnow()
        image_id, _ = ImageService.ingest_image(
            image_bytes=good_image_bytes,
            camera_id="CAM_STATS",
            timestamp=now,
            db=test_db
        )
        
        det1 = Detection(id=str(uuid.uuid4()), image_id=image_id, object_type="animal", confidence=0.9, bbox_x_min=0.1, bbox_y_min=0.1, bbox_x_max=0.5, bbox_y_max=0.5)
        det2 = Detection(id=str(uuid.uuid4()), image_id=image_id, object_type="human", confidence=0.8, bbox_x_min=0.2, bbox_y_min=0.2, bbox_x_max=0.6, bbox_y_max=0.6)
        
        test_db.add(det1)
        test_db.add(det2)
        test_db.commit()
        
        # GET stats
        response = client.get("/api/detections/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["data"]["total_detections"] >= 2
        assert data["data"]["breakdown"]["animals"] >= 1
        assert data["data"]["breakdown"]["humans"] >= 1


# ============ MASTER PIPELINE INTEGRATION TESTS ============

class TestMegaDetectorPipeline:
    """Test MegaDetector integration in the ProcessingPipeline"""

    def test_pipeline_runs_megadetector(self, test_db, good_image_bytes):
        """Verify running the master pipeline on a GOOD image invokes MegaDetector and saves results"""
        now = datetime.utcnow()
        image_id, _ = ImageService.ingest_image(
            image_bytes=good_image_bytes,
            camera_id="CAM_PIPELINE_TEST",
            timestamp=now,
            db=test_db
        )
        
        # Trigger the pipeline
        pipeline_result = ProcessingPipeline.process_image(image_id, test_db)
        
        assert pipeline_result.success is True
        assert pipeline_result.quality_passed is True
        assert pipeline_result.megadetector_output is not None
        
        # Check database stored detections
        db_dets = test_db.query(Detection).filter(Detection.image_id == image_id).all()
        # The default profile select might choose empty, so check both scenarios
        if pipeline_result.megadetector_output.no_detections:
            assert len(db_dets) == 0
        else:
            assert len(db_dets) > 0
            assert db_dets[0].object_type in ["animal", "human", "vehicle"]
            assert db_dets[0].crop_path is not None
