"""
VanRakshak AI - Phase 7 Test Suite
SAM/SAM2 Wildlife Segmentation Tests
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

from db.models import Base, Image, Detection, Classification, Verification, Segmentation, Decision, ImageQuality
from services.image_service import ImageService
from core.quality_gate import QualityGateService
from core.megadetector import MegaDetectorService
from core.species_classifier import SpeciesClassifierService
from core.semantic_verifier import SemanticVerifierService
from core.segmentation import SegmentationService, SegmentationResult
from core.decision_engine import DecisionEngineService
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
    """Create a tiger-like image"""
    img = PILImage.new('RGB', (640, 480), color=(185, 110, 45))
    draw = ImageDraw.Draw(img)
    for x in range(0, 640, 30):
        draw.line([(x, 0), (x + 20, 480)], fill=(15, 15, 15), width=6)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


# ============ SAM2 CORE SEGMENTATION TESTS ============

class TestSAM2SegmentationCore:
    """Test SAM2 model initialization, box prompts, mask quality, and multi-animal segmentation"""

    def test_sam2_pkl_model_exists(self):
        """Verify the serialized SAM2 model file exists"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(base_dir, "..", "models", "sam2.pkl")
        assert os.path.exists(pkl_path) or os.path.exists("models/sam2.pkl")

    def test_device_selection(self):
        """Verify device selection returns a valid compute device"""
        device = SegmentationService.get_device()
        assert device in ["cpu", "cuda", "mps"]

    def test_single_animal_box_prompt_segmentation(self, tmp_path, tiger_image_bytes):
        """Test SAM2 segmentation produces mask, transparent crop, and quality score"""
        img_path = tmp_path / "tiger_seg.jpg"
        img_path.write_bytes(tiger_image_bytes)
        
        det_id = str(uuid.uuid4())
        res = SegmentationService.segment(
            image_path=str(img_path),
            image_id="img-seg-01",
            detection_id=det_id,
            bbox_x_min=0.1,
            bbox_y_min=0.1,
            bbox_x_max=0.9,
            bbox_y_max=0.9,
            species="Bengal Tiger"
        )
        
        assert isinstance(res, SegmentationResult)
        assert res.image_id == "img-seg-01"
        assert res.detection_id == det_id
        assert os.path.exists(res.mask_path)
        assert os.path.exists(res.segmented_crop_path)
        assert res.mask_quality >= 0.80
        assert res.confidence >= 0.80
        assert res.flank_crop_path is not None
        assert os.path.exists(res.flank_crop_path)

    def test_multi_animal_segmentation(self, tmp_path, tiger_image_bytes):
        """Test multi-animal segmentation coordinates multiple independent masks and crops"""
        img_path = tmp_path / "multi_animal.jpg"
        img_path.write_bytes(tiger_image_bytes)
        
        detections = [
            {
                "id": str(uuid.uuid4()),
                "bbox": {"x_min": 0.05, "y_min": 0.1, "x_max": 0.45, "y_max": 0.8},
                "species": "Spotted Deer (Chital)"
            },
            {
                "id": str(uuid.uuid4()),
                "bbox": {"x_min": 0.55, "y_min": 0.1, "x_max": 0.95, "y_max": 0.8},
                "species": "Sambar Deer"
            }
        ]
        
        results = SegmentationService.segment_all_detections(
            image_path=str(img_path),
            image_id="img-multi-01",
            detections=detections
        )
        
        assert len(results) == 2
        assert results[0].detection_id != results[1].detection_id
        assert results[0].mask_path != results[1].mask_path
        assert os.path.exists(results[0].mask_path)
        assert os.path.exists(results[1].mask_path)

    def test_segmentation_fallback_on_invalid_path(self):
        """Test graceful failure handling when image path is invalid"""
        res = SegmentationService.segment(
            image_path="/invalid/nonexistent/image.jpg",
            image_id="none",
            detection_id="none",
            bbox_x_min=0.0,
            bbox_y_min=0.0,
            bbox_x_max=1.0,
            bbox_y_max=1.0
        )
        assert res is None


# ============ REST API SEGMENTATION TESTS ============

class TestSAM2API:
    """Test SAM2 REST endpoints under /api/detections/ and /api/reidentification/"""

    def test_segment_detection_endpoint(self, client, test_db, tiger_image_bytes):
        """Test POST /api/detections/segment/{detection_id}"""
        now = datetime.utcnow()
        img_id, _ = ImageService.ingest_image(
            image_bytes=tiger_image_bytes,
            camera_id="CAM_SEG_01",
            timestamp=now,
            db=test_db
        )
        
        det_id = str(uuid.uuid4())
        det = Detection(
            id=det_id,
            image_id=img_id,
            object_type="animal",
            confidence=0.95,
            bbox_x_min=0.1,
            bbox_y_min=0.1,
            bbox_x_max=0.9,
            bbox_y_max=0.9
        )
        test_db.add(det)
        test_db.commit()
        
        response = client.post(f"/api/detections/segment/{det_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "mask_path" in data["data"]
        assert "segmented_crop_path" in data["data"]
        assert data["data"]["mask_quality"] >= 0.80

    def test_get_image_segmentations_endpoint(self, client, test_db, tiger_image_bytes):
        """Test GET /api/detections/segmentations/{image_id}"""
        now = datetime.utcnow()
        img_id, _ = ImageService.ingest_image(
            image_bytes=tiger_image_bytes,
            camera_id="CAM_SEG_GET",
            timestamp=now,
            db=test_db
        )
        
        det_id = str(uuid.uuid4())
        seg = Segmentation(
            id=str(uuid.uuid4()),
            image_id=img_id,
            detection_id=det_id,
            mask_path="storage/segmented/mask_test.png",
            segmented_crop_path="storage/segmented/crop_test.png",
            model_name="sam2_wildlife_v1"
        )
        test_db.add(seg)
        test_db.commit()
        
        response = client.get(f"/api/detections/segmentations/{img_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["segmentations"]) == 1

    def test_get_reid_segmented_crops_endpoint(self, client, test_db, tiger_image_bytes):
        """Test GET /api/reidentification/crops/{image_id}"""
        now = datetime.utcnow()
        img_id, _ = ImageService.ingest_image(
            image_bytes=tiger_image_bytes,
            camera_id="CAM_REID_CROP",
            timestamp=now,
            db=test_db
        )
        
        det_id = str(uuid.uuid4())
        seg = Segmentation(
            id=str(uuid.uuid4()),
            image_id=img_id,
            detection_id=det_id,
            mask_path="storage/segmented/mask_reid.png",
            segmented_crop_path="storage/segmented/crop_reid.png",
            model_name="sam2_wildlife_v1"
        )
        test_db.add(seg)
        test_db.commit()
        
        response = client.get(f"/api/reidentification/crops/{img_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["crops"]) == 1


# ============ FULL PIPELINE WITH SAM2 INTEGRATION ============

class TestSAM2Pipeline:
    """Test full pipeline orchestration with SAM2 segmentation enabled"""

    def test_full_pipeline_runs_sam2(self, test_db, tiger_image_bytes):
        """Verify full pipeline: Gate → MegaDetector → SpeciesNet → OpenCLIP → SAM2 → Decision → Re-ID"""
        now = datetime.utcnow()
        img_id, _ = ImageService.ingest_image(
            image_bytes=tiger_image_bytes,
            camera_id="CAM_PIPE_SAM2",
            timestamp=now,
            db=test_db
        )
        
        result = ProcessingPipeline.process_image(img_id, test_db)
        
        assert result.success is True
        assert result.quality_passed is True
        assert result.megadetector_output is not None
        
        # Verify SAM2 segmentations exist
        if settings.ENABLE_SAM_SEGMENTATION and len(result.megadetector_output.animal_detections) > 0:
            assert len(result.segmentations) > 0
            
            # Check Segmentation record in database
            db_segs = test_db.query(Segmentation).filter(Segmentation.image_id == img_id).all()
            assert len(db_segs) > 0
            assert os.path.exists(db_segs[0].mask_path)
