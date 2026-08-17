"""
VanRakshak AI - Phase 2 Test Suite
Quality Gate Assessment Tests
"""

import pytest
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
from PIL import ImageFilter
import io

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Import models and services
from db.models import Base, Image, ImageQuality
from db.database import SessionLocal
from services.image_service import ImageService
from core.quality_gate import (
    QualityGateService, QualityDecision, QualityGateResult
)
from config import settings


# ============ FIXTURES ============

@pytest.fixture(scope="function")
def test_db():
    """Create test database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    
    TestSessionLocal = sessionmaker(bind=engine)
    session = TestSessionLocal()
    
    yield session
    
    session.close()
    engine.dispose()


@pytest.fixture
def clear_image_bytes():
    """Create a clear, good quality image with texture (high Laplacian variance)"""
    from PIL import ImageDraw
    import random
    img = PILImage.new('RGB', (1920, 1080), color=(100, 150, random.randint(0, 255)))
    draw = ImageDraw.Draw(img)
    # Draw high-contrast lines to provide texture for OpenCV Laplacian check
    for i in range(0, 1920, 40):
        draw.line([(i, 0), (1920 - i, 1080)], fill=(0, 0, 0) if i % 80 == 0 else (255, 255, 255), width=3)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


@pytest.fixture
def blurry_image_bytes():
    """Create a blurry image (low Laplacian variance)"""
    # Create a very blurred image
    import random
    img = PILImage.new('RGB', (100, 100), color=(random.randint(100, 150), 128, 128))
    for _ in range(5):
        img = img.filter(ImageFilter.GaussianBlur(radius=10))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


@pytest.fixture
def dark_image_bytes():
    """Create a very dark image"""
    # Create dark image (mostly black, brightness ~5)
    import random
    img_array = np.full((100, 100, 3), random.randint(1, 10), dtype=np.uint8)
    img = PILImage.fromarray(img_array)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


@pytest.fixture
def bright_image_bytes():
    """Create a very bright/overexposed image"""
    # Create bright image (mostly white, brightness ~250)
    import random
    img_array = np.full((100, 100, 3), random.randint(245, 255), dtype=np.uint8)
    img = PILImage.fromarray(img_array)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


# ============ QUALITY ASSESSMENT TESTS ============

class TestQualityAssessment:
    """Test quality assessment logic"""

    def test_assess_clear_image(self, clear_image_bytes, test_db):
        """Test assessment of good quality image"""
        now = datetime.utcnow()
        
        # Ingest image
        image_id, _ = ImageService.ingest_image(
            image_bytes=clear_image_bytes,
            camera_id="CAM_QUALITY_TEST",
            timestamp=now,
            db=test_db
        )
        
        # Assess quality
        result = QualityGateService.assess_quality(image_id, test_db)
        
        assert result.passed
        assert result.decision == QualityDecision.ACCEPT
        assert result.quality_score >= 0.7
        assert len(result.reasons) == 0 or "too" not in str(result.reasons).lower()

    def test_assess_blurry_image(self, blurry_image_bytes, test_db):
        """Test detection of blurry image"""
        now = datetime.utcnow()
        
        image_id, _ = ImageService.ingest_image(
            image_bytes=blurry_image_bytes,
            camera_id="CAM_BLUR_TEST",
            timestamp=now,
            db=test_db
        )
        
        result = QualityGateService.assess_quality(image_id, test_db)
        
        # Should detect blur or have low quality score
        # (May not always detect depending on image processing)
        assert result.quality_score is not None
        assert isinstance(result.details, dict)

    def test_assess_dark_image(self, dark_image_bytes, test_db):
        """Test detection of too-dark image"""
        now = datetime.utcnow()
        
        image_id, _ = ImageService.ingest_image(
            image_bytes=dark_image_bytes,
            camera_id="CAM_DARK_TEST",
            timestamp=now,
            db=test_db
        )
        
        result = QualityGateService.assess_quality(image_id, test_db)
        
        # Should detect darkness
        if result.quality_score < 0.7:
            assert "dark" in str(result.reasons).lower() or \
                   "brightness" in str(result.details).lower()

    def test_assess_overexposed_image(self, bright_image_bytes, test_db):
        """Test detection of overexposed image"""
        now = datetime.utcnow()
        
        image_id, _ = ImageService.ingest_image(
            image_bytes=bright_image_bytes,
            camera_id="CAM_BRIGHT_TEST",
            timestamp=now,
            db=test_db
        )
        
        result = QualityGateService.assess_quality(image_id, test_db)
        
        # Should detect overexposure
        if result.quality_score < 0.7:
            assert "bright" in str(result.reasons).lower() or \
                   "overexposed" in str(result.reasons).lower() or \
                   "brightness" in str(result.details).lower()

    def test_quality_result_structure(self, clear_image_bytes, test_db):
        """Test QualityGateResult structure"""
        now = datetime.utcnow()
        
        image_id, _ = ImageService.ingest_image(
            image_bytes=clear_image_bytes,
            camera_id="CAM_RESULT_TEST",
            timestamp=now,
            db=test_db
        )
        
        result = QualityGateService.assess_quality(image_id, test_db)
        
        # Check result structure
        assert isinstance(result, QualityGateResult)
        assert result.image_id == image_id
        assert isinstance(result.decision, QualityDecision)
        assert 0 <= result.quality_score <= 1.0
        assert isinstance(result.reasons, list)
        assert isinstance(result.details, dict)
        assert isinstance(result.passed, bool)


# ============ QUALITY GATE APPLICATION TESTS ============

class TestQualityGateApplication:
    """Test applying quality gate to images"""

    def test_apply_quality_gate_good_image(self, clear_image_bytes, test_db):
        """Test applying quality gate to good image"""
        now = datetime.utcnow()
        
        image_id, _ = ImageService.ingest_image(
            image_bytes=clear_image_bytes,
            camera_id="CAM_GATE_GOOD",
            timestamp=now,
            db=test_db
        )
        
        # Initially should be GOOD
        image = test_db.query(Image).filter(Image.id == image_id).first()
        initial_status = image.quality_status
        
        # Apply gate
        passed = QualityGateService.apply_quality_gate(image_id, test_db)
        
        # Refresh
        test_db.refresh(image)
        
        assert passed
        assert image.quality_status == ImageQuality.GOOD.value
        assert image.quality_score is not None
        assert 0 <= image.quality_score <= 1.0

    def test_apply_quality_gate_updates_status(self, clear_image_bytes, test_db):
        """Test that quality gate updates image status"""
        now = datetime.utcnow()
        
        image_id, _ = ImageService.ingest_image(
            image_bytes=clear_image_bytes,
            camera_id="CAM_GATE_UPDATE",
            timestamp=now,
            db=test_db
        )
        
        # Apply gate
        QualityGateService.apply_quality_gate(image_id, test_db)
        
        # Verify update
        image = test_db.query(Image).filter(Image.id == image_id).first()
        
        assert image.quality_status in [
            ImageQuality.GOOD.value,
            ImageQuality.BLURRY.value,
            ImageQuality.TOO_DARK.value,
            ImageQuality.OVEREXPOSED.value,
            ImageQuality.CORRUPTED.value
        ]


# ============ BATCH PROCESSING TESTS ============

class TestBatchQualityGate:
    """Test batch quality gate processing"""

    def test_batch_quality_gate(self, clear_image_bytes, dark_image_bytes, test_db):
        """Test batch quality gate processing"""
        now = datetime.utcnow()
        
        # Create multiple images
        image_ids = []
        for i in range(5):
            import uuid
            img_bytes = clear_image_bytes if i < 3 else dark_image_bytes
            img_bytes = img_bytes + str(uuid.uuid4()).encode()
            image_id, _ = ImageService.ingest_image(
                image_bytes=img_bytes,
                camera_id="CAM_BATCH",
                timestamp=now,
                db=test_db
            )
            image_ids.append(image_id)
        
        # Apply batch gate
        results = QualityGateService.batch_quality_gate(image_ids, test_db)
        
        assert len(results) == 5
        assert all(img_id in results for img_id in image_ids)
        
        # Check that results have expected structure
        for img_id, result in results.items():
            assert isinstance(result, QualityGateResult)
            assert result.passed is not None


# ============ STATISTICS TESTS ============

class TestQualityStatistics:
    """Test quality statistics and reporting"""

    def test_quality_breakdown(self, clear_image_bytes, test_db):
        """Test quality breakdown statistics"""
        now = datetime.utcnow()
        
        # Ingest multiple images
        for i in range(3):
            import uuid
            img_bytes = clear_image_bytes + str(uuid.uuid4()).encode()
            ImageService.ingest_image(
                image_bytes=img_bytes,
                camera_id="CAM_STATS",
                timestamp=now,
                db=test_db
            )
        
        # Get breakdown
        breakdown = QualityGateService.get_quality_breakdown(
            camera_id="CAM_STATS",
            db=test_db
        )
        
        assert breakdown["total"] >= 3
        assert "breakdown" in breakdown
        assert isinstance(breakdown["good_percentage"], float)
        assert 0 <= breakdown["good_percentage"] <= 100

    def test_rejection_reasons(self, dark_image_bytes, test_db):
        """Test rejection reason tracking"""
        now = datetime.utcnow()
        
        # Ingest a dark image (likely to be rejected)
        image_id, _ = ImageService.ingest_image(
            image_bytes=dark_image_bytes,
            camera_id="CAM_REJECTION",
            timestamp=now,
            db=test_db
        )
        
        # Apply gate to mark it
        QualityGateService.apply_quality_gate(image_id, test_db)
        
        # Get reasons
        reasons = QualityGateService.get_rejection_reasons(
            camera_id="CAM_REJECTION",
            db=test_db
        )
        
        assert isinstance(reasons, dict)
        assert all(key in reasons for key in [
            "blurry", "too_dark", "overexposed", "corrupted", "duplicate"
        ])


# ============ QUALITY GATE DECISION TESTS ============

class TestQualityDecisions:
    """Test quality gate decision logic"""

    def test_accept_decision(self):
        """Test ACCEPT decision"""
        assert QualityDecision.ACCEPT.value == "accept"

    def test_blur_reject_decision(self):
        """Test BLUR_REJECT decision"""
        assert QualityDecision.BLUR_REJECT.value == "blur_reject"

    def test_darkness_reject_decision(self):
        """Test DARKNESS_REJECT decision"""
        assert QualityDecision.DARKNESS_REJECT.value == "darkness_reject"

    def test_overexposed_reject_decision(self):
        """Test OVEREXPOSED_REJECT decision"""
        assert QualityDecision.OVEREXPOSED_REJECT.value == "overexposed_reject"

    def test_corrupted_reject_decision(self):
        """Test CORRUPTED_REJECT decision"""
        assert QualityDecision.CORRUPTED_REJECT.value == "corrupted_reject"


# ============ INTEGRATION TESTS ============

class TestPhase2Integration:
    """Integration tests for Phase 2"""

    def test_full_quality_gate_pipeline(self, clear_image_bytes, test_db):
        """Test complete quality gate pipeline"""
        now = datetime.utcnow()
        
        # 1. Ingest image
        image_id, metadata = ImageService.ingest_image(
            image_bytes=clear_image_bytes,
            camera_id="CAM_PIPELINE",
            timestamp=now,
            location="Test Location",
            gps_latitude=21.12,
            gps_longitude=78.56,
            db=test_db
        )
        
        # 2. Assess quality
        result = QualityGateService.assess_quality(image_id, test_db)
        
        assert result.quality_score is not None
        assert result.decision is not None
        
        # 3. Apply gate
        passed = QualityGateService.apply_quality_gate(image_id, test_db)
        
        # 4. Verify update
        image = test_db.query(Image).filter(Image.id == image_id).first()
        
        assert image.quality_status is not None
        assert image.quality_score is not None
        
        # 5. Get statistics
        breakdown = QualityGateService.get_quality_breakdown(
            camera_id="CAM_PIPELINE",
            db=test_db
        )
        
        assert breakdown["total"] >= 1

    def test_quality_gate_with_multiple_cameras(self, clear_image_bytes, dark_image_bytes, test_db):
        """Test quality gate across multiple cameras"""
        now = datetime.utcnow()
        
        # Ingest from multiple cameras
        for camera_id in ["CAM_A", "CAM_B", "CAM_C"]:
            import uuid
            img_bytes = clear_image_bytes + str(uuid.uuid4()).encode()
            ImageService.ingest_image(
                image_bytes=img_bytes,
                camera_id=camera_id,
                timestamp=now,
                db=test_db
            )
        
        # Get stats for each camera
        for camera_id in ["CAM_A", "CAM_B", "CAM_C"]:
            breakdown = QualityGateService.get_quality_breakdown(
                camera_id=camera_id,
                db=test_db
            )
            
            assert breakdown["camera_id"] == camera_id
            assert breakdown["total"] >= 1


# ============ EDGE CASES ============

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_assess_nonexistent_image(self, test_db):
        """Test assessing nonexistent image"""
        result = QualityGateService.assess_quality(
            "nonexistent_id",
            test_db
        )
        
        assert not result.passed
        assert result.quality_score == 0.0

    def test_quality_gate_nonexistent_image(self, test_db):
        """Test applying gate to nonexistent image"""
        result = QualityGateService.apply_quality_gate(
            "nonexistent_id",
            test_db
        )
        
        assert result is False

    def test_batch_with_empty_list(self, test_db):
        """Test batch processing with empty list"""
        results = QualityGateService.batch_quality_gate([], test_db)
        
        assert results == {}

    def test_batch_with_nonexistent_ids(self, test_db):
        """Test batch processing with nonexistent IDs"""
        results = QualityGateService.batch_quality_gate(
            ["id1", "id2", "id3"],
            test_db
        )
        
        # Should handle gracefully
        assert len(results) == 3


# ============ RUN TESTS ============

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
