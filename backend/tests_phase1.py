"""
VanRakshak AI - Phase 1 Test Suite
Image Ingestion & Storage Tests
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from PIL import Image as PILImage
import numpy as np
import io

# Import models and services
from db.models import Base, Image, ImageQuality
from db.database import SessionLocal
from services.image_service import ImageService
from utils.image_utils import ImageUtils, PerceptualHash
from config import settings


# ============ FIXTURES ============

@pytest.fixture(scope="session")
def test_db():
    """Create test database"""
    # Use in-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    
    TestSessionLocal = sessionmaker(bind=engine)
    session = TestSessionLocal()
    
    yield session
    
    session.close()
    engine.dispose()


@pytest.fixture
def sample_image_bytes():
    """Create a sample valid image for testing"""
    # Create a simple RGB image
    img = PILImage.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


@pytest.fixture
def blurry_image_bytes():
    """Create a blurry image for testing"""
    # Create a blurred image
    img = PILImage.new('RGB', (100, 100), color='gray')
    img = img.filter(PILImage.BLUR)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


@pytest.fixture
def dark_image_bytes():
    """Create a very dark image"""
    # Create dark image (mostly black)
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img = PILImage.fromarray(img_array)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


# ============ IMAGE UTILS TESTS ============

class TestImageUtils:
    """Test image utility functions"""

    def test_image_hash_calculation(self, sample_image_bytes):
        """Test SHA256 hash calculation"""
        hash1 = ImageUtils.get_image_hash(sample_image_bytes)
        hash2 = ImageUtils.get_image_hash(sample_image_bytes)
        
        # Same image should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 is 64 hex chars
        assert isinstance(hash1, str)

    def test_image_hash_different_images(self, sample_image_bytes, dark_image_bytes):
        """Test that different images produce different hashes"""
        hash1 = ImageUtils.get_image_hash(sample_image_bytes)
        hash2 = ImageUtils.get_image_hash(dark_image_bytes)
        
        assert hash1 != hash2

    def test_image_dimensions(self, sample_image_bytes, tmp_path):
        """Test image dimension extraction"""
        # Save image to temp file
        img_path = tmp_path / "test_image.png"
        img_path.write_bytes(sample_image_bytes)
        
        width, height, file_size = ImageUtils.get_image_dimensions(str(img_path))
        
        assert width == 100
        assert height == 100
        assert file_size > 0

    def test_brightness_calculation(self, sample_image_bytes, tmp_path):
        """Test brightness calculation"""
        img_path = tmp_path / "test_image.png"
        img_path.write_bytes(sample_image_bytes)
        
        brightness = ImageUtils.get_brightness(str(img_path))
        
        # Red image should have high brightness in red channel
        assert 0 <= brightness <= 255

    def test_contrast_calculation(self, sample_image_bytes, tmp_path):
        """Test contrast calculation"""
        img_path = tmp_path / "test_image.png"
        img_path.write_bytes(sample_image_bytes)
        
        contrast = ImageUtils.get_contrast(str(img_path))
        
        assert contrast >= 0

    def test_blur_score_calculation(self, sample_image_bytes, blurry_image_bytes, tmp_path):
        """Test blur score calculation"""
        clear_path = tmp_path / "clear.png"
        blurry_path = tmp_path / "blurry.png"
        
        clear_path.write_bytes(sample_image_bytes)
        blurry_path.write_bytes(blurry_image_bytes)
        
        clear_score = ImageUtils.get_blur_score(str(clear_path))
        blurry_score = ImageUtils.get_blur_score(str(blurry_path))
        
        # Blurry images should have lower Laplacian variance
        assert clear_score >= blurry_score

    def test_corruption_detection(self, tmp_path):
        """Test image corruption detection"""
        # Valid image
        valid_path = tmp_path / "valid.png"
        img = PILImage.new('RGB', (100, 100), color='red')
        img.save(str(valid_path))
        
        assert not ImageUtils.is_corrupted(str(valid_path))
        
        # Corrupted file
        corrupt_path = tmp_path / "corrupt.png"
        corrupt_path.write_text("this is not an image")
        
        assert ImageUtils.is_corrupted(str(corrupt_path))

    def test_quality_metrics(self, sample_image_bytes, tmp_path):
        """Test complete quality metrics calculation"""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)
        
        metrics = ImageUtils.get_image_quality_metrics(str(img_path))
        
        assert "blur_score" in metrics
        assert "brightness" in metrics
        assert "contrast" in metrics
        assert "is_corrupted" in metrics
        
        assert isinstance(metrics["blur_score"], (int, float))
        assert isinstance(metrics["brightness"], (int, float))
        assert isinstance(metrics["contrast"], (int, float))
        assert isinstance(metrics["is_corrupted"], bool)


# ============ PERCEPTUAL HASH TESTS ============

class TestPerceptualHash:
    """Test perceptual hashing for duplicate detection"""

    def test_phash_calculation(self, sample_image_bytes, tmp_path):
        """Test pHash calculation"""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)
        
        phash = PerceptualHash.calculate_phash(str(img_path))
        
        assert len(phash) > 0
        assert all(c in '0123456789abcdef' for c in phash)

    def test_hamming_distance(self):
        """Test Hamming distance calculation"""
        hash1 = "0f00ff00"
        hash2 = "0f00ff00"
        hash3 = "ff00ff00"
        
        # Identical hashes
        distance1 = PerceptualHash.hamming_distance(hash1, hash2)
        assert distance1 == 0
        
        # Different hashes
        distance2 = PerceptualHash.hamming_distance(hash1, hash3)
        assert distance2 > 0

    def test_duplicate_detection(self, sample_image_bytes, tmp_path):
        """Test duplicate image detection"""
        img_path1 = tmp_path / "test1.png"
        img_path2 = tmp_path / "test2.png"
        
        # Same image saved twice
        img_path1.write_bytes(sample_image_bytes)
        img_path2.write_bytes(sample_image_bytes)
        
        hash1 = PerceptualHash.calculate_phash(str(img_path1))
        hash2 = PerceptualHash.calculate_phash(str(img_path2))
        
        # Should be detected as duplicates
        is_duplicate = PerceptualHash.is_duplicate(hash1, hash2, threshold=5)
        assert is_duplicate

    def test_non_duplicate_images(self, sample_image_bytes, dark_image_bytes, tmp_path):
        """Test that different images are not marked as duplicates"""
        img_path1 = tmp_path / "test1.png"
        img_path2 = tmp_path / "test2.png"
        
        img_path1.write_bytes(sample_image_bytes)
        img_path2.write_bytes(dark_image_bytes)
        
        hash1 = PerceptualHash.calculate_phash(str(img_path1))
        hash2 = PerceptualHash.calculate_phash(str(img_path2))
        
        # Should NOT be detected as duplicates
        is_duplicate = PerceptualHash.is_duplicate(hash1, hash2, threshold=5)
        assert not is_duplicate


# ============ IMAGE SERVICE TESTS ============

class TestImageService:
    """Test image ingestion and management service"""

    def test_image_ingestion(self, sample_image_bytes, test_db):
        """Test basic image ingestion"""
        now = datetime.utcnow()
        
        image_id, metadata = ImageService.ingest_image(
            image_bytes=sample_image_bytes,
            camera_id="CAM001",
            timestamp=now,
            gps_latitude=21.1234,
            gps_longitude=78.5678,
            location="Zone A",
            db=test_db
        )
        
        assert image_id is not None
        assert metadata["camera_id"] == "CAM001"
        assert metadata["width"] == 100
        assert metadata["height"] == 100
        assert metadata["file_size"] > 0

    def test_image_retrieval(self, sample_image_bytes, test_db):
        """Test image retrieval from database"""
        now = datetime.utcnow()
        
        image_id, _ = ImageService.ingest_image(
            image_bytes=sample_image_bytes,
            camera_id="CAM002",
            timestamp=now,
            db=test_db
        )
        
        # Retrieve image
        image = ImageService.get_image(image_id, test_db)
        
        assert image is not None
        assert image.id == image_id
        assert image.camera_id == "CAM002"

    def test_images_by_camera(self, sample_image_bytes, test_db):
        """Test retrieving images by camera"""
        now = datetime.utcnow()
        
        # Ingest multiple images from same camera
        for i in range(3):
            ImageService.ingest_image(
                image_bytes=sample_image_bytes,
                camera_id="CAM003",
                timestamp=now,
                db=test_db
            )
        
        # Retrieve images
        images = ImageService.get_images_by_camera("CAM003", test_db)
        
        assert len(images) >= 3
        assert all(img.camera_id == "CAM003" for img in images)

    def test_quality_metadata_storage(self, sample_image_bytes, test_db):
        """Test that quality metrics are stored correctly"""
        now = datetime.utcnow()
        
        image_id, metadata = ImageService.ingest_image(
            image_bytes=sample_image_bytes,
            camera_id="CAM004",
            timestamp=now,
            db=test_db
        )
        
        # Retrieve from database
        image = ImageService.get_image(image_id, test_db)
        
        assert image.blur_score is not None
        assert image.brightness is not None
        assert image.contrast is not None


# ============ INTEGRATION TESTS ============

class TestPhase1Integration:
    """Integration tests for Phase 1"""

    def test_full_image_ingestion_pipeline(self, sample_image_bytes, test_db):
        """Test complete image ingestion workflow"""
        now = datetime.utcnow()
        
        # Ingest image
        image_id, metadata = ImageService.ingest_image(
            image_bytes=sample_image_bytes,
            camera_id="CAM_INTEGRATION",
            timestamp=now,
            gps_latitude=21.12,
            gps_longitude=78.56,
            location="Integration Test Zone",
            db=test_db
        )
        
        # Verify ingestion
        assert image_id is not None
        
        # Retrieve and verify
        image = ImageService.get_image(image_id, test_db)
        assert image is not None
        assert image.camera_id == "CAM_INTEGRATION"
        assert image.quality_status == ImageQuality.GOOD.value
        
        # Verify metrics
        assert image.image_width == 100
        assert image.image_height == 100
        assert image.blur_score is not None
        assert image.brightness is not None

    def test_multiple_images_processing(self, sample_image_bytes, dark_image_bytes, test_db):
        """Test processing multiple images"""
        now = datetime.utcnow()
        
        # Process multiple images
        ids = []
        for i in range(5):
            img_bytes = sample_image_bytes if i % 2 == 0 else dark_image_bytes
            image_id, _ = ImageService.ingest_image(
                image_bytes=img_bytes,
                camera_id="CAM_MULTI",
                timestamp=now,
                db=test_db
            )
            ids.append(image_id)
        
        # Verify all were stored
        assert len(ids) == 5
        assert len(set(ids)) == 5  # All unique


# ============ RUN TESTS ============

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
