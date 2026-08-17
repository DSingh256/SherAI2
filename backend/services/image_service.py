"""
VanRakshak AI - Image Service
Core image ingestion and management
"""

from datetime import datetime
from sqlalchemy.orm import Session
from db.models import Image
from db.schemas import ImageMetadata
from utils.image_utils import ImageUtils, PerceptualHash
from config import settings
from typing import Optional, Tuple


class ImageService:
    """Service for image management and ingestion"""

    @staticmethod
    def ingest_image(
        image_bytes: bytes,
        camera_id: str,
        timestamp: datetime,
        gps_latitude: Optional[float] = None,
        gps_longitude: Optional[float] = None,
        location: Optional[str] = None,
        db: Session = None
    ) -> Tuple[str, dict]:
        """
        Ingest a new camera-trap image
        
        Process:
        1. Save image to RAW storage
        2. Extract image metadata
        3. Calculate quality metrics
        4. Check for duplicates
        5. Store in database
        
        Args:
            image_bytes: Image file bytes
            camera_id: Camera identifier
            timestamp: Image capture timestamp
            gps_latitude: GPS latitude (optional)
            gps_longitude: GPS longitude (optional)
            location: Location name (optional)
            db: Database session
        
        Returns:
            Tuple of (image_id, metadata_dict)
        """
        
        # 1. Save image to RAW storage
        image_path = ImageUtils.save_image(image_bytes, settings.RAW_STORAGE_PATH)
        
        # 2. Calculate image hash for duplicate detection
        image_hash = ImageUtils.get_image_hash(image_bytes)
        
        # 3. Extract image metadata
        width, height, file_size = ImageUtils.get_image_dimensions(image_path)
        
        # 4. Calculate quality metrics
        quality_metrics = ImageUtils.get_image_quality_metrics(image_path)
        
        # 5. Create database record
        image = Image(
            camera_id=camera_id,
            timestamp=timestamp,
            gps_latitude=gps_latitude,
            gps_longitude=gps_longitude,
            location=location,
            image_path=image_path,
            image_hash=image_hash,
            image_width=width,
            image_height=height,
            file_size=file_size,
            blur_score=quality_metrics["blur_score"],
            brightness=quality_metrics["brightness"],
            contrast=quality_metrics["contrast"],
        )
        
        db.add(image)
        db.commit()
        db.refresh(image)
        
        return image.id, {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "image_path": image_path,
            "width": width,
            "height": height,
            "file_size": file_size,
            "blur_score": quality_metrics["blur_score"],
            "brightness": quality_metrics["brightness"],
            "contrast": quality_metrics["contrast"],
        }

    @staticmethod
    def get_image(image_id: str, db: Session) -> Optional[Image]:
        """Get image record by ID"""
        return db.query(Image).filter(Image.id == image_id).first()

    @staticmethod
    def get_images_by_camera(camera_id: str, db: Session, limit: int = 100) -> list:
        """Get images from a specific camera"""
        return db.query(Image).filter(
            Image.camera_id == camera_id
        ).order_by(Image.timestamp.desc()).limit(limit).all()

    @staticmethod
    def get_images_requiring_review(db: Session, limit: int = 50) -> list:
        """Get images pending human review"""
        from db.models import Decision
        
        # Images with MEDIUM or LOW confidence decisions
        return db.query(Image).join(
            Decision, Image.id == Decision.image_id
        ).filter(
            Decision.decision.in_(["human_review", "uncertain"])
        ).order_by(Image.created_at.desc()).limit(limit).all()

    @staticmethod
    def mark_as_duplicate(image_id: str, db: Session) -> bool:
        """Mark image as duplicate"""
        from db.models import ImageQuality
        
        image = ImageService.get_image(image_id, db)
        if image:
            image.quality_status = ImageQuality.DUPLICATE.value
            db.commit()
            return True
        return False

    @staticmethod
    def check_duplicate_images(new_image_id: str, db: Session, threshold: int = 5) -> list:
        """
        Check if an image is a duplicate of existing images
        
        Args:
            new_image_id: ID of image to check
            db: Database session
            threshold: Max Hamming distance for duplicate
        
        Returns:
            List of potential duplicate image IDs
        """
        new_image = ImageService.get_image(new_image_id, db)
        if not new_image or not new_image.image_hash:
            return []
        
        new_hash = PerceptualHash.calculate_phash(new_image.image_path)
        
        duplicates = []
        existing_images = db.query(Image).filter(
            Image.id != new_image_id
        ).all()
        
        for img in existing_images:
            if img.image_hash:
                existing_hash = PerceptualHash.calculate_phash(img.image_path)
                if PerceptualHash.is_duplicate(new_hash, existing_hash, threshold):
                    duplicates.append(img.id)
        
        return duplicates
