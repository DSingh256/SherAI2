"""
VanRakshak AI - Database Models
SQLAlchemy ORM models for all core entities
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, ForeignKey, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
import uuid

Base = declarative_base()


class ImageQuality(str, enum.Enum):
    """Image quality assessment"""
    GOOD = "good"
    BLURRY = "blurry"
    TOO_DARK = "too_dark"
    OVEREXPOSED = "overexposed"
    CORRUPTED = "corrupted"
    DUPLICATE = "duplicate"


class DetectionType(str, enum.Enum):
    """Types of objects detected by MegaDetector"""
    ANIMAL = "animal"
    HUMAN = "human"
    VEHICLE = "vehicle"


class ConfidenceLevel(str, enum.Enum):
    """Confidence routing levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DecisionStatus(str, enum.Enum):
    """Decision engine routing decision"""
    AUTO_ACCEPT = "auto_accept"
    HUMAN_REVIEW = "human_review"
    UNCERTAIN = "uncertain"


class Image(Base):
    """Core image entity - every camera-trap image"""
    __tablename__ = "images"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    gps_latitude = Column(Float, nullable=True)
    gps_longitude = Column(Float, nullable=True)
    location = Column(String, nullable=True)
    image_path = Column(String, nullable=False, unique=True)
    image_hash = Column(String, nullable=True, unique=True)  # For duplicate detection
    
    # Image metadata
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)
    
    # Quality assessment
    quality_status = Column(String, default=ImageQuality.GOOD.value)
    quality_score = Column(Float, nullable=True)  # 0-1
    blur_score = Column(Float, nullable=True)
    brightness = Column(Float, nullable=True)
    contrast = Column(Float, nullable=True)
    
    # Pipeline tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    detections = relationship("Detection", back_populates="image", cascade="all, delete-orphan")
    classifications = relationship("Classification", back_populates="image", cascade="all, delete-orphan")
    human_reviews = relationship("HumanReview", back_populates="image", cascade="all, delete-orphan")
    decision = relationship("Decision", back_populates="image", uselist=False, cascade="all, delete-orphan")


class Detection(Base):
    """MegaDetector V6 detections - animal/human/vehicle"""
    __tablename__ = "detections"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String, ForeignKey("images.id"), nullable=False, index=True)
    
    # Detection type
    object_type = Column(String, nullable=False)  # animal, human, vehicle
    confidence = Column(Float, nullable=False)
    
    # Bounding box (normalized 0-1)
    bbox_x_min = Column(Float, nullable=False)
    bbox_y_min = Column(Float, nullable=False)
    bbox_x_max = Column(Float, nullable=False)
    bbox_y_max = Column(Float, nullable=False)
    
    # Crop information for downstream processing
    crop_path = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    image = relationship("Image", back_populates="detections")
    classifications = relationship("Classification", back_populates="detection", cascade="all, delete-orphan")


class Classification(Base):
    """SpeciesNet species classification"""
    __tablename__ = "classifications"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String, ForeignKey("images.id"), nullable=False, index=True)
    detection_id = Column(String, ForeignKey("detections.id"), nullable=True)
    
    # Primary prediction
    species = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    
    # Alternative predictions (JSON: [{species: str, confidence: float}])
    alternative_predictions = Column(JSON, nullable=True)
    
    # Model metadata
    model_name = Column(String, default="speciesnet_v1")
    model_version = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    image = relationship("Image", back_populates="classifications")
    detection = relationship("Detection", back_populates="classifications")


class Verification(Base):
    """OpenCLIP semantic verification (Phase 5)"""
    __tablename__ = "verifications"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String, ForeignKey("images.id"), nullable=False, index=True)
    
    # Semantic scores
    semantic_scores = Column(JSON, nullable=False)  # {species: similarity_score}
    primary_prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    
    model_name = Column(String, default="openclip_v1")
    created_at = Column(DateTime, default=datetime.utcnow)


class Segmentation(Base):
    """SAM/SAM2 segmentation masks (Phase 6)"""
    __tablename__ = "segmentations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String, ForeignKey("images.id"), nullable=False, index=True)
    detection_id = Column(String, ForeignKey("detections.id"), nullable=False, index=True)
    
    # Segmentation mask path
    mask_path = Column(String, nullable=False)
    segmented_crop_path = Column(String, nullable=True)
    
    model_name = Column(String, default="sam_v1")
    created_at = Column(DateTime, default=datetime.utcnow)


class Decision(Base):
    """Decision engine output - final routing decision"""
    __tablename__ = "decisions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String, ForeignKey("images.id"), nullable=False, index=True, unique=True)
    
    # Final prediction
    species = Column(String, nullable=True)
    confidence = Column(Float, nullable=False)
    
    # Routing decision
    decision = Column(String, nullable=False)  # auto_accept, human_review, uncertain
    confidence_level = Column(String, nullable=False)  # high, medium, low
    
    # Reasoning (JSON: list of signals)
    reasoning = Column(JSON, nullable=False)
    signals = Column(JSON, nullable=True)  # Detailed signal breakdown
    
    # Is this a tiger?
    is_tiger = Column(Boolean, default=False, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    image = relationship("Image", back_populates="decision")


class HumanReview(Base):
    """Human-in-the-loop review and correction"""
    __tablename__ = "human_reviews"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String, ForeignKey("images.id"), nullable=False, index=True)
    
    # AI prediction
    ai_prediction = Column(String, nullable=False)
    ai_confidence = Column(Float, nullable=False)
    
    # Human decision
    human_prediction = Column(String, nullable=False)
    human_confidence = Column(Float, nullable=True)
    
    # Metadata
    reviewer_id = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Is this a tiger?
    human_is_tiger = Column(Boolean, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    image = relationship("Image", back_populates="human_reviews")


class TigerReidentification(Base):
    """Potential tiger re-identification via embedding similarity (Phase 11)"""
    __tablename__ = "tiger_reidentifications"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id_1 = Column(String, ForeignKey("images.id"), nullable=False, index=True)
    image_id_2 = Column(String, ForeignKey("images.id"), nullable=False, index=True)
    
    # Similarity score
    similarity = Column(Float, nullable=False)
    
    # Status
    verified = Column(Boolean, default=False)  # Human verification
    
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditTrail(Base):
    """Complete audit trail for every important decision"""
    __tablename__ = "audit_trails"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String, ForeignKey("images.id"), nullable=False, index=True)
    
    # Event type
    event_type = Column(String, nullable=False)  # quality_check, detection, classification, decision, human_review
    event_status = Column(String, nullable=False)  # pass, fail, pending
    
    # Details (JSON)
    details = Column(JSON, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class Camera(Base):
    """Camera trap registry and health tracking"""
    __tablename__ = "cameras"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=True)
    location = Column(String, nullable=True)
    gps_latitude = Column(Float, nullable=True)
    gps_longitude = Column(Float, nullable=True)
    zone = Column(String, nullable=True)
    
    # Status tracking
    status = Column(String, default="active")  # active, inactive, maintenance
    last_image_time = Column(DateTime, nullable=True)
    total_images = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    """System and conservation alerts"""
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    alert_type = Column(String, nullable=False, index=True)  # tiger_sighting, unusual_activity, threat, camera_failure
    severity = Column(String, nullable=False)  # low, medium, high, critical
    
    title = Column(String, nullable=False)
    message = Column(String, nullable=False)
    
    camera_id = Column(String, nullable=True, index=True)
    image_id = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    
    # Workflow
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
