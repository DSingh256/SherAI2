"""
VanRakshak AI - Pydantic Schemas for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============ IMAGE SCHEMAS ============

class ImageMetadata(BaseModel):
    """Image metadata from camera"""
    camera_id: str
    timestamp: datetime
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    location: Optional[str] = None


class ImageUploadRequest(BaseModel):
    """Image upload request"""
    camera_id: str
    timestamp: datetime
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    location: Optional[str] = None
    # Image file is handled separately via multipart/form-data


class ImageQualityResponse(BaseModel):
    """Image quality assessment response"""
    image_id: str
    quality_status: str  # good, blurry, too_dark, overexposed, corrupted, duplicate
    quality_score: float
    blur_score: Optional[float]
    brightness: Optional[float]
    contrast: Optional[float]


# ============ DETECTION SCHEMAS ============

class BoundingBox(BaseModel):
    """Bounding box in normalized coordinates (0-1)"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "x_min": 0.1,
                "y_min": 0.2,
                "x_max": 0.9,
                "y_max": 0.8
            }
        }


class DetectionResult(BaseModel):
    """Single detection result"""
    object_type: str  # animal, human, vehicle
    confidence: float
    bbox: BoundingBox
    crop_path: Optional[str] = None


class DetectionsResponse(BaseModel):
    """MegaDetector results for an image"""
    image_id: str
    detections: List[DetectionResult]
    no_detections: bool = False
    processing_time_ms: float


# ============ CLASSIFICATION SCHEMAS ============

class AlternativePrediction(BaseModel):
    """Alternative species prediction"""
    species: str
    confidence: float


class ClassificationResult(BaseModel):
    """SpeciesNet classification for a detection"""
    species: str
    confidence: float
    alternatives: List[AlternativePrediction] = []
    model_name: str = "speciesnet_v1"


class ClassificationsResponse(BaseModel):
    """Classification results for an image"""
    image_id: str
    classifications: List[ClassificationResult]


# ============ VERIFICATION SCHEMAS ============

class VerificationResult(BaseModel):
    """OpenCLIP semantic verification (Phase 5)"""
    primary_prediction: str
    confidence: float
    semantic_scores: Dict[str, float]


# ============ DECISION SCHEMAS ============

class DecisionSignals(BaseModel):
    """Signals used in decision making"""
    megadetector_confidence: Optional[float] = None
    speciesnet_confidence: Optional[float] = None
    openclip_confidence: Optional[float] = None
    image_quality: Optional[float] = None
    model_agreement: Optional[float] = None
    is_tiger: Optional[bool] = None


class DecisionEngineOutput(BaseModel):
    """Decision engine final output"""
    image_id: str
    species: Optional[str]
    confidence: float
    decision: str  # auto_accept, human_review, uncertain
    confidence_level: str  # high, medium, low
    reasoning: List[str]
    signals: DecisionSignals
    is_tiger: bool = False


# ============ HUMAN REVIEW SCHEMAS ============

class HumanReviewRequest(BaseModel):
    """Human review submission"""
    image_id: str
    ai_prediction: str
    ai_confidence: float
    human_prediction: str
    human_confidence: Optional[float] = None
    human_is_tiger: Optional[bool] = None
    reviewer_id: Optional[str] = None
    notes: Optional[str] = None


class HumanReviewResponse(BaseModel):
    """Human review stored response"""
    id: str
    image_id: str
    ai_prediction: str
    human_prediction: str
    created_at: datetime


# ============ TIGER DETECTION SCHEMAS ============

class TigerDetection(BaseModel):
    """Tiger detection result"""
    image_id: str
    camera_id: str
    timestamp: datetime
    location: Optional[str]
    gps_latitude: Optional[float]
    gps_longitude: Optional[float]
    confidence: float
    species: str


class TigerReidentification(BaseModel):
    """Potential tiger re-identification"""
    image_id_1: str
    image_id_2: str
    similarity: float
    verified: bool = False


# ============ ANALYTICS SCHEMAS ============

class ImageStats(BaseModel):
    """Image processing statistics"""
    total_images: int
    processed_images: int
    pending_review: int
    auto_accepted: int
    humans_reviewed: int


class SpeciesCount(BaseModel):
    """Count of species detected"""
    species: str
    count: int


class CameraActivity(BaseModel):
    """Activity for a single camera"""
    camera_id: str
    total_images: int
    animals_detected: int
    tigers_detected: int
    humans_detected: int
    vehicles_detected: int
    last_image_time: Optional[datetime]


# ============ AUDIT TRAIL SCHEMAS ============

class AuditEvent(BaseModel):
    """Single audit trail event"""
    event_type: str
    event_status: str
    details: Dict[str, Any]
    created_at: datetime


class ImageAuditTrail(BaseModel):
    """Complete audit trail for an image"""
    image_id: str
    events: List[AuditEvent]


# ============ API RESPONSE SCHEMAS ============

class APIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
