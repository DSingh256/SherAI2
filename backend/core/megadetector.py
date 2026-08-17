"""
VanRakshak AI - MegaDetector V6 Integration
Detects: ANIMAL, HUMAN, VEHICLE with bounding boxes and confidence scores.

Architecture:
    - SimulatedMegaDetector: Returns realistic mock results based on image characteristics
    - RealMegaDetector: Interface ready for production MegaDetector V6 drop-in

The simulated mode uses image hash seeding to produce deterministic but varied results,
ensuring the same image always gives the same detections for demo consistency.
"""

import os
import hashlib
import random
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image as PILImage
import numpy as np
import pickle

from config import settings


class DetectionCategory(str, Enum):
    """MegaDetector detection categories"""
    ANIMAL = "animal"
    HUMAN = "human"
    VEHICLE = "vehicle"


@dataclass
class BoundingBox:
    """Normalized bounding box (0-1 coordinates)"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_dict(self) -> dict:
        return {
            "x_min": round(self.x_min, 4),
            "y_min": round(self.y_min, 4),
            "x_max": round(self.x_max, 4),
            "y_max": round(self.y_max, 4),
        }

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class MegaDetectorResult:
    """Result from a single MegaDetector detection"""
    object_type: DetectionCategory
    confidence: float
    bbox: BoundingBox
    crop_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "object_type": self.object_type.value,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict(),
            "crop_path": self.crop_path,
        }


@dataclass
class MegaDetectorOutput:
    """Complete MegaDetector output for one image"""
    image_id: str
    detections: List[MegaDetectorResult] = field(default_factory=list)
    no_detections: bool = False
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "detections": [d.to_dict() for d in self.detections],
            "no_detections": self.no_detections,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

    @property
    def has_animal(self) -> bool:
        return any(d.object_type == DetectionCategory.ANIMAL for d in self.detections)

    @property
    def has_human(self) -> bool:
        return any(d.object_type == DetectionCategory.HUMAN for d in self.detections)

    @property
    def has_vehicle(self) -> bool:
        return any(d.object_type == DetectionCategory.VEHICLE for d in self.detections)

    @property
    def animal_detections(self) -> List[MegaDetectorResult]:
        return [d for d in self.detections if d.object_type == DetectionCategory.ANIMAL]

    @property
    def human_detections(self) -> List[MegaDetectorResult]:
        return [d for d in self.detections if d.object_type == DetectionCategory.HUMAN]

    @property
    def max_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return max(d.confidence for d in self.detections)


class MockMegaDetectorModel:
    """Mock MegaDetector V6 model structure for serialization"""
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.categories = ["animal", "human", "vehicle"]
        self.profiles = ["animal_only", "multi_animal", "animal_human", "human_only", "vehicle_only", "empty"]
        self.profile_weights = [0.55, 0.15, 0.05, 0.08, 0.04, 0.13]

    def predict(self, image_path: str, image_id: str = "") -> dict:
        import hashlib
        seed_str = f"{image_path}_{image_id}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        profile = rng.choices(self.profiles, weights=self.profile_weights, k=1)[0]
        detections = []
        if profile == "empty":
            pass
        elif profile == "multi_animal":
            num_animals = rng.randint(2, 4)
            for i in range(num_animals):
                conf = rng.uniform(0.75, 0.99)
                bbox = self._gen_bbox(rng, i, num_animals)
                detections.append({
                    "category": "animal",
                    "confidence": conf,
                    "bbox": bbox
                })
        else:
            categories_to_check = []
            if "animal" in profile:
                categories_to_check.append("animal")
            if "human" in profile:
                categories_to_check.append("human")
            if "vehicle" in profile:
                categories_to_check.append("vehicle")
            for cat in categories_to_check:
                conf = rng.uniform(0.60, 0.98)
                if conf >= self.confidence_threshold:
                    bbox = self._gen_bbox(rng)
                    detections.append({
                        "category": cat,
                        "confidence": conf,
                        "bbox": bbox
                    })
        return {
            "detections": detections,
            "processing_time_ms": rng.uniform(180, 420)
        }

    def _gen_bbox(self, rng, index=0, total=1):
        if total == 1:
            cx = rng.uniform(0.3, 0.7)
            cy = rng.uniform(0.3, 0.7)
            w = rng.uniform(0.2, 0.4)
            h = rng.uniform(0.2, 0.5)
        else:
            segment = 1.0 / total
            cx = segment * index + segment * rng.uniform(0.3, 0.7)
            cy = rng.uniform(0.3, 0.7)
            w = rng.uniform(0.12, 0.22)
            h = rng.uniform(0.18, 0.38)
        return [
            max(0.0, cx - w/2),
            max(0.0, cy - h/2),
            min(1.0, cx + w/2),
            min(1.0, cy + h/2)
        ]


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "MockMegaDetectorModel":
            return MockMegaDetectorModel
        return super().find_class(module, name)


class MegaDetectorService:
    """
    MegaDetector V6 service.

    Uses simulated detection for hackathon prototype.
    Architecture supports drop-in replacement with real MegaDetector V6.
    """

    # Detection probability weights for simulation
    # Biased toward animals since these are camera traps in a reserve
    DETECTION_PROFILES = {
        "animal_only": {"animal": 1.0, "human": 0.0, "vehicle": 0.0},
        "animal_human": {"animal": 0.7, "human": 0.3, "vehicle": 0.0},
        "human_only": {"animal": 0.0, "human": 1.0, "vehicle": 0.0},
        "vehicle_only": {"animal": 0.0, "human": 0.0, "vehicle": 1.0},
        "human_vehicle": {"animal": 0.0, "human": 0.5, "vehicle": 0.5},
        "multi_animal": {"animal": 1.0, "human": 0.0, "vehicle": 0.0},
        "empty": {"animal": 0.0, "human": 0.0, "vehicle": 0.0},
    }

    # Weighted profile selection (wildlife camera bias)
    PROFILE_WEIGHTS = {
        "animal_only": 0.50,
        "multi_animal": 0.15,
        "animal_human": 0.05,
        "human_only": 0.08,
        "vehicle_only": 0.04,
        "human_vehicle": 0.03,
        "empty": 0.15,
    }

    @staticmethod
    def detect(image_path: str, image_id: str = "") -> MegaDetectorOutput:
        """
        Run MegaDetector on an image.

        In simulated mode, generates realistic detections based on
        image hash for deterministic but varied results.

        Args:
            image_path: Path to the image file
            image_id: Image identifier for tracking

        Returns:
            MegaDetectorOutput with detections
        """
        start_time = time.time()

        # Check if pickled model exists
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pkl_path = os.path.join(base_dir, "models", "megadetector.pkl")
        if not os.path.exists(pkl_path):
            pkl_path = os.path.join("models", "megadetector.pkl")
            
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    model = SafeUnpickler(f).load()
                
                res = model.predict(image_path, image_id)
                detections = []
                for d in res["detections"]:
                    bbox = BoundingBox(
                        x_min=d["bbox"][0],
                        y_min=d["bbox"][1],
                        x_max=d["bbox"][2],
                        y_max=d["bbox"][3]
                    )
                    detections.append(MegaDetectorResult(
                        object_type=DetectionCategory(d["category"]),
                        confidence=d["confidence"],
                        bbox=bbox
                    ))
                return MegaDetectorOutput(
                    image_id=image_id,
                    detections=detections,
                    no_detections=len(detections) == 0,
                    processing_time_ms=res["processing_time_ms"]
                )
            except Exception as e:
                print(f"Error loading megadetector pickle model: {e}. Falling back to default simulation.")

        # Use image hash for deterministic random seeding
        seed = MegaDetectorService._get_seed(image_path, image_id)
        rng = random.Random(seed)

        # Select detection profile
        profile_name = MegaDetectorService._select_profile(rng)
        profile = MegaDetectorService.DETECTION_PROFILES[profile_name]

        detections = []

        # Generate detections based on profile
        if profile_name == "empty":
            # No detections — blank/empty frame
            pass

        elif profile_name == "multi_animal":
            # Multiple animal detections (herd/group)
            num_animals = rng.randint(2, 4)
            for i in range(num_animals):
                confidence = rng.uniform(0.75, 0.99)
                bbox = MegaDetectorService._generate_bbox(rng, index=i, total=num_animals)
                detections.append(MegaDetectorResult(
                    object_type=DetectionCategory.ANIMAL,
                    confidence=confidence,
                    bbox=bbox,
                ))

        else:
            # Generate detections per category
            for category_str, prob in profile.items():
                if prob > 0 and rng.random() < max(prob, 0.8):
                    category = DetectionCategory(category_str)
                    confidence = MegaDetectorService._generate_confidence(rng, category)

                    # Only include if above threshold
                    if confidence >= settings.MEGADETECTOR_CONFIDENCE_THRESHOLD:
                        bbox = MegaDetectorService._generate_bbox(rng)
                        detections.append(MegaDetectorResult(
                            object_type=category,
                            confidence=confidence,
                            bbox=bbox,
                        ))

        processing_time = (time.time() - start_time) * 1000

        # Simulate realistic processing time (200-800ms)
        simulated_time = rng.uniform(200, 800)

        output = MegaDetectorOutput(
            image_id=image_id,
            detections=detections,
            no_detections=len(detections) == 0,
            processing_time_ms=simulated_time,
        )

        return output

    @staticmethod
    def crop_detection(
        image_path: str,
        bbox: BoundingBox,
        output_dir: str = None,
        detection_id: str = ""
    ) -> Optional[str]:
        """
        Crop a detected region from the image.

        Args:
            image_path: Path to original image
            bbox: Bounding box to crop
            output_dir: Directory to save crop
            detection_id: Unique ID for the crop filename

        Returns:
            Path to cropped image, or None if failed
        """
        try:
            if output_dir is None:
                output_dir = settings.PROCESSED_STORAGE_PATH

            os.makedirs(output_dir, exist_ok=True)

            img = PILImage.open(image_path)
            w, h = img.size

            # Convert normalized bbox to pixel coordinates
            left = int(bbox.x_min * w)
            top = int(bbox.y_min * h)
            right = int(bbox.x_max * w)
            bottom = int(bbox.y_max * h)

            # Ensure valid bounds
            left = max(0, left)
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)

            # Crop
            crop = img.crop((left, top, right, bottom))

            # Save
            crop_filename = f"crop_{detection_id or hashlib.md5(str(bbox.to_dict()).encode()).hexdigest()[:12]}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            crop.save(crop_path, "JPEG", quality=95)

            return crop_path

        except Exception as e:
            print(f"Error cropping detection: {e}")
            return None

    @staticmethod
    def _get_seed(image_path: str, image_id: str) -> int:
        """Generate deterministic seed from image path/id"""
        seed_str = f"{image_path}_{image_id}"
        return int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)

    @staticmethod
    def _select_profile(rng: random.Random) -> str:
        """Select a detection profile based on weighted probabilities"""
        profiles = list(MegaDetectorService.PROFILE_WEIGHTS.keys())
        weights = list(MegaDetectorService.PROFILE_WEIGHTS.values())
        return rng.choices(profiles, weights=weights, k=1)[0]

    @staticmethod
    def _generate_confidence(rng: random.Random, category: DetectionCategory) -> float:
        """Generate realistic confidence score for a detection category"""
        if category == DetectionCategory.ANIMAL:
            # Animals usually detected with high confidence in camera traps
            return rng.uniform(0.70, 0.99)
        elif category == DetectionCategory.HUMAN:
            # Humans have varied confidence
            return rng.uniform(0.55, 0.97)
        else:
            # Vehicles
            return rng.uniform(0.60, 0.95)

    @staticmethod
    def _generate_bbox(rng: random.Random, index: int = 0, total: int = 1) -> BoundingBox:
        """
        Generate a realistic bounding box.

        For multiple detections, spaces them across the image.
        """
        if total == 1:
            # Single detection — centered-ish with variation
            cx = rng.uniform(0.25, 0.75)
            cy = rng.uniform(0.25, 0.75)
            w = rng.uniform(0.15, 0.45)
            h = rng.uniform(0.20, 0.50)
        else:
            # Multiple detections — space across image
            segment = 1.0 / total
            cx = segment * index + segment * rng.uniform(0.3, 0.7)
            cy = rng.uniform(0.30, 0.70)
            w = rng.uniform(0.10, 0.25)
            h = rng.uniform(0.15, 0.35)

        x_min = max(0.0, cx - w / 2)
        y_min = max(0.0, cy - h / 2)
        x_max = min(1.0, cx + w / 2)
        y_max = min(1.0, cy + h / 2)

        return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
