"""
VanRakshak AI - MegaDetector V6 Integration (YOLOv8 fallback)
Detects: ANIMAL, HUMAN, VEHICLE with bounding boxes and confidence scores.

Architecture:
    - RealMegaDetector: Uses YOLOv8s as an offline drop-in for MegaDetector V6
"""

import os
import time
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image as PILImage
import hashlib

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

class YOLOMegaDetector:
    _model = None

    @classmethod
    def get_device(cls) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @classmethod
    def get_model(cls):
        if cls._model is None:
            try:
                from ultralytics import YOLO
                # YOLOv8m (medium) — higher mAP than YOLOv8s for better accuracy
                print("Loading YOLOv8m (Medium) for MegaDetector...")
                cls._model = YOLO('yolov8m.pt')
                print("YOLOv8m loaded successfully.")
            except ImportError:
                print("Ultralytics YOLO not installed. Please install it.")
                return None
        return cls._model

    @staticmethod
    def map_coco_class(class_id: int) -> Optional[DetectionCategory]:
        """Maps COCO classes to MegaDetector categories."""
        # 0: person
        if class_id == 0:
            return DetectionCategory.HUMAN
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        elif class_id in [2, 3, 5, 7]:
            return DetectionCategory.VEHICLE
        # Animals: bird(14), cat(15), dog(16), horse(17), sheep(18), cow(19),
        # elephant(20), bear(21), zebra(22), giraffe(23)
        elif 14 <= class_id <= 23:
            return DetectionCategory.ANIMAL
        return None


class MegaDetectorService:
    """
    MegaDetector V6 service using YOLOv8 fallback.
    """

    @staticmethod
    def detect(image_path: str, image_id: str = "") -> MegaDetectorOutput:
        start_time = time.time()
        
        model = YOLOMegaDetector.get_model()
        detections = []
        
        if model:
            device = YOLOMegaDetector.get_device()
            use_half = device in ["cuda", "mps"]
            results = model.predict(image_path, verbose=False, device=device, half=use_half)
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                img_h, img_w = result.orig_shape
                
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < settings.MEGADETECTOR_CONFIDENCE_THRESHOLD:
                        continue
                        
                    class_id = int(box.cls[0])
                    category = YOLOMegaDetector.map_coco_class(class_id)
                    
                    if category:
                        # Normalize bounding box
                        xyxy = box.xyxy[0].cpu().numpy()
                        bbox = BoundingBox(
                            x_min=max(0.0, float(xyxy[0]) / img_w),
                            y_min=max(0.0, float(xyxy[1]) / img_h),
                            x_max=min(1.0, float(xyxy[2]) / img_w),
                            y_max=min(1.0, float(xyxy[3]) / img_h)
                        )
                        detections.append(MegaDetectorResult(
                            object_type=category,
                            confidence=conf,
                            bbox=bbox
                        ))
        
        processing_time = (time.time() - start_time) * 1000

        output = MegaDetectorOutput(
            image_id=image_id,
            detections=detections,
            no_detections=len(detections) == 0,
            processing_time_ms=processing_time,
        )

        return output

    @staticmethod
    def crop_detection(
        image_path: str,
        bbox: BoundingBox,
        output_dir: str = None,
        detection_id: str = ""
    ) -> Optional[str]:
        try:
            if output_dir is None:
                output_dir = settings.PROCESSED_STORAGE_PATH

            os.makedirs(output_dir, exist_ok=True)

            img = PILImage.open(image_path)
            w, h = img.size

            left = int(bbox.x_min * w)
            top = int(bbox.y_min * h)
            right = int(bbox.x_max * w)
            bottom = int(bbox.y_max * h)

            left = max(0, left)
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)

            crop = img.crop((left, top, right, bottom))

            crop_filename = f"crop_{detection_id or hashlib.md5(str(bbox.to_dict()).encode()).hexdigest()[:12]}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            crop.save(crop_path, "JPEG", quality=95)

            return crop_path

        except Exception as e:
            print(f"Error cropping detection: {e}")
            return None
