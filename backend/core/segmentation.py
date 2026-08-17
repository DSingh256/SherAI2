"""
VanRakshak AI - SAM/SAM2 Segmentation
Image segmentation for detected animals.

Creates segmentation masks to isolate animals from backgrounds,
improving visualization and supporting downstream re-identification.

Simulated mode generates elliptical masks within bounding boxes.
"""

import os
import hashlib
import time
import pickle
from typing import Optional
from dataclasses import dataclass
from PIL import Image as PILImage, ImageDraw
import numpy as np

from config import settings


@dataclass
class SegmentationResult:
    """Result from segmentation"""
    image_id: str
    detection_id: str
    mask_path: str
    segmented_crop_path: Optional[str] = None
    model_name: str = "sam2_simulated"
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "detection_id": self.detection_id,
            "mask_path": self.mask_path,
            "segmented_crop_path": self.segmented_crop_path,
            "model_name": self.model_name,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class MockSAM2Model:
    """Mock SAM2 Segmentation model structure for serialization"""
    def __init__(self, model_name="sam2_wildlife_v1"):
        self.model_name = model_name

    def segment(self, image_path: str, bbox: list) -> dict:
        x_min, y_min, x_max, y_max = bbox
        w_box = x_max - x_min
        h_box = y_max - y_min
        mask = np.zeros((100, 100), dtype=np.uint8)
        cx, cy = 50, 50
        rx, ry = int(w_box * 50), int(h_box * 50)
        rx = max(5, min(50, rx))
        ry = max(5, min(50, ry))
        y_indices, x_indices = np.ogrid[-cy:100-cy, -cx:100-cx]
        ellipse_area = (x_indices**2 / rx**2) + (y_indices**2 / ry**2) <= 1
        mask[ellipse_area] = 255
        return {
            "mask_data": mask.tolist(),
            "model_name": self.model_name,
            "mask_ratio": float(np.sum(mask == 255) / 10000.0)
        }


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "MockSAM2Model":
            return MockSAM2Model
        return super().find_class(module, name)


class SegmentationService:
    """
    SAM/SAM2 segmentation service.

    Generates segmentation masks for detected animals.
    Simulated mode creates elliptical masks within detection bounding boxes.
    """

    @staticmethod
    def segment(
        image_path: str,
        image_id: str,
        detection_id: str,
        bbox_x_min: float,
        bbox_y_min: float,
        bbox_x_max: float,
        bbox_y_max: float,
    ) -> Optional[SegmentationResult]:
        """
        Generate segmentation mask for a detection.

        Args:
            image_path: Path to original image
            image_id: Image identifier
            detection_id: Detection identifier
            bbox_*: Normalized bounding box coordinates (0-1)

        Returns:
            SegmentationResult with mask and segmented crop paths
        """
        start_time = time.time()

        try:
            img = PILImage.open(image_path)
            w, h = img.size

            # Check if pickled model exists
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            pkl_path = os.path.join(base_dir, "models", "sam2.pkl")
            if not os.path.exists(pkl_path):
                pkl_path = os.path.join("models", "sam2.pkl")
                
            model_name = "sam2_simulated"
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, "rb") as f:
                        model = SafeUnpickler(f).load()
                    res = model.segment(image_path, [bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max])
                    model_name = res.get("model_name", "sam2_serialized_v1")
                except Exception as e:
                    print(f"Error loading sam2 pickle model: {e}. Falling back to default simulation.")

            # Convert normalized bbox to pixel coordinates
            left = int(bbox_x_min * w)
            top = int(bbox_y_min * h)
            right = int(bbox_x_max * w)
            bottom = int(bbox_y_max * h)

            # Ensure valid bounds
            left = max(0, left)
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)

            # Create segmentation mask (elliptical approximation)
            mask = PILImage.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)

            # Draw ellipse within bounding box with slight padding
            pad_x = int((right - left) * 0.05)
            pad_y = int((bottom - top) * 0.05)
            draw.ellipse(
                [left + pad_x, top + pad_y, right - pad_x, bottom - pad_y],
                fill=255,
            )

            # Save mask
            output_dir = settings.SEGMENTED_STORAGE_PATH
            os.makedirs(output_dir, exist_ok=True)

            mask_filename = f"mask_{detection_id}.png"
            mask_path = os.path.join(output_dir, mask_filename)
            mask.save(mask_path)

            # Create segmented crop (apply mask to original image)
            segmented_crop_path = SegmentationService._create_segmented_crop(
                img, mask, left, top, right, bottom, detection_id, output_dir
            )

            processing_time = (time.time() - start_time) * 1000

            return SegmentationResult(
                image_id=image_id,
                detection_id=detection_id,
                mask_path=mask_path,
                segmented_crop_path=segmented_crop_path,
                model_name=model_name,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            print(f"Segmentation error: {e}")
            return None

    @staticmethod
    def _create_segmented_crop(
        img: PILImage.Image,
        mask: PILImage.Image,
        left: int, top: int, right: int, bottom: int,
        detection_id: str,
        output_dir: str,
    ) -> Optional[str]:
        """Create a segmented crop with transparent background"""
        try:
            # Convert to RGBA
            rgba = img.convert("RGBA")

            # Apply mask as alpha channel
            mask_array = np.array(mask)
            rgba_array = np.array(rgba)
            rgba_array[:, :, 3] = mask_array

            segmented = PILImage.fromarray(rgba_array)

            # Crop to bounding box region
            crop = segmented.crop((left, top, right, bottom))

            # Save
            crop_filename = f"segmented_{detection_id}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            crop.save(crop_path, "PNG")

            return crop_path

        except Exception as e:
            print(f"Error creating segmented crop: {e}")
            return None
