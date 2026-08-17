"""
VanRakshak AI - MobileSAM Wildlife Segmentation Service
High-precision image segmentation for detected animals.
"""

import os
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from PIL import Image as PILImage
import numpy as np

from config import settings

@dataclass
class SegmentationResult:
    """Result from SAM/SAM2 segmentation"""
    image_id: str
    detection_id: str
    mask_path: str
    segmented_crop_path: Optional[str] = None
    flank_crop_path: Optional[str] = None
    mask_quality: float = 0.90
    confidence: float = 0.90
    species: Optional[str] = None
    model_name: str = "mobile_sam"
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "detection_id": self.detection_id,
            "mask_path": self.mask_path,
            "segmented_crop_path": self.segmented_crop_path,
            "flank_crop_path": self.flank_crop_path,
            "mask_quality": round(self.mask_quality, 4),
            "confidence": round(self.confidence, 4),
            "species": self.species,
            "model_name": self.model_name,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

class RealSAMModel:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            try:
                from ultralytics import SAM
                # Use MobileSAM for fast offline inference
                cls._model = SAM('mobile_sam.pt')
            except ImportError:
                print("Ultralytics MobileSAM not installed.")
                return None
        return cls._model

class SegmentationService:
    @staticmethod
    def get_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def segment(
        image_path: str,
        image_id: str,
        detection_id: str,
        bbox_x_min: float,
        bbox_y_min: float,
        bbox_x_max: float,
        bbox_y_max: float,
        species: Optional[str] = None,
        point_prompts: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[SegmentationResult]:
        start_time = time.time()

        if not image_path or not os.path.exists(image_path):
            return None

        try:
            img = PILImage.open(image_path)
            w, h = img.size

            left = max(0, int(bbox_x_min * w))
            top = max(0, int(bbox_y_min * h))
            right = min(w, int(bbox_x_max * w))
            bottom = min(h, int(bbox_y_max * h))

            if right <= left or bottom <= top:
                return None

            output_dir = settings.SEGMENTED_STORAGE_PATH
            os.makedirs(output_dir, exist_ok=True)
            
            mask_filename = f"mask_{detection_id}.png"
            mask_path = os.path.join(output_dir, mask_filename)

            model = RealSAMModel.get_model()
            
            mask_quality = 0.85
            confidence = 0.85
            segmented_crop_path = None
            flank_crop_path = None

            if model:
                # Provide bounding box as prompt to SAM
                bboxes = [[left, top, right, bottom]]
                results = model.predict(image_path, bboxes=bboxes, verbose=False)
                
                if results and len(results) > 0 and results[0].masks is not None:
                    # Get the binary mask
                    mask_data = results[0].masks.data[0].cpu().numpy()
                    
                    # Convert to PIL Image (L mode)
                    mask_img = PILImage.fromarray((mask_data * 255).astype(np.uint8)).resize((w, h), PILImage.NEAREST)
                    mask_img.save(mask_path)
                    
                    mask_quality = 0.95
                    confidence = 0.95
                    
                    # Create segmented crop
                    segmented_crop_path = SegmentationService._create_segmented_crop(
                        img, mask_img, left, top, right, bottom, detection_id, output_dir
                    )
                    
                    # Extract flank crop if Tiger/Leopard
                    if species in ["Bengal Tiger", "Indian Leopard"]:
                        flank_crop_path = SegmentationService._extract_flank_crop(
                            img, mask_img, left, top, right, bottom, detection_id, output_dir
                        )
                else:
                    return None
            else:
                return None

            processing_time = (time.time() - start_time) * 1000

            return SegmentationResult(
                image_id=image_id,
                detection_id=detection_id,
                mask_path=mask_path,
                segmented_crop_path=segmented_crop_path,
                flank_crop_path=flank_crop_path,
                mask_quality=mask_quality,
                confidence=confidence,
                species=species,
                model_name="mobile_sam",
                processing_time_ms=processing_time,
            )

        except Exception as e:
            print(f"Error in SAM segmentation: {e}")
            return None

    @staticmethod
    def _create_segmented_crop(
        original_img: PILImage.Image,
        mask_img: PILImage.Image,
        left: int, top: int, right: int, bottom: int,
        detection_id: str,
        output_dir: str
    ) -> str:
        transparent_bg = PILImage.new("RGBA", original_img.size, (0, 0, 0, 0))
        rgba_img = original_img.convert("RGBA")
        
        composite = PILImage.composite(rgba_img, transparent_bg, mask_img)
        crop = composite.crop((left, top, right, bottom))
        
        crop_path = os.path.join(output_dir, f"segcrop_{detection_id}.png")
        crop.save(crop_path, "PNG")
        return crop_path

    @staticmethod
    def _extract_flank_crop(
        original_img: PILImage.Image,
        mask_img: PILImage.Image,
        left: int, top: int, right: int, bottom: int,
        detection_id: str,
        output_dir: str
    ) -> str:
        box_w = right - left
        box_h = bottom - top
        
        f_left = left + int(box_w * 0.3)
        f_top = top + int(box_h * 0.3)
        f_right = right - int(box_w * 0.3)
        f_bottom = bottom - int(box_h * 0.3)
        
        if f_right <= f_left or f_bottom <= f_top:
            f_left, f_top, f_right, f_bottom = left, top, right, bottom
            
        flank = original_img.crop((f_left, f_top, f_right, f_bottom))
        flank_path = os.path.join(output_dir, f"flank_{detection_id}.jpg")
        flank.save(flank_path, "JPEG", quality=95)
        return flank_path
