"""
VanRakshak AI - Privacy Protection
Human face detection and anonymization.

When MegaDetector detects a human, this module:
1. Detects faces using OpenCV Haar cascades
2. Applies Gaussian blur to face regions
3. Saves a privacy-safe copy

Protects: forest workers, researchers, tourists, villagers
"""

import os
import hashlib
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import cv2
import numpy as np

from config import settings


@dataclass
class FaceRegion:
    """Detected face region"""
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class PrivacyProtectionResult:
    """Result of privacy protection processing"""
    image_id: str
    faces_detected: int
    faces_blurred: int
    privacy_safe_path: Optional[str] = None
    face_regions: List[FaceRegion] = field(default_factory=list)
    applied: bool = False

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "faces_detected": self.faces_detected,
            "faces_blurred": self.faces_blurred,
            "privacy_safe_path": self.privacy_safe_path,
            "face_regions": [f.to_dict() for f in self.face_regions],
            "applied": self.applied,
        }


class PrivacyProtectionService:
    """
    Privacy protection service for human detections.

    Detects faces and applies blurring to protect identities
    of forest workers, researchers, tourists, and villagers.
    """

    # OpenCV Haar cascade for face detection
    FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    @staticmethod
    def protect(
        image_path: str,
        image_id: str,
        bbox_x_min: float = 0.0,
        bbox_y_min: float = 0.0,
        bbox_x_max: float = 1.0,
        bbox_y_max: float = 1.0,
    ) -> PrivacyProtectionResult:
        """
        Detect and blur faces in an image.

        Args:
            image_path: Path to original image
            image_id: Image identifier
            bbox_*: Human detection bounding box (normalized)

        Returns:
            PrivacyProtectionResult with blurred image path
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return PrivacyProtectionResult(
                    image_id=image_id,
                    faces_detected=0,
                    faces_blurred=0,
                    applied=False,
                )

            h, w = img.shape[:2]

            # Focus face detection on the human bounding box region
            x1 = int(bbox_x_min * w)
            y1 = int(bbox_y_min * h)
            x2 = int(bbox_x_max * w)
            y2 = int(bbox_y_max * h)

            # Ensure valid bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Extract ROI for face detection
            roi = img[y1:y2, x1:x2]

            # Detect faces
            face_cascade = cv2.CascadeClassifier(PrivacyProtectionService.FACE_CASCADE_PATH)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray_roi,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(20, 20),
            )

            face_regions = []
            faces_blurred = 0

            # If no faces detected by cascade, blur the upper portion
            # of the human detection as a safety measure
            if len(faces) == 0:
                # Blur upper 40% of human bounding box (likely head area)
                head_height = int((y2 - y1) * 0.4)
                head_region = img[y1:y1 + head_height, x1:x2]
                if head_region.size > 0:
                    blurred = cv2.GaussianBlur(head_region, (99, 99), 30)
                    img[y1:y1 + head_height, x1:x2] = blurred
                    faces_blurred = 1
                    face_regions.append(FaceRegion(x=x1, y=y1, width=x2-x1, height=head_height))
            else:
                # Blur each detected face
                for (fx, fy, fw, fh) in faces:
                    # Convert ROI coordinates back to full image
                    abs_x = x1 + fx
                    abs_y = y1 + fy

                    # Add padding
                    pad = int(max(fw, fh) * 0.2)
                    px1 = max(0, abs_x - pad)
                    py1 = max(0, abs_y - pad)
                    px2 = min(w, abs_x + fw + pad)
                    py2 = min(h, abs_y + fh + pad)

                    # Apply Gaussian blur
                    face_area = img[py1:py2, px1:px2]
                    if face_area.size > 0:
                        blurred = cv2.GaussianBlur(face_area, (99, 99), 30)
                        img[py1:py2, px1:px2] = blurred
                        faces_blurred += 1

                    face_regions.append(FaceRegion(
                        x=abs_x, y=abs_y, width=fw, height=fh
                    ))

            # Save privacy-safe copy
            output_dir = settings.PROCESSED_STORAGE_PATH
            os.makedirs(output_dir, exist_ok=True)
            safe_filename = f"privacy_safe_{image_id[:12]}.jpg"
            safe_path = os.path.join(output_dir, safe_filename)
            cv2.imwrite(safe_path, img)

            return PrivacyProtectionResult(
                image_id=image_id,
                faces_detected=len(faces) if len(faces) > 0 else 1,
                faces_blurred=faces_blurred,
                privacy_safe_path=safe_path,
                face_regions=face_regions,
                applied=True,
            )

        except Exception as e:
            print(f"Privacy protection error: {e}")
            return PrivacyProtectionResult(
                image_id=image_id,
                faces_detected=0,
                faces_blurred=0,
                applied=False,
            )
