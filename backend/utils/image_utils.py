"""
VanRakshak AI - Image Utilities
Image processing, hashing, validation
"""

import os
import hashlib
from pathlib import Path
from PIL import Image as PILImage
import cv2
import numpy as np
from typing import Tuple, Optional
from config import settings


class ImageUtils:
    """Utility functions for image processing"""

    @staticmethod
    def save_image(image_bytes: bytes, storage_path: str) -> str:
        """
        Save image bytes to disk
        
        Args:
            image_bytes: Image file bytes
            storage_path: Path to save to (from config)
        
        Returns:
            Full file path
        """
        # Create storage directory if needed
        Path(storage_path).mkdir(parents=True, exist_ok=True)
        
        # Generate filename from hash
        file_hash = hashlib.sha256(image_bytes).hexdigest()
        filename = f"{file_hash}.jpg"
        full_path = os.path.join(storage_path, filename)
        
        # Save file
        with open(full_path, "wb") as f:
            f.write(image_bytes)
        
        return full_path

    @staticmethod
    def get_image_hash(image_bytes: bytes) -> str:
        """
        Calculate SHA256 hash of image for duplicate detection
        
        Args:
            image_bytes: Image file bytes
        
        Returns:
            Hex hash string
        """
        return hashlib.sha256(image_bytes).hexdigest()

    @staticmethod
    def get_image_dimensions(image_path: str) -> Tuple[int, int, int]:
        """
        Get image width, height, and file size
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (width, height, file_size_bytes)
        """
        try:
            img = PILImage.open(image_path)
            width, height = img.size
            file_size = os.path.getsize(image_path)
            return width, height, file_size
        except Exception as e:
            raise ValueError(f"Failed to read image dimensions: {e}")

    @staticmethod
    def get_brightness(image_path: str) -> float:
        """
        Calculate average brightness of image
        
        Args:
            image_path: Path to image file
        
        Returns:
            Average brightness (0-255)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            return float(brightness)
        except Exception as e:
            print(f"Error calculating brightness: {e}")
            return 0.0

    @staticmethod
    def get_contrast(image_path: str) -> float:
        """
        Calculate contrast of image (standard deviation of pixel values)
        
        Args:
            image_path: Path to image file
        
        Returns:
            Contrast score (standard deviation)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray)
            return float(contrast)
        except Exception as e:
            print(f"Error calculating contrast: {e}")
            return 0.0

    @staticmethod
    def get_blur_score(image_path: str) -> float:
        """
        Calculate blur score using Laplacian variance
        Higher variance = less blurry
        
        Args:
            image_path: Path to image file
        
        Returns:
            Laplacian variance (blur score)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(laplacian_var)
        except Exception as e:
            print(f"Error calculating blur score: {e}")
            return 0.0

    @staticmethod
    def is_corrupted(image_path: str) -> bool:
        """
        Check if image file is corrupted/unreadable
        
        Args:
            image_path: Path to image file
        
        Returns:
            True if corrupted, False if valid
        """
        try:
            with PILImage.open(image_path) as img:
                img.verify()
            return False
        except Exception:
            return True

    @staticmethod
    def get_image_quality_metrics(image_path: str) -> dict:
        """
        Calculate all image quality metrics
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary of quality metrics
        """
        return {
            "blur_score": ImageUtils.get_blur_score(image_path),
            "brightness": ImageUtils.get_brightness(image_path),
            "contrast": ImageUtils.get_contrast(image_path),
            "is_corrupted": ImageUtils.is_corrupted(image_path),
        }


class PerceptualHash:
    """Perceptual hashing for duplicate image detection"""

    @staticmethod
    def calculate_phash(image_path: str, hash_size: int = 8) -> str:
        """
        Calculate perceptual hash (pHash) of image
        Same/similar images will have similar hashes
        
        Args:
            image_path: Path to image file
            hash_size: Hash resolution (default 8x8)
        
        Returns:
            Hex hash string
        """
        try:
            img = PILImage.open(image_path)
            img = img.convert("L")  # Grayscale
            img = img.resize((hash_size + 1, hash_size), PILImage.LANCZOS)
            
            pixels = list(img.getdata())
            
            # Calculate average
            avg = sum(pixels) / len(pixels)
            
            # Create hash
            hash_bits = "".join("1" if pixel >= avg else "0" for pixel in pixels)
            return format(int(hash_bits, 2), f"0{hash_size * hash_size // 4}x")
        except Exception as e:
            print(f"Error calculating pHash: {e}")
            return ""

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """
        Calculate Hamming distance between two hashes
        Lower distance = more similar
        
        Args:
            hash1: First hash hex string
            hash2: Second hash hex string
        
        Returns:
            Hamming distance (0-64 typically)
        """
        if len(hash1) != len(hash2):
            return 64
        
        xor = int(hash1, 16) ^ int(hash2, 16)
        return bin(xor).count("1")

    @staticmethod
    def is_duplicate(hash1: str, hash2: str, threshold: int = 5) -> bool:
        """
        Check if two images are duplicates based on perceptual hash
        
        Args:
            hash1: First image hash
            hash2: Second image hash
            threshold: Max Hamming distance for duplicate (default 5)
        
        Returns:
            True if images are likely duplicates
        """
        distance = PerceptualHash.hamming_distance(hash1, hash2)
        return distance <= threshold
