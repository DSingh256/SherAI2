"""
VanRakshak AI - SpeciesNet Classification
Species classification for detected animals.

Takes cropped animal detections from MegaDetector and predicts species
with confidence scores and top-K alternatives.

Simulated mode uses a pool of 15+ Indian wildlife species with weighted
random selection seeded by image hash for reproducibility.
"""

import hashlib
import random
import time
import os
import pickle
import numpy as np
import cv2
from PIL import Image as PILImage
from typing import List, Dict, Optional
from dataclasses import dataclass, field


# Indian wildlife species pool with relative abundance weights
SPECIES_POOL = {
    "Bengal Tiger": 0.08,
    "Indian Leopard": 0.07,
    "Sambar Deer": 0.15,
    "Spotted Deer (Chital)": 0.16,
    "Wild Boar": 0.12,
    "Asian Elephant": 0.05,
    "Sloth Bear": 0.04,
    "Indian Gaur": 0.06,
    "Nilgai": 0.05,
    "Indian Muntjac (Barking Deer)": 0.04,
    "Langur": 0.06,
    "Rhesus Macaque": 0.04,
    "Indian Porcupine": 0.02,
    "Jungle Cat": 0.02,
    "Indian Hare": 0.02,
    "Peafowl": 0.02,
}

# Tiger-specific species for confusion matrix (species commonly confused with tigers)
TIGER_CONFUSION = {
    "Bengal Tiger": 0.70,
    "Indian Leopard": 0.15,
    "Jungle Cat": 0.05,
    "Sambar Deer": 0.03,
    "Other": 0.07,
}


from config import settings


@dataclass
class SpeciesPrediction:
    """A single species prediction with confidence"""
    species: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "species": self.species,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class SpeciesClassificationResult:
    """Complete classification result for one detection"""
    detection_id: str
    primary_species: str
    primary_confidence: float
    alternatives: List[SpeciesPrediction] = field(default_factory=list)
    is_tiger: bool = False
    model_name: str = "speciesnet_v1_simulated"
    processing_time_ms: float = 0.0
    passes_threshold: bool = True
    confidence_level: str = "high"
    requires_human_review: bool = False

    def to_dict(self) -> dict:
        return {
            "detection_id": self.detection_id,
            "primary_species": self.primary_species,
            "primary_confidence": round(self.primary_confidence, 4),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "is_tiger": self.is_tiger,
            "model_name": self.model_name,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "passes_threshold": self.passes_threshold,
            "confidence_level": self.confidence_level,
            "requires_human_review": self.requires_human_review,
        }

    @property
    def top_predictions(self) -> List[SpeciesPrediction]:
        """All predictions including primary, sorted by confidence"""
        all_preds = [SpeciesPrediction(self.primary_species, self.primary_confidence)]
        all_preds.extend(self.alternatives)
        return sorted(all_preds, key=lambda p: p.confidence, reverse=True)


class SpeciesClassifierService:
    """
    SpeciesNet species classification service.

    Simulated mode generates realistic species predictions
    for Indian wildlife based on image characteristics.
    """

    @staticmethod
    def preprocess_image(image_path: str) -> Optional[List[float]]:
        """
        Validates and preprocesses an image crop into standardized feature vector.
        Standardizes to 224x224 RGB image representation and extracts:
        [mean_r, mean_g, mean_b, std_r, std_g, std_b, contrast, brightness, width, height, aspect_ratio]
        """
        if not image_path or not os.path.exists(image_path):
            return None
        
        try:
            img_pil = PILImage.open(image_path)
            w, h = img_pil.size
            aspect_ratio = float(w) / float(h) if h > 0 else 1.0
            
            # Standardize crop size for feature consistency
            resized = img_pil.resize((224, 224), PILImage.BILINEAR)
            img_rgb = np.array(resized.convert("RGB"))
            
            mean_r = float(np.mean(img_rgb[:, :, 0]))
            mean_g = float(np.mean(img_rgb[:, :, 1]))
            mean_b = float(np.mean(img_rgb[:, :, 2]))
            std_r = float(np.std(img_rgb[:, :, 0]))
            std_g = float(np.std(img_rgb[:, :, 1]))
            std_b = float(np.std(img_rgb[:, :, 2]))
            
            gray = np.mean(img_rgb, axis=2)
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            return [
                mean_r, mean_g, mean_b,
                std_r, std_g, std_b,
                contrast, brightness,
                float(w), float(h), aspect_ratio
            ]
        except Exception:
            return None

    @staticmethod
    def classify(
        image_path: str,
        detection_id: str = "",
        top_k: int = 5,
    ) -> SpeciesClassificationResult:
        """
        Classify species from a cropped animal image.

        Args:
            image_path: Path to cropped animal image
            detection_id: Detection identifier
            top_k: Number of top predictions to return

        Returns:
            SpeciesClassificationResult with species predictions
        """
        start_time = time.time()

        # 1. Preprocessing & Input Validation
        features = SpeciesClassifierService.preprocess_image(image_path)

        # 2. Check if pickled model exists
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pkl_path = os.path.join(base_dir, "models", "speciesnet.pkl")
        if not os.path.exists(pkl_path):
            pkl_path = os.path.join("models", "speciesnet.pkl")

        if features is not None and os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    model_dict = pickle.load(f)
                
                clf = model_dict["classifier"]
                species_list = model_dict["species_list"]
                
                probs = clf.predict_proba([features])[0]
                top_indices = np.argsort(probs)[::-1]
                
                primary_idx = top_indices[0]
                primary_species = species_list[primary_idx]
                primary_confidence = float(probs[primary_idx])
                
                alternatives = []
                for idx in top_indices[1:top_k]:
                    alternatives.append(SpeciesPrediction(
                        species=species_list[idx],
                        confidence=float(probs[idx])
                    ))
                
                is_tiger = primary_species == "Bengal Tiger"
                passes_threshold = primary_confidence >= settings.SPECIESNET_CONFIDENCE_THRESHOLD
                if primary_confidence >= settings.HIGH_CONFIDENCE_THRESHOLD:
                    confidence_level = "high"
                elif primary_confidence >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
                    confidence_level = "medium"
                else:
                    confidence_level = "low"
                
                requires_human_review = not passes_threshold or (is_tiger and confidence_level != "high")
                
                return SpeciesClassificationResult(
                    detection_id=detection_id,
                    primary_species=primary_species,
                    primary_confidence=primary_confidence,
                    alternatives=alternatives,
                    is_tiger=is_tiger,
                    model_name="speciesnet_random_forest_v1",
                    processing_time_ms=(time.time() - start_time) * 1000,
                    passes_threshold=passes_threshold,
                    confidence_level=confidence_level,
                    requires_human_review=requires_human_review
                )
            except Exception as e:
                print(f"Error classifying with pickle model: {e}. Falling back to default simulation.")

        # Seed RNG for deterministic results (Fallback)
        seed = SpeciesClassifierService._get_seed(image_path or "empty", detection_id)
        rng = random.Random(seed)

        # Select primary species using weighted distribution
        species_list = list(SPECIES_POOL.keys())
        weights = list(SPECIES_POOL.values())
        primary_species = rng.choices(species_list, weights=weights, k=1)[0]

        # Generate primary confidence
        if primary_species == "Bengal Tiger":
            primary_confidence = rng.uniform(0.60, 0.98)
        elif primary_species == "Indian Leopard":
            primary_confidence = rng.uniform(0.55, 0.95)
        else:
            primary_confidence = rng.uniform(0.70, 0.97)

        # Generate alternative predictions
        alternatives = SpeciesClassifierService._generate_alternatives(
            rng, primary_species, primary_confidence, top_k - 1
        )

        is_tiger = primary_species == "Bengal Tiger"
        passes_threshold = primary_confidence >= settings.SPECIESNET_CONFIDENCE_THRESHOLD
        if primary_confidence >= settings.HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = "high"
        elif primary_confidence >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        requires_human_review = not passes_threshold or (is_tiger and confidence_level != "high")
        simulated_time = rng.uniform(150, 500)

        return SpeciesClassificationResult(
            detection_id=detection_id,
            primary_species=primary_species,
            primary_confidence=primary_confidence,
            alternatives=alternatives,
            is_tiger=is_tiger,
            model_name="speciesnet_v1_simulated",
            processing_time_ms=simulated_time,
            passes_threshold=passes_threshold,
            confidence_level=confidence_level,
            requires_human_review=requires_human_review
        )

    @staticmethod
    def classify_with_tiger_bias(
        image_path: str,
        detection_id: str = "",
        top_k: int = 5,
    ) -> SpeciesClassificationResult:
        """
        Classify with higher probability of tiger detection.
        Used for demo/testing purposes.
        """
        seed = SpeciesClassifierService._get_seed(image_path, detection_id)
        rng = random.Random(seed)

        # Use tiger confusion matrix for more tiger results
        species_list = list(TIGER_CONFUSION.keys())
        weights = list(TIGER_CONFUSION.values())
        primary_species = rng.choices(species_list, weights=weights, k=1)[0]

        if primary_species == "Other":
            # Pick from general pool excluding big cats
            other_species = [s for s in SPECIES_POOL.keys()
                           if s not in ["Bengal Tiger", "Indian Leopard", "Jungle Cat"]]
            primary_species = rng.choice(other_species)

        primary_confidence = rng.uniform(0.65, 0.98)

        alternatives = SpeciesClassifierService._generate_alternatives(
            rng, primary_species, primary_confidence, top_k - 1
        )

        is_tiger = primary_species == "Bengal Tiger"

        return SpeciesClassificationResult(
            detection_id=detection_id,
            primary_species=primary_species,
            primary_confidence=primary_confidence,
            alternatives=alternatives,
            is_tiger=is_tiger,
            processing_time_ms=rng.uniform(150, 500),
        )

    @staticmethod
    def _generate_alternatives(
        rng: random.Random,
        primary_species: str,
        primary_confidence: float,
        num_alternatives: int,
    ) -> List[SpeciesPrediction]:
        """Generate alternative species predictions"""
        alternatives = []
        remaining_confidence = 1.0 - primary_confidence

        # Get other species (excluding primary)
        other_species = [s for s in SPECIES_POOL.keys() if s != primary_species]
        rng.shuffle(other_species)

        for i, species in enumerate(other_species[:num_alternatives]):
            if i == num_alternatives - 1:
                # Last alternative gets remaining confidence
                conf = remaining_confidence
            else:
                # Distribute remaining confidence
                conf = remaining_confidence * rng.uniform(0.2, 0.6)
                remaining_confidence -= conf

            alternatives.append(SpeciesPrediction(
                species=species,
                confidence=max(0.001, conf),
            ))

        # Sort by confidence descending
        alternatives.sort(key=lambda p: p.confidence, reverse=True)
        return alternatives

    @staticmethod
    def _get_seed(image_path: str, detection_id: str) -> int:
        """Generate deterministic seed"""
        seed_str = f"speciesnet_{image_path}_{detection_id}"
        return int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
