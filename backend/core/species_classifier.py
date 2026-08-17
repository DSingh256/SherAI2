"""
VanRakshak AI - Species Classification
Uses OpenCLIP Zero-Shot classification for highly accurate specialized wildlife detection.
"""

import time
import os
from typing import List
from dataclasses import dataclass, field
from PIL import Image as PILImage
import torch

from config import settings
from core.openclip_model import SharedOpenCLIP

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
    model_name: str = "openclip_vit_b32_zeroshot"
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
        all_preds = [SpeciesPrediction(self.primary_species, self.primary_confidence)]
        all_preds.extend(self.alternatives)
        return sorted(all_preds, key=lambda p: p.confidence, reverse=True)


class SpeciesClassifierService:
    """
    Highly efficient Zero-Shot Species Classification using OpenCLIP.
    """

    # Target specialized species list for Indian wildlife
    SPECIES_LIST = [
        "Bengal Tiger",
        "Indian Leopard",
        "Snow Leopard",
        "Asiatic Lion",
        "Asian Elephant",
        "Gaur (Indian Bison)",
        "Sambar Deer",
        "Chital (Spotted Deer)",
        "Nilgai (Blue Bull)",
        "Sloth Bear",
        "Wild Boar",
        "Dhole (Wild Dog)",
        "Indian Rhinoceros",
        "Blackbuck",
        "Macaque",
        "Langur"
    ]
    
    # Pre-computed text prompts
    _TEXT_PROMPTS = [f"A camera trap photo of a {species} in the wild" for species in SPECIES_LIST]
    _text_features = None

    @classmethod
    def _get_text_features(cls, model, tokenizer, device):
        if cls._text_features is None:
            with torch.no_grad():
                text = tokenizer(cls._TEXT_PROMPTS).to(device)
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                cls._text_features = text_features
        return cls._text_features

    @staticmethod
    def classify(
        image_path: str,
        detection_id: str = "",
        top_k: int = 5,
    ) -> SpeciesClassificationResult:
        start_time = time.time()
        
        model, preprocess, tokenizer = SharedOpenCLIP.get_model()
        device = SharedOpenCLIP.get_device()
        
        if not model or not os.path.exists(image_path):
            return SpeciesClassificationResult(
                detection_id=detection_id,
                primary_species="Unknown",
                primary_confidence=0.0
            )
            
        try:
            img_pil = PILImage.open(image_path).convert("RGB")
            image = preprocess(img_pil).unsqueeze(0).to(device)
            
            # Use half precision on GPU/MPS for efficiency
            if device in ["cuda", "mps"]:
                try:
                    image = image.half()
                except Exception:
                    pass
            
            with torch.no_grad():
                text_features = SpeciesClassifierService._get_text_features(model, tokenizer, device)
                
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Cosine similarity as logits
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probabilities = similarity[0]
                
            top_prob, top_indices = torch.topk(probabilities, min(top_k, len(SpeciesClassifierService.SPECIES_LIST)))
            
            predictions = []
            for i in range(top_indices.size(0)):
                idx = top_indices[i].item()
                prob = top_prob[i].item()
                species = SpeciesClassifierService.SPECIES_LIST[idx]
                predictions.append(SpeciesPrediction(species, prob))
                
            if not predictions:
                predictions.append(SpeciesPrediction("Other / Unknown", 0.0))
                
        except Exception as e:
            print(f"Classification error: {e}")
            predictions = [SpeciesPrediction("Error", 0.0)]
            
        primary = predictions[0]
        alternatives = predictions[1:]
        
        is_tiger = "Tiger" in primary.species
        
        passes_threshold = primary.confidence >= settings.SPECIESNET_CONFIDENCE_THRESHOLD
        
        if primary.confidence >= settings.HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = "high"
            requires_review = False
        elif primary.confidence >= settings.LOW_CONFIDENCE_THRESHOLD:
            confidence_level = "medium"
            requires_review = True
        else:
            confidence_level = "low"
            requires_review = True

        processing_time = (time.time() - start_time) * 1000

        return SpeciesClassificationResult(
            detection_id=detection_id,
            primary_species=primary.species,
            primary_confidence=primary.confidence,
            alternatives=alternatives,
            is_tiger=is_tiger,
            model_name="openclip_vit_b32_zeroshot",
            processing_time_ms=processing_time,
            passes_threshold=passes_threshold,
            confidence_level=confidence_level,
            requires_human_review=requires_review,
        )
