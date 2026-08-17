"""
VanRakshak AI - OpenCLIP Semantic Verification
Independent semantic verification layer using shared OpenCLIP model.
Uses the same model instance as the Species Classifier for memory efficiency.
"""

import time
import os
from typing import Dict, List
from dataclasses import dataclass, field
from PIL import Image as PILImage
import torch

from config import settings
from core.openclip_model import SharedOpenCLIP

WILDLIFE_CONCEPTS = {
    "Bengal Tiger": "a camera trap photo of a Bengal tiger in the wild",
    "Indian Leopard": "a camera trap photo of a leopard with spotted fur in the jungle",
    "Snow Leopard": "a camera trap photo of a snow leopard in the mountains",
    "Asiatic Lion": "a camera trap photo of an Asiatic lion",
    "Sambar Deer": "a camera trap photo of a large brown sambar deer",
    "Chital (Spotted Deer)": "a camera trap photo of a spotted chital deer",
    "Wild Boar": "a camera trap photo of a wild boar",
    "Asian Elephant": "a camera trap photo of an Asian elephant",
    "Sloth Bear": "a camera trap photo of a sloth bear",
    "Gaur (Indian Bison)": "a camera trap photo of an Indian gaur wild cattle",
    "Nilgai (Blue Bull)": "a camera trap photo of a nilgai blue bull antelope",
    "Dhole (Wild Dog)": "a camera trap photo of a dhole Asiatic wild dog",
    "Indian Rhinoceros": "a camera trap photo of an Indian rhinoceros",
    "Blackbuck": "a camera trap photo of a blackbuck antelope",
    "Macaque": "a camera trap photo of a rhesus macaque monkey",
    "Langur": "a camera trap photo of a gray langur monkey",
    "Human": "a photo of a person or human",
    "Vehicle": "a photo of a vehicle car truck",
}

CONCEPT_SPECIES = list(WILDLIFE_CONCEPTS.keys())
CONCEPT_PROMPTS = list(WILDLIFE_CONCEPTS.values())

@dataclass
class SemanticScore:
    concept: str
    similarity: float

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "similarity": round(self.similarity, 4),
        }

@dataclass
class SemanticVerificationResult:
    image_id: str
    primary_prediction: str
    primary_similarity: float
    scores: Dict[str, float] = field(default_factory=dict)
    agrees_with_speciesnet: bool = True
    agreement_score: float = 0.0
    model_name: str = "openclip_vit_b_32"
    processing_time_ms: float = 0.0
    passes_threshold: bool = True
    confidence_level: str = "high"

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "primary_prediction": self.primary_prediction,
            "primary_similarity": round(self.primary_similarity, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "agrees_with_speciesnet": self.agrees_with_speciesnet,
            "agreement_score": round(self.agreement_score, 4),
            "model_name": self.model_name,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "passes_threshold": self.passes_threshold,
            "confidence_level": self.confidence_level,
        }

    @property
    def top_predictions(self) -> List[SemanticScore]:
        sorted_concepts = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        return [SemanticScore(concept=k, similarity=v) for k, v in sorted_concepts]


class SemanticVerifierService:
    """Uses the shared OpenCLIP model singleton — no duplicate model loading."""
    
    _text_features = None

    @classmethod
    def _get_text_features(cls, model, tokenizer, device):
        """Pre-compute and cache text features for verification concepts."""
        if cls._text_features is None:
            with torch.no_grad():
                text = tokenizer(CONCEPT_PROMPTS).to(device)
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                cls._text_features = text_features
        return cls._text_features

    @staticmethod
    def verify(
        image_path: str,
        speciesnet_prediction: str,
        image_id: str = "",
    ) -> SemanticVerificationResult:
        start_time = time.time()

        model, preprocess, tokenizer = SharedOpenCLIP.get_model()
        device = SharedOpenCLIP.get_device()

        if not model or not os.path.exists(image_path):
            return SemanticVerificationResult(
                image_id=image_id,
                primary_prediction="Unknown",
                primary_similarity=0.0
            )

        try:
            image = PILImage.open(image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            if device in ["cuda", "mps"]:
                try:
                    image_input = image_input.half()
                except Exception:
                    pass

            with torch.no_grad():
                text_features = SemanticVerifierService._get_text_features(model, tokenizer, device)

                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                similarity = similarity.squeeze().cpu().float().numpy()

            scores = {}
            for i, species in enumerate(CONCEPT_SPECIES):
                scores[species] = float(similarity[i])

            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            primary_prediction = sorted_scores[0][0]
            primary_similarity = sorted_scores[0][1]

            # Normalize SpeciesNet prediction for matching
            sn_pred = speciesnet_prediction
            if "Tiger" in sn_pred:
                sn_pred = "Bengal Tiger"
            elif "Leopard" in sn_pred and "Snow" not in sn_pred:
                sn_pred = "Indian Leopard"

            agreement_score = scores.get(sn_pred, 0.0)
            agrees_with_speciesnet = (primary_prediction == sn_pred) or (agreement_score > 0.1)

        except Exception as e:
            print(f"OpenCLIP verification error: {e}")
            primary_prediction = "Error"
            primary_similarity = 0.0
            scores = {}
            agrees_with_speciesnet = False
            agreement_score = 0.0

        passes_threshold = primary_similarity >= settings.SEMANTIC_CONFIDENCE_THRESHOLD

        if primary_similarity >= settings.HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = "high"
        elif primary_similarity >= settings.LOW_CONFIDENCE_THRESHOLD:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        processing_time = (time.time() - start_time) * 1000

        return SemanticVerificationResult(
            image_id=image_id,
            primary_prediction=primary_prediction,
            primary_similarity=primary_similarity,
            scores=scores,
            agrees_with_speciesnet=agrees_with_speciesnet,
            agreement_score=agreement_score,
            model_name="openclip_vit_b_32",
            processing_time_ms=processing_time,
            passes_threshold=passes_threshold,
            confidence_level=confidence_level,
        )
