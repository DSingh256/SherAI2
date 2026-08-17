"""
VanRakshak AI - OpenCLIP Semantic Verification
Independent semantic verification layer using image-text similarity.

Compares image embeddings against text concept prompts to provide
an independent cross-check of SpeciesNet predictions.

Agreement between SpeciesNet and OpenCLIP increases overall confidence.
Disagreement triggers human review.
"""

import hashlib
import random
import time
from typing import Dict, Optional
from dataclasses import dataclass, field


# Text prompts for semantic matching
WILDLIFE_CONCEPTS = {
    "Bengal Tiger": [
        "a Bengal tiger in the wild",
        "a tiger walking through forest",
        "orange and black striped big cat",
    ],
    "Indian Leopard": [
        "a leopard with spotted fur",
        "a leopard in forest",
    ],
    "Sambar Deer": [
        "a large brown deer",
        "sambar deer in forest",
    ],
    "Spotted Deer (Chital)": [
        "a spotted deer with white spots",
        "chital deer grazing",
    ],
    "Wild Boar": [
        "a wild boar",
        "wild pig in forest",
    ],
    "Asian Elephant": [
        "an Asian elephant",
        "elephant in forest",
    ],
    "Sloth Bear": [
        "a sloth bear",
        "black bear with white chest mark",
    ],
    "Indian Gaur": [
        "an Indian gaur",
        "large wild cattle",
    ],
    "Human": [
        "a person walking",
        "human figure in forest",
    ],
    "Vehicle": [
        "a vehicle on road",
        "car or truck",
    ],
}

# Simplified concept list for scoring
CONCEPT_SPECIES = list(WILDLIFE_CONCEPTS.keys())


@dataclass
class SemanticScore:
    """Semantic similarity score for a concept"""
    concept: str
    similarity: float

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "similarity": round(self.similarity, 4),
        }


@dataclass
class SemanticVerificationResult:
    """Complete semantic verification result"""
    image_id: str
    primary_prediction: str
    primary_similarity: float
    scores: Dict[str, float] = field(default_factory=dict)
    agrees_with_speciesnet: bool = True
    agreement_score: float = 0.0
    model_name: str = "openclip_simulated"
    processing_time_ms: float = 0.0

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
        }


class SemanticVerifierService:
    """
    OpenCLIP semantic verification service.

    Provides independent semantic verification of species classifications
    using image-text similarity scoring.

    Simulated mode generates correlated but not identical scores to
    SpeciesNet, with ~85% agreement rate for realistic demo scenarios.
    """

    AGREEMENT_RATE = 0.85  # How often OpenCLIP agrees with SpeciesNet

    @staticmethod
    def verify(
        image_path: str,
        image_id: str = "",
        speciesnet_prediction: str = "",
        speciesnet_confidence: float = 0.0,
    ) -> SemanticVerificationResult:
        """
        Run semantic verification on an image.

        Args:
            image_path: Path to the image
            image_id: Image identifier
            speciesnet_prediction: What SpeciesNet predicted (for agreement check)
            speciesnet_confidence: SpeciesNet's confidence

        Returns:
            SemanticVerificationResult with similarity scores
        """
        start_time = time.time()

        seed = SemanticVerifierService._get_seed(image_path, image_id)
        rng = random.Random(seed)

        # Decide if OpenCLIP will agree with SpeciesNet
        will_agree = rng.random() < SemanticVerifierService.AGREEMENT_RATE

        # Generate similarity scores for all concepts
        scores = {}

        if will_agree and speciesnet_prediction:
            # Agreement scenario: OpenCLIP's top prediction matches SpeciesNet
            scores = SemanticVerifierService._generate_agreeing_scores(
                rng, speciesnet_prediction, speciesnet_confidence
            )
        elif speciesnet_prediction:
            # Disagreement scenario: OpenCLIP picks a different top prediction
            scores = SemanticVerifierService._generate_disagreeing_scores(
                rng, speciesnet_prediction
            )
        else:
            # No SpeciesNet prediction to compare against
            scores = SemanticVerifierService._generate_independent_scores(rng)

        # Find the top prediction from OpenCLIP
        primary_prediction = max(scores, key=scores.get)
        primary_similarity = scores[primary_prediction]

        # Calculate agreement
        agrees = primary_prediction == speciesnet_prediction
        if speciesnet_prediction and speciesnet_prediction in scores:
            agreement_score = scores[speciesnet_prediction]
        else:
            agreement_score = 0.0

        simulated_time = rng.uniform(300, 900)

        return SemanticVerificationResult(
            image_id=image_id,
            primary_prediction=primary_prediction,
            primary_similarity=primary_similarity,
            scores=scores,
            agrees_with_speciesnet=agrees,
            agreement_score=agreement_score,
            processing_time_ms=simulated_time,
        )

    @staticmethod
    def _generate_agreeing_scores(
        rng: random.Random,
        speciesnet_prediction: str,
        speciesnet_confidence: float,
    ) -> Dict[str, float]:
        """Generate scores where OpenCLIP agrees with SpeciesNet"""
        scores = {}

        # The agreed-upon species gets a high but slightly different similarity
        offset = rng.uniform(-0.08, 0.05)
        target_score = min(0.99, max(0.5, speciesnet_confidence + offset))
        scores[speciesnet_prediction] = target_score

        # Other species get lower scores
        for concept in CONCEPT_SPECIES:
            if concept not in scores:
                # Generate a score lower than the primary
                max_other = target_score - 0.15
                scores[concept] = rng.uniform(0.05, max(0.06, max_other))

        return scores

    @staticmethod
    def _generate_disagreeing_scores(
        rng: random.Random,
        speciesnet_prediction: str,
    ) -> Dict[str, float]:
        """Generate scores where OpenCLIP disagrees with SpeciesNet"""
        scores = {}

        # Pick a different species as OpenCLIP's top prediction
        other_species = [s for s in CONCEPT_SPECIES
                        if s != speciesnet_prediction and s not in ["Human", "Vehicle"]]
        top_pick = rng.choice(other_species) if other_species else speciesnet_prediction

        # Top pick gets moderate-to-high similarity
        scores[top_pick] = rng.uniform(0.55, 0.85)

        # SpeciesNet's prediction gets a similar but lower score (close contest)
        scores[speciesnet_prediction] = scores[top_pick] - rng.uniform(0.05, 0.20)

        # Fill in remaining concepts
        for concept in CONCEPT_SPECIES:
            if concept not in scores:
                scores[concept] = rng.uniform(0.05, 0.35)

        return scores

    @staticmethod
    def _generate_independent_scores(rng: random.Random) -> Dict[str, float]:
        """Generate independent scores without SpeciesNet reference"""
        scores = {}
        for concept in CONCEPT_SPECIES:
            scores[concept] = rng.uniform(0.05, 0.80)
        return scores

    @staticmethod
    def _get_seed(image_path: str, image_id: str) -> int:
        """Generate deterministic seed"""
        seed_str = f"openclip_{image_path}_{image_id}"
        return int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
