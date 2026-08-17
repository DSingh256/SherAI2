"""
VanRakshak AI - Tiger Re-identification
Potential tiger re-identification via embeddings.

Uses image embeddings (simulated via hashing/random vectors for hackathon)
to find similar past tiger sightings.

IMPORTANT: Always labeled as "potential re-identification" requiring human validation.
"""

import hashlib
import random
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    import numpy as np
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss-cpu not installed, re-id will use basic similarity")


@dataclass
class ReIDMatch:
    """A potential re-identification match"""
    match_image_id: str
    match_camera_id: str
    match_timestamp: datetime
    similarity_score: float
    verified: bool = False

    def to_dict(self) -> dict:
        return {
            "match_image_id": self.match_image_id,
            "match_camera_id": self.match_camera_id,
            "match_timestamp": self.match_timestamp.isoformat() if self.match_timestamp else None,
            "similarity_score": round(self.similarity_score, 4),
            "verified": self.verified,
        }


@dataclass
class ReIDResult:
    """Complete re-identification result"""
    image_id: str
    matches: List[ReIDMatch] = field(default_factory=list)
    has_matches: bool = False
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "matches": [m.to_dict() for m in self.matches],
            "has_matches": self.has_matches,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class ReIdentificationService:
    """
    Tiger re-identification service.
    
    Simulates embedding generation and FAISS vector search.
    Generates realistic "potential matches" for demo purposes.
    """
    
    # Configurable similarity threshold
    SIMILARITY_THRESHOLD = 0.85
    
    @staticmethod
    def extract_embedding(image_path: str, detection_bbox: dict = None) -> List[float]:
        """
        Simulate extracting a 512-d embedding vector for a tiger.
        In production, this would be a ResNet/EfficientNet feature extractor.
        """
        # Generate a deterministic but pseudo-random vector based on file path
        seed = int(hashlib.md5(image_path.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        
        # Generate 512-d vector and normalize it
        vector = rng.randn(512).astype(np.float32)
        faiss.normalize_L2(np.array([vector])) if FAISS_AVAILABLE else None
        # Basic normalization if faiss isn't available
        if not FAISS_AVAILABLE:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
                
        return vector.tolist()

    @staticmethod
    def search_similar(
        image_id: str,
        embedding: List[float],
        camera_id: str,
        timestamp: datetime,
        top_k: int = 3
    ) -> ReIDResult:
        """
        Search for similar tigers using the embedding.
        
        For the hackathon, we simulate a vector search that sometimes finds
        matches and sometimes doesn't, based on the image_id hash.
        
        Args:
            image_id: Source image ID
            embedding: 512-d embedding vector
            camera_id: Source camera ID
            timestamp: Source timestamp
            top_k: Max matches to return
            
        Returns:
            ReIDResult with potential matches
        """
        start_time = time.time()
        
        seed = int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        
        matches = []
        
        # ~30% chance of finding a match for demo purposes
        if rng.random() < 0.30:
            num_matches = rng.randint(1, top_k)
            
            for i in range(num_matches):
                # Generate realistic similarity score (0.85 to 0.98)
                similarity = rng.uniform(
                    ReIdentificationService.SIMILARITY_THRESHOLD, 0.98
                )
                
                # Generate a plausible past timestamp (1 to 30 days ago)
                days_ago = rng.randint(1, 30)
                hours_ago = rng.randint(0, 23)
                if timestamp:
                    past_ts = datetime.fromtimestamp(
                        timestamp.timestamp() - (days_ago * 86400) - (hours_ago * 3600)
                    )
                else:
                    past_ts = datetime.now()
                
                # Pick a camera (could be same or different)
                cameras = ["CAM001", "CAM004", "CAM007", "CAM008", "CAM012", "CAM014", "CAM017", "CAM022"]
                # 60% chance it's the same camera
                match_camera = camera_id if rng.random() < 0.6 else rng.choice(cameras)
                
                matches.append(ReIDMatch(
                    match_image_id=f"sim_match_{seed}_{i}",
                    match_camera_id=match_camera,
                    match_timestamp=past_ts,
                    similarity_score=similarity,
                    verified=False
                ))
                
            # Sort by similarity
            matches.sort(key=lambda m: m.similarity_score, reverse=True)
            
        processing_time = (time.time() - start_time) * 1000
        simulated_time = rng.uniform(50, 150) # FAISS search is fast
            
        return ReIDResult(
            image_id=image_id,
            matches=matches,
            has_matches=len(matches) > 0,
            processing_time_ms=simulated_time
        )
