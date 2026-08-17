import os
import pickle
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Ensure target models directory exists
os.makedirs("models", exist_ok=True)

print("🚀 Starting mock model generation and training...")

# ==========================================
# 1. MEGADETECTOR MODEL SERIALIZATION
# ==========================================
print("📝 Creating MegaDetector model...")

class MockMegaDetectorModel:
    """Mock MegaDetector V6 model structure for serialization"""
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.categories = ["animal", "human", "vehicle"]
        # Different scenario profiles for mock data generation
        self.profiles = ["animal_only", "multi_animal", "animal_human", "human_only", "vehicle_only", "empty"]
        self.profile_weights = [0.55, 0.15, 0.05, 0.08, 0.04, 0.13]

    def predict(self, image_path: str, image_id: str = "") -> dict:
        import hashlib
        # Deterministic seeding based on image identifiers
        seed_str = f"{image_path}_{image_id}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        
        profile = rng.choices(self.profiles, weights=self.profile_weights, k=1)[0]
        detections = []
        
        if profile == "empty":
            pass
        elif profile == "multi_animal":
            num_animals = rng.randint(2, 4)
            for i in range(num_animals):
                conf = rng.uniform(0.75, 0.99)
                bbox = self._gen_bbox(rng, i, num_animals)
                detections.append({
                    "category": "animal",
                    "confidence": conf,
                    "bbox": bbox
                })
        else:
            categories_to_check = []
            if "animal" in profile:
                categories_to_check.append("animal")
            if "human" in profile:
                categories_to_check.append("human")
            if "vehicle" in profile:
                categories_to_check.append("vehicle")
                
            for cat in categories_to_check:
                conf = rng.uniform(0.60, 0.98)
                if conf >= self.confidence_threshold:
                    bbox = self._gen_bbox(rng)
                    detections.append({
                        "category": cat,
                        "confidence": conf,
                        "bbox": bbox
                    })
        return {
            "detections": detections,
            "processing_time_ms": rng.uniform(180, 420)
        }

    def _gen_bbox(self, rng, index=0, total=1):
        if total == 1:
            cx = rng.uniform(0.3, 0.7)
            cy = rng.uniform(0.3, 0.7)
            w = rng.uniform(0.2, 0.4)
            h = rng.uniform(0.2, 0.5)
        else:
            segment = 1.0 / total
            cx = segment * index + segment * rng.uniform(0.3, 0.7)
            cy = rng.uniform(0.3, 0.7)
            w = rng.uniform(0.12, 0.22)
            h = rng.uniform(0.18, 0.38)
            
        return [
            max(0.0, cx - w/2),
            max(0.0, cy - h/2),
            min(1.0, cx + w/2),
            min(1.0, cy + h/2)
        ]

megadetector_model = MockMegaDetectorModel()
with open("models/megadetector.pkl", "wb") as f:
    pickle.dump(megadetector_model, f)
print("✓ MegaDetector model saved to models/megadetector.pkl")


# ==========================================
# 2. SPECIESNET MODEL TRAINING
# ==========================================
print("\n📝 Training SpeciesNet RandomForest model...")

# 15 Target Indian reserve species
SPECIES_LIST = [
    "Bengal Tiger", "Indian Leopard", "Asian Elephant", "Sambar Deer", 
    "Spotted Deer", "Wild Boar", "Sloth Bear", "Indian Gaur", 
    "Nilgai", "Golden Jackal", "Dhole", "Striped Hyena", 
    "Jungle Cat", "Rhesus Macaque", "Common Langur"
]

# Generate synthetic features for model training
# Features: [mean_r, mean_g, mean_b, std_r, std_g, std_b, contrast, brightness, width, height, aspect_ratio]
np.random.seed(42)
X_data = []
y_data = []

# Define statistical feature profiles (color, contrast, dimensions) for each species
PROFILES = {
    # Tiger: Orange-ish (high R, medium G, low B), high contrast (stripes), medium-large size
    "Bengal Tiger":   {"mean": [180, 110, 50, 45, 35, 25, 60, 115, 450, 350, 1.3], "std": [15, 10, 10, 5, 5, 5, 8, 10, 50, 40, 0.1]},
    # Leopard: Golden-ish, high contrast (spots)
    "Indian Leopard": {"mean": [165, 125, 65, 40, 30, 20, 55, 118, 380, 280, 1.35], "std": [15, 10, 10, 5, 5, 5, 8, 10, 40, 30, 0.1]},
    # Elephant: Gray (balanced channels), low contrast, very large
    "Asian Elephant": {"mean": [110, 110, 110, 15, 15, 15, 20, 110, 950, 800, 1.2], "std": [10, 10, 10, 3, 3, 3, 4, 8, 80, 70, 0.05]},
    # Sambar Deer: Dark brown, moderate size
    "Sambar Deer":    {"mean": [100, 75, 50, 25, 20, 15, 30, 75, 350, 420, 0.83], "std": [12, 8, 6, 4, 3, 2, 5, 8, 35, 40, 0.08]},
    # Spotted Deer: Light brown/spotted
    "Spotted Deer":   {"mean": [130, 95, 60, 35, 25, 18, 45, 95, 260, 280, 0.93], "std": [12, 10, 8, 4, 4, 3, 6, 8, 25, 30, 0.08]},
    # Wild Boar: Very dark/black, low height
    "Wild Boar":      {"mean": [60, 55, 50, 12, 10, 8, 15, 55, 280, 190, 1.47], "std": [8, 8, 8, 2, 2, 2, 3, 6, 30, 20, 0.1]},
    # Sloth Bear: Shaggy black (low values, low contrast), large
    "Sloth Bear":     {"mean": [45, 45, 45, 10, 10, 10, 12, 45, 420, 380, 1.1], "std": [6, 6, 6, 2, 2, 2, 3, 5, 45, 40, 0.07]},
    # Indian Gaur: Very dark/black but massive
    "Indian Gaur":    {"mean": [65, 60, 55, 15, 15, 12, 22, 60, 800, 750, 1.07], "std": [8, 8, 8, 3, 3, 3, 4, 6, 70, 65, 0.06]},
    # Nilgai: Grayish-blue, tall
    "Nilgai":         {"mean": [95, 100, 105, 20, 20, 20, 25, 100, 400, 480, 0.83], "std": [10, 10, 10, 3, 3, 3, 4, 8, 40, 45, 0.07]},
    # Jackal: Sandy brown, small
    "Golden Jackal":  {"mean": [140, 110, 80, 25, 20, 15, 35, 110, 220, 160, 1.38], "std": [12, 9, 8, 3, 3, 2, 5, 8, 20, 15, 0.1]},
    # Dhole: Reddish-brown, small
    "Dhole":          {"mean": [155, 85, 45, 30, 20, 12, 38, 95, 210, 150, 1.4], "std": [12, 8, 6, 3, 2, 2, 5, 8, 20, 15, 0.1]},
    # Hyena: Striped gray, sloped back
    "Striped Hyena":  {"mean": [115, 110, 100, 30, 25, 20, 42, 108, 290, 250, 1.16], "std": [10, 10, 9, 4, 3, 3, 6, 8, 30, 25, 0.08]},
    # Jungle Cat: Sandy small cat
    "Jungle Cat":     {"mean": [135, 115, 90, 22, 18, 15, 28, 113, 150, 120, 1.25], "std": [10, 8, 8, 3, 2, 2, 4, 8, 15, 12, 0.08]},
    # Macaque: Brown/tan monkey, small
    "Rhesus Macaque": {"mean": [120, 105, 85, 18, 15, 12, 24, 103, 130, 130, 1.0], "std": [9, 8, 7, 2, 2, 2, 3, 6, 12, 12, 0.05]},
    # Langur: Gray body, black face, tall/long tail
    "Common Langur":  {"mean": [105, 105, 105, 28, 25, 22, 36, 105, 140, 220, 0.64], "std": [8, 8, 8, 3, 3, 2, 5, 6, 15, 20, 0.05]},
}

# Build dataset of 150 samples per class
for label_idx, species in enumerate(SPECIES_LIST):
    profile = PROFILES[species]
    for _ in range(150):
        sample = []
        for mean, std in zip(profile["mean"], profile["std"]):
            val = np.random.normal(mean, std)
            sample.append(max(0.0, val))
        X_data.append(sample)
        y_data.append(label_idx)

X_data = np.array(X_data)
y_data = np.array(y_data)

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42)
rf_clf.fit(X_data, y_data)

# Check training accuracy
train_acc = rf_clf.score(X_data, y_data)
print(f"✓ Model trained successfully. Training Accuracy: {train_acc * 100:.2f}%")

# Save model object along with classes map
speciesnet_model = {
    "classifier": rf_clf,
    "species_list": SPECIES_LIST
}

with open("models/speciesnet.pkl", "wb") as f:
    pickle.dump(speciesnet_model, f)
print("✓ SpeciesNet model saved to models/speciesnet.pkl")


# ==========================================
# 3. SAM2 SEGMENTATION MODEL SERIALIZATION
# ==========================================
print("\n📝 Creating SAM2 segmentation model...")

class MockSAM2Model:
    """Mock SAM2 Segmentation model structure for serialization"""
    def __init__(self, model_name="sam2_wildlife_v1"):
        self.model_name = model_name

    def segment(self, image_path: str, bbox: list, point_prompts: list = None) -> dict:
        x_min, y_min, x_max, y_max = bbox
        w_box = max(0.01, x_max - x_min)
        h_box = max(0.01, y_max - y_min)
        
        aspect = w_box / h_box
        quality = 0.92 if 0.5 <= aspect <= 2.5 else 0.85
        
        return {
            "model_name": self.model_name,
            "mask_quality": quality,
            "confidence": float(min(0.99, quality + 0.03)),
            "coverage_ratio": float(w_box * h_box * 0.78)
        }

sam2_model = MockSAM2Model()
with open("models/sam2.pkl", "wb") as f:
    pickle.dump(sam2_model, f)
print("✓ SAM2 model saved to models/sam2.pkl")


# ==========================================
# 4. OPENCLIP SEMANTIC MODEL SERIALIZATION
# ==========================================
print("\n📝 Creating OpenCLIP Semantic Verification model...")

class MockOpenCLIPModel:
    """Mock OpenCLIP model for vision-language semantic verification"""
    def __init__(self, concepts_dict=None):
        self.concepts = concepts_dict or {
            "Bengal Tiger": ["a Bengal tiger in the wild", "orange and black striped big cat", "tiger in forest"],
            "Indian Leopard": ["a leopard with spotted fur", "golden leopard in forest"],
            "Asian Elephant": ["an Asian elephant", "large gray elephant in forest"],
            "Sambar Deer": ["a large brown deer", "sambar deer in forest"],
            "Spotted Deer (Chital)": ["a spotted deer with white spots", "chital deer grazing"],
            "Wild Boar": ["a wild boar", "wild pig in forest"],
            "Sloth Bear": ["a sloth bear", "black bear with white chest mark"],
            "Indian Gaur": ["an Indian gaur", "large wild cattle"],
            "Nilgai": ["a nilgai blue bull", "tall grayish-blue antelope"],
            "Golden Jackal": ["a golden jackal", "small wild canine"],
            "Dhole": ["an Asiatic wild dog", "reddish-brown dhole pack animal"],
            "Striped Hyena": ["a striped hyena", "scavenger with striped coat"],
            "Jungle Cat": ["a jungle cat", "small wild feline"],
            "Rhesus Macaque": ["a rhesus macaque monkey", "brown monkey in tree"],
            "Common Langur": ["a gray langur with black face", "tall monkey in forest"],
            "Human": ["a human person walking", "forest ranger or poacher"],
            "Vehicle": ["a motor vehicle", "jeep or patrol truck"]
        }
        
        # Build normalized prototype embedding vectors for each concept
        np.random.seed(42)
        self.concept_embeddings = {}
        for sp in self.concepts.keys():
            vec = np.random.randn(32)
            self.concept_embeddings[sp] = (vec / np.linalg.norm(vec)).tolist()

    def predict_similarities(self, features: list, speciesnet_hint: str = None) -> dict:
        """
        Computes cosine similarities between visual feature projection and text concepts.
        """
        seed_val = int(abs(features[0] * 100 + features[1])) if len(features) >= 2 else 42
        rng = random.Random(seed_val)
        
        scores = {}
        for sp in self.concepts.keys():
            # If visual features match tiger profile (high red, contrast)
            if sp == "Bengal Tiger" and features[0] > 150 and features[6] > 40:
                scores[sp] = float(rng.uniform(0.85, 0.98))
            elif sp == "Asian Elephant" and features[0] < 120 and features[1] < 120 and features[2] < 120 and features[8] > 300:
                scores[sp] = float(rng.uniform(0.82, 0.95))
            elif sp == speciesnet_hint and rng.random() < 0.85:
                scores[sp] = float(rng.uniform(0.75, 0.94))
            else:
                scores[sp] = float(rng.uniform(0.10, 0.55))
                
        return scores

openclip_model = MockOpenCLIPModel()
with open("models/openclip.pkl", "wb") as f:
    pickle.dump(openclip_model, f)
print("✓ OpenCLIP model saved to models/openclip.pkl")

print("\n🎉 All 4 models created, trained, and serialized successfully!")
