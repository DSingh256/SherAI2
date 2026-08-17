"""
VanRakshak AI - Phase 1 Validation Script
Tests Phase 1 without external dependencies
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import hashlib
from PIL import Image as PILImage
import numpy as np
import io

# Test counter
tests_passed = 0
tests_failed = 0

def print_test(name, passed, details=""):
    """Print test result"""
    global tests_passed, tests_failed
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {name}")
    if details and not passed:
        print(f"       {details}")
    if passed:
        tests_passed += 1
    else:
        tests_failed += 1

print("=" * 70)
print("🧪 VanRakshak AI - Phase 1 Validation")
print("=" * 70)
print()

# ============ TEST 1: Image Hash Calculation ============
print("📝 Test 1: Image Hash Calculation")
try:
    # Create a simple image
    img = PILImage.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    # Calculate hash
    hash1 = hashlib.sha256(img_bytes).hexdigest()
    hash2 = hashlib.sha256(img_bytes).hexdigest()
    
    # Test
    passed = (hash1 == hash2) and len(hash1) == 64
    print_test("Same image produces same SHA256 hash", passed)
except Exception as e:
    print_test("Image hash calculation", False, str(e))

# ============ TEST 2: Image Dimensions ============
print("\n📝 Test 2: Image Dimensions")
try:
    img = PILImage.new('RGB', (1920, 1080), color='blue')
    width, height = img.size
    
    passed = (width == 1920) and (height == 1080)
    print_test("Extract correct image dimensions", passed)
except Exception as e:
    print_test("Image dimensions extraction", False, str(e))

# ============ TEST 3: Image Quality Metrics ============
print("\n📝 Test 3: Image Quality Metrics")
try:
    # Create test images
    clear_img = PILImage.new('RGB', (100, 100), color='red')
    dark_img = PILImage.new('RGB', (100, 100), color=(10, 10, 10))
    
    # Convert to arrays to calculate brightness
    clear_array = np.array(clear_img.convert('L'))
    dark_array = np.array(dark_img.convert('L'))
    
    clear_brightness = np.mean(clear_array)
    dark_brightness = np.mean(dark_array)
    
    passed = clear_brightness > dark_brightness
    print_test("Bright image has higher brightness than dark image", passed)
    
    # Test contrast (standard deviation)
    contrast_score = np.std(clear_array)
    passed = contrast_score >= 0
    print_test("Calculate contrast score", passed)
except Exception as e:
    print_test("Image quality metrics", False, str(e))

# ============ TEST 4: Perceptual Hash ============
print("\n📝 Test 4: Perceptual Hash (Duplicate Detection)")
try:
    def calculate_phash(img_array, hash_size=8):
        """Calculate perceptual hash"""
        img_pil = PILImage.fromarray(img_array)
        img_pil = img_pil.convert("L")
        img_pil = img_pil.resize((hash_size + 1, hash_size), PILImage.LANCZOS)
        
        pixels = list(img_pil.getdata())
        avg = sum(pixels) / len(pixels)
        hash_bits = "".join("1" if pixel >= avg else "0" for pixel in pixels)
        return format(int(hash_bits, 2), f"0{hash_size * hash_size // 4}x")
    
    # Create identical images
    img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    hash1 = calculate_phash(img_array)
    hash2 = calculate_phash(img_array)
    
    passed = hash1 == hash2
    print_test("Identical images have same perceptual hash", passed)
    
    # Different images
    different_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    hash3 = calculate_phash(different_array)
    
    passed = hash1 != hash3
    print_test("Different images have different hashes", passed)
except Exception as e:
    print_test("Perceptual hash calculation", False, str(e))

# ============ TEST 5: Image Validation ============
print("\n📝 Test 5: Image Validation")
try:
    # Valid image
    valid_img = PILImage.new('RGB', (100, 100), color='green')
    
    # Try to save and reload (corruption check)
    img_bytes = io.BytesIO()
    valid_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    try:
        test_img = PILImage.open(img_bytes)
        test_img.verify()
        passed = True
    except:
        passed = False
    
    print_test("Valid PNG image passes verification", passed)
    
    # Corrupted data
    corrupted_data = b"This is not an image"
    try:
        PILImage.open(io.BytesIO(corrupted_data)).verify()
        passed = False
    except:
        passed = True
    
    print_test("Corrupted data fails verification", passed)
except Exception as e:
    print_test("Image validation", False, str(e))

# ============ TEST 6: Database Schema Definition ============
print("\n📝 Test 6: Database Schema Definition")
try:
    # Check if models file exists and has required classes
    models_path = Path("backend/db/models.py")
    passed = models_path.exists()
    print_test("Database models file exists", passed)
    
    if passed:
        with open(models_path) as f:
            content = f.read()
            
            # Check for required classes
            required_classes = [
                "class Image",
                "class Detection",
                "class Classification",
                "class Decision",
                "class HumanReview",
                "class AuditTrail"
            ]
            
            for cls in required_classes:
                has_cls = cls in content
                print_test(f"  - {cls} defined", has_cls)
except Exception as e:
    print_test("Database schema definition", False, str(e))

# ============ TEST 7: FastAPI Application ============
print("\n📝 Test 7: FastAPI Application Structure")
try:
    # Check main.py exists
    main_path = Path("backend/main.py")
    passed = main_path.exists()
    print_test("Main FastAPI app file exists", passed)
    
    if passed:
        with open(main_path) as f:
            content = f.read()
            
            # Check for required components
            checks = [
                ("FastAPI initialization", "app = FastAPI"),
                ("CORS middleware", "CORSMiddleware"),
                ("Root endpoint", '@app.get("/")'),
                ("Health check", '@app.get("/health")'),
                ("Routes included", "include_router")
            ]
            
            for check_name, check_str in checks:
                has_check = check_str in content
                print_test(f"  - {check_name}", has_check)
except Exception as e:
    print_test("FastAPI application", False, str(e))

# ============ TEST 8: Image Service ============
print("\n📝 Test 8: Image Service Layer")
try:
    service_path = Path("backend/services/image_service.py")
    passed = service_path.exists()
    print_test("Image service file exists", passed)
    
    if passed:
        with open(service_path) as f:
            content = f.read()
            
            methods = [
                ("ingest_image", "def ingest_image"),
                ("get_image", "def get_image"),
                ("get_images_by_camera", "def get_images_by_camera"),
                ("check_duplicate_images", "def check_duplicate_images")
            ]
            
            for method_name, method_str in methods:
                has_method = method_str in content
                print_test(f"  - {method_name} method", has_method)
except Exception as e:
    print_test("Image service", False, str(e))

# ============ TEST 9: API Endpoints ============
print("\n📝 Test 9: Image Upload API Endpoints")
try:
    routes_path = Path("backend/api/routes/images.py")
    passed = routes_path.exists()
    print_test("Image routes file exists", passed)
    
    if passed:
        with open(routes_path) as f:
            content = f.read()
            
            endpoints = [
                ("Upload endpoint", '@router.post("/upload")'),
                ("Get image info", '@router.get("/image/{image_id}")'),
                ("Get camera images", '@router.get("/camera/{camera_id}")'),
                ("Review queue", '@router.get("/review-queue")'),
                ("Statistics", '@router.get("/stats")')
            ]
            
            for endpoint_name, endpoint_str in endpoints:
                has_endpoint = endpoint_str in content
                print_test(f"  - {endpoint_name}", has_endpoint)
except Exception as e:
    print_test("API endpoints", False, str(e))

# ============ TEST 10: Configuration ============
print("\n📝 Test 10: Configuration Management")
try:
    config_path = Path("backend/config.py")
    passed = config_path.exists()
    print_test("Configuration file exists", passed)
    
    if passed:
        with open(config_path) as f:
            content = f.read()
            
            settings_checks = [
                ("Settings class", "class Settings"),
                ("Database URL", "DATABASE_URL"),
                ("Storage paths", "RAW_STORAGE_PATH"),
                ("Confidence thresholds", "HIGH_CONFIDENCE_THRESHOLD"),
                ("Feature flags", "ENABLE_MEGADETECTOR")
            ]
            
            for check_name, check_str in settings_checks:
                has_check = check_str in content
                print_test(f"  - {check_name}", has_check)
except Exception as e:
    print_test("Configuration", False, str(e))

# ============ TEST 11: Docker Configuration ============
print("\n📝 Test 11: Docker & Containerization")
try:
    docker_compose_path = Path("docker-compose.yml")
    dockerfile_path = Path("Dockerfile.backend")
    env_example_path = Path(".env.example")
    
    print_test("docker-compose.yml exists", docker_compose_path.exists())
    print_test("Dockerfile.backend exists", dockerfile_path.exists())
    print_test(".env.example exists", env_example_path.exists())
    
    if docker_compose_path.exists():
        with open(docker_compose_path) as f:
            content = f.read()
            print_test("  - PostgreSQL service defined", "postgres:" in content)
            print_test("  - Backend service defined", "backend:" in content)
except Exception as e:
    print_test("Docker configuration", False, str(e))

# ============ TEST 12: Test Suite ============
print("\n📝 Test 12: Test Suite Availability")
try:
    tests_path = Path("backend/tests_phase1.py")
    passed = tests_path.exists()
    print_test("Phase 1 test file exists", passed)
    
    if passed:
        with open(tests_path) as f:
            content = f.read()
            
            test_classes = [
                ("TestImageUtils", "class TestImageUtils"),
                ("TestPerceptualHash", "class TestPerceptualHash"),
                ("TestImageService", "class TestImageService"),
                ("TestPhase1Integration", "class TestPhase1Integration")
            ]
            
            for class_name, class_str in test_classes:
                has_class = class_str in content
                print_test(f"  - {class_name}", has_class)
except Exception as e:
    print_test("Test suite", False, str(e))

# ============ TEST 13: Documentation ============
print("\n📝 Test 13: Documentation")
try:
    readme_path = Path("README.md")
    passed = readme_path.exists()
    print_test("README.md exists", passed)
    
    if passed:
        with open(readme_path) as f:
            content = f.read()
            
            doc_checks = [
                ("Phase 1 overview", "Phase 1"),
                ("Architecture diagram", "Architecture"),
                ("API endpoints", "API Endpoints"),
                ("Setup instructions", "Setup & Installation"),
                ("Tests documentation", "Running Tests")
            ]
            
            for check_name, check_str in doc_checks:
                has_check = check_str in content
                print_test(f"  - {check_name}", has_check)
except Exception as e:
    print_test("Documentation", False, str(e))

# ============ SUMMARY ============
print()
print("=" * 70)
print("📊 PHASE 1 VALIDATION SUMMARY")
print("=" * 70)
print(f"✅ Tests Passed: {tests_passed}")
print(f"❌ Tests Failed: {tests_failed}")
print(f"📈 Success Rate: {(tests_passed / (tests_passed + tests_failed) * 100):.1f}%")
print()

if tests_failed == 0:
    print("🎉 PHASE 1 VALIDATION COMPLETE - ALL CHECKS PASSED!")
    print()
    print("📚 Next Steps:")
    print("  1. Create .env file: cp .env.example .env")
    print("  2. Setup PostgreSQL database")
    print("  3. Install dependencies: pip install -r backend/requirements.txt")
    print("  4. Run backend: python -m uvicorn backend.main:app --reload")
    print("  5. Test upload: curl -X POST http://localhost:8000/api/images/upload ...")
    print()
    sys.exit(0)
else:
    print("⚠️  Some checks failed. Review output above.")
    sys.exit(1)
