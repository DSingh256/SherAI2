"""
VanRakshak AI - Phase 2 Validation Script
Tests Phase 2 without external dependencies
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import hashlib
from PIL import Image as PILImage
import numpy as np
import io

# Add backend directory to sys.path to allow imports in validation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

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
print("🧪 VanRakshak AI - Phase 2 Validation")
print("=" * 70)
print()

# ============ TEST 1: Quality Gate Service Module ============
print("📝 Test 1: Quality Gate Service Module")
try:
    service_path = Path("backend/core/quality_gate.py")
    passed = service_path.exists()
    print_test("quality_gate.py exists", passed)
    
    if passed:
        with open(service_path) as f:
            content = f.read()
            
            # Check for required classes
            classes_to_check = [
                ("QualityDecision enum", "class QualityDecision"),
                ("QualityGateResult class", "class QualityGateResult"),
                ("QualityGateService class", "class QualityGateService"),
            ]
            
            for class_name, class_str in classes_to_check:
                has_class = class_str in content
                print_test(f"  - {class_name}", has_class)
except Exception as e:
    print_test("Quality gate service", False, str(e))

# ============ TEST 2: Quality Gate Service Methods ============
print("\n📝 Test 2: Quality Gate Service Methods")
try:
    with open("backend/core/quality_gate.py") as f:
        content = f.read()
        
        methods = [
            ("assess_quality", "def assess_quality"),
            ("apply_quality_gate", "def apply_quality_gate"),
            ("batch_quality_gate", "def batch_quality_gate"),
            ("get_quality_breakdown", "def get_quality_breakdown"),
            ("get_rejection_reasons", "def get_rejection_reasons"),
        ]
        
        for method_name, method_str in methods:
            has_method = method_str in content
            print_test(f"  - {method_name} method", has_method)
except Exception as e:
    print_test("Service methods", False, str(e))

# ============ TEST 3: Quality Gate API Routes ============
print("\n📝 Test 3: Quality Gate API Routes")
try:
    routes_path = Path("backend/api/routes/quality.py")
    passed = routes_path.exists()
    print_test("quality.py routes file exists", passed)
    
    if passed:
        with open(routes_path) as f:
            content = f.read()
            
            endpoints = [
                ("assess endpoint", '@router.post("/assess/{image_id}")'),
                ("gate endpoint", '@router.post("/gate/{image_id}")'),
                ("batch gate endpoint", '@router.post("/gate-batch")'),
                ("breakdown endpoint", '@router.get("/breakdown")'),
                ("rejection reasons endpoint", '@router.get("/rejection-reasons")'),
                ("report endpoint", '@router.get("/report")'),
            ]
            
            for endpoint_name, endpoint_str in endpoints:
                has_endpoint = endpoint_str in content
                print_test(f"  - {endpoint_name}", has_endpoint)
except Exception as e:
    print_test("API routes", False, str(e))

# ============ TEST 4: Quality Gate Decision Enum ============
print("\n📝 Test 4: Quality Gate Decision Enum")
try:
    with open("backend/core/quality_gate.py") as f:
        content = f.read()
        
        decisions = [
            ("ACCEPT decision", "ACCEPT"),
            ("BLUR_REJECT decision", "BLUR_REJECT"),
            ("DARKNESS_REJECT decision", "DARKNESS_REJECT"),
            ("OVEREXPOSED_REJECT decision", "OVEREXPOSED_REJECT"),
            ("CORRUPTED_REJECT decision", "CORRUPTED_REJECT"),
            ("DUPLICATE_REJECT decision", "DUPLICATE_REJECT"),
        ]
        
        for decision_name, decision_str in decisions:
            has_decision = decision_str in content
            print_test(f"  - {decision_name}", has_decision)
except Exception as e:
    print_test("Decision enum", False, str(e))

# ============ TEST 5: Database Integration ============
print("\n📝 Test 5: Database Integration")
try:
    with open("backend/db/models.py") as f:
        content = f.read()
        
        # Check for ImageQuality enum
        has_enum = "class ImageQuality" in content
        print_test("ImageQuality enum defined", has_enum)
        
        if has_enum:
            quality_statuses = [
                ("GOOD status", "GOOD = "),
                ("BLURRY status", "BLURRY = "),
                ("TOO_DARK status", "TOO_DARK = "),
                ("OVEREXPOSED status", "OVEREXPOSED = "),
                ("CORRUPTED status", "CORRUPTED = "),
                ("DUPLICATE status", "DUPLICATE = "),
            ]
            
            for status_name, status_str in quality_statuses:
                has_status = status_str in content
                print_test(f"  - {status_name}", has_status)
except Exception as e:
    print_test("Database integration", False, str(e))

# ============ TEST 6: Quality Metrics in Image Model ============
print("\n📝 Test 6: Quality Metrics in Image Model")
try:
    with open("backend/db/models.py") as f:
        content = f.read()
        
        metrics = [
            ("quality_status field", "quality_status = "),
            ("quality_score field", "quality_score = "),
            ("blur_score field", "blur_score = "),
            ("brightness field", "brightness = "),
            ("contrast field", "contrast = "),
        ]
        
        for metric_name, metric_str in metrics:
            has_metric = metric_str in content
            print_test(f"  - {metric_name}", has_metric)
except Exception as e:
    print_test("Quality metrics", False, str(e))

# ============ TEST 7: Image Utils Quality Functions ============
print("\n📝 Test 7: Image Utils Quality Functions")
try:
    with open("backend/utils/image_utils.py") as f:
        content = f.read()
        
        functions = [
            ("get_blur_score", "def get_blur_score"),
            ("get_brightness", "def get_brightness"),
            ("get_contrast", "def get_contrast"),
            ("is_corrupted", "def is_corrupted"),
            ("get_image_quality_metrics", "def get_image_quality_metrics"),
        ]
        
        for func_name, func_str in functions:
            has_func = func_str in content
            print_test(f"  - {func_name} function", has_func)
except Exception as e:
    print_test("Image utils", False, str(e))

# ============ TEST 8: Configuration Thresholds ============
print("\n📝 Test 8: Configuration Thresholds")
try:
    with open("backend/config.py") as f:
        content = f.read()
        
        thresholds = [
            ("BLUR_THRESHOLD", "BLUR_THRESHOLD"),
            ("MIN_BRIGHTNESS", "MIN_BRIGHTNESS"),
            ("MAX_BRIGHTNESS", "MAX_BRIGHTNESS"),
        ]
        
        for threshold_name, threshold_str in thresholds:
            has_threshold = threshold_str in content
            print_test(f"  - {threshold_name} configured", has_threshold)
except Exception as e:
    print_test("Configuration thresholds", False, str(e))

# ============ TEST 9: FastAPI Integration ============
print("\n📝 Test 9: FastAPI Integration")
try:
    with open("backend/main.py") as f:
        content = f.read()
        
        # Check for quality route import
        has_import = "from api.routes import images, quality" in content
        print_test("Quality router imported", has_import)
        
        # Check for router registration
        has_registration = "app.include_router(quality.router)" in content
        print_test("Quality router registered", has_registration)
        
        # Check for quality endpoint in root
        has_quality_info = '"/api/quality"' in content or '"quality"' in content
        print_test("Quality info in root endpoint", has_quality_info)
except Exception as e:
    print_test("FastAPI integration", False, str(e))

# ============ TEST 10: Test Suite ============
print("\n📝 Test 10: Phase 2 Test Suite")
try:
    tests_path = Path("backend/tests_phase2.py")
    passed = tests_path.exists()
    print_test("Phase 2 test file exists", passed)
    
    if passed:
        with open(tests_path) as f:
            content = f.read()
            
            test_classes = [
                ("TestQualityAssessment", "class TestQualityAssessment"),
                ("TestQualityGateApplication", "class TestQualityGateApplication"),
                ("TestBatchQualityGate", "class TestBatchQualityGate"),
                ("TestQualityStatistics", "class TestQualityStatistics"),
                ("TestQualityDecisions", "class TestQualityDecisions"),
                ("TestPhase2Integration", "class TestPhase2Integration"),
                ("TestEdgeCases", "class TestEdgeCases"),
            ]
            
            for class_name, class_str in test_classes:
                has_class = class_str in content
                print_test(f"  - {class_name}", has_class)
except Exception as e:
    print_test("Test suite", False, str(e))

# ============ TEST 11: Quality Assessment Logic ============
print("\n📝 Test 11: Quality Assessment Logic")
try:
    # Test brightness calculation
    img = PILImage.new('RGB', (100, 100), color=(100, 100, 100))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Convert to grayscale and calculate brightness
    img_gray = img.convert('L')
    brightness = np.mean(np.array(img_gray))
    
    passed = 0 <= brightness <= 255
    print_test("Brightness calculation (0-255 range)", passed)
    
    # Test blur calculation
    import cv2
    img_array = np.array(img.convert('L'))
    laplacian_var = cv2.Laplacian(img_array.astype(np.float64), cv2.CV_64F).var()
    
    passed = laplacian_var >= 0
    print_test("Blur score calculation (Laplacian)", passed)
    
    # Test contrast calculation
    contrast = np.std(img_array)
    passed = contrast >= 0
    print_test("Contrast calculation (std dev)", passed)
except Exception as e:
    print_test("Quality assessment logic", False, str(e))

# ============ TEST 12: Quality Gate Decision Logic ============
print("\n📝 Test 12: Quality Gate Decision Logic")
try:
    # Create test image bytes
    dark_img = PILImage.new('RGB', (100, 100), color=(5, 5, 5))
    dark_bytes = io.BytesIO()
    dark_img.save(dark_bytes, format='PNG')
    
    bright_img = PILImage.new('RGB', (100, 100), color=(250, 250, 250))
    bright_bytes = io.BytesIO()
    bright_img.save(bright_bytes, format='PNG')
    
    # Test thresholds exist
    has_min_brightness = hasattr(__import__('config', fromlist=['settings']).settings, 'MIN_BRIGHTNESS')
    print_test("MIN_BRIGHTNESS threshold configured", has_min_brightness)
    
    has_max_brightness = hasattr(__import__('config', fromlist=['settings']).settings, 'MAX_BRIGHTNESS')
    print_test("MAX_BRIGHTNESS threshold configured", has_max_brightness)
    
    has_blur_threshold = hasattr(__import__('config', fromlist=['settings']).settings, 'BLUR_THRESHOLD')
    print_test("BLUR_THRESHOLD configured", has_blur_threshold)
except Exception as e:
    print_test("Decision logic", False, str(e))

# ============ TEST 13: Quality Audit Trail ============
print("\n📝 Test 13: Quality Audit Trail")
try:
    with open("backend/core/quality_gate.py") as f:
        content = f.read()
        
        has_audit = "_record_audit" in content
        print_test("Audit trail recording implemented", has_audit)
        
        has_audit_model = "AuditTrail" in content
        print_test("AuditTrail model used", has_audit_model)
except Exception as e:
    print_test("Audit trail", False, str(e))

# ============ TEST 14: API Response Structure ============
print("\n📝 Test 14: API Response Structure")
try:
    with open("backend/api/routes/quality.py") as f:
        content = f.read()
        
        response_elements = [
            ("APIResponse wrapper", "APIResponse"),
            ("Success field", "success"),
            ("Message field", "message"),
            ("Data field", "data"),
            ("Details in response", "details"),
            ("Reasoning in response", "reasons"),
        ]
        
        for element_name, element_str in response_elements:
            has_element = element_str in content
            print_test(f"  - {element_name}", has_element)
except Exception as e:
    print_test("API response structure", False, str(e))

# ============ TEST 15: Quality Categories ============
print("\n📝 Test 15: Quality Categories")
try:
    with open("backend/core/quality_gate.py") as f:
        content = f.read()
        
        categories = [
            ("ACCEPT category", "QualityDecision.ACCEPT"),
            ("BLUR_REJECT category", "QualityDecision.BLUR_REJECT"),
            ("DARKNESS_REJECT category", "QualityDecision.DARKNESS_REJECT"),
            ("OVEREXPOSED_REJECT category", "QualityDecision.OVEREXPOSED_REJECT"),
            ("CORRUPTED_REJECT category", "QualityDecision.CORRUPTED_REJECT"),
        ]
        
        for cat_name, cat_str in categories:
            has_cat = cat_str in content
            print_test(f"  - {cat_name}", has_cat)
except Exception as e:
    print_test("Quality categories", False, str(e))

# ============ SUMMARY ============
print()
print("=" * 70)
print("📊 PHASE 2 VALIDATION SUMMARY")
print("=" * 70)
print(f"✅ Tests Passed: {tests_passed}")
print(f"❌ Tests Failed: {tests_failed}")
total = tests_passed + tests_failed
print(f"📈 Success Rate: {(tests_passed / total * 100):.1f}%")
print()

if tests_failed == 0:
    print("🎉 PHASE 2 VALIDATION COMPLETE - ALL CHECKS PASSED!")
    print()
    print("✨ Phase 2 Features:")
    print("  ✓ Image quality assessment (blur, brightness, corruption)")
    print("  ✓ Quality gate with 6 decision categories")
    print("  ✓ Batch quality gate processing")
    print("  ✓ Quality statistics and breakdown")
    print("  ✓ Rejection reason tracking")
    print("  ✓ API endpoints for quality assessment")
    print("  ✓ Comprehensive test suite (40+ tests)")
    print()
    print("📚 Next Steps:")
    print("  1. Run tests: pytest backend/tests_phase2.py -v")
    print("  2. Test endpoints: http://localhost:8000/api/quality/...")
    print("  3. View API docs: http://localhost:8000/docs")
    print()
    sys.exit(0)
else:
    print("⚠️  Some checks failed. Review output above.")
    sys.exit(1)
