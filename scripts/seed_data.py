"""
VanRakshak AI - Demo Data Seeder
Populates the database with realistic hackathon demo data.
Runs the AI pipeline on mock images to generate a rich dashboard.
"""

import os
import sys
import uuid
import shutil
import asyncio
from datetime import datetime, timedelta
import random

# Add backend dir to path so we can import from core modules
backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
sys.path.append(backend_dir)

from db.database import SessionLocal, init_db, engine
from db.models import Base, Camera, Image, Alert
from services.image_service import ImageService
from core.pipeline import ProcessingPipeline


# Demo configurations
NUM_DEMO_IMAGES = 150
CAMERAS = [
    {"id": "CAM001", "name": "North Gate", "zone": "Buffer Zone", "lat": 21.1458, "lon": 79.0882},
    {"id": "CAM007", "name": "Waterhole Alpha", "zone": "Core Zone", "lat": 21.1500, "lon": 79.0950},
    {"id": "CAM012", "name": "Tiger Trail East", "zone": "Core Zone", "lat": 21.1350, "lon": 79.1000},
    {"id": "CAM022", "name": "Village Border South", "zone": "Buffer Zone", "lat": 21.1200, "lon": 79.0800},
]

def create_mock_image_files():
    """Create placeholder images in the ingestion directory"""
    print("Creating mock image files...")
    
    upload_dir = "storage/raw_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs("storage/processed", exist_ok=True)
    os.makedirs("storage/segmented", exist_ok=True)
    
    # We'll just create empty files for the demo since the simulated AI
    # uses the filename hash to generate deterministic detections
    image_paths = []
    for i in range(NUM_DEMO_IMAGES):
        filename = f"demo_img_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(upload_dir, filename)
        
        # Touch the file
        with open(filepath, 'w') as f:
            f.write("mock image data")
            
        image_paths.append(filepath)
        
    return image_paths

def seed_cameras(db):
    """Seed camera trap inventory"""
    print("Seeding cameras...")
    for cam in CAMERAS:
        c = Camera(
            camera_id=cam["id"],
            name=cam["name"],
            zone=cam["zone"],
            gps_latitude=cam["lat"],
            gps_longitude=cam["lon"],
            status="active"
        )
        db.add(c)
    db.commit()
    print(f"Added {len(CAMERAS)} cameras.")

def process_demo_images(db, image_paths):
    """Ingest and process all mock images to generate rich AI data"""
    print(f"Processing {len(image_paths)} images through AI pipeline (simulated mode)...")
    
    # Time range: Last 7 days
    now = datetime.utcnow()
    
    success_count = 0
    for i, path in enumerate(image_paths):
        # Assign random timestamp in last 7 days
        days_ago = random.uniform(0, 7)
        timestamp = now - timedelta(days=days_ago)
        
        # Random camera
        camera = random.choice(CAMERAS)["id"]
        
        # 1. Ingest image
        with open(path, 'rb') as f:
            file_data = f.read()
            
        # Manually create image record (bypassing fastapi UploadFile)
        filename = os.path.basename(path)
        img_id = str(uuid.uuid4())
        
        img = Image(
            id=img_id,
            camera_id=camera,
            location=camera,
            timestamp=timestamp,
            image_path=path,
            file_size=len(file_data)
        )
        db.add(img)
        db.commit()
        
        # 2. Run Pipeline
        try:
            result = ProcessingPipeline.process_image(img_id, db)
            if result.success:
                success_count += 1
                
                # Update image status based on pipeline
                img = db.query(Image).filter(Image.id == img_id).first()
                if img:
                    img.status = "processed"
                    db.commit()
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_paths)}...")
            
    print(f"Successfully processed {success_count} images.")

def generate_system_alerts(db):
    """Generate some demo alerts for the dashboard"""
    print("Generating demo alerts...")
    
    alerts = [
        {
            "type": "threat",
            "severity": "high",
            "title": "Human + Vehicle Activity at 3AM",
            "msg": "Multiple humans and a vehicle detected in Core Zone during restricted hours.",
            "cam": "CAM007"
        },
        {
            "type": "tiger_sighting",
            "severity": "low",
            "title": "Tiger Detected",
            "msg": "Bengal Tiger detected on Tiger Trail East.",
            "cam": "CAM012"
        },
        {
            "type": "camera_failure",
            "severity": "medium",
            "title": "Camera Offline",
            "msg": "No heartbeat received from North Gate camera for 24 hours.",
            "cam": "CAM001"
        },
        {
            "type": "unusual_activity",
            "severity": "medium",
            "title": "Unusual Animal Activity Drop",
            "msg": "Sudden drop in animal detections at Waterhole Alpha (-2.8 z-score).",
            "cam": "CAM007"
        }
    ]
    
    for a in alerts:
        alert = Alert(
            alert_type=a["type"],
            severity=a["severity"],
            title=a["title"],
            message=a["msg"],
            camera_id=a["cam"],
            acknowledged=random.choice([True, False])
        )
        db.add(alert)
        
    db.commit()
    print(f"Generated {len(alerts)} alerts.")

def run_seeder():
    print("=== VanRakshak AI Demo Data Seeder ===")
    
    # 1. Reset Database
    print("Resetting database...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # 2. Get DB session
    db = SessionLocal()
    
    try:
        # 3. Clean storage dirs
        for d in ["storage/raw_uploads", "storage/processed", "storage/segmented"]:
            if os.path.exists(d):
                shutil.rmtree(d)
                
        # 4. Create mock files
        image_paths = create_mock_image_files()
        
        # 5. Seed Cameras
        seed_cameras(db)
        
        # 6. Process Images (Generates AI data)
        process_demo_images(db, image_paths)
        
        # 7. Generate Alerts
        generate_system_alerts(db)
        
        print("\n✅ Demo data seeding complete!")
        print("You can now start the backend server and explore the dashboard.")
        
    finally:
        db.close()

if __name__ == "__main__":
    run_seeder()
