import sys
import os
sys.path.insert(0, os.path.abspath("backend"))

from db.database import SessionLocal
from db.models import Image, Detection, Decision, QualityGate

db = SessionLocal()
print(f"Total images in DB: {db.query(Image).count()}")
imgs = db.query(Image).order_by(Image.created_at.desc()).limit(10).all()
for img in imgs:
    print(f"\nImage ID: {img.id}")
    print(f"  Path: {img.image_path}")
    print(f"  Quality: {img.quality_status}, Score: {img.quality_score}")
    dets = db.query(Detection).filter(Detection.image_id == img.id).all()
    print(f"  Detections ({len(dets)}): {[(d.object_type, d.confidence) for d in dets]}")
    dec = db.query(Decision).filter(Decision.image_id == img.id).first()
    if dec:
        print(f"  Decision: species={dec.species}, decision={dec.decision}, conf={dec.confidence}, reasons={dec.reasoning}")
db.close()
