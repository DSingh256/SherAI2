"""
VanRakshak AI - Re-identification Routes
API endpoints for tiger re-identification features.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from db.database import get_db
from db.schemas import APIResponse
from db.models import TigerReidentification, Image

router = APIRouter(prefix="/api/reidentification", tags=["reidentification"])


@router.get("/matches")
async def get_reid_matches(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get potential tiger re-identification matches"""
    
    matches = db.query(TigerReidentification).order_by(
        desc(TigerReidentification.created_at)
    ).limit(limit).all()
    
    results = []
    for match in matches:
        # Get image 1 details
        img1 = db.query(Image).filter(Image.id == match.image_id_1).first()
        # Get image 2 details
        img2 = db.query(Image).filter(Image.id == match.image_id_2).first()
        
        if img1 and img2:
            results.append({
                "match_id": match.id,
                "similarity": match.similarity,
                "verified": match.verified,
                "created_at": match.created_at.isoformat(),
                "image1": {
                    "id": img1.id,
                    "path": img1.image_path,
                    "camera_id": img1.camera_id,
                    "timestamp": img1.timestamp.isoformat() if img1.timestamp else None
                },
                "image2": {
                    "id": img2.id,
                    "path": img2.image_path,
                    "camera_id": img2.camera_id,
                    "timestamp": img2.timestamp.isoformat() if img2.timestamp else None
                }
            })
            
    return APIResponse(
        success=True,
        message=f"Retrieved {len(results)} potential matches",
        data={"matches": results}
    ).model_dump()


@router.post("/verify/{match_id}")
async def verify_match(
    match_id: str,
    db: Session = Depends(get_db)
):
    """Mark a potential match as verified by a human"""
    
    match = db.query(TigerReidentification).filter(
        TigerReidentification.id == match_id
    ).first()
    
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
        
    match.verified = True
    db.commit()
    
    return APIResponse(
        success=True,
        message="Match verified successfully"
    ).model_dump()
