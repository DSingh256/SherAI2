"""
VanRakshak AI - Audit Routes
API endpoints for decision audit trails.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.schemas import APIResponse
from db.models import AuditTrail

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("/image/{image_id}")
async def get_image_audit_trail(
    image_id: str,
    db: Session = Depends(get_db)
):
    """Get the complete audit trail for an image"""
    
    events = db.query(AuditTrail).filter(
        AuditTrail.image_id == image_id
    ).order_by(AuditTrail.created_at).all()
    
    if not events:
        raise HTTPException(status_code=404, detail="No audit trail found for this image")
        
    return APIResponse(
        success=True,
        message="Audit trail retrieved",
        data={
            "image_id": image_id,
            "events": [
                {
                    "id": event.id,
                    "event_type": event.event_type,
                    "event_status": event.event_status,
                    "details": event.details,
                    "created_at": event.created_at.isoformat()
                }
                for event in events
            ]
        }
    ).model_dump()
