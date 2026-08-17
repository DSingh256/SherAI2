"""
VanRakshak AI - Alerts Routes
API endpoints for system alerts.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.schemas import APIResponse
from services.alert_service import AlertService

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/")
async def get_alerts(
    limit: int = 50,
    include_acknowledged: bool = False,
    db: Session = Depends(get_db)
):
    """Get recent alerts"""
    
    alerts = AlertService.get_recent_alerts(db, limit, include_acknowledged)
    
    return APIResponse(
        success=True,
        message=f"Retrieved {len(alerts)} alerts",
        data={"alerts": alerts}
    ).model_dump()


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    user: str = "system_user",
    db: Session = Depends(get_db)
):
    """Acknowledge an alert"""
    
    success = AlertService.acknowledge_alert(db, alert_id, user)
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found or could not be acknowledged")
        
    return APIResponse(
        success=True,
        message="Alert acknowledged successfully"
    ).model_dump()
