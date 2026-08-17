"""
VanRakshak AI - Alert Service
Manages real-time alerts for conservation events.
"""

from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session

# Try to import Alert model, but handle if it's not yet created
try:
    from db.models import Alert
    ALERT_MODEL_AVAILABLE = True
except ImportError:
    ALERT_MODEL_AVAILABLE = False


class AlertService:
    """
    Service for generating and managing alerts.
    """

    @staticmethod
    def create_alert(
        db: Session,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        camera_id: str,
        image_id: Optional[str] = None,
        details: Dict = None
    ) -> bool:
        """Create a new alert in the system"""
        if not ALERT_MODEL_AVAILABLE:
            return False
            
        try:
            alert = Alert(
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message,
                camera_id=camera_id,
                image_id=image_id,
                details=details or {}
            )
            db.add(alert)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"Failed to create alert: {e}")
            return False

    @staticmethod
    def get_recent_alerts(db: Session, limit: int = 50, include_acknowledged: bool = False) -> List[Dict]:
        """Get recent alerts"""
        if not ALERT_MODEL_AVAILABLE:
            return []
            
        query = db.query(Alert)
        
        if not include_acknowledged:
            query = query.filter(Alert.acknowledged == False)
            
        alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()
        
        return [
            {
                "id": a.id,
                "type": a.alert_type,
                "severity": a.severity,
                "title": a.title,
                "message": a.message,
                "camera_id": a.camera_id,
                "image_id": a.image_id,
                "created_at": a.created_at.isoformat(),
                "acknowledged": a.acknowledged
            }
            for a in alerts
        ]
        
    @staticmethod
    def acknowledge_alert(db: Session, alert_id: str, user: str = "system") -> bool:
        """Mark an alert as acknowledged"""
        if not ALERT_MODEL_AVAILABLE:
            return False
            
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.acknowledged = True
            alert.acknowledged_by = user
            db.commit()
            return True
        return False
