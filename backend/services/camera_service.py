"""
VanRakshak AI - Camera Service
Manages camera trap inventory, health, and activity statistics.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from db.models import Image, Decision


class CameraService:
    """
    Service for camera management and health monitoring.
    """

    @staticmethod
    def get_camera_activity(db: Session) -> List[Dict]:
        """
        Get activity statistics for all cameras.
        """
        # Get latest image time and total count per camera
        camera_stats = db.query(
            Image.camera_id,
            func.count(Image.id).label('total_images'),
            func.max(Image.timestamp).label('last_active')
        ).group_by(
            Image.camera_id
        ).all()
        
        results = []
        for stat in camera_stats:
            camera_id = stat.camera_id
            
            # Get tiger count for this camera
            tiger_count = db.query(Decision).join(
                Image, Decision.image_id == Image.id
            ).filter(
                Image.camera_id == camera_id,
                Decision.is_tiger == True
            ).count()
            
            # Determine status based on last active time (offline if > 24h)
            is_online = True
            if stat.last_active:
                time_diff = datetime.utcnow() - stat.last_active
                if time_diff.total_seconds() > 86400:  # 24 hours
                    is_online = False
            
            results.append({
                "camera_id": camera_id,
                "total_images": stat.total_images,
                "last_active": stat.last_active.isoformat() if stat.last_active else None,
                "tiger_sightings": tiger_count,
                "status": "online" if is_online else "offline"
            })
            
        return sorted(results, key=lambda x: x["tiger_sightings"], reverse=True)
