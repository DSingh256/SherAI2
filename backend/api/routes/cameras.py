"""
VanRakshak AI - Cameras Routes
API endpoints for camera management.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.schemas import APIResponse
from services.camera_service import CameraService
from db.models import Camera, Image

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("/")
async def get_all_cameras(db: Session = Depends(get_db)):
    """Get all cameras and their activity stats"""
    
    activity = CameraService.get_camera_activity(db)
    
    return APIResponse(
        success=True,
        message="Cameras retrieved",
        data={"cameras": activity}
    ).model_dump()


@router.get("/{camera_id}")
async def get_camera_detail(
    camera_id: str, 
    db: Session = Depends(get_db)
):
    """Get detailed stats for a specific camera"""
    
    camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    
    if not camera:
        # Check if we have images for it even if not registered
        img = db.query(Image).filter(Image.camera_id == camera_id).first()
        if not img:
            raise HTTPException(status_code=404, detail="Camera not found")
            
        camera_data = {
            "camera_id": camera_id,
            "status": "unregistered_active",
            "name": f"Camera {camera_id}"
        }
    else:
        camera_data = {
            "camera_id": camera.camera_id,
            "name": camera.name,
            "location": camera.location,
            "gps_latitude": camera.gps_latitude,
            "gps_longitude": camera.gps_longitude,
            "zone": camera.zone,
            "status": camera.status
        }
        
    return APIResponse(
        success=True,
        message="Camera details retrieved",
        data=camera_data
    ).model_dump()
