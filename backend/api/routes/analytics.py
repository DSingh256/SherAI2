"""
VanRakshak AI - Analytics Routes
API endpoints for dashboard metrics and insights.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.database import get_db
from db.schemas import APIResponse
from services.analytics_service import AnalyticsService

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/overview")
async def get_overview(db: Session = Depends(get_db)):
    """Get high-level dashboard metrics"""
    
    stats = AnalyticsService.get_overview_stats(db)
    
    return APIResponse(
        success=True,
        message="Overview stats retrieved",
        data=stats
    ).model_dump()


@router.get("/species")
async def get_species_distribution(db: Session = Depends(get_db)):
    """Get species counts"""
    
    distribution = AnalyticsService.get_species_distribution(db)
    
    return APIResponse(
        success=True,
        message="Species distribution retrieved",
        data={"distribution": distribution}
    ).model_dump()


@router.get("/temporal")
async def get_activity_timeline(
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get activity counts over time"""
    
    timeline = AnalyticsService.get_activity_timeline(db, days)
    
    return APIResponse(
        success=True,
        message="Activity timeline retrieved",
        data={"timeline": timeline}
    ).model_dump()


@router.get("/detection-types")
async def get_detection_types(db: Session = Depends(get_db)):
    """Get breakdown of detection types (animal/human/vehicle)"""
    
    types = AnalyticsService.get_detection_types(db)
    
    return APIResponse(
        success=True,
        message="Detection types retrieved",
        data={"types": types}
    ).model_dump()
