"""
VanRakshak AI - Analytics Routes
API endpoints for dashboard metrics, species distributions, temporal activity, camera hotspots, tiger analytics, and report export.
"""

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session
from typing import Optional

from db.database import get_db
from db.schemas import APIResponse
from services.analytics_service import AnalyticsService

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/overview")
@router.get("/summary")
def get_overview_metrics(db: Session = Depends(get_db)):
    """Get high-level dashboard metrics and KPI stats"""
    stats = AnalyticsService.get_overview_stats(db)
    return APIResponse(
        success=True,
        message="Overview stats retrieved",
        data=stats
    ).model_dump()


@router.get("/species")
def get_species_distribution(
    limit: int = Query(15, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get species population and frequency breakdown"""
    distribution = AnalyticsService.get_species_distribution(db, limit=limit)
    return APIResponse(
        success=True,
        message="Species distribution retrieved",
        data={"distribution": distribution}
    ).model_dump()


@router.get("/temporal")
def get_activity_timeline(
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db)
):
    """Get temporal activity timelines, hourly circadian distributions, and day/night split"""
    temporal = AnalyticsService.get_temporal_activity(db, days=days)
    return APIResponse(
        success=True,
        message="Temporal activity retrieved",
        data=temporal
    ).model_dump()


@router.get("/cameras")
def get_camera_hotspots(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get ranked camera trap wildlife activity hotspots"""
    hotspots = AnalyticsService.get_camera_hotspots(db, limit=limit)
    return APIResponse(
        success=True,
        message="Camera hotspots retrieved",
        data={"hotspots": hotspots}
    ).model_dump()


@router.get("/tigers")
def get_tiger_analytics(db: Session = Depends(get_db)):
    """Get dedicated tiger tracking and sighting analytics"""
    tiger_data = AnalyticsService.get_tiger_analytics(db)
    return APIResponse(
        success=True,
        message="Tiger analytics retrieved",
        data=tiger_data
    ).model_dump()


@router.get("/export")
def export_analytics_data(
    format: str = Query("json", pattern="^(json|csv)$"),
    db: Session = Depends(get_db)
):
    """Export complete wildlife intelligence dataset as JSON or CSV"""
    export_content = AnalyticsService.export_data(db, format=format)
    
    media_type = "text/csv" if format == "csv" else "application/json"
    filename = f"vanrakshak_wildlife_report_{format}.{format}"
    
    return Response(
        content=export_content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
