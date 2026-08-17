"""
VanRakshak AI - Analytics Service
Calculates dashboard metrics and conservation insights.
"""

from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, extract
from datetime import datetime, timedelta

from db.models import Image, Decision, Detection


class AnalyticsService:
    """
    Service for calculating dashboard metrics and wildlife statistics.
    """

    @staticmethod
    def get_overview_stats(db: Session) -> Dict:
        """Get high-level dashboard metrics"""
        total_images = db.query(Image).count()
        
        # Get decision breakdowns
        decisions = db.query(
            Decision.decision, 
            func.count(Decision.id)
        ).group_by(Decision.decision).all()
        
        decision_counts = {d[0]: d[1] for d in decisions}
        
        # Get tiger count
        tiger_count = db.query(Decision).filter(Decision.is_tiger == True).count()
        
        return {
            "total_images": total_images,
            "auto_accepted": decision_counts.get("auto_accept", 0),
            "pending_review": decision_counts.get("human_review", 0) + decision_counts.get("uncertain", 0),
            "human_reviewed": decision_counts.get("human_reviewed", 0),
            "tigers_detected": tiger_count
        }

    @staticmethod
    def get_species_distribution(db: Session) -> List[Dict]:
        """Get counts of each species detected"""
        species_counts = db.query(
            Decision.species,
            func.count(Decision.id).label('count')
        ).filter(
            Decision.species.isnot(None),
            Decision.species != 'none'
        ).group_by(
            Decision.species
        ).order_by(
            desc('count')
        ).limit(10).all()
        
        return [{"species": s[0], "count": s[1]} for s in species_counts]
        
    @staticmethod
    def get_activity_timeline(db: Session, days: int = 7) -> List[Dict]:
        """Get detection counts grouped by day"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Query images with detections by date
        daily_activity = db.query(
            func.date(Image.timestamp).label('date'),
            func.count(Image.id).label('total'),
        ).filter(
            Image.timestamp >= cutoff_date
        ).group_by(
            func.date(Image.timestamp)
        ).order_by(
            func.date(Image.timestamp)
        ).all()
        
        return [
            {
                "date": str(d.date),
                "count": d.total
            }
            for d in daily_activity
        ]
        
    @staticmethod
    def get_detection_types(db: Session) -> Dict:
        """Get breakdown of animal vs human vs vehicle"""
        types = db.query(
            Detection.object_type,
            func.count(Detection.id)
        ).group_by(Detection.object_type).all()
        
        return {t[0]: t[1] for t in types}
