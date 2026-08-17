"""
VanRakshak AI - Anomaly Detector
Statistical anomaly detection for camera traps.

Maintains per-camera baselines (rolling averages of animal/human/vehicle
percentages). Compares current period against baseline using z-score.
Flags significant deviations for ranger attention.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AnomalyMetric:
    """A single metric analyzed for anomalies"""
    metric_name: str
    current_value: float
    baseline_avg: float
    baseline_std: float
    z_score: float
    is_anomalous: bool
    description: str

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "current_value": round(self.current_value, 4),
            "baseline_avg": round(self.baseline_avg, 4),
            "baseline_std": round(self.baseline_std, 4),
            "z_score": round(self.z_score, 4),
            "is_anomalous": self.is_anomalous,
            "description": self.description,
        }


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection for a camera"""
    camera_id: str
    period_start: datetime
    period_end: datetime
    is_anomalous: bool
    anomaly_score: float  # 0.0 to 1.0, max of normalized z-scores
    metrics: List[AnomalyMetric] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "is_anomalous": self.is_anomalous,
            "anomaly_score": round(self.anomaly_score, 4),
            "metrics": [m.to_dict() for m in self.metrics],
        }


class AnomalyDetectorService:
    """
    Statistical anomaly detection service.
    
    In a real system, this would query the database for historical data
    and calculate real z-scores. For the hackathon, we simulate baselines
    but calculate actual z-scores based on provided current values.
    """
    
    # Configurable anomaly threshold (z-score)
    Z_SCORE_THRESHOLD = 2.5
    
    # Simulated baselines per camera (for hackathon demo)
    # Format: {camera_id: {metric: (avg, std_dev)}}
    SIMULATED_BASELINES = {
        "CAM001": {
            "animal_count_daily": (15.0, 3.5),
            "human_count_daily": (0.5, 0.8),
            "vehicle_count_daily": (0.1, 0.3),
        },
        "CAM007": {  # High activity zone
            "animal_count_daily": (45.0, 12.0),
            "human_count_daily": (2.0, 1.5),
            "vehicle_count_daily": (1.0, 0.5),
        },
        "CAM012": {  # Core zone - humans rare
            "animal_count_daily": (25.0, 8.0),
            "human_count_daily": (0.1, 0.2),
            "vehicle_count_daily": (0.05, 0.1),
        },
        "CAM022": {  # Near boundary - more humans
            "animal_count_daily": (10.0, 4.0),
            "human_count_daily": (5.0, 2.5),
            "vehicle_count_daily": (2.5, 1.2),
        }
    }
    
    # Default baseline for unknown cameras
    DEFAULT_BASELINE = {
        "animal_count_daily": (20.0, 5.0),
        "human_count_daily": (1.0, 1.0),
        "vehicle_count_daily": (0.5, 0.5),
    }

    @staticmethod
    def detect_anomalies(
        camera_id: str,
        current_metrics: Dict[str, float],
        period_start: datetime,
        period_end: datetime,
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies by comparing current metrics to historical baseline.
        
        Args:
            camera_id: Camera identifier
            current_metrics: Dict mapping metric names to current values
            period_start: Start of analysis period
            period_end: End of analysis period
            
        Returns:
            AnomalyDetectionResult
        """
        
        # Get baseline for this camera
        baseline = AnomalyDetectorService.SIMULATED_BASELINES.get(
            camera_id, AnomalyDetectorService.DEFAULT_BASELINE
        )
        
        analyzed_metrics = []
        is_overall_anomalous = False
        max_z_score = 0.0
        
        for metric_name, current_val in current_metrics.items():
            if metric_name in baseline:
                avg, std_dev = baseline[metric_name]
                
                # Prevent division by zero
                if std_dev < 0.01:
                    std_dev = 0.01
                    
                # Calculate z-score
                # (current - average) / standard_deviation
                z_score = (current_val - avg) / std_dev
                
                # We typically only care about positive anomalies (spikes in activity)
                # except maybe for animals where a sudden drop could indicate poaching/disturbance
                is_anomalous = False
                
                if metric_name.startswith("animal"):
                    # For animals, both spikes and sudden drops are anomalous
                    is_anomalous = abs(z_score) > AnomalyDetectorService.Z_SCORE_THRESHOLD
                else:
                    # For humans/vehicles, we mainly care about spikes
                    is_anomalous = z_score > AnomalyDetectorService.Z_SCORE_THRESHOLD
                    
                if is_anomalous:
                    is_overall_anomalous = True
                    
                abs_z = abs(z_score)
                if abs_z > max_z_score:
                    max_z_score = abs_z
                    
                # Generate description
                if is_anomalous:
                    direction = "Spike" if z_score > 0 else "Drop"
                    desc = f"⚠ {direction} in {metric_name.replace('_', ' ')} (z={z_score:.1f})"
                else:
                    desc = f"✓ {metric_name.replace('_', ' ')} normal"
                    
                analyzed_metrics.append(AnomalyMetric(
                    metric_name=metric_name,
                    current_value=current_val,
                    baseline_avg=avg,
                    baseline_std=std_dev,
                    z_score=z_score,
                    is_anomalous=is_anomalous,
                    description=desc
                ))
        
        # Normalize max z-score to 0-1 range for anomaly_score
        # A z-score of 0 -> 0.0, z-score of 3 -> 0.6, z-score of 5+ -> 1.0
        anomaly_score = min(1.0, max_z_score / 5.0)
        
        return AnomalyDetectionResult(
            camera_id=camera_id,
            period_start=period_start,
            period_end=period_end,
            is_anomalous=is_overall_anomalous,
            anomaly_score=anomaly_score,
            metrics=analyzed_metrics
        )
