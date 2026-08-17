"""
VanRakshak AI - Main FastAPI Application
Entry point for backend API
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
import logging

# Import configuration and database
from config import settings
from db.database import init_db, health_check, get_db

# Import routes
from api.routes import images, quality, review, analytics, cameras, alerts, audit, reidentification, detections
from core.pipeline import ProcessingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    # Startup
    logger.info("🚀 VanRakshak AI Starting...")
    try:
        init_db()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
    
    if health_check():
        logger.info("✓ Database connection healthy")
    else:
        logger.warning("⚠ Database connection check failed")
    
    yield
    
    # Shutdown
    logger.info("🛑 VanRakshak AI Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered wildlife conservation platform",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ ROOT ENDPOINTS ============

@app.get("/")
async def root():
    """API root - system information"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "images": "/api/images",
            "quality": "/api/quality",
            "detections": "/api/detections",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    if health_check():
        return {
            "status": "healthy",
            "database": "connected"
        }
    else:
        raise HTTPException(status_code=503, detail="Database connection failed")


# ============ REGISTER ROUTES ============

app.include_router(images.router)
app.include_router(quality.router)
app.include_router(detections.router)
app.include_router(review.router)
app.include_router(analytics.router)
app.include_router(cameras.router)
app.include_router(alerts.router)
app.include_router(audit.router)
app.include_router(reidentification.router)


# ============ PIPELINE TRIGGER ============

@app.post("/api/pipeline/process/{image_id}")
async def process_image_pipeline(
    image_id: str,
    db: Session = Depends(get_db)
):
    """Trigger the full AI processing pipeline for an image"""
        
    result = ProcessingPipeline.process_image(image_id, db)
    
    if not result.success:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Pipeline failed: {result.error_message}",
                "data": result.to_dict()
            }
        )
        
    return {
        "success": True,
        "message": "Pipeline completed successfully",
        "data": result.to_dict()
    }


# ============ ERROR HANDLERS ============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc)
        }
    )


# ============ STARTUP MESSAGE ============

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {settings.APP_NAME} on {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
