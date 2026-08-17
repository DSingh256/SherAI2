"""
VanRakshak AI - Database Setup and Session Management
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from config import settings
from db.models import Base

# Configure connect_args based on database type
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False
else:
    connect_args["connect_timeout"] = 10

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.SQLALCHEMY_ECHO,
    pool_pre_ping=True,  # Verify connections before using
    connect_args=connect_args
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully")


def get_db() -> Session:
    """Get database session - for FastAPI dependency injection"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Context manager for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reset_db():
    """Reset database - DROP ALL TABLES (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
    print("✓ Database tables dropped")


def health_check():
    """Test database connection"""
    try:
        with get_db_context() as db:
            db.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False
