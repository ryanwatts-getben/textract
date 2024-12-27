import os
import logging
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from contextlib import contextmanager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("[db] DATABASE_URL environment variable is not set")

# SQLAlchemy engine configuration
ENGINE_CONFIG = {
    'pool_size': 5,
    'max_overflow': 10,
    'pool_timeout': 30,
    'pool_recycle': 1800,
    'pool_pre_ping': True,
    'poolclass': QueuePool
}

# Create engine and session factory
try:
    engine = create_engine(DATABASE_URL, **ENGINE_CONFIG)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("[db] Database engine initialized successfully")
except Exception as e:
    logger.error(f"[db] Failed to initialize database engine: {str(e)}")
    raise

@contextmanager
def get_session() -> Session:
    """
    Context manager for database sessions.
    Ensures proper handling of sessions including error cases.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"[db] Database session error: {str(e)}")
        raise
    finally:
        session.close()

# Export commonly used objects
__all__ = ['get_session']