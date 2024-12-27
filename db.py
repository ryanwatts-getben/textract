import os
import logging
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

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
    'pool_size': 5,  # Maximum number of connections to keep in the pool
    'max_overflow': 10,  # Maximum number of connections that can be created beyond pool_size
    'pool_timeout': 30,  # Seconds to wait before giving up on getting a connection from the pool
    'pool_recycle': 1800,  # Recycle connections after 30 minutes
    'pool_pre_ping': True,  # Enable connection health checks
    'poolclass': QueuePool  # Use QueuePool for connection pooling
}

class DatabaseConnection:
    _engine: Optional[Engine] = None
    _SessionLocal: Optional[sessionmaker] = None

    @classmethod
    def initialize(cls) -> None:
        """Initialize the database engine and session factory."""
        try:
            if not cls._engine:
                logger.info("[db] Initializing database engine")
                cls._engine = create_engine(
                    DATABASE_URL,
                    **ENGINE_CONFIG
                )
                cls._SessionLocal = sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=cls._engine
                )
                logger.info("[db] Database engine initialized successfully")
        except Exception as e:
            logger.error(f"[db] Failed to initialize database engine: {str(e)}")
            raise

    @classmethod
    def get_engine(cls) -> Engine:
        """Get the SQLAlchemy engine instance."""
        if not cls._engine:
            cls.initialize()
        return cls._engine

    @classmethod
    def get_session(cls) -> Session:
        """Create a new database session."""
        if not cls._SessionLocal:
            cls.initialize()
        return cls._SessionLocal()

def get_db() -> Session:
    """
    Get a database session with automatic cleanup.
    Use this as a context manager or in a try-finally block.
    
    Example:
        with get_db() as db:
            result = db.query(Model).all()
    """
    db = DatabaseConnection.get_session()
    try:
        yield db
    finally:
        try:
            db.close()
        except SQLAlchemyError as e:
            logger.error(f"[db] Error closing database session: {str(e)}")

# Initialize the database connection on module import
try:
    DatabaseConnection.initialize()
except Exception as e:
    logger.error(f"[db] Failed to initialize database connection: {str(e)}")
    raise

# Export commonly used objects
__all__ = ['DatabaseConnection', 'get_db']
