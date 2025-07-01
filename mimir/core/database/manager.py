"""Database manager for async SQLAlchemy operations."""

import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.pool import StaticPool

from ...config import get_settings
from ..models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine: AsyncEngine = None
        self.session_factory: async_sessionmaker = None
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        if self.engine is not None:
            return  # Already initialized
        
        database_url = self.settings.database.url
        
        # Convert SQLite URL to async version
        if database_url.startswith("sqlite:///"):
            database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        elif database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        
        # Create engine
        engine_kwargs = {
            "echo": self.settings.database.echo,
        }
        
        # SQLite specific settings
        if "sqlite" in database_url:
            engine_kwargs.update({
                "poolclass": StaticPool,
                "connect_args": {
                    "check_same_thread": False,
                }
            })
        else:
            # PostgreSQL settings
            engine_kwargs.update({
                "pool_size": self.settings.database.pool_size,
                "max_overflow": self.settings.database.max_overflow,
            })
        
        self.engine = create_async_engine(database_url, **engine_kwargs)
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        # Create tables
        await self.create_tables()
        
        logger.info(f"Database initialized: {database_url}")
    
    async def create_tables(self) -> None:
        """Create all tables defined in models."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")
    
    async def close(self) -> None:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session context manager."""
        if not self.session_factory:
            await self.initialize()
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


# Global database manager instance
_db_manager: DatabaseManager = None


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session.
    
    This is a dependency function for FastAPI that provides
    database sessions to route handlers.
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    
    async with _db_manager.get_session() as session:
        yield session