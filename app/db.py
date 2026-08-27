"""
SQLAlchemy engine/session setup.

SQLite for this hackathon build — see the stack-lock
rationale: zero ops for a few hundred rows, and the engine is created from a
single DATABASE_URL string, so swapping to Postgres later is a config change,
not a rewrite.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

# check_same_thread=False is required for SQLite when FastAPI's test client
# and app run in different threads (e.g. under pytest + TestClient).
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they don't already exist."""
    from app import models  # noqa: F401  (ensures models are registered on Base)
    Base.metadata.create_all(bind=engine)
