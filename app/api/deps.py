from typing import Generator
from app.db.session import SessionLocal

def get_db() -> Generator:
    """
    Creates a new SQLAlchemy session for each request.
    Yields the session to the endpoint, then closes it.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()