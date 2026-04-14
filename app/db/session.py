from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
engine = create_engine(settings.SQLALCHEMY_DATABASE_URL,pool_timeout=30, pool_recycle=1800, pool_pre_ping=True,echo=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)