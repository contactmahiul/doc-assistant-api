from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends,HTTPException, status
from jose import jwt, JWTError
from datetime import datetime , timedelta , timezone
from sqlalchemy.orm import Session
from app.api import deps
from app.model import User
from app.core.config import settings

pwd_context= CryptContext(schemes=["bcrypt"], deprecated="auto")
def hash_password(password: str)-> str:
    return pwd_context.hash(password)
def verify_password(plain_password:str, hashed_password: str)-> bool:
    return pwd_context.verify(plain_password, hashed_password)




def create_token(data:dict)-> str:
    to_encode= data.copy()
    expire= datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(deps.get_db)
        ):
    
    credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail = "invalid or expired token")
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, settings.ALGORITHM)
        user_id = payload.get("sub")
        if not user_id:
            raise credential_exception
        
    except JWTError:
        raise credential_exception
    
    user = db.query(User).filter(User.id == int(user_id), User.is_deleted == False).first()
    if not user:
        raise credential_exception
    return user



