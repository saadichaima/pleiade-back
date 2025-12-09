# app/auth_ms.py
import requests
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
from app.models.auth import AppUser
from app.services.users_service import get_user_by_email

security = HTTPBearer(auto_error=False)

_jwks = None

def _get_jwks():
    global _jwks
    if _jwks is None:
        if not settings.AAD_JWKS_URL:
            raise RuntimeError("AAD_JWKS_URL non configuré.")
        r = requests.get(settings.AAD_JWKS_URL, timeout=10)
        r.raise_for_status()
        _jwks = r.json()
    return _jwks

def _decode_aad_token(token: str) -> dict:
    jwks = _get_jwks()
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide (header).")

    kid = unverified_header.get("kid")
    key = None
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            key = k
            break
    if not key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clé de signature introuvable.")

    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=settings.AAD_CLIENT_ID,
            issuer=settings.AAD_ISSUER,
        )
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token invalide: {e}")

    return claims

async def get_aad_claims(creds: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if creds is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Non authentifié.")
    token = creds.credentials
    return _decode_aad_token(token)

async def get_current_user(claims: dict = Depends(get_aad_claims)) -> AppUser:
    email = (
        (claims.get("preferred_username") or claims.get("email") or "").lower().strip()
    )
    if not email:
        raise HTTPException(status_code=403, detail="Email absent du token Microsoft.")

    user = get_user_by_email(email)
    if not user or not user.active:
        raise HTTPException(
            status_code=403,
            detail="Vous n'êtes pas autorisé à utiliser cette application.",
        )
    return user

async def require_admin(user: AppUser = Depends(get_current_user)) -> AppUser:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès réservé aux administrateurs.")
    return user
