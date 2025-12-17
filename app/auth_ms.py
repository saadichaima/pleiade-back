# app/auth_ms.py
import time
import requests
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.models.auth import AppUser
from app.services.users_service import get_user_by_email

security = HTTPBearer(auto_error=False)

# Cache JWKS + TTL (rotation clés)
_JWKS_CACHE = {"value": None, "ts": 0.0}
_JWKS_TTL_SECONDS = 6 * 60 * 60  # 6h


def _fetch_jwks() -> dict:
    if not settings.AAD_JWKS_URL:
        raise RuntimeError("AAD_JWKS_URL non configuré.")
    r = requests.get(settings.AAD_JWKS_URL, timeout=10)
    r.raise_for_status()
    return r.json()


def _get_jwks(force_refresh: bool = False) -> dict:
    now = time.time()
    if (
        force_refresh
        or _JWKS_CACHE["value"] is None
        or (now - float(_JWKS_CACHE["ts"])) > _JWKS_TTL_SECONDS
    ):
        _JWKS_CACHE["value"] = _fetch_jwks()
        _JWKS_CACHE["ts"] = now
    return _JWKS_CACHE["value"]


def _pick_signing_key(jwks: dict, kid: str | None):
    if not kid:
        return None
    for k in jwks.get("keys", []) or []:
        if k.get("kid") == kid:
            return k
    return None


def _expected_audience() -> str:
    # Audience API (recommandé) ; fallback sur client id
    aud = (getattr(settings, "AAD_AUDIENCE", "") or "").strip()
    if aud:
        return aud
    return (settings.AAD_CLIENT_ID or "").strip()


def _expected_issuers() -> set[str]:
    """
    Accepte issuer v2 OU v1.
    - v2: https://login.microsoftonline.com/<tenantId>/v2.0
    - v1: https://sts.windows.net/<tenantId>/
    """
    issuers: set[str] = set()

    iss_v2 = (settings.AAD_ISSUER or "").strip()
    if iss_v2:
        issuers.add(iss_v2)

    tenant = (settings.AAD_TENANT_ID or "").strip()
    if tenant:
        issuers.add(f"https://sts.windows.net/{tenant}/")

    return issuers


def _decode_aad_token(token: str) -> dict:
    # 1) Header
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide (header).",
        )

    kid = unverified_header.get("kid")

    # 2) JWKS (cache + refresh si kid introuvable)
    jwks = _get_jwks(force_refresh=False)
    key = _pick_signing_key(jwks, kid)
    if not key:
        jwks = _get_jwks(force_refresh=True)
        key = _pick_signing_key(jwks, kid)

    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé de signature introuvable (JWKS).",
        )

    aud = _expected_audience()
    if not aud:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration AAD incomplète: AAD_AUDIENCE/AAD_CLIENT_ID manquant.",
        )

    allowed_issuers = _expected_issuers()
    if not allowed_issuers:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration AAD incomplète: AAD_TENANT_ID/AAD_ISSUER manquant.",
        )

    # 3) Decode en vérifiant signature + audience, et vérification issuer manuelle
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=aud,
            options={"verify_iss": False},
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token invalide: {e}",
        )

    token_iss = (claims.get("iss") or "").strip()
    if token_iss not in allowed_issuers:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Issuer invalide: {token_iss}",
        )

    return claims


def _extract_email_from_claims(claims: dict) -> str:
    """
    Sources possibles selon tenant/config :
    - preferred_username (courant)
    - email
    - upn
    - unique_name
    - emails (liste)
    """
    candidates = [
        claims.get("preferred_username"),
        claims.get("email"),
        claims.get("upn"),
        claims.get("unique_name"),
    ]

    emails = claims.get("emails")
    if isinstance(emails, list) and emails:
        candidates.insert(0, emails[0])

    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.lower().strip()

    return ""


async def get_aad_claims(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Non authentifié.",
        )
    token = creds.credentials
    return _decode_aad_token(token)


async def get_current_user(claims: dict = Depends(get_aad_claims)) -> AppUser:
    email = _extract_email_from_claims(claims)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email absent du token Microsoft.",
        )

    user = get_user_by_email(email)
    if not user or not user.active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'êtes pas autorisé à utiliser cette application.",
        )
    return user


async def require_admin(user: AppUser = Depends(get_current_user)) -> AppUser:
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès réservé aux administrateurs.",
        )
    return user
