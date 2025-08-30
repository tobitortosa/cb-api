import time, httpx
from jose import jwt
from functools import lru_cache
from app.core.config import settings

@lru_cache(maxsize=1)
def _jwks_cached():
    # Cachea JWKS ~5 min por proceso (simple para MVP)
    return {"jwks": None, "ts": 0}

async def get_jwks():
    cache = _jwks_cached()
    if not cache["jwks"] or time.time() - cache["ts"] > 300:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(settings.supabase_jwks_url)
            r.raise_for_status()
            jwks_data = r.json()
            cache["jwks"] = jwks_data
            cache["ts"] = time.time()
    return cache["jwks"]

async def verify_supabase_token(token: str) -> dict:
    try:
        # Primero intentamos obtener el header para ver el algoritmo
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg", "")
        
        if algorithm == "HS256":
            # Token firmado con HMAC - usar el JWT secret
            claims = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                issuer=f"{settings.supabase_project_url}/auth/v1"
            )
            return claims
            
        elif algorithm.startswith("RS"):
            # Token firmado con RSA - usar JWKS
            jwks = await get_jwks()
            
            key = None
            for k in jwks["keys"]:
                if k["kid"] == header.get("kid"):
                    key = k
                    break
            if not key:
                raise ValueError("JWKS key not found")

            claims = jwt.decode(
                token,
                key,
                algorithms=[key["alg"]],
                audience="authenticated",
                options={"verify_exp": True},
                issuer=f"{settings.supabase_project_url}/auth/v1"
            )
            return claims
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
    except Exception as e:
        raise