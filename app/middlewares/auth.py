from fastapi import Header, HTTPException, Depends
from app.core.security import verify_supabase_token

async def current_user(authorization: str = Header(...)):
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        claims = await verify_supabase_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    # devolvemos claims mínimos
    return {"sub": claims["sub"], "email": claims.get("email")}