# app/routers/auth_ms.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Literal
from pydantic import BaseModel, EmailStr

from app.auth_ms import get_current_user, require_admin
from app.models.auth import AppUser
from app.services.users_service import list_users, create_user, update_user, delete_user

router = APIRouter()

@router.get("/auth/me", response_model=AppUser)
def get_me(user: AppUser = Depends(get_current_user)):
    return user

class NewUser(BaseModel):
    email: EmailStr
    name: str
    role: Literal["admin", "consultant", "manager"]
    active: bool = True

@router.get("/admin/users", response_model=list[AppUser])
def admin_list_users(_: AppUser = Depends(require_admin)):
    return list_users()

@router.post("/admin/users", response_model=AppUser)
def admin_create_user(body: NewUser, _: AppUser = Depends(require_admin)):
    u = AppUser(
        id=body.email.lower(),
        email=body.email.lower(),
        name=body.name,
        role=body.role,
        active=body.active,
    )
    return create_user(u)

@router.patch("/admin/users/{email}", response_model=AppUser)
def admin_update_user(email: str, body: NewUser, _: AppUser = Depends(require_admin)):
    upd = {
        "name": body.name,
        "role": body.role,
        "active": body.active,
    }
    u = update_user(email, upd)
    if not u:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    return u

@router.delete("/admin/users/{email}")
def admin_delete_user(email: str, _: AppUser = Depends(require_admin)):
    ok = delete_user(email)
    if not ok:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    return {"ok": True}
