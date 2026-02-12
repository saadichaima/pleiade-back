# app/models/auth.py
from pydantic import BaseModel, EmailStr
from typing import Literal, Optional

Role = Literal["admin", "consultant", "manager"]

class AppUser(BaseModel):
    id: str
    email: EmailStr
    name: str
    role: Role
    active: bool = True
    team_id: Optional[str] = None
    avatar_url: Optional[str] = None
