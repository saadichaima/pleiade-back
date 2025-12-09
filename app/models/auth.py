# app/models/auth.py
from pydantic import BaseModel, EmailStr
from typing import Literal

Role = Literal["admin", "consultant"]

class AppUser(BaseModel):
    id: str
    email: EmailStr
    name: str
    role: Role
    active: bool = True
