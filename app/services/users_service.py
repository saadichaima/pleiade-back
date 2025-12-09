# app/services/users_service.py
from typing import Optional, List
from app.models.auth import AppUser
from app.services.cosmos_client import get_users_container

def get_user_by_email(email: str) -> Optional[AppUser]:
    email = (email or "").lower().strip()
    if not email:
        return None
    container = get_users_container()
    try:
        doc = container.read_item(item=email, partition_key=email)
        return AppUser(**doc)
    except Exception:
        return None

def list_users() -> List[AppUser]:
    container = get_users_container()
    items = container.query_items(
        query="SELECT * FROM c ORDER BY c.email",
        enable_cross_partition_query=True,
    )
    return [AppUser(**it) for it in items]

def create_user(user: AppUser) -> AppUser:
    container = get_users_container()
    doc = user.model_dump()
    container.create_item(doc)
    return user

def update_user(email: str, data: dict) -> Optional[AppUser]:
    email = email.lower()
    container = get_users_container()
    try:
        doc = container.read_item(item=email, partition_key=email)
    except Exception:
        return None
    doc.update(data)
    container.replace_item(item=doc["id"], body=doc)
    return AppUser(**doc)

def delete_user(email: str) -> bool:
    email = email.lower()
    container = get_users_container()
    try:
        container.delete_item(item=email, partition_key=email)
        return True
    except Exception:
        return False
