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


def assign_user_to_team(user_email: str, team_id: str) -> Optional[AppUser]:
    """Assigne un utilisateur à une équipe."""
    return update_user(user_email, {"team_id": team_id})


def remove_user_from_team(user_email: str) -> Optional[AppUser]:
    """Retire un utilisateur de son équipe."""
    return update_user(user_email, {"team_id": None})


def get_users_by_team(team_id: str) -> List[AppUser]:
    """Récupère tous les utilisateurs d'une équipe."""
    container = get_users_container()
    items = container.query_items(
        query="SELECT * FROM c WHERE c.team_id = @team_id",
        parameters=[{"name": "@team_id", "value": team_id}],
        enable_cross_partition_query=True,
    )
    return [AppUser(**it) for it in items]


def get_users_without_team() -> List[AppUser]:
    """Récupère tous les consultants sans équipe."""
    container = get_users_container()
    items = container.query_items(
        query="SELECT * FROM c WHERE (NOT IS_DEFINED(c.team_id) OR c.team_id = null) AND c.role = 'consultant'",
        enable_cross_partition_query=True,
    )
    return [AppUser(**it) for it in items]
