# app/services/teams_service.py
"""
Service de gestion des équipes dans Cosmos DB.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from azure.cosmos import exceptions

from app.services.cosmos_client import get_db
from app.models.team import Team
from app.config import settings


def _get_teams_container():
    """Récupère le conteneur Cosmos pour les équipes."""
    db = get_db()
    return db.get_container_client(settings.COSMOS_CONTAINER_TEAMS)


def create_team(name: str, description: Optional[str] = None, manager_id: Optional[str] = None) -> Team:
    """Crée une nouvelle équipe."""
    container = _get_teams_container()

    team_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    team_data = {
        "id": team_id,
        "team_id": team_id,  # partition key
        "name": name,
        "description": description or "",
        "manager_id": manager_id,
        "members": [],
        "created_at": now,
        "updated_at": now,
        "active": True,
    }

    container.create_item(body=team_data)
    return Team(**team_data)


def get_team(team_id: str) -> Optional[Team]:
    """Récupère une équipe par son ID."""
    container = _get_teams_container()

    try:
        item = container.read_item(item=team_id, partition_key=team_id)
        return Team(**item)
    except exceptions.CosmosResourceNotFoundError:
        return None


def list_all_teams() -> List[Team]:
    """Liste toutes les équipes actives."""
    container = _get_teams_container()

    query = "SELECT * FROM c WHERE c.active = true ORDER BY c.name"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))

    return [Team(**item) for item in items]


def update_team(team_id: str, data: Dict[str, Any]) -> Optional[Team]:
    """Met à jour les informations d'une équipe."""
    container = _get_teams_container()

    try:
        # Récupérer l'équipe existante
        existing = container.read_item(item=team_id, partition_key=team_id)

        # Mettre à jour les champs autorisés
        if "name" in data and data["name"] is not None:
            existing["name"] = data["name"]
        if "description" in data:
            existing["description"] = data["description"]
        if "manager_id" in data:
            existing["manager_id"] = data["manager_id"]
        if "active" in data:
            existing["active"] = data["active"]

        existing["updated_at"] = datetime.utcnow().isoformat() + "Z"

        # Sauvegarder
        updated = container.replace_item(item=team_id, body=existing)
        return Team(**updated)
    except exceptions.CosmosResourceNotFoundError:
        return None


def delete_team(team_id: str) -> bool:
    """Supprime une équipe (soft delete)."""
    container = _get_teams_container()

    try:
        existing = container.read_item(item=team_id, partition_key=team_id)
        existing["active"] = False
        existing["updated_at"] = datetime.utcnow().isoformat() + "Z"
        container.replace_item(item=team_id, body=existing)
        return True
    except exceptions.CosmosResourceNotFoundError:
        return False


def add_member_to_team(team_id: str, user_email: str) -> Optional[Team]:
    """Ajoute un membre à une équipe."""
    container = _get_teams_container()

    try:
        existing = container.read_item(item=team_id, partition_key=team_id)

        # Vérifier si le membre n'est pas déjà dans l'équipe
        if user_email not in existing["members"]:
            existing["members"].append(user_email)
            existing["updated_at"] = datetime.utcnow().isoformat() + "Z"
            updated = container.replace_item(item=team_id, body=existing)
            return Team(**updated)

        return Team(**existing)
    except exceptions.CosmosResourceNotFoundError:
        return None


def remove_member_from_team(team_id: str, user_email: str) -> Optional[Team]:
    """Retire un membre d'une équipe."""
    container = _get_teams_container()

    try:
        existing = container.read_item(item=team_id, partition_key=team_id)

        # Retirer le membre s'il est présent
        if user_email in existing["members"]:
            existing["members"].remove(user_email)
            existing["updated_at"] = datetime.utcnow().isoformat() + "Z"
            updated = container.replace_item(item=team_id, body=existing)
            return Team(**updated)

        return Team(**existing)
    except exceptions.CosmosResourceNotFoundError:
        return None


def get_team_by_manager(manager_id: str) -> Optional[Team]:
    """Récupère l'équipe gérée par un manager."""
    container = _get_teams_container()

    query = "SELECT * FROM c WHERE c.manager_id = @manager_id AND c.active = true"
    parameters = [{"name": "@manager_id", "value": manager_id}]

    items = list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    ))

    if items:
        return Team(**items[0])
    return None


def get_teams_by_member(user_email: str) -> List[Team]:
    """Récupère toutes les équipes dont un utilisateur est membre."""
    container = _get_teams_container()

    query = "SELECT * FROM c WHERE ARRAY_CONTAINS(c.members, @email) AND c.active = true"
    parameters = [{"name": "@email", "value": user_email}]

    items = list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    ))

    return [Team(**item) for item in items]
