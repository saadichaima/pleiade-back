# app/models/team.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class Team(BaseModel):
    """Modèle d'équipe dans Cosmos DB"""
    id: str                              # team_id (UUID)
    team_id: str                         # partition key (même valeur que id)
    name: str                            # Nom de l'équipe
    description: Optional[str] = None    # Description de l'équipe
    manager_id: Optional[str] = None     # Email du manager
    members: List[str] = []              # Liste des emails des consultants
    created_at: str                      # ISO 8601 timestamp
    updated_at: str                      # ISO 8601 timestamp
    active: bool = True                  # Statut de l'équipe


class CreateTeamInput(BaseModel):
    """Input pour créer une équipe"""
    name: str
    description: Optional[str] = None
    manager_id: Optional[str] = None


class UpdateTeamInput(BaseModel):
    """Input pour mettre à jour une équipe"""
    name: Optional[str] = None
    description: Optional[str] = None
    manager_id: Optional[str] = None
    active: Optional[bool] = None


class AddMemberInput(BaseModel):
    """Input pour ajouter un membre à une équipe"""
    email: str
