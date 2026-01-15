# app/routers/teams.py
"""
Endpoints pour la gestion des équipes.
- Admins : CRUD complet sur toutes les équipes
- Managers : Gestion de leur propre équipe
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict

from app.auth_ms import require_admin, require_manager, get_current_user
from app.models.auth import AppUser
from app.models.team import Team, CreateTeamInput, UpdateTeamInput, AddMemberInput
from app.services.teams_service import (
    create_team,
    get_team,
    list_all_teams,
    update_team,
    delete_team,
    add_member_to_team,
    remove_member_from_team,
    get_team_by_manager,
)
from app.services.users_service import (
    assign_user_to_team,
    remove_user_from_team,
    get_users_by_team,
    get_users_without_team,
    update_user,
)
from app.services.cosmos_client import get_projects_container

router = APIRouter()


# ========== ENDPOINTS ADMIN ==========

@router.get("/admin/teams", response_model=List[Team])
async def admin_list_teams(_: AppUser = Depends(require_admin)):
    """Liste toutes les équipes (admin only)."""
    return list_all_teams()


@router.get("/admin/teams/stats")
async def admin_get_teams_stats(_: AppUser = Depends(require_admin)):
    """
    Récupère les statistiques de toutes les équipes (nombre de dossiers générés par équipe).
    Retourne un dictionnaire {team_id: count}.
    """
    teams = list_all_teams()
    projects_container = get_projects_container()

    stats: Dict[str, int] = {}

    for team in teams:
        # Récupérer tous les emails de l'équipe (membres + manager)
        all_emails = team.members.copy()
        if team.manager_id and team.manager_id not in all_emails:
            all_emails.append(team.manager_id)

        if not all_emails:
            stats[team.team_id] = 0
            continue

        # Query pour compter les projets
        placeholders = ", ".join([f"@email{i}" for i in range(len(all_emails))])
        query = f"""
        SELECT VALUE COUNT(1)
        FROM c
        WHERE c.user_id IN ({placeholders})
        """

        parameters = [
            {"name": f"@email{i}", "value": email}
            for i, email in enumerate(all_emails)
        ]

        result = list(projects_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        ))

        count = result[0] if result else 0
        stats[team.team_id] = count

    return stats


@router.post("/admin/teams", response_model=Team)
async def admin_create_team(
    body: CreateTeamInput,
    _: AppUser = Depends(require_admin)
):
    """
    Crée une nouvelle équipe (admin only).
    Si un manager_id est fourni et que l'utilisateur est consultant,
    son rôle sera automatiquement changé en "manager".
    """
    team = create_team(
        name=body.name,
        description=body.description,
        manager_id=body.manager_id,
    )

    # Si un manager est assigné, mettre à jour son rôle et team_id
    if body.manager_id:
        update_user(body.manager_id, {
            "role": "manager",
            "team_id": team.team_id
        })

    return team


@router.get("/admin/teams/{team_id}", response_model=Team)
async def admin_get_team(
    team_id: str,
    _: AppUser = Depends(require_admin)
):
    """Récupère une équipe par son ID (admin only)."""
    team = get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Équipe introuvable.")
    return team


@router.patch("/admin/teams/{team_id}", response_model=Team)
async def admin_update_team(
    team_id: str,
    body: UpdateTeamInput,
    _: AppUser = Depends(require_admin)
):
    """
    Met à jour une équipe (admin only).
    Si le manager change, met à jour les rôles en conséquence.
    """
    # Récupérer l'équipe actuelle
    current_team = get_team(team_id)
    if not current_team:
        raise HTTPException(status_code=404, detail="Équipe introuvable.")

    old_manager_id = current_team.manager_id
    new_manager_id = body.manager_id if body.manager_id is not None else old_manager_id

    # Mettre à jour l'équipe
    update_data = {}
    if body.name is not None:
        update_data["name"] = body.name
    if body.description is not None:
        update_data["description"] = body.description
    if body.manager_id is not None:
        update_data["manager_id"] = body.manager_id
    if body.active is not None:
        update_data["active"] = body.active

    team = update_team(team_id, update_data)
    if not team:
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour.")

    # Si le manager a changé
    if old_manager_id != new_manager_id:
        # Retirer l'ancien manager (le mettre en consultant si pas admin)
        if old_manager_id:
            old_manager = update_user(old_manager_id, {
                "team_id": None,
                "role": "consultant"  # Rétrogradation
            })

        # Promouvoir le nouveau manager
        if new_manager_id:
            update_user(new_manager_id, {
                "role": "manager",
                "team_id": team_id
            })

    return team


@router.delete("/admin/teams/{team_id}")
async def admin_delete_team(
    team_id: str,
    _: AppUser = Depends(require_admin)
):
    """
    Supprime une équipe (soft delete - admin only).
    Retire tous les membres de l'équipe et rétrograde le manager en consultant.
    """
    team = get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Équipe introuvable.")

    # Retirer tous les membres
    for member_email in team.members:
        remove_user_from_team(member_email)

    # Rétrograder le manager
    if team.manager_id:
        update_user(team.manager_id, {
            "role": "consultant",
            "team_id": None
        })

    # Soft delete de l'équipe
    success = delete_team(team_id)
    if not success:
        raise HTTPException(status_code=500, detail="Erreur lors de la suppression.")

    return {"ok": True}


@router.post("/admin/teams/{team_id}/members", response_model=Team)
async def admin_add_member(
    team_id: str,
    body: AddMemberInput,
    _: AppUser = Depends(require_admin)
):
    """Ajoute un consultant à une équipe (admin only)."""
    # Vérifier que l'équipe existe
    team = get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Équipe introuvable.")

    # Ajouter le membre à l'équipe
    team = add_member_to_team(team_id, body.email)
    if not team:
        raise HTTPException(status_code=500, detail="Erreur lors de l'ajout du membre.")

    # Mettre à jour le team_id de l'utilisateur
    assign_user_to_team(body.email, team_id)

    return team


@router.delete("/admin/teams/{team_id}/members/{email}")
async def admin_remove_member(
    team_id: str,
    email: str,
    _: AppUser = Depends(require_admin)
):
    """Retire un consultant d'une équipe (admin only)."""
    # Vérifier que l'équipe existe
    team = get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Équipe introuvable.")

    # Retirer le membre de l'équipe
    team = remove_member_from_team(team_id, email)
    if not team:
        raise HTTPException(status_code=500, detail="Erreur lors du retrait du membre.")

    # Mettre à jour le team_id de l'utilisateur
    remove_user_from_team(email)

    return {"ok": True}


@router.get("/admin/teams/{team_id}/members", response_model=List[AppUser])
async def admin_get_team_members(
    team_id: str,
    _: AppUser = Depends(require_admin)
):
    """Récupère la liste des membres d'une équipe (admin only)."""
    team = get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Équipe introuvable.")

    return get_users_by_team(team_id)


@router.get("/admin/consultants-without-team", response_model=List[AppUser])
async def admin_get_consultants_without_team(_: AppUser = Depends(require_admin)):
    """Récupère la liste des consultants sans équipe (admin only)."""
    return get_users_without_team()


# ========== ENDPOINTS MANAGER ==========

@router.get("/manager/my-team", response_model=Team)
async def manager_get_my_team(user: AppUser = Depends(require_manager)):
    """Récupère l'équipe du manager connecté."""
    team = get_team_by_manager(user.email)
    if not team:
        raise HTTPException(
            status_code=404,
            detail="Vous n'êtes assigné à aucune équipe en tant que manager."
        )
    return team


@router.get("/manager/my-team/members", response_model=List[AppUser])
async def manager_get_team_members(user: AppUser = Depends(require_manager)):
    """Récupère la liste des membres de l'équipe du manager."""
    team = get_team_by_manager(user.email)
    if not team:
        raise HTTPException(
            status_code=404,
            detail="Vous n'êtes assigné à aucune équipe en tant que manager."
        )

    return get_users_by_team(team.team_id)


@router.post("/manager/my-team/members", response_model=Team)
async def manager_add_member(
    body: AddMemberInput,
    user: AppUser = Depends(require_manager)
):
    """Ajoute un consultant à l'équipe du manager (manager only)."""
    team = get_team_by_manager(user.email)
    if not team:
        raise HTTPException(
            status_code=404,
            detail="Vous n'êtes assigné à aucune équipe en tant que manager."
        )

    # Ajouter le membre à l'équipe
    team = add_member_to_team(team.team_id, body.email)
    if not team:
        raise HTTPException(status_code=500, detail="Erreur lors de l'ajout du membre.")

    # Mettre à jour le team_id de l'utilisateur
    assign_user_to_team(body.email, team.team_id)

    return team


@router.delete("/manager/my-team/members/{email}")
async def manager_remove_member(
    email: str,
    user: AppUser = Depends(require_manager)
):
    """Retire un consultant de l'équipe du manager (manager only)."""
    team = get_team_by_manager(user.email)
    if not team:
        raise HTTPException(
            status_code=404,
            detail="Vous n'êtes assigné à aucune équipe en tant que manager."
        )

    # Vérifier que le membre est bien dans l'équipe
    if email not in team.members:
        raise HTTPException(
            status_code=400,
            detail="Ce consultant ne fait pas partie de votre équipe."
        )

    # Retirer le membre de l'équipe
    team = remove_member_from_team(team.team_id, email)
    if not team:
        raise HTTPException(status_code=500, detail="Erreur lors du retrait du membre.")

    # Mettre à jour le team_id de l'utilisateur
    remove_user_from_team(email)

    return {"ok": True}


@router.get("/manager/consultants-without-team", response_model=List[AppUser])
async def manager_get_consultants_without_team(_: AppUser = Depends(require_manager)):
    """Récupère la liste des consultants sans équipe (manager only)."""
    return get_users_without_team()
