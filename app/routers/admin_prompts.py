# app/routers/admin_prompts.py
"""
Endpoints pour la gestion des prompts système (admin uniquement).
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List

from app.auth_ms import require_admin
from app.models.auth import AppUser
from app.services.prompts_admin_service import (
    list_all_prompts,
    get_prompt,
    update_prompt,
)

router = APIRouter()


class PromptConfigResponse(BaseModel):
    """Schéma de réponse pour un prompt."""
    section: str
    type_dossier: str
    content: str
    blob_url: str | None = None
    last_modified: str | None = None


class UpdatePromptRequest(BaseModel):
    """Schéma de requête pour mettre à jour un prompt."""
    content: str


@router.get("/admin/prompts", response_model=List[PromptConfigResponse])
def admin_list_prompts(_: AppUser = Depends(require_admin)):
    """
    Liste tous les prompts système disponibles (CIR, CII, OTHERS).
    Requiert le rôle admin.
    """
    prompts = list_all_prompts()
    return [
        PromptConfigResponse(
            section=p.section,
            type_dossier=p.type_dossier,
            content=p.content,
            blob_url=p.blob_url,
            last_modified=p.last_modified.isoformat() if p.last_modified else None,
        )
        for p in prompts
    ]


@router.get(
    "/admin/prompts/{type_dossier}/{section}",
    response_model=PromptConfigResponse
)
def admin_get_prompt(
    type_dossier: str,
    section: str,
    _: AppUser = Depends(require_admin)
):
    """
    Récupère un prompt spécifique par type de dossier et section.
    Requiert le rôle admin.
    """
    prompt = get_prompt(section, type_dossier)
    if not prompt:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt introuvable: {type_dossier}/{section}"
        )

    return PromptConfigResponse(
        section=prompt.section,
        type_dossier=prompt.type_dossier,
        content=prompt.content,
        blob_url=prompt.blob_url,
        last_modified=prompt.last_modified.isoformat() if prompt.last_modified else None,
    )


@router.patch(
    "/admin/prompts/{type_dossier}/{section}",
    response_model=PromptConfigResponse
)
def admin_update_prompt(
    type_dossier: str,
    section: str,
    body: UpdatePromptRequest,
    _: AppUser = Depends(require_admin)
):
    """
    Met à jour le contenu d'un prompt spécifique.
    Requiert le rôle admin.
    """
    if not body.content.strip():
        raise HTTPException(
            status_code=400,
            detail="Le contenu du prompt ne peut pas être vide."
        )

    prompt = update_prompt(section, type_dossier, body.content)
    if not prompt:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la mise à jour du prompt: {type_dossier}/{section}"
        )

    return PromptConfigResponse(
        section=prompt.section,
        type_dossier=prompt.type_dossier,
        content=prompt.content,
        blob_url=prompt.blob_url,
        last_modified=prompt.last_modified.isoformat() if prompt.last_modified else None,
    )
