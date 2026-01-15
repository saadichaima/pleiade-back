# app/routers/admin_templates.py
"""
Endpoints pour la gestion des templates Word (admin uniquement).
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from datetime import datetime
import io
import os

from app.auth_ms import require_admin
from app.models.auth import AppUser
from app.services.templates_service import (
    get_template_info,
    get_template_content,
    update_template,
    list_template_history,
    restore_template_version,
    update_template_variables,
)

router = APIRouter()


class TemplateInfoResponse(BaseModel):
    """Schéma de réponse pour les informations d'une template."""
    type: str  # "cir" ou "cii"
    filename: str
    size_bytes: int
    last_modified: str
    version: int
    blob_url: str | None = None


class TemplateHistoryItem(BaseModel):
    """Élément d'historique d'une template."""
    version: int
    filename: str
    size_bytes: int
    modified_at: str
    modified_by: str | None = None


@router.get("/admin/templates", response_model=List[TemplateInfoResponse])
def admin_list_templates(_: AppUser = Depends(require_admin)):
    """
    Liste les templates Word disponibles (CIR et CII).
    Requiert le rôle admin.
    """
    templates = []
    for template_type in ["cir", "cii"]:
        try:
            info = get_template_info(template_type)
            templates.append(info)
        except Exception as e:
            print(f"Erreur lors de la récupération de la template {template_type}: {e}")

    return templates


@router.get("/admin/templates/{template_type}/info", response_model=TemplateInfoResponse)
def admin_get_template_info(template_type: str, _: AppUser = Depends(require_admin)):
    """
    Récupère les informations détaillées d'une template spécifique.
    """
    if template_type.lower() not in ["cir", "cii"]:
        raise HTTPException(status_code=400, detail="Type de template invalide. Utilisez 'cir' ou 'cii'.")

    try:
        return get_template_info(template_type.lower())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template {template_type.upper()} non trouvée")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/admin/templates/{template_type}/download")
def admin_download_template(template_type: str, _: AppUser = Depends(require_admin)):
    """
    Télécharge le fichier Word de la template actuelle.
    """
    if template_type.lower() not in ["cir", "cii"]:
        raise HTTPException(status_code=400, detail="Type de template invalide")

    try:
        content, filename = get_template_content(template_type.lower())
        return StreamingResponse(
            io.BytesIO(content),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Template non trouvée")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.put("/admin/templates/{template_type}")
async def admin_update_template(
    template_type: str,
    file: UploadFile = File(...),
    user: AppUser = Depends(require_admin)
):
    """
    Met à jour une template Word en uploadant un nouveau fichier.
    Crée automatiquement une sauvegarde de l'ancienne version.
    """
    if template_type.lower() not in ["cir", "cii"]:
        raise HTTPException(status_code=400, detail="Type de template invalide")

    # Vérification du type de fichier
    if not file.filename or not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Le fichier doit être au format .docx")

    try:
        content = await file.read()
        result = update_template(
            template_type.lower(),
            content,
            user.email or "unknown"
        )
        return {
            "success": True,
            "message": f"Template {template_type.upper()} mise à jour avec succès",
            "version": result["version"],
            "filename": result["filename"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la mise à jour: {str(e)}")


@router.get("/admin/templates/{template_type}/history", response_model=List[TemplateHistoryItem])
def admin_get_template_history(template_type: str, _: AppUser = Depends(require_admin)):
    """
    Récupère l'historique des versions d'une template (5 dernières versions).
    """
    if template_type.lower() not in ["cir", "cii"]:
        raise HTTPException(status_code=400, detail="Type de template invalide")

    try:
        return list_template_history(template_type.lower())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.post("/admin/templates/{template_type}/restore/{version}")
def admin_restore_template(
    template_type: str,
    version: int,
    user: AppUser = Depends(require_admin)
):
    """
    Restaure une ancienne version d'une template.
    """
    if template_type.lower() not in ["cir", "cii"]:
        raise HTTPException(status_code=400, detail="Type de template invalide")

    try:
        result = restore_template_version(
            template_type.lower(),
            version,
            user.email or "unknown"
        )
        return {
            "success": True,
            "message": f"Version {version} restaurée avec succès",
            "new_version": result["version"]
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version {version} non trouvée")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/admin/templates/{template_type}/preview")
def admin_preview_template(
    template_type: str,
    _: AppUser = Depends(require_admin)
):
    """
    Extrait le texte du template pour prévisualisation et modification des variables Jinja.
    """
    if template_type.lower() not in ["cir", "cii"]:
        raise HTTPException(status_code=400, detail="Type de template invalide")

    try:
        from app.services.templates_service import extract_text_from_template
        text_content = extract_text_from_template(template_type.lower())
        return {
            "text": text_content,
            "template_type": template_type.lower()
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template {template_type.upper()} non trouvée")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


class UpdateVariablesRequest(BaseModel):
    """Schéma pour la mise à jour des variables."""
    old_text: str
    new_text: str


@router.post("/admin/templates/{template_type}/save-variables")
async def admin_save_template_variables(
    template_type: str,
    request: UpdateVariablesRequest,
    user: AppUser = Depends(require_admin)
):
    """
    Sauvegarde les modifications des variables Jinja dans le template.
    """
    if template_type.lower() not in ["cir", "cii"]:
        raise HTTPException(status_code=400, detail="Type de template invalide")

    try:
        result = update_template_variables(
            template_type.lower(),
            request.old_text,
            request.new_text,
            user.email or "unknown"
        )
        return {
            "success": True,
            "message": f"Variables du template {template_type.upper()} mises à jour avec succès",
            "version": result["version"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde: {str(e)}")
