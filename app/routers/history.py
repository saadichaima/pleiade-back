from fastapi import APIRouter, Depends

from app.auth_ms import get_current_user, require_admin
from app.models.auth import AppUser
from app.services.cosmos_client import get_projects_container, get_outputs_container
from app.config import settings

router = APIRouter()


def _add_outputs_sas(url: str | None) -> str | None:
    """
    Ajoute le SAS du conteneur outputs à l'URL si STORAGE_OUTPUTS_SAS est défini.
    Si pas de SAS configuré, renvoie l'URL telle quelle.
    """
    if not url:
        return url

    sas = (settings.STORAGE_OUTPUTS_SAS or "").lstrip("?").strip()
    if not sas:
        return url

    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{sas}"


def _attach_output_meta(items, user_filter: str | None = None):
    """
    Pour chaque projet, va chercher dans le container 'outputs' le fichier généré
    (filename + blob_url) et les ajoute au dictionnaire renvoyé au front.

    Si un SAS de conteneur outputs est configuré (STORAGE_OUTPUTS_SAS),
    il est ajouté à l'URL renvoyée au front.
    """
    projects = list(items)  # matérialise le générateur Cosmos
    outputs_container = get_outputs_container()

    for proj in projects:
        pid = proj.get("project_id") or proj.get("id")
        if not pid:
            continue

        params = [{"name": "@pid", "value": pid}]
        query = "SELECT TOP 1 c.filename, c.blob_url FROM c WHERE c.project_id = @pid"

        out_items = outputs_container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True,
        )
        out = next(out_items, None)
        if out:
            proj["filename"] = out.get("filename")
            raw_url = out.get("blob_url")
            proj["blob_url"] = _add_outputs_sas(raw_url)
        else:
            proj["filename"] = None
            proj["blob_url"] = None

    return projects


@router.get("/history/me")
def my_history(user: AppUser = Depends(get_current_user)):
    projects_container = get_projects_container()
    query = """
    SELECT c.id, c.project_id, c.created_at, c.type_dossier,
           c.payload.info.societe AS societe,
           c.payload.info.projet_name AS projet,
           c.payload.info.annee AS annee
    FROM c
    WHERE c.user_id = @uid
    ORDER BY c.created_at DESC
    """
    items = projects_container.query_items(
        query=query,
        parameters=[{"name": "@uid", "value": user.email}],
        enable_cross_partition_query=True,
    )
    enriched = _attach_output_meta(items, user_filter=user.email)
    return {"items": enriched}


@router.get("/history/all")
def all_history(_: AppUser = Depends(require_admin)):
    projects_container = get_projects_container()
    query = """
    SELECT c.id, c.project_id, c.created_at, c.type_dossier,
           c.user_id,
           c.payload.info.societe AS societe,
           c.payload.info.projet_name AS projet,
           c.payload.info.annee AS annee
    FROM c
    ORDER BY c.created_at DESC
    """
    items = projects_container.query_items(
        query=query,
        enable_cross_partition_query=True,
    )
    enriched = _attach_output_meta(items, user_filter=None)
    return {"items": enriched}
