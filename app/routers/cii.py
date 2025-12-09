# app/routers/cii.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
import json

from app.models.schemas import (
    Competitor,
    FormInfo,
    CiiAnalyseRequest,
    CiiSuggestRequest,
    CiiSuggestResponse,
    SuggestedCompetitor,
)
from app.services.prompts import prompt_cii_analyse, prompt_cii_suggest_competitors
from Core import rag, document
from openai import BadRequestError  # <-- à ajouter


router = APIRouter()


@router.post(
    "/analyse",
    summary="Analyse concurrentielle CII",
    description="Génère une analyse textuelle structurée (prompt + LLM).",
)
def cii_analyse(body: CiiAnalyseRequest):
    """
    Génère une analyse concurrentielle textuelle pour CII (pas de RAG, prompt pur).
    """
    info: FormInfo = body.info
    competitors: List[Competitor] = body.competitors or []
    axes: List[str] = body.axes or []

    concur_lines = []
    for c in competitors:
        line = f"- {c.name}"
        if c.site:
            line += f" ({c.site})"
        if c.axes:
            line += f" — axes: {', '.join(c.axes)}"
        concur_lines.append(line)
    txt_comp = "\n".join(concur_lines) if concur_lines else "- (aucun concurrent déclaré)"

    tpl = prompt_cii_analyse()
    prompt = tpl.format(
        societe=info.societe,
        projet=info.projet_name,
        annee=info.annee,
        concurrents=txt_comp,
        axes=", ".join(axes or []),
    )
    out = rag.call_ai(prompt, meta="CII Analyse")
    return {"analyse": out}

@router.post(
    "/suggest",
    summary="Suggestion de concurrents (CII, IA + docs client)",
    description=(
        "Analyse les documents client (techniques + administratifs) et les infos projet, "
        "puis propose jusqu’à 8 concurrents avec leurs limites et l’avantage du projet client."
    ),
    response_model=CiiSuggestResponse,
)
async def cii_suggest_competitors(
    payload: str = Form(..., description="JSON de CiiSuggestRequest"),
    docs_client: List[UploadFile] = File(
        [], description="Documents techniques (mêmes fichiers que pour la génération du dossier)"
    ),
    docs_admin: List[UploadFile] = File(
        [], description="Documents administratifs (présentation entreprise, etc.)"
    ),
):
    # 0) Parse du JSON
    try:
        req: CiiSuggestRequest = CiiSuggestRequest.model_validate_json(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Payload invalide: {e}")

    info = req.info
    axes = req.axes or []
    axes_str = ", ".join(axes) if axes else "non précisés"

    # 1) Extraction du texte à partir des docs (comme /generate/docx)
    text_client, text_admin = "", ""

    for f in docs_client:
        data = await f.read()
        if not data:
            continue
        f.file.seek(0)
        text_client += "\n" + document.extract_text(f.file)

    for f in docs_admin:
        data = await f.read()
        if not data:
            continue
        f.file.seek(0)
        text_admin += "\n" + document.extract_text(f.file)

    full_text = (text_client + "\n" + text_admin).strip()
    ctx = full_text[:8000] if full_text else ""

    # 2) Récupération du prompt dans le Blob + formatage
    try:
        tpl = prompt_cii_suggest_competitors()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Placeholders attendus dans cii_suggest_competitors.txt :
    # {societe}, {projet}, {site_client}, {annee}, {axes}, {ctx}
    try:
        prompt = tpl.format(
            societe=info.societe,
            projet=info.projet_name,
            site_client=info.site_web or "(non renseigné)",
            annee=info.annee,
            axes=axes_str,
            ctx=ctx,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur formatage prompt CII suggest: {e}",
        )

    # Log debug (optionnel, utile pour vérifier le prompt réel)
    print("[CII /suggest] Prompt envoyé à Azure (début) :")
    print(prompt[:2000])

    # 3) Appel IA avec gestion du content_filter Azure
    try:
        raw = rag.call_ai(prompt, meta="CII Suggest competitors")
    except BadRequestError as e:
        # Azure a filtré le prompt (content_filter, jailbreak, etc.)
        # On renvoie une erreur claire côté API, sans crasher le serveur.
        msg = getattr(e, "message", None) or str(e)
        raise HTTPException(
            status_code=400,
            detail=(
                "La requête IA a été rejetée par la politique de contenu Azure OpenAI. "
                "Merci de vérifier le contenu des documents et/ou d'adoucir le prompt. "
                f"Détail technique: {msg}"
            ),
        )

    # 4) Parsing JSON
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON racine != liste")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Impossible de parser la réponse IA en JSON: {e}. Réponse brute: {raw[:500]}",
        )

    out_items: List[SuggestedCompetitor] = []
    for item in data:
        try:
            name = (item.get("name") or "").trim()
        except AttributeError:
            name = (item.get("name") or "").strip()
        if not name:
            continue
        site = (item.get("site") or "").strip()
        weakness = (item.get("weakness") or "").strip()
        client_advantage = (item.get("client_advantage") or "").strip()
        if not weakness or not client_advantage:
            continue

        out_items.append(
            SuggestedCompetitor(
                name=name,
                site=site,
                weakness=weakness,
                client_advantage=client_advantage,
            )
        )

    return CiiSuggestResponse(competitors=out_items[:8])
