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
from Core.serper import search_competitors_websites, search_competitors_from_serper, SERPER_API_KEY
from openai import BadRequestError


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

    # 2) Recherche de concurrents réels via Serper Google Search
    serper_competitors: List[dict] = []
    if SERPER_API_KEY:
        print(f"[CII /suggest] Recherche Serper de concurrents pour '{info.projet_name}'...")
        serper_competitors = search_competitors_from_serper(
            societe=info.societe,
            projet=info.projet_name,
            secteur_keywords=axes_str,
        )
        print(f"[CII /suggest] {len(serper_competitors)} concurrents trouvés via Serper")

    # 3) Construire le prompt LLM avec les concurrents Serper comme contexte
    try:
        tpl = prompt_cii_suggest_competitors()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Ajouter les résultats Serper au contexte pour guider le LLM
    serper_context = ""
    if serper_competitors:
        lines = []
        for sc in serper_competitors:
            lines.append(f"- {sc['name']} ({sc['site']}): {sc['snippet']}")
        serper_context = "\n\nConcurrents/solutions trouvés sur le marché (résultats de recherche web):\n" + "\n".join(lines)

    try:
        prompt = tpl.format(
            societe=info.societe,
            projet=info.projet_name,
            site_client=info.site_web or "(non renseigné)",
            annee=info.annee,
            axes=axes_str,
            ctx=ctx + serper_context,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur formatage prompt CII suggest: {e}",
        )

    print("[CII /suggest] Prompt envoyé à Azure (début) :")
    print(prompt[:2000])

    # 4) Appel IA avec gestion du content_filter Azure
    try:
        raw = rag.call_ai(prompt, meta="CII Suggest competitors")
    except BadRequestError as e:
        msg = getattr(e, "message", None) or str(e)
        raise HTTPException(
            status_code=400,
            detail=(
                "La requête IA a été rejetée par la politique de contenu Azure OpenAI. "
                "Merci de vérifier le contenu des documents et/ou d'adoucir le prompt. "
                f"Détail technique: {msg}"
            ),
        )

    # 5) Parsing JSON
    raw_clean = raw.strip()
    if raw_clean.startswith("```json"):
        raw_clean = raw_clean[7:]
    elif raw_clean.startswith("```"):
        raw_clean = raw_clean[3:]
    if raw_clean.endswith("```"):
        raw_clean = raw_clean[:-3]
    raw_clean = raw_clean.strip()

    try:
        data = json.loads(raw_clean)
        if not isinstance(data, list):
            raise ValueError("JSON racine != liste")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Impossible de parser la réponse IA en JSON: {e}. Réponse brute: {raw[:500]}",
        )

    # 6) Extraire les concurrents du LLM
    parsed_items = []
    for item in data:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        weakness = (item.get("weakness") or "").strip()
        client_advantage = (item.get("client_advantage") or "").strip()
        if not weakness or not client_advantage:
            continue
        parsed_items.append({
            "name": name,
            "site": "",
            "weakness": weakness,
            "client_advantage": client_advantage,
        })
    parsed_items = parsed_items[:8]

    # 7) Associer les URLs réelles : d'abord depuis Serper, sinon recherche individuelle
    if SERPER_API_KEY and parsed_items:
        # Construire un index des URLs Serper par nom (matching souple)
        serper_url_map: dict = {}
        for sc in serper_competitors:
            serper_url_map[sc["name"].lower()] = sc["site"]
            # Aussi indexer par domaine simplifié pour matching partiel
            from urllib.parse import urlparse
            domain = urlparse(sc["site"]).netloc.lower().replace("www.", "")
            if domain:
                serper_url_map[domain] = sc["site"]

        # Noms sans URL Serper directe → recherche individuelle
        names_to_search = []
        for it in parsed_items:
            name_lower = it["name"].lower()
            # Essayer matching partiel avec les résultats Serper
            matched = False
            for serper_key, serper_url in serper_url_map.items():
                if name_lower in serper_key or serper_key in name_lower:
                    it["site"] = serper_url
                    matched = True
                    break
            if not matched:
                names_to_search.append(it["name"])

        # Recherche individuelle pour les noms non matchés
        if names_to_search:
            print(f"[CII /suggest] Recherche individuelle pour {len(names_to_search)} concurrents...")
            individual_urls = search_competitors_websites(names_to_search)
            for it in parsed_items:
                if not it["site"] and it["name"] in individual_urls:
                    it["site"] = individual_urls[it["name"]]

        found = sum(1 for it in parsed_items if it["site"])
        print(f"[CII /suggest] URLs finales: {found}/{len(parsed_items)} concurrents avec URL")

    out_items: List[SuggestedCompetitor] = []
    for item in parsed_items:
        out_items.append(
            SuggestedCompetitor(
                name=item["name"],
                site=item["site"],
                weakness=item["weakness"],
                client_advantage=item["client_advantage"],
            )
        )

    return CiiSuggestResponse(competitors=out_items)
