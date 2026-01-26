# app/routers/generate.py

import asyncio
import json
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse

from app.auth_ms import get_current_user, get_current_user_with_query_token
from app.config import settings
from app.models.auth import AppUser
from app.models.schemas import GenerateRequest

from app.services.builder import build_rag_indexes, build_sections_cir
from app.services.builder_cii import build_sections_cii
from app.services.figures_planner import prepare_figures_for_cir, prepare_figures_for_cii
from app.services.web_scraper import extract_website_context

from app.services.cosmos_client import get_projects_container, get_outputs_container
from app.services.blob_client import upload_bytes_to_blob

from Core import document, embeddings, rag, writer
from Core import rag as core_rag
from Core.images_figures import insert_images_by_reference_live

router = APIRouter()

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# =========================
# In-memory job registry
# =========================

@dataclass
class _Job:
    id: str
    user_email: str
    loop: asyncio.AbstractEventLoop
    events: List[dict] = field(default_factory=list)
    done: bool = False
    error: Optional[str] = None
    result: Optional[dict] = None  # {"content": bytes, "filename": str, "project_id": str}
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def emit(self, payload: dict) -> None:
        async with self.cond:
            self.events.append(payload)
            self.cond.notify_all()

    def emit_threadsafe(self, payload: dict) -> None:
        # Called from thread
        def _schedule():
            asyncio.create_task(self.emit(payload))
        self.loop.call_soon_threadsafe(_schedule)


_JOBS: Dict[str, _Job] = {}


def _log_progress(step: str, label: str, percent: int) -> None:
    print(f"[PROGRESS] {percent}% — {step} — {label}")


def _persist_generation(
    *,
    project_id: str,
    user_email: str,
    req: GenerateRequest,
    created_at: str,
    filename: str,
    content: bytes,
) -> Dict[str, str]:
    """
    Persist project + output (Cosmos) + file (Blob).

    IMPORTANT: Cosmos containers 'projects' and 'outputs' are partitioned on /project_id.
    Therefore every item MUST include 'project_id' with a non-empty string.
    """
    if not project_id:
        raise ValueError("project_id is required for Cosmos partition key (/project_id)")

    # 1) Upload to Blob
    blob_name = f"{project_id}/{filename}"
    blob_url = upload_bytes_to_blob(
        container_name=settings.STORAGE_CONTAINER_OUTPUTS,
        blob_name=blob_name,
        data=content,
        content_type=DOCX_MIME,
    )

    projects = get_projects_container()
    outputs = get_outputs_container()

    # 2) Project item: id = project_id (simple & stable)
    project_item = {
        "id": project_id,
        "project_id": project_id,              # partition key
        "user_id": user_email,
        "type_dossier": req.info.type_dossier,
        "created_at": created_at,
        "updated_at": created_at,
        "payload": req.model_dump(),           # history.py lit payload.info.*
    }

    # 3) Output item: must include project_id (partition key)
    output_item = {
        "id": str(uuid4()),
        "project_id": project_id,              # partition key
        "user_id": user_email,
        "created_at": created_at,
        "filename": filename,
        "blob_url": blob_url,
        "content_type": DOCX_MIME,
    }

    # NB: si vous voulez une garantie "historique complet", si create_item échoue => on FAIL le job.
    # Donc on ne swallow pas l'exception.
    projects.create_item(project_item)
    outputs.create_item(output_item)

    print("[PERSIST] OK", project_id, filename)
    return {"blob_url": blob_url, "blob_name": blob_name}


# =========================
# Core sync generator (run in thread)
# =========================

def _generate_docx_sync(
    req: GenerateRequest,
    docs_client_data: List[dict],
    docs_admin_data: List[dict],
    cvs_data: List[dict],
    logo_bytes: Optional[bytes],
    articles_pdfs_data: List[dict],
    emit: Callable[[str, str, int], Any],
) -> Tuple[bytes, str]:
    info = req.info
    type_dossier = info.type_dossier

    emit("init", "Initialisation de la génération", 5)

    # 1) Read & extract docs
    emit("read_docs", "Analyse des documents fournis", 10)
    text_client = ""
    text_admin = ""
    for d in docs_client_data:
        text_client += "\n" + document.extract_text_from_bytes(d["data"], d.get("filename", "") or "")
    for d in docs_admin_data:
        text_admin += "\n" + document.extract_text_from_bytes(d["data"], d.get("filename", "") or "")

    # Enrichir avec le site web si nécessaire (docs admin insuffisants)
    site_web = info.site_web or ""
    text_admin = extract_website_context(site_web, text_admin, min_words=500)

    # 2) Build RAG
    emit("rag_index", "Construction des index de recherche (RAG)", 25)
    pack_client, pack_admin, pack_mix = build_rag_indexes(text_client, text_admin)

    # 3) Articles
    emit("articles", "Analyse des articles scientifiques", 35)
    text_articles = ""
    articles_texts: List[str] = []
    for d in articles_pdfs_data:
        txt = document.extract_text_from_bytes(d["data"], d.get("filename", "") or "")
        text_articles += "\n" + txt
        articles_texts.append(txt)

    chunks_art = [c for c in document.chunk_text(text_articles) if c.strip()]
    if chunks_art:
        index_art, vectors_art = embeddings.build_index(chunks_art)
        pack_articles = (index_art, chunks_art, vectors_art)
    else:
        pack_articles = (None, [], [])

    # 4) Generate sections
    emit("sections", "Génération du contenu scientifique", 45)

    enriched_articles = core_rag.enrich_manual_articles_iso(
        [a.model_dump() for a in req.articles],
        articles_texts,
    )

    src_docx_figures = None
    captions_text = None

    if type_dossier == "CIR":
        sections = build_sections_cir(
            pack_client, pack_mix, pack_admin, pack_articles,
            "", "", info.annee, info.societe, info.site_web or "",
            enriched_articles, req.doc_complete, req.externalises
        )
        emit("figures", "Analyse et planification des figures", 55)
        sections, src_docx_figures, captions_text = prepare_figures_for_cir(
            docs_client_data, sections, max_figures=5, min_side=400
        )
    else:
        sections = build_sections_cii(
            pack_client, pack_mix, pack_admin,
            info=info,
            axes_cibles=req.performance_types,
            concurrents=[c.model_dump() for c in req.competitors],
            total_heures="",
            contexte_societe="",
            secteur="",
            visee_generale="",
            performance_type="",
            doc_complete=req.doc_complete,
        )
        emit("figures", "Analyse et planification des figures", 55)
        sections, src_docx_figures, captions_text = prepare_figures_for_cii(
            docs_client_data, sections, max_figures=5, min_side=400
        )

    # 5) RH
    emit("rh", "Analyse des ressources humaines", 65)
    cvs_texts = [
        document.extract_text_from_bytes(d["data"], d.get("filename", "") or "")
        for d in cvs_data
        if d.get("data")
    ]
    rh = rag.generate_ressources_humaines_from_cvs(cvs_texts) if cvs_texts else []

    # 6) DOCX
    emit("docx", "Génération du document Word", 75)

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
    doc_payload = {
        "info": info.model_dump(),
        "ressources_humaines": rh,
        "sections": sections,
        **sections,
    }
    if type_dossier.upper() == "CII":
        doc_payload["cii"] = dict(sections)
    else:
        doc_payload["cir"] = dict(sections)

    writer.generate_docx(
        template_path=settings.TEMPLATE_CIR if type_dossier == "CIR" else settings.TEMPLATE_CII,
        output_path=out_path,
        d=doc_payload,
        branding_tokens={},
        logo_bytes=logo_bytes,
    )

    # 7) Insert figures
    emit("insert_figures", "Insertion des figures dans le document", 85)
    if src_docx_figures:
        insert_images_by_reference_live(
            src_docx_figures,
            out_path,
            out_path,
            captions_text=captions_text,
            caption_label="Figure",
        )

    # 8) Finalize
    emit("finalize", "Finalisation et sauvegarde", 95)
    with open(out_path, "rb") as f:
        content = f.read()

    filename = f"{info.societe}_{type_dossier}_{info.annee}_VIA.docx"
    emit("done", "Dossier prêt", 100)
    return content, filename


# =========================
# SSE JOBS: start/events/download
# =========================

@router.post("/jobs", summary="Démarre une génération asynchrone (retourne un job_id)")
async def start_generate_job(
    payload: str = Form(...),
    docs_client: List[UploadFile] = File([]),
    docs_admin: List[UploadFile] = File([]),
    cvs: List[UploadFile] = File([]),
    logo: Optional[UploadFile] = File(None),
    articles_pdfs: List[UploadFile] = File([]),
    current_user: AppUser = Depends(get_current_user),
):
    req: GenerateRequest = GenerateRequest.model_validate_json(payload)
    user_email = current_user.email

    loop = asyncio.get_running_loop()
    job_id = str(uuid4())

    # Cosmos partition key is /project_id => create it upfront and keep stable
    project_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    job = _Job(id=job_id, user_email=user_email, loop=loop)
    _JOBS[job_id] = job

    async def _read_list(files: List[UploadFile]) -> List[dict]:
        out: List[dict] = []
        for f in files:
            data = await f.read()
            if not data:
                continue
            out.append({"filename": f.filename or "", "data": data, "content_type": f.content_type or ""})
        return out

    docs_client_data = await _read_list(docs_client)
    docs_admin_data = await _read_list(docs_admin)
    cvs_data = await _read_list(cvs)
    articles_pdfs_data = await _read_list(articles_pdfs)
    logo_bytes = (await logo.read()) if logo else None

    async def _runner():
        try:
            def emit(step: str, label: str, percent: int):
                _log_progress(step, label, percent)
                job.emit_threadsafe({"type": "progress", "step": step, "label": label, "percent": percent})

            content, filename = await anyio.to_thread.run_sync(
                _generate_docx_sync,
                req,
                docs_client_data,
                docs_admin_data,
                cvs_data,
                logo_bytes,
                articles_pdfs_data,
                emit,
            )

            # Persist (CRITICAL): guarantees History completeness
            _persist_generation(
                project_id=project_id,
                user_email=user_email,
                req=req,
                created_at=created_at,
                filename=filename,
                content=content,
            )

            job.result = {"filename": filename, "content": content, "project_id": project_id}
            job.done = True
            await job.emit({"type": "done", "step": "done", "label": "Dossier prêt", "percent": 100})

        except Exception as e:
            job.error = str(e)
            job.done = True
            await job.emit({"type": "error", "message": str(e), "trace": traceback.format_exc()})
        finally:
            async with job.cond:
                job.cond.notify_all()

    asyncio.create_task(_runner())
    return {"job_id": job_id, "project_id": project_id}


@router.get("/jobs/{job_id}/events", summary="Flux SSE des étapes du job")
async def generate_job_events(
    job_id: str,
    current_user: AppUser = Depends(get_current_user_with_query_token)
):
    """
    Endpoint SSE pour suivre la progression d'un job.

    Authentification :
    - Via header Authorization: Bearer <token> (préféré)
    - Via query param ?token=<token> (fallback pour EventSource)

    Note : EventSource (navigateur) ne supporte pas les headers customs,
    donc on accepte aussi le token en query param.
    """
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job.user_email != current_user.email:
        raise HTTPException(status_code=403, detail="Accès interdit")

    async def _stream():
        idx = 0
        while True:
            async with job.cond:
                if idx >= len(job.events) and not job.done:
                    try:
                        await asyncio.wait_for(job.cond.wait(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue

                while idx < len(job.events):
                    ev = job.events[idx]
                    idx += 1
                    yield "event: message\n"
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

                if job.done:
                    return

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.get("/jobs/{job_id}/download", summary="Télécharge le DOCX final du job")
async def download_generate_job(job_id: str, current_user: AppUser = Depends(get_current_user)):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job.user_email != current_user.email:
        raise HTTPException(status_code=403, detail="Accès interdit")
    if not job.done:
        raise HTTPException(status_code=202, detail="Pas prêt")
    if job.error:
        raise HTTPException(status_code=400, detail=job.error)
    if not job.result:
        raise HTTPException(status_code=500, detail="Résultat manquant")

    content: bytes = job.result["content"]
    filename: str = job.result["filename"]

    # Libération mémoire après download
    _JOBS.pop(job_id, None)

    return Response(
        content,
        media_type=DOCX_MIME,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
