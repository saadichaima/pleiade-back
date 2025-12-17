# app/routers/generate.py

from fastapi import APIRouter, UploadFile, File, Form, Response
from typing import List, Optional, Dict
import tempfile
from io import BytesIO
from uuid import uuid4
from datetime import datetime, timezone

from app.models.schemas import GenerateRequest
from app.config import settings
from app.services.builder import build_rag_indexes, build_sections_cir
from app.services.builder_cii import build_sections_cii
from app.services.cosmos_client import (
    get_projects_container,
    get_documents_container,
    get_outputs_container,
)
from app.services.blob_client import upload_bytes_to_blob
from app.services.figures_planner import (
    prepare_figures_for_cir,
    prepare_figures_for_cii,
)

from Core import document, rag, writer
from Core import embeddings
from Core import rag as core_rag
from Core.images_figures import insert_images_by_reference_live
from fastapi import APIRouter, UploadFile, File, Form, Response, Depends
from app.auth_ms import get_current_user
from app.models.auth import AppUser

router = APIRouter()


@router.post(
    "/docx",
    summary="Génération du dossier DOCX (CIR/CII)",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "payload": {
                                "type": "string",
                                "description": "JSON de GenerateRequest",
                            },
                            "user_id": {
                                "type": "string",
                                "description": "Identifiant utilisateur (consultant)",
                            },
                            "docs_client": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            },
                            "docs_admin": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            },
                            "cvs": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            },
                            "logo": {
                                "type": "string",
                                "format": "binary",
                            },
                            "articles_pdfs": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            },
                        },
                        "required": ["payload"],
                    },
                    "encoding": {"payload": {"contentType": "application/json"}},
                }
            }
        }
    },
)
async def generate_docx(
    payload: str = Form(...),
    user_id: str = Form("anonymous"),
    docs_client: List[UploadFile] = File([]),
    docs_admin: List[UploadFile] = File([]),
    cvs: List[UploadFile] = File([]),
    logo: Optional[UploadFile] = File(None),
    articles_pdfs: List[UploadFile] = File([]),
    current_user: AppUser = Depends(get_current_user),

):
    # ===== 0) Parse du payload & métadonnées globales =====
    req: GenerateRequest = GenerateRequest.model_validate_json(payload)
    info = req.info
    type_dossier = info.type_dossier  # "CIR" | "CII"
    user_id = current_user.email 

    project_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    uploaded_docs_meta: List[dict] = []
    print(
        f"[generate.docx] Début génération projet={project_id}, "
        f"type_dossier={type_dossier}"
    )

    # On garde tous les docs clients en mémoire pour le planner de figures
    docs_client_data: List[Dict[str, bytes]] = []

    # ===== 1) Extraction texte + upload Blob pour docs client / admin =====
    text_client, text_admin = "", ""

    # Documents techniques (docs_client)
    for idx, f in enumerate(docs_client):
        data = await f.read()
        if not data:
            continue

        filename = f.filename or ""
        text_client += "\n" + document.extract_text_from_bytes(data, filename)

        docs_client_data.append(
            {
                "filename": filename,
                "data": data,
                "content_type": f.content_type or "",
            }
        )

        print(
            f"[generate.docx] docs_client[{idx}] = {filename} "
            f"(content_type={f.content_type})"
        )

        blob_name = f"{project_id}/docs_client/{idx}_{filename or 'doc'}"
        blob_url = upload_bytes_to_blob(
            settings.STORAGE_CONTAINER_UPLOADS,
            blob_name,
            data,
            content_type=f.content_type or "application/octet-stream",
        )

        uploaded_docs_meta.append(
            {
                "id": f"{project_id}:docs_client:{idx}",
                "project_id": project_id,
                "user_id": user_id,
                "kind": "docs_client",
                "filename": filename,
                "content_type": f.content_type,
                "size": len(data),
                "blob_url": blob_url,
                "created_at": now,
            }
        )

    # Documents administratifs (docs_admin)
    for idx, f in enumerate(docs_admin):
        data = await f.read()
        if not data:
            continue

        filename = f.filename or ""
        text_admin += "\n" + document.extract_text_from_bytes(data, filename)

        blob_name = f"{project_id}/docs_admin/{idx}_{filename or 'doc'}"
        blob_url = upload_bytes_to_blob(
            settings.STORAGE_CONTAINER_UPLOADS,
            blob_name,
            data,
            content_type=f.content_type or "application/octet-stream",
        )

        uploaded_docs_meta.append(
            {
                "id": f"{project_id}:docs_admin:{idx}",
                "project_id": project_id,
                "user_id": user_id,
                "kind": "docs_admin",
                "filename": filename,
                "content_type": f.content_type,
                "size": len(data),
                "blob_url": blob_url,
                "created_at": now,
            }
        )

    # ===== 2) Index RAG : client / admin / mix =====
    pack_client, pack_admin, pack_mix = build_rag_indexes(text_client, text_admin)

    # Articles PDF (upload + RAG)
    text_articles = ""
    articles_texts: List[str] = []
    for idx, f in enumerate(articles_pdfs):
        data = await f.read()
        if not data:
            continue

        filename = f.filename or ""
        txt = document.extract_text_from_bytes(data, filename)
        text_articles += "\n" + txt
        articles_texts.append(txt)

        blob_name = f"{project_id}/articles/{idx}_{filename or 'article.pdf'}"
        blob_url = upload_bytes_to_blob(
            settings.STORAGE_CONTAINER_UPLOADS,
            blob_name,
            data,
            content_type=f.content_type or "application/pdf",
        )

        uploaded_docs_meta.append(
            {
                "id": f"{project_id}:article:{idx}",
                "project_id": project_id,
                "user_id": user_id,
                "kind": "article_pdf",
                "filename": filename,
                "content_type": f.content_type,
                "size": len(data),
                "blob_url": blob_url,
                "created_at": now,
            }
        )

    chunks_art = [c for c in document.chunk_text(text_articles) if c.strip()]
    if chunks_art:
        index_art, vectors_art = embeddings.build_index(chunks_art)
        pack_articles = (index_art, chunks_art, vectors_art)
    else:
        pack_articles = (None, [], [])

    # ===== 3) Sections (CIR / CII) =====
    objectif = ""
    sections_cir = None
    sections_cii = None

    enriched_articles = core_rag.enrich_manual_articles_iso(
        [a.model_dump() for a in req.articles],
        articles_texts,
    )
    articles_for_cir = enriched_articles

    # Ces variables serviront à l'insertion d'images pour les deux types de dossiers
    src_docx_figures: Optional[str] = None
    captions_text: Optional[str] = None

    if type_dossier == "CIR":
        sections_cir = build_sections_cir(
            pack_client,
            pack_mix,
            pack_admin,
            pack_articles,
            objectif,
            "",
            info.annee,
            info.societe,
            info.site_web or "",
            articles_for_cir,
            req.doc_complete,
            req.externalises,
        )

        # Planification dynamique des figures pour CIR
        try:
            sections_cir, src_docx_figures, captions_text = prepare_figures_for_cir(
                docs_client_data,
                sections_cir,
                max_figures=5,
                min_side=400,
            )
        except Exception as e:
            print(f"[generate.docx] ERREUR prepare_figures_for_cir: {e!r}")
            src_docx_figures = None
            captions_text = None

    else:
        axes_from_perf = [ax for ax in (req.performance_types or []) if ax]
        axes_from_comp = sorted(
            {ax for comp in req.competitors for ax in (comp.axes or [])}
        )
        axes_cibles = axes_from_perf or axes_from_comp or [
            "Fonctionnelles",
            "Techniques",
            "Ergonomiques",
            "Écologiques",
        ]

        perf_type = (
            axes_cibles[0] if axes_cibles else "Fonctionnelles"
        ).strip()

        sections_cii = build_sections_cii(
            pack_client,
            pack_mix,
            pack_admin,
            info=info,
            axes_cibles=axes_cibles,
            concurrents=[c.model_dump() for c in req.competitors],
            total_heures="",
            contexte_societe="",
            secteur="",
            visee_generale="",
            performance_type=perf_type,
            doc_complete=req.doc_complete,
        )

        # Planification dynamique des figures pour CII
        try:
            sections_cii, src_docx_figures, captions_text = prepare_figures_for_cii(
                docs_client_data,
                sections_cii,
                max_figures=5,
                min_side=400,
            )
        except Exception as e:
            print(f"[generate.docx] ERREUR prepare_figures_for_cii: {e!r}")
            src_docx_figures = None
            captions_text = None

    # ===== 4) RH depuis CV =====
    cvs_texts = []
    for idx, f in enumerate(cvs):
        data = await f.read()
        if not data:
            continue

        filename = f.filename or ""
        txt = document.extract_text_from_bytes(data, filename)
        cvs_texts.append(txt)

        blob_name = f"{project_id}/cvs/{idx}_{filename or 'cv'}"
        blob_url = upload_bytes_to_blob(
            settings.STORAGE_CONTAINER_UPLOADS,
            blob_name,
            data,
            content_type=f.content_type or "application/octet-stream",
        )

        uploaded_docs_meta.append(
            {
                "id": f"{project_id}:cv:{idx}",
                "project_id": project_id,
                "user_id": user_id,
                "kind": "cv",
                "filename": filename,
                "content_type": f.content_type,
                "size": len(data),
                "blob_url": blob_url,
                "created_at": now,
            }
        )

    rh = rag.generate_ressources_humaines_from_cvs(cvs_texts) if cvs_texts else []

    # ===== 5) Logo =====
    logo_bytes: Optional[bytes] = None
    if logo:
        logo_bytes = await logo.read()
        if logo_bytes:
            filename = logo.filename or ""
            blob_name = f"{project_id}/logo/{filename or 'logo'}"
            blob_url = upload_bytes_to_blob(
                settings.STORAGE_CONTAINER_UPLOADS,
                blob_name,
                logo_bytes,
                content_type=logo.content_type or "image/png",
            )
            uploaded_docs_meta.append(
                {
                    "id": f"{project_id}:logo",
                    "project_id": project_id,
                    "user_id": user_id,
                    "kind": "logo",
                    "filename": filename,
                    "content_type": logo.content_type,
                    "size": len(logo_bytes),
                    "blob_url": blob_url,
                    "created_at": now,
                }
            )

    # 6) Période + info
    period = (
        f"{info.date_debut or ''} — {info.date_fin or ''}".strip(" —")
        if (info.date_debut or info.date_fin)
        else str(info.annee)
    )
    info_dict = info.model_dump()
    info_dict["period"] = period

    if logo_bytes:
        try:
            from PIL import Image as PILImage

            img = PILImage.open(BytesIO(logo_bytes))
            out = BytesIO()
            img.save(out, format="PNG")
            png_bytes = out.getvalue()
        except Exception:
            png_bytes = logo_bytes
        ftmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        ftmp.write(png_bytes)
        ftmp.close()
        info_dict["logo"] = ftmp.name

    # 7) Payload "d" pour la template
    if type_dossier == "CIR":
        d = {
            "info": info_dict,
            "context": [sections_cir["contexte"]],
            "verrou": [sections_cir["verrou"]],
            "travaux": [sections_cir["travaux"]],
            "contribution": [sections_cir["contribution"]],
            "indicateurs": [sections_cir["indicateurs"]],
            "partenariat": [sections_cir["partenariat"]],
            "biblio": sections_cir["biblio"].split("\n")
            if isinstance(sections_cir["biblio"], str)
            else sections_cir["biblio"],
            "entreprise": [sections_cir["entreprise"]],
            "objectif": [sections_cir["objet"]],
            "resume": [sections_cir["resume"]],
            "ressources_humaines": rh,
        }
    else:
        d = {
            "info": info_dict,
            "cii": sections_cii,
            "ressources_humaines": rh,
        }

    # ===== 8) Enregistrement dans Cosmos DB =====
    projects_container = get_projects_container()
    documents_container = get_documents_container()

    project_doc = {
        "id": project_id,
        "project_id": project_id,
        "user_id": user_id,
        "type_dossier": type_dossier,
        "created_at": now,
        "updated_at": now,
        "payload": req.model_dump(),
    }
    projects_container.create_item(project_doc)

    for doc_meta in uploaded_docs_meta:
        documents_container.create_item(doc_meta)

    # ===== 9) Génération DOCX (template + mise en forme + notes) =====
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
    print(f"[generate.docx] Appel writer.generate_docx, out_path={out_path}")
    writer.generate_docx(
        template_path=(
            settings.TEMPLATE_CIR
            if type_dossier == "CIR"
            else settings.TEMPLATE_CII
        ),
        output_path=out_path,
        d=d,
        branding_tokens={
            "CLIENT": info.societe or "",
            "20XX": str(info.annee),
            "SITE_WEB": info.site_web or "",
            "EMAIL": info.email or "",
            "TELEPHONE": info.telephone or "",
            "RESPONSABLE": info.responsable_innovation or "",
            "TITRE_RESP": info.titre_resp or "",
            "DIPLOME": info.diplome or "",
            "DATE_DEBUT": info.date_debut or "",
            "DATE_FIN": info.date_fin or "",
            "TEMPS_OPERATION": info.temps_operation or "",
        },
        logo_bytes=logo_bytes,
    )

    # ===== 10) Insertion des figures dynamiques (CIR et CII) =====
    if src_docx_figures:
        try:
            print(
                f"[generate.docx] Insertion des figures depuis "
                f"{src_docx_figures} dans {out_path}"
            )
            insert_images_by_reference_live(
                src_docx=src_docx_figures,
                dst_docx=out_path,
                out_docx=out_path,
                include_header_footer_src=False,
                captions_text=captions_text,
                caption_label="Figure",
                caption_style="Caption",
                nbspace_before_colon=True,
                renumber_references=True,
                target_label="Figure",
            )
        except Exception as e:
            print(f"[generate.docx] ERREUR insertion figures: {e!r}")
    else:
        print("[generate.docx] Pas de figures dynamiques à insérer.")

    # Lecture du DOCX final
    with open(out_path, "rb") as f:
        content = f.read()
    year_suffix = str(info.annee)[-2:]

    filename = (
        f'{info.projet_name}_{type_dossier}_{year_suffix}_VIA.docx'
    )

    # ===== 11) Upload du DOCX généré dans Blob + enregistrement Cosmos outputs =====
    output_blob_name = f"{project_id}/outputs/{filename}"
    output_blob_url = upload_bytes_to_blob(
        settings.STORAGE_CONTAINER_OUTPUTS,
        output_blob_name,
        content,
        content_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
    )

    outputs_container = get_outputs_container()
    output_doc = {
        "id": f"{project_id}:output",
        "project_id": project_id,
        "user_id": user_id,
        "type_dossier": type_dossier,
        "filename": filename,
        "blob_url": output_blob_url,
        "created_at": now,
    }
    outputs_container.create_item(output_doc)

    print(
        f"[generate.docx] Fin génération projet={project_id}, fichier={filename}"
    )

    return Response(
        content,
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        headers={
            "Content-Disposition": f'attachment; filename=\"{filename}\"',
            "X-Project-Id": project_id,
            "X-Output-Blob-Url": output_blob_url,
        },
    )
