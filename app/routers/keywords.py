# app/routers/keywords.py
from fastapi import APIRouter, UploadFile, File, Depends
from typing import List
from app.models.schemas import KeywordsResponse
from app.deps import ensure_openai
from Core import document, keywords as kw

router = APIRouter()

@router.post(
    "/keywords",
    response_model=KeywordsResponse,
    summary="Extraction de mots-clés",
    description="Envoie 1..N fichiers (PDF/DOCX/TXT/PPTX/XLSX) dans une requête **multipart/form-data** sous la clé `files`.",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"}
                            }
                        },
                        "required": ["files"]
                    }
                }
            }
        }
    }
)
async def extract_keywords(files: List[UploadFile] = File(...), _: bool = Depends(ensure_openai)):
    text = ""
    for f in files:
        data = await f.read()
        if not data:
            continue
        # on n'utilise PAS f.file.name (souvent /tmp/xxx), mais le vrai nom :
        text += "\n" + document.extract_text_from_bytes(data, f.filename or "")
        # remet le curseur si l'appelant réutilise f.file
        try:
            f.file.seek(0)
        except Exception:
            pass

    if not text.strip():
        return {"keywords": []}  # ou un message si tu veux

    kws = kw.extract_keywords(text, max_keywords=8)
    return {"keywords": kws}
