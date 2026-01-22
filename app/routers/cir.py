# app/routers/cir.py
"""
Endpoints spécifiques au CIR (Crédit d'Impôt Recherche).
Notamment : suggestion d'articles scientifiques à partir des documents techniques.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from typing import List, Optional
from pydantic import BaseModel
from app.deps import ensure_openai
from Core import document, keywords as kw, serper
from Core.document import chunk_text
from Core.embeddings import build_index

router = APIRouter()


class SuggestedArticle(BaseModel):
    """Article scientifique suggéré par Serper."""
    title: str
    url: str
    citations: int
    authors: str
    year: Optional[int]
    journal: str
    keyword: str
    selected: bool = True


class CirSuggestArticlesResponse(BaseModel):
    """Réponse de suggestion d'articles CIR."""
    keywords: List[str]
    articles: List[SuggestedArticle]


@router.post(
    "/suggest-articles",
    response_model=CirSuggestArticlesResponse,
    summary="Suggestion d'articles scientifiques (CIR, IA + Serper)",
    description=(
        "Analyse les documents techniques pour en extraire des mots-clés pertinents, "
        "puis recherche des articles scientifiques correspondants via Google Scholar (Serper). "
        "Retourne les mots-clés extraits et les articles trouvés."
    ),
)
async def cir_suggest_articles(
    annee: int = Form(..., description="Année de référence pour filtrer les articles (ex: 2024)"),
    docs_tech: List[UploadFile] = File(
        ..., description="Documents techniques (PDF/DOCX/TXT/PPTX/XLSX)"
    ),
    max_keywords: int = Form(5, description="Nombre maximum de mots-clés à extraire (défaut: 5)"),
    max_articles_per_kw: int = Form(2, description="Nombre d'articles par mot-clé (défaut: 2)"),
    _: bool = Depends(ensure_openai),
):
    """
    Workflow :
    1. Extraction du texte de tous les documents techniques
    2. Extraction des mots-clés pertinents via LLM (Azure OpenAI)
    3. Recherche d'articles sur Google Scholar via Serper pour chaque mot-clé
    4. Retour des mots-clés et articles triés par citations
    """
    # 1) Extraction du texte des documents
    text = ""
    for f in docs_tech:
        data = await f.read()
        if not data:
            continue
        text += "\n" + document.extract_text_from_bytes(data, f.filename or "")
        try:
            f.file.seek(0)
        except Exception:
            pass

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Aucun texte n'a pu être extrait des documents fournis."
        )

    # 2) Construction de l'index RAG pour une extraction intelligente
    print(f"[CIR] Texte extrait: {len(text)} caractères")
    chunks = chunk_text(text, size=800, overlap=150)
    print(f"[CIR] Chunks créés: {len(chunks)}")

    index, vectors = None, []
    if chunks:
        try:
            index, vectors = build_index(chunks)
            print(f"[CIR] Index RAG construit: {len(vectors)} vecteurs")
        except Exception as e:
            print(f"[CIR] Erreur construction index RAG: {e}, fallback sur texte brut")

    # 3) Extraction des mots-clés via LLM (avec RAG si disponible)
    try:
        if index is not None and vectors:
            # Utiliser le RAG pour trouver les passages pertinents
            keywords_list = kw.extract_keywords_with_rag(index, chunks, vectors, max_keywords=max_keywords)
        else:
            # Fallback : extraction sur le texte brut
            keywords_list = kw.extract_keywords(text, max_keywords=max_keywords)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'extraction des mots-clés: {e}"
        )

    if not keywords_list:
        return CirSuggestArticlesResponse(keywords=[], articles=[])

    # 4) Recherche d'articles via Serper
    try:
        articles_raw = serper.search_articles_serper(
            keywords=keywords_list,
            annee_reference=annee,
            max_per_kw=max_articles_per_kw,
        )
    except RuntimeError as e:
        # Clé API Serper manquante ou autre erreur
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la recherche Serper: {e}"
        )

    # 5) Conversion en modèle Pydantic
    articles = [
        SuggestedArticle(
            title=a["title"],
            url=a["url"],
            citations=a["citations"],
            authors=a["authors"],
            year=a["year"],
            journal=a["journal"],
            keyword=a["keyword"],
            selected=a.get("selected", True),
        )
        for a in articles_raw
    ]

    return CirSuggestArticlesResponse(keywords=keywords_list, articles=articles)
