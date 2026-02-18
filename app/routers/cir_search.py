# app/routers/cir_search.py
"""
Router pour la recherche dans les dossiers CIR existants.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional
from app.services.cir_search_service import get_cir_search_service, CirSearchResult
from Core.document import extract_text_from_bytes

router = APIRouter()


# ============ SCHEMAS ============

class CirSearchRequest(BaseModel):
    """Requête de recherche CIR."""
    query: str = Field(
        ...,
        min_length=20,
        description="Description du projet: problématique, verrous techniques, technologies, contexte"
    )
    max_results: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Nombre maximum de résultats (3-10)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Développement d'une plateforme de machine learning pour l'analyse prédictive de données industrielles. Verrous: traitement temps réel de données massives, optimisation des modèles deep learning, interprétabilité des prédictions.",
                "max_results": 5
            }
        }
    }


class CitedArticleResponse(BaseModel):
    """Article scientifique cité dans un dossier."""
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    source_document: str


class ExcerptResponse(BaseModel):
    """Extrait pertinent d'un document."""
    text: str
    document: str
    location: Optional[str] = None
    score: float


class CirSearchResultResponse(BaseModel):
    """Résultat de recherche pour un dossier CIR."""
    client_name: str = Field(..., description="Nom du client")
    team: str = Field(..., description="Équipe en charge (répertoire)")
    consultant: str = Field(..., description="Consultant en charge (depuis Excel)")
    relevant_excerpts: List[ExcerptResponse] = Field(
        default_factory=list,
        description="Extraits d'état de l'art pertinents"
    )
    cited_articles: List[CitedArticleResponse] = Field(
        default_factory=list,
        description="Articles scientifiques cités dans le dossier"
    )
    document_paths: List[str] = Field(
        default_factory=list,
        description="Chemins des documents sources"
    )
    relevance_score: float = Field(..., description="Score de pertinence (0-1)")


class CirSearchResponse(BaseModel):
    """Réponse de recherche CIR (fiche réponse)."""
    results: List[CirSearchResultResponse]
    total_found: int
    query: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [
                    {
                        "client_name": "ACME TECH",
                        "team": "Equipe Alexis",
                        "consultant": "Jean Dupont",
                        "relevant_excerpts": [
                            {
                                "text": "L'état de l'art montre que les approches de deep learning...",
                                "document": "ACME_TECH_CIR_24.pdf",
                                "location": "Page 12",
                                "score": 0.85
                            }
                        ],
                        "cited_articles": [
                            {
                                "title": "Deep Learning for Industrial IoT",
                                "authors": "Smith J., Jones A.",
                                "year": 2023,
                                "journal": "IEEE Trans.",
                                "source_document": "ACME_TECH_CIR_24.pdf"
                            }
                        ],
                        "document_paths": ["C:\\...\\ACME_TECH_CIR_24.pdf"],
                        "relevance_score": 0.85
                    }
                ],
                "total_found": 3,
                "query": "machine learning industriel..."
            }
        }
    }


class CirStatsResponse(BaseModel):
    """Statistiques de l'index CIR."""
    total_documents: int
    total_chunks: int
    total_clients_in_excel: int
    teams: List[str]


# ============ ENDPOINTS ============

@router.post(
    "/search",
    response_model=CirSearchResponse,
    summary="Recherche de dossiers CIR similaires",
    description="""
Recherche dans la base de ~100 dossiers CIR pour trouver des projets comparables.

**Entrée:** Description textuelle du nouveau projet (problématique, verrous, technologies, contexte)

**Sortie:** Fiche réponse avec 3-5 projets comparables incluant:
- Nom du client
- Équipe responsable
- Consultant en charge (depuis Excel)
- Extraits d'état de l'art pertinents avec citations
- Liste des articles scientifiques déjà cités
"""
)
async def search_cir_projects(body: CirSearchRequest) -> CirSearchResponse:
    """Recherche des dossiers CIR similaires au projet décrit."""
    try:
        service = get_cir_search_service()
        results = service.search(body.query, max_results=body.max_results)

        # Convertir en réponse API
        response_results = []
        for r in results:
            response_results.append(CirSearchResultResponse(
                client_name=r.client_name,
                team=r.team,
                consultant=r.consultant,
                relevant_excerpts=[
                    ExcerptResponse(
                        text=e['text'],
                        document=e['document'],
                        location=e.get('location'),
                        score=e['score']
                    )
                    for e in r.relevant_excerpts
                ],
                cited_articles=[
                    CitedArticleResponse(
                        title=a.title,
                        authors=a.authors,
                        year=a.year,
                        journal=a.journal,
                        source_document=a.source_document
                    )
                    for a in r.cited_articles
                ],
                document_paths=r.document_paths,
                relevance_score=r.relevance_score
            ))

        return CirSearchResponse(
            results=response_results,
            total_found=len(response_results),
            query=body.query
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche: {str(e)}")


@router.post(
    "/search-with-file",
    response_model=CirSearchResponse,
    summary="Recherche CIR avec fichier compte rendu",
    description="""
Recherche dans les dossiers CIR en combinant une description textuelle et/ou un fichier compte rendu (PDF/DOCX).
Le texte extrait du fichier est combiné avec la description pour la recherche sémantique.
"""
)
async def search_cir_with_file(
    query: str = Form(""),
    max_results: int = Form(5),
    file: Optional[UploadFile] = File(None),
) -> CirSearchResponse:
    """Recherche CIR avec description textuelle et/ou compte rendu uploadé."""
    try:
        # Extraire le texte du fichier si fourni
        file_text = ""
        if file and file.filename:
            data = await file.read()
            if data:
                file_text = extract_text_from_bytes(data, file.filename or "")

        # Combiner query texte + texte du fichier
        combined_query = (query.strip() + "\n" + file_text.strip()).strip()

        if len(combined_query) < 20:
            raise HTTPException(
                status_code=422,
                detail="La description et/ou le contenu du fichier doivent contenir au moins 20 caractères."
            )

        # Limiter la taille de la requête combinée (les embeddings ont une limite de tokens)
        if len(combined_query) > 8000:
            combined_query = combined_query[:8000]

        service = get_cir_search_service()
        results = service.search(combined_query, max_results=max_results)

        # Convertir en réponse API (même format que /search)
        response_results = []
        for r in results:
            response_results.append(CirSearchResultResponse(
                client_name=r.client_name,
                team=r.team,
                consultant=r.consultant,
                relevant_excerpts=[
                    ExcerptResponse(
                        text=e['text'],
                        document=e['document'],
                        location=e.get('location'),
                        score=e['score']
                    )
                    for e in r.relevant_excerpts
                ],
                cited_articles=[
                    CitedArticleResponse(
                        title=a.title,
                        authors=a.authors,
                        year=a.year,
                        journal=a.journal,
                        source_document=a.source_document
                    )
                    for a in r.cited_articles
                ],
                document_paths=r.document_paths,
                relevance_score=r.relevance_score
            ))

        return CirSearchResponse(
            results=response_results,
            total_found=len(response_results),
            query=query or (file.filename if file else "")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche: {str(e)}")


@router.get(
    "/stats",
    response_model=CirStatsResponse,
    summary="Statistiques de l'index CIR",
    description="Retourne des informations sur le nombre de documents indexés et les équipes."
)
async def get_cir_stats() -> CirStatsResponse:
    """Retourne les statistiques de l'index CIR."""
    try:
        service = get_cir_search_service()
        stats = service.get_stats()
        return CirStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get(
    "/health",
    summary="Health check du service CIR",
    description="Vérifie que le service de recherche CIR est opérationnel."
)
async def cir_health():
    """Health check du service CIR."""
    try:
        service = get_cir_search_service()
        stats = service.get_stats()
        return {
            "status": "ok",
            "documents_indexed": stats['total_documents'],
            "service": "cir_search"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "service": "cir_search"
        }
