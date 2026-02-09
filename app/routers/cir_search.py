# app/routers/cir_search.py
"""
Router pour la recherche dans les dossiers CIR existants.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from app.services.cir_search_service import get_cir_search_service, CirSearchResult

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
