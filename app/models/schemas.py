# app/models/schemas.py
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Literal
from pydantic import ConfigDict

DossierType = Literal["CIR", "CII"]


class FormInfo(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "societe": "Acme SAS",
            "projet_name": "Vision++",
            "annee": 2024,
            "site_web": "https://acme.ai",
            "responsable_innovation": "Jane Doe",
            "titre_resp": "CTO",
            "telephone": "+33 1 23 45 67 89",
            "email": "jane@acme.ai",
            "date_debut": "2024-01-01",
            "date_fin": "2024-12-31",
            "diplome": "PhD",
            "temps_operation": "12 mois",
            "type_dossier": "CIR"
        }
    })
    societe: str
    projet_name: str
    annee: int
    site_web: Optional[str] = ""
    responsable_innovation: str
    titre_resp: str
    telephone: str
    email: EmailStr
    date_debut: str
    date_fin: str
    diplome: Optional[str] = ""
    temps_operation: Optional[str] = ""
    type_dossier: DossierType


class Article(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "title": "A Survey on Vision Transformers",
        "year": 2022,
        "authors": "Dosovitskiy A., Brox T.",
        "journal": "ArXiv",
        "url": "https://arxiv.org/abs/2010.11929",
        "citations": 1234,
        "selected": True,
        "volume": "246",
        "pages": "3781-3790",
        "citation_iso": "ARSLAN, Muhammad, GHANEM, Hussam, MUNAWAR, Saba, et al. A Survey on RAG with LLMs. Procedia Computer Science, 2024, vol. 246, p. 3781-3790."
    }})
    title: str
    year: Optional[int] = None
    authors: Optional[str] = ""
    journal: Optional[str] = ""
    url: Optional[str] = ""
    citations: Optional[int] = 0
    selected: bool = True
    iso_citation: Optional[str] = ""


class Competitor(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "name": "RivalOne",
        "site": "https://rival.one",
        "axes": ["Fonctionnelles", "Techniques"],
        "weaknesses": "Interface peu ergonomique, pas de gestion temps réel.",
        "client_advantage": "Le projet Vision++ apporte une interface unifiée et une mise à jour des données en temps réel."
    }})
    name: str
    site: Optional[str] = ""
    axes: List[str] = []
    weaknesses: Optional[str] = ""
    client_advantage: Optional[str] = ""


class GenerateRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "info": FormInfo.model_config["json_schema_extra"]["example"],
        "doc_complete": True,
        "externalises": False,
        "articles": [Article.model_config["json_schema_extra"]["example"]],
        "competitors": [
            {
                "name": "RivalOne",
                "site": "https://rival.one",
                "axes": ["Fonctionnelles"],
                "weaknesses": "Pas de personnalisation fine.",
                "client_advantage": "Permet une personnalisation avancée des indicateurs."
            }
        ],
        "performance_types": ["Fonctionnelles", "Techniques"]
    }})
    info: FormInfo
    doc_complete: bool = True
    externalises: bool = False
    articles: List[Article] = []
    competitors: List[Competitor] = []
    performance_types: List[str] = []


class KeywordsResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "keywords": ["computer vision", "image segmentation"]
    }})
    keywords: List[str]


class ArticlesRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "keywords": ["computer vision", "time series"],
        "annee": 2024
    }})
    keywords: List[str]
    annee: int


class ArticlesResponse(BaseModel):
    articles: List[Article]


# Pour /cii/analyse
class CiiAnalyseRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "info": FormInfo.model_config["json_schema_extra"]["example"] | {"type_dossier": "CII"},
        "competitors": [{
            "name": "RivalOne",
            "site": "https://rival.one",
            "axes": ["UX", "temps réel"]
        }],
        "axes": ["fonctionnels", "techniques", "ergonomiques"]

    }})
    info: FormInfo
    competitors: List[Competitor] = []
    axes: List[str] = []


class ArticleMetadataResponse(BaseModel):
    title: str = ""
    authors: str = ""
    year: Optional[int] = None
    journal: str = ""


class CiiCompetitorsResponse(BaseModel):
    """Réponse de la route IA de suggestion de concurrents."""
    competitors: List[Competitor]
class SuggestedCompetitor(BaseModel):
    name: str
    site: Optional[str] = ""
    weakness: str
    client_advantage: str

class CiiSuggestRequest(BaseModel):
    """
    Utilisé pour /cii/suggest :
    - info : infos projet
    - axes : types de performances ciblées (fonctionnels / techniques / ergonomiques / écologiques...)
    """
    info: FormInfo
    axes: List[str] = []

class CiiSuggestResponse(BaseModel):
    competitors: List[SuggestedCompetitor]