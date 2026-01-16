# app/services/prompts.py
import requests
from urllib.parse import quote_plus
from app.config import settings

def _blob_base(container: str) -> str:
    # https://{account}.blob.core.windows.net/{container}
    return f"https://{settings.PROMPTS_ACCOUNT}.blob.core.windows.net/{container}".rstrip("/")

def _blob_url(container: str, name: str) -> str:
    base = _blob_base(container)
    if settings.PROMPTS_PUBLIC:
        return f"{base}/{quote_plus(name)}"

    # Sélectionne le SAS spécifique au conteneur si dispo, sinon le générique
    if container == settings.PROMPTS_CONTAINER_CIR and settings.PROMPTS_SAS_CIR:
        sas = settings.PROMPTS_SAS_CIR.lstrip("?")
    elif container == settings.PROMPTS_CONTAINER_CII and settings.PROMPTS_SAS_CII:
        sas = settings.PROMPTS_SAS_CII.lstrip("?")
    elif container == settings.PROMPTS_CONTAINER_OTHERS and settings.PROMPTS_SAS_OTHERS:
        sas = settings.PROMPTS_SAS_OTHERS.lstrip("?")
    else:
        sas = settings.PROMPTS_SAS.lstrip("?")

    return f"{base}/{quote_plus(name)}?{sas}" if sas else f"{base}/{quote_plus(name)}"


def _http_url(path: str) -> str:
    base = settings.PROMPTS_BASE_URL.rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def fetch_prompt(container: str, filename: str) -> str:
    """
    Récupère un prompt (texte) depuis Azure Blob Storage ou fallback HTTP.
    container: ex. settings.PROMPTS_CONTAINER_CIR
    filename : ex. 'objectifs.txt'
    """
    provider = (settings.PROMPTS_PROVIDER or "blob").lower()
    url = _blob_url(container, filename) if provider == "blob" else _http_url(filename)
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text.strip()
    except Exception as e:
        # Fallback optionnel sur HTTP si provider=blob
        if provider == "blob":
            try:
                r2 = requests.get(_http_url(filename), timeout=8)
                r2.raise_for_status()
                return r2.text.strip()
            except Exception:
                pass
        print(f"[prompts] fetch failed: {e} @ {url}")
        return ""


# ====== API publique : CIR / CII / AUTRES ======

def fetch_cir(name: str) -> str:
    # name = 'objectifs.txt' / 'travaux.txt' ...
    return fetch_prompt(settings.PROMPTS_CONTAINER_CIR, name)

def fetch_cii(name: str) -> str:
    return fetch_prompt(settings.PROMPTS_CONTAINER_CII, name)

def fetch_other(name: str) -> str:
    """
    Prompts divers (Vision, figures, keywords, etc.) stockés dans PROMPTS_CONTAINER_OTHERS.
    """
    return fetch_prompt(settings.PROMPTS_CONTAINER_OTHERS, name)

# Compat héritée (si tu l’appelles ailleurs)
def fetch(path: str) -> str:
    return fetch_cir(path)  # par défaut CIR


# --- Helpers CII ---------------------------------------------------------

def prompt_cii_analyse() -> str:
    """
    Charge le prompt d'analyse concurrentielle CII depuis le conteneur CII.
    Nom du fichier côté Blob: 'analyse_concurrence_justification.txt'
    """
    txt = fetch_cii("analyse_concurrence_justification.txt")
    if not txt:
        raise RuntimeError(
            "Prompt CII 'analyse_concurrence_justification.txt' introuvable dans le Blob."
        )
    return txt


def prompt_cii_suggest_competitors() -> str:
    """
    Prompt utilisé pour /cii/suggest, stocké dans le conteneur CII.
    Fichier attendu : 'cii_suggest_competitors.txt'.
    """
    txt = fetch_cii("cii_suggest_competitors.txt")  # adapte si tu n'as pas mis .txt
    if not txt:
        raise RuntimeError(
            "Prompt CII 'cii_suggest_competitors.txt' introuvable dans le Blob."
        )
    return txt


# --- Helpers "autres" (Vision, figures, keywords...) ---------------------

def prompt_vision_describe_images() -> str:
    """
    Prompt utilisé par Core.description_img pour décrire les images (Azure Vision).
    Fichier attendu dans PROMPTS_CONTAINER_OTHERS : 'vision_describe_images.txt'.
    """
    txt = fetch_other("vision_describe_images.txt")  # <== ICI: fetch_other, pas fetch_cii
    if not txt:
        raise RuntimeError(
            "Prompt 'vision_describe_images.txt' introuvable dans le Blob (conteneur PROMPTS_CONTAINER_OTHERS)."
        )
    return txt


def prompt_figures_plan() -> str:
    """
    Prompt utilisé par figures_planner._plan_figures_with_llm pour planifier les figures.
    Fichier attendu : 'figures_plan.txt' dans PROMPTS_CONTAINER_OTHERS.
    """
    txt = fetch_other("figures_plan.txt")
    if not txt:
        raise RuntimeError(
            "Prompt 'figures_plan.txt' introuvable dans le Blob (conteneur PROMPTS_CONTAINER_OTHERS)."
        )
    return txt
def prompt_evaluateur_travaux() -> str:
    """
    Prompt utilisé par les fonctions evaluateur_travaux (CIR/CII) pour enrichir la section Travaux
    avec des questions [[ROUGE: ...]].

    Fichier attendu dans PROMPTS_CONTAINER_OTHERS : 'evaluateur_travaux.txt'.
    """
    txt = fetch_other("evaluateur_travaux.txt")
    if not txt:
        raise RuntimeError(
            "Prompt 'evaluateur_travaux.txt' introuvable dans le Blob (conteneur PROMPTS_CONTAINER_OTHERS)."
        )
    return txt
def prompt_footnotes_glossary() -> str:
    """
    Prompt utilisé par Core.footnotes.extract_terms_with_llm pour générer le glossaire
    des termes techniques (notes de bas de page).

    Fichier attendu dans PROMPTS_CONTAINER_OTHERS : 'footnotes_glossary.txt'.
    """
    txt = fetch_other("footnotes_glossary.txt")
    if not txt:
        raise RuntimeError(
            "Prompt 'footnotes_glossary.txt' introuvable dans le Blob (conteneur PROMPTS_CONTAINER_OTHERS)."
        )
    return txt

def prompt_cir_resume() -> str:
    """
    Prompt utilisé par Core.rag.generate_resume_from_sections pour générer
    le résumé scientifique du document CIR à partir des sections.

    Fichier attendu dans PROMPTS_CONTAINER_CIR : 'resume.txt'.
    """
    txt = fetch_cir("resume.txt")
    if not txt:
        raise RuntimeError(
            "Prompt 'resume.txt' introuvable dans le Blob (conteneur PROMPTS_CONTAINER_CIR)."
        )
    return txt

def prompt_cii_resume() -> str:
    """
    Prompt utilisé par Core.rag_cii.gen_resume_from_sections pour générer
    le résumé scientifique du document CII à partir des sections.

    Fichier attendu dans PROMPTS_CONTAINER_CII : 'resume_scientifique.txt'.
    """
    txt = fetch_cii("resume_scientifique.txt")
    if not txt:
        raise RuntimeError(
            "Prompt 'resume_scientifique.txt' introuvable dans le Blob (conteneur PROMPTS_CONTAINER_CII)."
        )
    return txt
