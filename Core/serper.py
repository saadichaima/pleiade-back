# Core/serper.py
import os
import re
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()  # <-- pour charger .env

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

def _parse_publication_info(pub_info: str):
    """
    Ex: "JD O'Grady - 2008 - books.google.com" -> (authors, year, source)
    """
    if not pub_info:
        return None, None, None
    parts = [p.strip() for p in pub_info.split(" - ")]
    authors = parts[0] if parts else None
    year = int(parts[1]) if len(parts) > 1 and re.fullmatch(r"\d{4}", parts[1]) else None
    source = parts[2] if len(parts) > 2 else None
    return authors, year, source

def search_articles_serper(
    keywords: List[str],
    annee_reference: int,
    max_per_kw: int = 2,
) -> List[Dict[str, Any]]:
    """
    Interroge Google Scholar via Serper pour chaque mot-clé
    et retourne pour chacun les 'max_per_kw' articles les plus cités
    restreints sur [annee-3, annee-1] si l'année est détectable.

    Retour (liste d'articles) — champs normalisés:
      - title, url, citations, authors, year, journal, keyword, selected
    """
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY manquante dans l'environnement.")

    url = "https://google.serper.dev/scholar"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    start_year = annee_reference - 3
    end_year   = annee_reference - 1

    results: List[Dict[str, Any]] = []

    for kw in keywords:
        # On demande plus large puis on filtrera/ordonnera
        payload = {"q": kw, "num": max_per_kw * 5}

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=20)
            r.raise_for_status()
        except Exception as e:
            # on continue sur le mot-clé suivant
            print(f"[Serper] erreur pour '{kw}': {e}")
            continue

        organic = (r.json() or {}).get("organic", []) or []

        parsed: List[Dict[str, Any]] = []
        for item in organic:
            title = (item.get("title") or "").strip()
            link  = item.get("link") or ""
            if not title or not link:
                continue

            # Année: essayer 'year', sinon la deviner via publicationInfo
            year = item.get("year")
            if not year and item.get("publicationInfo"):
                m = re.search(r"\b(19|20)\d{2}\b", str(item["publicationInfo"]))
                if m:
                    year = int(m.group(0))

            # Filtre sur fenêtre si l'année est disponible
            if isinstance(year, int) and not (start_year <= year <= end_year):
                continue

            authors, y2, source = _parse_publication_info(item.get("publicationInfo", ""))
            if not year and y2:
                year = y2

            citations = int(item.get("citedBy") or 0)

            parsed.append({
                "title": title,
                "url": link,
                "citations": citations,
                "authors": authors or "",
                "year": year,
                "journal": source or "",
                "keyword": kw,
                "selected": True,
            })

        # Trier par citations et prendre top N
        parsed.sort(key=lambda x: x["citations"], reverse=True)
        results.extend(parsed[:max_per_kw])

    # Ordonner globalement par citations décroissantes
    results.sort(key=lambda x: x["citations"], reverse=True)
    return results
