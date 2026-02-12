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


# --------------- Recherche de sites web concurrents ---------------

def search_competitor_website(name: str, timeout: int = 10) -> str:
    """
    Cherche le site officiel d'un concurrent via Google Search (Serper).
    Retourne l'URL du domaine principal ou '' si non trouvé.
    """
    if not SERPER_API_KEY:
        return ""

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": f"{name} site officiel", "num": 5, "gl": "fr", "hl": "fr"}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        print(f"[Serper Web] Erreur recherche '{name}': {e}")
        return ""

    organic = (r.json() or {}).get("organic", []) or []
    if not organic:
        print(f"[Serper Web] Aucun résultat pour '{name}'")
        return ""

    # Prendre le premier résultat organique
    link = (organic[0].get("link") or "").strip()
    print(f"[Serper Web] '{name}' → {link}")
    return link


def search_competitors_websites(names: List[str], timeout: int = 10) -> Dict[str, str]:
    """
    Cherche les sites officiels de plusieurs concurrents en parallèle.
    Retourne un dict {nom: url}.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not SERPER_API_KEY or not names:
        return {n: "" for n in names}

    result: Dict[str, str] = {}
    max_workers = min(len(names), 5)

    def _search(name: str) -> tuple:
        return name, search_competitor_website(name, timeout=timeout)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for name, url in pool.map(lambda n: _search(n), names):
            result[name] = url

    return result


# --------------- Recherche directe de concurrents via Google ---------------

def search_competitors_from_serper(
    societe: str,
    projet: str,
    secteur_keywords: str = "",
    max_results: int = 10,
    timeout: int = 15,
) -> List[Dict[str, str]]:
    """
    Cherche directement des concurrents sur Google via Serper.
    Fait 2 requêtes avec des angles différents pour maximiser la couverture,
    puis déduplique par domaine.

    Retourne une liste de dicts: [{"name": ..., "site": ..., "snippet": ...}]
    """
    if not SERPER_API_KEY:
        return []

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    # 2 requêtes complémentaires pour couvrir large
    queries = [
        f"{projet} concurrents alternatives solutions",
        f"{societe} {projet} competitors marché",
    ]
    if secteur_keywords:
        queries.append(f"{secteur_keywords} solutions logiciels entreprises")

    seen_domains: set = set()
    results: List[Dict[str, str]] = []

    for q in queries:
        payload = {"q": q, "num": 10, "gl": "fr", "hl": "fr"}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json() or {}
        except Exception as e:
            print(f"[Serper Competitors] Erreur pour '{q}': {e}")
            continue

        for item in data.get("organic", []) or []:
            link = (item.get("link") or "").strip()
            title = (item.get("title") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            if not link or not title:
                continue

            # Extraire le domaine pour dédupliquer
            from urllib.parse import urlparse
            domain = urlparse(link).netloc.lower().replace("www.", "")

            # Ignorer le site du client lui-même
            client_domain = urlparse(f"https://{societe.lower().replace(' ', '')}.com").netloc
            if domain == client_domain:
                continue

            # Ignorer les sites généralistes (wikipedia, linkedin, comparateurs génériques)
            skip_domains = {
                "wikipedia.org", "linkedin.com", "facebook.com", "twitter.com",
                "youtube.com", "reddit.com", "quora.com", "medium.com",
                "amazon.com", "amazon.fr",
            }
            if any(sd in domain for sd in skip_domains):
                continue

            if domain in seen_domains:
                continue
            seen_domains.add(domain)

            results.append({
                "name": title,
                "site": link,
                "snippet": snippet,
            })

        if len(results) >= max_results:
            break

    print(f"[Serper Competitors] {len(results)} concurrents trouvés pour '{projet}'")
    return results[:max_results]
