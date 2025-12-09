# app/services/serper_proxy.py
import requests
from typing import List, Dict, Any
from app.config import settings

def search_serper_topN(keyword: str, ref_year: int, n: int = 2) -> List[Dict[str, Any]]:
    """
    Proxy côté serveur (optionnel) pour Google Scholar via Serper.
    Renvoie les N articles les plus cités pour un mot-clé (filtré [ref-3, ref-1]).
    """
    if not settings.SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY manquante")

    url = "https://google.serper.dev/scholar"
    headers = {"X-API-KEY": settings.SERPER_API_KEY, "Content-Type": "application/json"}
    body = {"q": keyword, "num": 10}

    r = requests.post(url, headers=headers, json=body, timeout=20)
    r.raise_for_status()
    json = r.json()
    organic = json.get("organic", []) or []

    start, end = ref_year - 3, ref_year - 1
    out = []
    for item in organic:
        title = (item.get("title") or "").strip()
        link = item.get("link") or ""
        if not title or not link:
            continue
        year = item.get("year")
        if not year and item.get("publicationInfo"):
            import re
            m = re.search(r"\b(19|20)\d{2}\b", item["publicationInfo"])
            if m:
                year = int(m.group(0))

        if isinstance(year, int):
            if year < start or year > end:
                continue

        authors, source = None, None
        pubinfo = item.get("publicationInfo")
        if pubinfo:
            parts = [p.strip() for p in pubinfo.split(" - ")]
            if parts: authors = parts[0]
            if len(parts) > 2: source = parts[2]

        cited = int(item.get("citedBy") or 0)
        out.append({
            "title": title, "url": link, "citations": cited, "authors": authors, "year": year, "journal": source, "selected": True
        })

    out.sort(key=lambda x: x["citations"], reverse=True)
    return out[:n]
