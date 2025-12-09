# app/routers/articles.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.deps import ensure_serper
from app.models.schemas import ArticlesRequest, ArticlesResponse, Article
from Core.serper import search_articles_serper

router = APIRouter()

@router.post(   "/articles/search",
    response_model=ArticlesResponse,
    summary="Recherche d’articles Google Scholar (Serper)",
    description="Pour chaque mot-clé, récupère les articles les plus cités sur la fenêtre [année-3, année-1].",)
def search_articles(body: ArticlesRequest, _: bool = Depends(ensure_serper)):
    try:
        raw_list: List[dict] = search_articles_serper(
            body.keywords[:5], annee_reference=body.annee, max_per_kw=2
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    mapped: List[Article] = []
    for a in raw_list or []:
        if not isinstance(a, dict):
            continue
        title = (a.get("title") or "").strip()
        url   = (a.get("url") or a.get("link") or "").strip()

        # on ignore les entrées sans titre OU sans url
        if not title or not url:
            continue

        try:
            mapped.append(Article(
                title=title,
                year=a.get("year"),
                authors=a.get("authors") or "",
                journal=a.get("journal") or "",
                url=url,
                citations=int(a.get("citations") or 0),
                selected=True,
            ))
        except Exception:
            # on skip toute entrée douteuse
            continue

    return {"articles": mapped}
