# Core/article_fetcher.py
"""
Utilitaire pour récupérer le contenu textuel d'un article scientifique
à partir de son URL HTML (page web) lorsque le pdfUrl direct est absent.

Stratégie par ordre de priorité :
  1. Dériver l'URL PDF selon les patterns connus (MDPI, arXiv, ACM, PLoS, bioRxiv…)
  2. Scraper la page HTML pour trouver un lien PDF
  3. Extraire l'abstract depuis la page HTML (BeautifulSoup)
"""
import re
import requests
from typing import Optional
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

from Core import document

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
_TIMEOUT = 15


# --------------------------------------------------------------------------- #
# 1. Patterns PDF connus par éditeur                                           #
# --------------------------------------------------------------------------- #

def _derive_pdf_url(html_url: str) -> Optional[str]:
    """
    Dérive l'URL PDF depuis l'URL HTML selon les patterns connus des éditeurs
    open-access. Retourne None si l'éditeur est inconnu ou payant.
    """
    u = html_url.rstrip("/")

    # MDPI (open access) — ex: .../2073-431X/12/4/78 → .../2073-431X/12/4/78/pdf
    if "mdpi.com" in u:
        return u + "/pdf"

    # arXiv — ex: /abs/2301.12345 → /pdf/2301.12345
    if "arxiv.org/abs/" in u:
        return u.replace("/abs/", "/pdf/")

    # bioRxiv / medRxiv — ex: /content/10.1101/XXX → /content/10.1101/XXX.full.pdf
    if ("biorxiv.org/content/" in u or "medrxiv.org/content/" in u) and not u.endswith(".full.pdf"):
        return u + ".full.pdf"

    # ACM DL — ex: /doi/10.1145/XXX → /doi/pdf/10.1145/XXX
    if "dl.acm.org/doi/" in u and "/doi/pdf/" not in u:
        return u.replace("/doi/", "/doi/pdf/", 1)

    # PLoS journals — ex: article?id=10.1371/XXX → article/file?id=10.1371/XXX&type=printable
    if "journals.plos.org" in u and "article?id=" in u:
        return u.replace("article?id=", "article/file?id=") + "&type=printable"

    # PNAS — ex: /doi/10.1073/XXX → /doi/pdf/10.1073/XXX
    if "pnas.org/doi/" in u and "/doi/pdf/" not in u:
        return u.replace("/doi/", "/doi/pdf/", 1)

    # PeerJ (open access) — ex: /articles/NNNNN → /articles/NNNNN.pdf
    if "peerj.com/articles/" in u and not u.endswith(".pdf"):
        return u + ".pdf"

    # F1000Research / Wellcome Open Research
    if "f1000research.com" in u and not u.endswith(".pdf"):
        return u + "/pdf"

    return None


# --------------------------------------------------------------------------- #
# 2. Helpers                                                                   #
# --------------------------------------------------------------------------- #

def _try_fetch_pdf(pdf_url: str) -> Optional[bytes]:
    """Tente de télécharger et valider un PDF. Retourne les bytes ou None."""
    try:
        r = requests.get(pdf_url, timeout=_TIMEOUT, headers=_HEADERS, allow_redirects=True)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            return r.content
    except Exception as e:
        print(f"[FETCHER] Échec téléchargement PDF {pdf_url}: {e}")
    return None


def _scrape_pdf_link_from_html(base_url: str, soup: BeautifulSoup) -> Optional[bytes]:
    """
    Cherche un lien vers un PDF dans la page HTML déjà parsée et tente
    de le télécharger.
    """
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    for a in soup.find_all("a", href=True):
        href = str(a.get("href", ""))
        link_text = a.get_text(strip=True).lower()
        if "pdf" not in href.lower() and "pdf" not in link_text:
            continue
        full_url = urljoin(base, href) if not href.startswith("http") else href
        pdf_bytes = _try_fetch_pdf(full_url)
        if pdf_bytes:
            return pdf_bytes
    return None


def _extract_abstract_from_html(soup: BeautifulSoup) -> str:
    """
    Extrait l'abstract/résumé depuis le HTML avec une série de sélecteurs
    couvrant les principaux éditeurs scientifiques.
    """
    selectors = [
        "section.abstract",
        "#abstract",
        ".abstract-content",
        ".article-abstract",
        "div.abstract",
        "[data-testid='abstract']",
        ".abstractSection",
        "p.abstract",
        ".abstract-text",
        "#abstracts",
        ".c-article-section__content",  # Springer/Nature
        ".articleBody .section-paragraph",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(separator=" ", strip=True)
            if len(text) > 100:
                return text[:4000]

    # Fallback meta description (souvent = abstract)
    meta = (
        soup.find("meta", attrs={"name": "description"})
        or soup.find("meta", attrs={"property": "og:description"})
    )
    if meta:
        content = str(meta.get("content", ""))
        if len(content) > 100:
            return content[:2000]

    return ""


# --------------------------------------------------------------------------- #
# 3. Fonction principale                                                       #
# --------------------------------------------------------------------------- #

def fetch_article_content(url: str) -> str:
    """
    Tente de récupérer le texte d'un article scientifique depuis son URL HTML.

    Ordre de priorité :
      1. Pattern PDF connu → texte complet via PyMuPDF
      2. Scraping HTML pour trouver un lien PDF → texte complet
      3. Extraction abstract depuis le HTML

    Retourne le texte extrait ou "" si toutes les stratégies échouent.
    """
    if not url or not url.startswith("http"):
        return ""

    # -- Stratégie 1 : pattern PDF connu ------------------------------------
    pdf_url = _derive_pdf_url(url)
    if pdf_url:
        pdf_bytes = _try_fetch_pdf(pdf_url)
        if pdf_bytes:
            txt = document.extract_text_from_bytes(pdf_bytes, "article.pdf")
            if txt.strip():
                print(f"[FETCHER] Texte complet via pattern PDF ({len(txt)} chars) : {url}")
                return txt

    # -- Stratégies 2 et 3 : scraping HTML ----------------------------------
    try:
        r = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS, allow_redirects=True)
        if r.status_code != 200:
            print(f"[FETCHER] HTTP {r.status_code} pour {url}")
            return ""

        soup = BeautifulSoup(r.text, "lxml")

        # Stratégie 2 : lien PDF dans la page
        pdf_bytes = _scrape_pdf_link_from_html(url, soup)
        if pdf_bytes:
            txt = document.extract_text_from_bytes(pdf_bytes, "article.pdf")
            if txt.strip():
                print(f"[FETCHER] Texte complet via lien PDF dans HTML ({len(txt)} chars) : {url}")
                return txt

        # Stratégie 3 : abstract HTML
        abstract = _extract_abstract_from_html(soup)
        if abstract:
            print(f"[FETCHER] Abstract HTML extrait ({len(abstract)} chars) : {url}")
            return abstract

    except Exception as e:
        print(f"[FETCHER] Erreur scraping {url}: {e}")

    print(f"[FETCHER] Aucun contenu récupérable pour : {url}")
    return ""
