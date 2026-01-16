# app/services/web_scraper.py
"""
Module de scraping de sites web pour enrichir le contexte RAG.
Utilisé lorsque aucun document administratif n'est fourni.
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional
from urllib.parse import urljoin, urlparse


def scrape_website(url: str, max_pages: int = 3, timeout: int = 10) -> str:
    """
    Scrape le site web et retourne le texte principal.

    Args:
        url: URL du site (ex: https://entreprise.com)
        max_pages: Nombre max de pages à scraper (pour l'instant 1 seule)
        timeout: Timeout en secondes pour chaque requête

    Returns:
        Texte extrait (title + description + contenu principal)
    """
    if not url or not url.startswith("http"):
        print(f"[Web Scraper] URL invalide : {url}")
        return ""

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (PLEÏADES Bot/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        print(f"[Web Scraper] Début scraping : {url}")

        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
            verify=True,  # Vérifier SSL
        )
        response.raise_for_status()

        # Vérifier le Content-Type
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            print(f"[Web Scraper] Content-Type non HTML : {content_type}")
            return ""

        soup = BeautifulSoup(response.content, "html.parser")

        # 1. Extraire metadata
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else ""

        description = (
            soup.find("meta", attrs={"name": "description"}) or
            soup.find("meta", attrs={"property": "og:description"})
        )
        desc_text = description.get("content", "").strip() if description else ""

        # 2. Extraire contenu principal
        # Priorité : <main>, <article>, ou <div role="main">, sinon <body>
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", attrs={"role": "main"}) or
            soup.find("div", class_=lambda c: c and any(x in c.lower() for x in ["content", "main"])) or
            soup.find("body")
        )

        if not main_content:
            print("[Web Scraper] Aucun contenu principal trouvé")
            return f"Site web : {url}\n\nTitre : {title_text}\n\nDescription : {desc_text}"

        # Nettoyer (supprimer scripts, styles, nav, footer, header, aside)
        for tag in main_content.find_all([
            "script", "style", "nav", "footer", "header", "aside",
            "noscript", "iframe", "svg", "form"
        ]):
            tag.decompose()

        # Supprimer aussi les divs de navigation/menu
        for tag in main_content.find_all("div", class_=lambda c: c and any(
            x in c.lower() for x in ["menu", "nav", "sidebar", "cookie", "popup"]
        )):
            tag.decompose()

        # Extraire texte
        content_text = main_content.get_text(separator="\n", strip=True)

        # Nettoyer les lignes vides multiples
        lines = [line.strip() for line in content_text.split("\n") if line.strip()]
        content_text = "\n".join(lines)

        # Limiter la taille (max ~2000 mots pour éviter coûts embeddings)
        words = content_text.split()
        word_count = len(words)

        if word_count > 2000:
            print(f"[Web Scraper] Contenu tronqué : {word_count} mots → 2000 mots")
            words = words[:2000]
            content_text = " ".join(words)
        else:
            print(f"[Web Scraper] Contenu extrait : {word_count} mots")

        result = f"""=== INFORMATIONS EXTRAITES DU SITE WEB ===

URL : {url}

Titre : {title_text}

Description : {desc_text}

Contenu principal :
{content_text}

=== FIN CONTENU SITE WEB ===
"""

        return result

    except requests.exceptions.Timeout:
        print(f"[Web Scraper] Timeout lors du scraping de {url}")
        return ""
    except requests.exceptions.SSLError as e:
        print(f"[Web Scraper] Erreur SSL pour {url}: {e}")
        return ""
    except requests.exceptions.ConnectionError as e:
        print(f"[Web Scraper] Erreur de connexion pour {url}: {e}")
        return ""
    except requests.exceptions.HTTPError as e:
        print(f"[Web Scraper] Erreur HTTP {e.response.status_code} pour {url}")
        return ""
    except Exception as e:
        print(f"[Web Scraper] Erreur inattendue lors du scraping de {url}: {type(e).__name__}: {e}")
        return ""


def extract_website_context(site_web: str, text_admin: str, min_words: int = 500) -> str:
    """
    Enrichit text_admin avec le contenu scrapé du site web si nécessaire.

    Args:
        site_web: URL du site
        text_admin: Texte des documents administratifs
        min_words: Seuil en nombre de mots (si text_admin < min_words, on scrape)

    Returns:
        text_admin enrichi (ou original si scraping échoue)
    """
    # Compter les mots dans text_admin
    word_count = len((text_admin or "").split())

    # Si docs admin suffisants, on ne scrape pas
    if word_count >= min_words:
        print(f"[Web Scraper] Docs admin suffisants ({word_count} mots), pas de scraping nécessaire")
        return text_admin

    # Si pas de site web fourni, on ne peut pas scraper
    if not site_web or not site_web.strip():
        print(f"[Web Scraper] Docs admin insuffisants ({word_count} mots) mais aucun site web fourni")
        return text_admin

    # Scraping nécessaire
    print(f"[Web Scraper] Docs admin insuffisants ({word_count} mots), scraping de {site_web}...")

    scraped_text = scrape_website(site_web)

    if scraped_text:
        scraped_words = len(scraped_text.split())
        print(f"[Web Scraper] Succès : {scraped_words} mots extraits du site web")

        # Combiner les deux sources
        if text_admin and text_admin.strip():
            result = text_admin + "\n\n" + scraped_text
        else:
            result = scraped_text

        return result
    else:
        print(f"[Web Scraper] Échec du scraping, utilisation des docs admin seuls ({word_count} mots)")
        return text_admin


def validate_url(url: str) -> bool:
    """
    Valide qu'une URL est bien formée et accessible.

    Args:
        url: URL à valider

    Returns:
        True si l'URL semble valide, False sinon
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url)
        return all([parsed.scheme in ["http", "https"], parsed.netloc])
    except Exception:
        return False
