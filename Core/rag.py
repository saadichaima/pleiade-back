# Core/rag.py
import os, json
from typing import Optional, Dict, Any, List, Callable
import re

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from Core.embeddings import embed_texts
from app.services.prompts import fetch_cir, fetch_cii, prompt_evaluateur_travaux

load_dotenv()

# Détection du type d'API : Responses API (GPT-5.2+) ou Chat Completions API (GPT-4.1)
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
USE_RESPONSES_API = API_VERSION.startswith("2025-") or "2025" in API_VERSION

# Timeout pour les appels LLM (en secondes)
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))  # 2 minutes par defaut

if USE_RESPONSES_API:
    # Nouvelle API Responses pour GPT-5.2
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    # Garder cognitiveservices.azure.com pour Azure
    base_url = endpoint.rstrip("/") + "/openai/v1/"

    print(f"[DEBUG] Using Responses API with base_url: {base_url}")

    client = OpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        base_url=base_url,
        timeout=LLM_TIMEOUT,
    )

    print(f"[DEBUG] Client type: {type(client)}, has responses: {hasattr(client, 'responses')}, timeout: {LLM_TIMEOUT}s")
else:
    # Ancienne API Chat Completions pour GPT-4.1
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=API_VERSION,
        timeout=LLM_TIMEOUT,
    )

GPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

TOKENS_SINK: Optional[Callable[[Dict[str, Any]], None]] = None
def set_tokens_sink(fn: Callable[[Dict[str, Any]], None]):
    global TOKENS_SINK
    TOKENS_SINK = fn

def _extract_usage(resp):
    try:
        if hasattr(resp, "model_dump"):
            return (resp.model_dump() or {}).get("usage", {}) or {}
        return dict(getattr(resp, "usage", {}) or {})
    except Exception:
        return {}

def call_ai(prompt: str, *, meta: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 40000) -> str:
    import time
    start_time = time.time()
    prompt_len = len(prompt)
    print(f"[CALL_AI] Debut appel LLM - meta={meta}, prompt={prompt_len} chars, max_tokens={max_tokens}")

    try:
        if USE_RESPONSES_API:
            # Nouvelle API Responses pour GPT-5.2
            # Note: GPT-5.2 ne supporte pas le paramètre temperature
            r = client.responses.create(
                model=GPT_DEPLOYMENT,
                input=[
                    {"role": "system", "content": "Tu es un expert rédaction scientifique."},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=max_tokens,
            )
            txt = r.output_text or ""

            # Extraction des tokens pour TOKENS_SINK
            if TOKENS_SINK:
                try:
                    usage_data = {
                        "meta": meta,
                        "prompt_tokens": getattr(r.usage, "input_tokens", 0) if hasattr(r, "usage") else 0,
                        "completion_tokens": getattr(r.usage, "output_tokens", 0) if hasattr(r, "usage") else 0,
                        "total_tokens": getattr(r.usage, "total_tokens", 0) if hasattr(r, "usage") else 0,
                    }
                    TOKENS_SINK(usage_data)
                except Exception:
                    pass
        else:
            # Ancienne API Chat Completions pour GPT-4.1
            r = client.chat.completions.create(
                model=GPT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "Tu es un expert rédaction scientifique."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            txt = r.choices[0].message.content or ""

            if TOKENS_SINK:
                u = _extract_usage(r)
                try:
                    TOKENS_SINK(
                        {
                            "meta": meta,
                            **{k: int(u.get(k, 0)) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                        }
                    )
                except Exception:
                    pass

        elapsed = time.time() - start_time
        print(f"[CALL_AI] Succes - meta={meta}, reponse={len(txt)} chars, duree={elapsed:.1f}s")
        return txt

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[CALL_AI] ERREUR - meta={meta}, duree={elapsed:.1f}s, erreur={type(e).__name__}: {e}")
        raise

# -------------------- RAG helpers --------------------
def search_similar_chunks(query: str, index, chunks: List[str], vectors, top_k: int = 3):
    if not chunks or index is None:
        return []
    q = embed_texts([query])[0]
    import numpy as np
    q = np.array([q], dtype=np.float32).reshape(1, -1)
    actual_k = min(top_k, len(chunks))
    d, idx = index.kneighbors(q, n_neighbors=actual_k)
    return [chunks[i] for i in idx[0]]

def _build(template: str, **kw) -> str:
    class _Safe(dict):
        def __missing__(self, k): return ""
    try:
        return (template or "").format_map(_Safe(**kw))
    except Exception:
        return template or ""

def _articles_list_str(articles: Optional[List[dict]]) -> str:
    if not articles:
        return "- Aucune référence sélectionnée."
    lines: List[str] = []
    for a in (articles or [])[:25]:
        authors = (a.get("authors") or "?").strip()
        year = a.get("year") or "?"
        title = (a.get("title") or "").strip()
        journal = (a.get("journal") or "").strip()
        url = (a.get("url") or "").strip()
        s = f"- {authors} ({year}). {title}" if title else f"- {authors} ({year})."
        if journal:
            s += f" — {journal}"
        if url:
            s += f" — {url}"
        lines.append(s)
    return "\n".join(lines) if lines else "- Aucune référence sélectionnée."

# -------------------- Prompt Manager (Azure Blob) --------------------
def _tmpl(name: str) -> str:
    mapping = {
        "objectif_unique": "objectif_unique.txt",  # NOUVEAU
        "verrou_unique": "verrou_unique.txt",      # NOUVEAU

        "contexte": "prompt_contexte.txt",
        "indicateurs": "indicateurs.txt",
        "objectifs": "objectifs.txt",
        "travaux": "travaux.txt",
        "contribution": "contribution.txt",
        "partenariat": "partenariat.txt",
        "verrou": "verrou.txt",  # description longue du verrou
        "entreprise": "entreprise.txt",
        "gestion": "gestion_recherche.txt",
        "resume": "resume.txt",
        "ressources": "ressources_humaines.txt",
        "bibliographie": "bibliographie.txt",
        "footnotes": "footnotes.txt",
    }
    filename = mapping.get(name, f"{name}.txt")
    return fetch_cir(filename)

def generate_section_with_rag(title: str, instruction: str, index, chunks, vectors, *, top_k: int = 3, temperature: float = 0.2) -> str:
    ctx = "\n".join(search_similar_chunks(title or "section", index, chunks, vectors, top_k=top_k)) if index else ""
    prompt = f"""Rédige la section "{title}".
Contexte:
\"\"\"{ctx}\"\"\"
Consignes:
{instruction}
"""
    return call_ai(prompt, meta=title, temperature=temperature)

# -------------------- Génération CANONIQUE (objectif/verrou uniques) --------------------
def generate_objectif_unique(index, chunks, vectors, *, annee: int, societe: str, articles: Optional[List[dict]] = None) -> str:
    """
    Génère l'objectif canonique (stable) AVANT le reste.
    Recommandé: température 0.
    """
    instruction = _build(
        _tmpl("objectif_unique"),
        annee=annee,
        societe=societe,
        liste_articles=_articles_list_str(articles),
    )
    txt = generate_section_with_rag(
        "Objectif unique (canonique)",
        instruction,
        index, chunks, vectors,
        top_k=5,
        temperature=0.0,
    )
    return (txt or "").strip()

def generate_verrou_unique(index, chunks, vectors, *, objectif_unique: str, annee: int, societe: str, articles: Optional[List[dict]] = None) -> str:
    """
    Génère le verrou canonique (stable) APRÈS l'objectif.
    Recommandé: température 0.
    """
    instruction = _build(
        _tmpl("verrou_unique"),
        objectif_unique=objectif_unique,
        annee=annee,
        societe=societe,
        liste_articles=_articles_list_str(articles),
    )
    txt = generate_section_with_rag(
        "Verrou unique (canonique)",
        instruction,
        index, chunks, vectors,
        top_k=5,
        temperature=0.0,
    )
    return (txt or "").strip()

# -------------------- Sections CIR (toutes ancrées sur objectif_unique/verrou_unique) --------------------
def prompt_objectifs_filtre(objectif_unique: str, verrou_unique: str, annee: int, societe: str, articles: Optional[List[dict]] = None) -> str:
    tpl = _tmpl("objectifs")
    return _build(
        tpl,
        objectif_unique=objectif_unique,
        verrou_unique=verrou_unique,
        # compat si vos prompts utilisent encore objectif/verrou
        objectif=objectif_unique,
        verrou=verrou_unique,
        annee=annee,
        societe=societe,
        liste_articles=_articles_list_str(articles),
        annee_debut=annee - 5,
        annee_fin=annee - 1,
    )

def generate_objectifs_section(index, chunks, vectors, *, objectif_unique: str, verrou_unique: str, annee: int, societe: str, articles: Optional[List[dict]] = None) -> str:
    return generate_section_with_rag(
        "Objet de l’opération de R&D",
        prompt_objectifs_filtre(objectif_unique, verrou_unique, annee, societe, articles),
        index, chunks, vectors,
        temperature=0.2,
    )

def generate_verrou_section(index, chunks, vectors, *, objectif_unique: str, verrou_unique: str, annee: int, societe: str, articles: Optional[List[dict]] = None) -> str:
    """
    Description longue du verrou.
    IMPORTANT: le prompt verrou.txt doit afficher verrou_unique à l'identique (copier-coller).
    """
    return generate_section_with_rag(
        "Verrou technique (description)",
        _build(
            _tmpl("verrou"),
            objectif_unique=objectif_unique,
            verrou_unique=verrou_unique,
            # compat
            objectif=objectif_unique,
            verrou=verrou_unique,
            annee=annee,
            societe=societe,
            liste_articles=_articles_list_str(articles),
        ),
        index, chunks, vectors,
        temperature=0.2,
    )

def generate_contexte_section(index, chunks, vectors, *, objectif_unique: str, verrou_unique: str, annee: int, societe: str, articles: Optional[List[dict]] = None) -> str:
    return generate_section_with_rag(
        "Contexte de l’opération de R&D",
        _build(
            _tmpl("contexte"),
            objectif_unique=objectif_unique,
            verrou_unique=verrou_unique,
            # compat
            objectif=objectif_unique,
            verrou=verrou_unique,
            annee=annee,
            societe=societe,
            liste_articles=_articles_list_str(articles),
        ),
        index, chunks, vectors,
        temperature=0.2,
    )

def generate_indicateurs_section(index, chunks, vectors, *, objectif_unique: str, verrou_unique: str, annee: int, societe: str) -> str:
    return generate_section_with_rag(
        "Indicateurs de R&D",
        _build(
            _tmpl("indicateurs"),
            objectif_unique=objectif_unique,
            verrou_unique=verrou_unique,
            objectif=objectif_unique,
            verrou=verrou_unique,
            annee=annee,
            societe=societe,
        ),
        index, chunks, vectors,
        temperature=0.2,
    )

def generate_travaux_section(index, chunks, vectors, *, objectif_unique: str, verrou_unique: str, annee: int, societe: str) -> str:
    return generate_section_with_rag(
        "Description de la démarche suivie et des travaux réalisés",
        _build(
            _tmpl("travaux"),
            objectif_unique=objectif_unique,
            verrou_unique=verrou_unique,
            objectif=objectif_unique,
            verrou=verrou_unique,
            annee=annee,
            societe=societe,
        ),
        index, chunks, vectors,
        temperature=0.2,
    )

def generate_contribution_section(index, chunks, vectors, *, objectif_unique: str, verrou_unique: str, annee: int, societe: str) -> str:
    return generate_section_with_rag(
        "Contribution scientifique, technique ou technologique",
        _build(
            _tmpl("contribution"),
            objectif_unique=objectif_unique,
            verrou_unique=verrou_unique,
            objectif=objectif_unique,
            verrou=verrou_unique,
            annee=annee,
            societe=societe,
        ),
        index, chunks, vectors,
        temperature=0.2,
    )

def generate_partenariat_section(index, chunks, vectors, *, objectif_unique: str, verrou_unique: str, annee: int, societe: str) -> str:
    return generate_section_with_rag(
        "Partenariat scientifique et recherche confiée",
        _build(
            _tmpl("partenariat"),
            objectif_unique=objectif_unique,
            verrou_unique=verrou_unique,
            objectif=objectif_unique,
            verrou=verrou_unique,
            annee=annee,
            societe=societe,
        ),
        index, chunks, vectors,
        temperature=0.2,
    )

def generate_entreprise_section(index, chunks, vectors, *, objectif_unique: str, verrou_unique: str, annee: int, societe: str, style=None, site_web: str = "") -> str:
    return generate_section_with_rag(
        "L’entreprise",
        _build(
            _tmpl("entreprise"),
            objectif_unique=objectif_unique,
            verrou_unique=verrou_unique,
            objectif=objectif_unique,
            verrou=verrou_unique,
            annee=annee,
            societe=societe,
            style=(style or ""),
            site_web=site_web or "(site non renseigné)",
        ),
        index, chunks, vectors,
        temperature=0.2,
    )

def generate_gestion_recherche_section(index, chunks, vectors, *, objectif_unique: str, verrou_unique: str, annee: int, societe: str) -> str:
    return generate_section_with_rag(
        "Gestion de la recherche",
        _build(
            _tmpl("gestion"),
            objectif_unique=objectif_unique,
            verrou_unique=verrou_unique,
            objectif=objectif_unique,
            verrou=verrou_unique,
            annee=annee,
            societe=societe,
        ),
        index, chunks, vectors,
        temperature=0.2,
    )

def generate_resume_section(index, chunks, vectors, *, objectif_unique: str, verrou_unique: str, annee: int, societe: str, articles: Optional[List[dict]] = None) -> str:
    return generate_section_with_rag(
        "Résumé scientifique de l'opération",
        _build(
            _tmpl("resume"),
            objectif_unique=objectif_unique,
            verrou_unique=verrou_unique,
            objectif=objectif_unique,
            verrou=verrou_unique,
            annee=annee,
            societe=societe,
            liste_articles=_articles_list_str(articles),
        ),
        index, chunks, vectors,
        temperature=0.2,
    )

def generate_resume_from_sections(sections: dict, *, objectif_unique: str, verrou_unique: str, annee: int, societe: str, articles: Optional[List[dict]] = None) -> str:
    """
    Génère un résumé basé sur les sections déjà générées du document CIR.
    Garantit que le résumé reflète exactement le contenu du document final.
    """
    import time
    from app.services.prompts import prompt_cir_resume

    print("[RESUME] Debut generation du resume...")

    # Construire le contenu des sections pour le prompt
    sections_content = ""

    if sections.get("entreprise"):
        sections_content += f"\n\n=== PRÉSENTATION DE L'ENTREPRISE ===\n{sections['entreprise']}"

    if sections.get("contexte"):
        sections_content += f"\n\n=== CONTEXTE ===\n{sections['contexte']}"

    if sections.get("objectifs"):
        sections_content += f"\n\n=== OBJECTIFS ===\n{sections['objectifs']}"

    if sections.get("verrous"):
        sections_content += f"\n\n=== VERROUS SCIENTIFIQUES ET TECHNIQUES ===\n{sections['verrous']}"

    if sections.get("travaux"):
        sections_content += f"\n\n=== TRAVAUX RÉALISÉS ===\n{sections['travaux']}"

    if sections.get("contribution"):
        sections_content += f"\n\n=== CONTRIBUTION SCIENTIFIQUE ===\n{sections['contribution']}"

    if sections.get("indicateurs"):
        sections_content += f"\n\n=== INDICATEURS ===\n{sections['indicateurs']}"

    if sections.get("partenariat"):
        sections_content += f"\n\n=== PARTENARIATS ===\n{sections['partenariat']}"

    if sections.get("gestion"):
        sections_content += f"\n\n=== GESTION DE LA RECHERCHE ===\n{sections['gestion']}"

    # Limiter la taille du contenu des sections pour eviter les timeouts
    MAX_SECTION_CHARS = 25000  # ~6000 tokens
    if len(sections_content) > MAX_SECTION_CHARS:
        print(f"[RESUME] ATTENTION: Contenu trop long ({len(sections_content)} chars), truncation a {MAX_SECTION_CHARS}")
        sections_content = sections_content[:MAX_SECTION_CHARS] + "\n\n[... contenu tronqué pour le résumé ...]"

    print(f"[RESUME] Taille du contenu des sections: {len(sections_content)} caracteres")

    # Récupérer le template du prompt depuis le blob
    print("[RESUME] Chargement du template prompt depuis blob...")
    try:
        prompt_template = prompt_cir_resume()
        print(f"[RESUME] Template charge: {len(prompt_template)} caracteres")
    except Exception as e:
        print(f"[RESUME] ERREUR chargement template: {type(e).__name__}: {e}")
        raise

    # Remplacer les variables dans le prompt
    print("[RESUME] Formatage du prompt...")
    try:
        articles_str = _articles_list_str(articles)
        prompt = prompt_template.format(
            sections_content=sections_content,
            societe=societe,
            annee=annee,
            objectif_unique=objectif_unique,
            verrou_unique=verrou_unique,
            articles=articles_str,
            liste_articles=articles_str,  # Le template utilise {liste_articles}
        )
        print(f"[RESUME] Prompt formate: {len(prompt)} caracteres")
    except Exception as e:
        print(f"[RESUME] ERREUR formatage prompt: {type(e).__name__}: {e}")
        raise

    print(f"[RESUME] Taille totale du prompt: {len(prompt)} caracteres (~{len(prompt)//4} tokens)")
    print("[RESUME] Appel LLM en cours...")

    start_time = time.time()
    try:
        result = call_ai(prompt, meta="resume_from_sections", temperature=0.3)
        elapsed = time.time() - start_time
        print(f"[RESUME] Resume genere avec succes en {elapsed:.1f}s ({len(result)} chars)")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[RESUME] ERREUR apres {elapsed:.1f}s: {type(e).__name__}: {e}")
        raise

# -------------------- Bibliographie, RH, evaluateur_travaux : gardez votre code existant --------------------
# (Je ne le recopie pas ici pour éviter d'introduire des divergences non nécessaires.)


# -------------------- Bibliographie & RH --------------------
def _parse_authors(authors_raw: str):
    if not authors_raw:
        return []
    tmp = authors_raw.replace(" and ", ",").replace(" et ", ",")
    parts = [p.strip() for p in tmp.split(",") if p.strip()]
    authors = []
    for p in parts:
        tokens = p.split()
        if not tokens:
            continue
        surname = tokens[-1].upper()
        firstnames = " ".join(tokens[:-1])
        authors.append((surname, firstnames))
    return authors

def _format_authors_iso(authors_raw: str) -> str:
    authors = _parse_authors(authors_raw or "")
    if not authors:
        return ""
    if len(authors) == 1:
        s, f = authors[0]
        return f"{s}, {f}" if f else s

    formatted = []
    for (s, f) in authors[:3]:
        formatted.append(f"{s}, {f}" if f else s)
    if len(authors) > 3:
        formatted[-1] = formatted[-1] + ", et al."
    return "; ".join(formatted)

def format_iso_690(a: dict) -> str:
    titre = (a.get("title") or "").strip()
    auteurs = (a.get("authors") or "").strip()
    journal = (a.get("journal") or "").strip()

    # Article manuel ISO déjà prêt
    if titre and not auteurs and not journal:
        return titre

    auteurs_iso = _format_authors_iso(auteurs)
    year = a.get("year") or "s.d."
    return f"{auteurs_iso}. {titre or 'Sans titre'}. {journal or 'Revue inconnue'}, {year}."

def generate_biblio_section(articles: Optional[List[dict]] = None) -> str:
    if not articles:
        return "Aucune référence sélectionnée."

    uniq: List[dict] = []
    seen_keys = set()

    for a in articles:
        title = (a.get("title") or "").strip().lower()
        authors = (a.get("authors") or "").strip().lower()
        journal = (a.get("journal") or "").strip().lower()

        if not authors and not journal:
            uniq.append(a)
            continue

        key = (title, authors, journal)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        uniq.append(a)

    uniq.sort(key=lambda x: (x.get("authors") or "").lower())
    return "\n".join(format_iso_690(x) for x in uniq)

def generate_ressources_humaines_from_cvs(cv_texts: List[str]):
    ctx = "\n\n".join((cv_texts or [])[:5])
    prompt = f"""
Tu reçois le contenu brut de CVs. Pour chaque personne, extrais les informations REELLES trouvées dans les CVs.

IMPORTANT: Ne génère PAS de placeholder comme "[À compléter par le client]".
Si une information n'est PAS présente dans les CVs, utilise une chaîne vide "".

Renvoie un JSON avec cette structure exacte (tableau de personnes):
```json
[
  {{
    "nom_prenom": "NOM Prénom réel du CV",
    "diplome": "Diplôme le plus élevé trouvé dans le CV ou vide",
    "fonction": "Fonction dans l'opération trouvée dans le CV ou vide",
    "contribution": "Contribution décrite dans le CV ou vide",
    "temps": "Temps/heures mentionné dans le CV ou vide"
  }}
]
```

CVs:
\"\"\"{ctx}\"\"\"

Renvoie UNIQUEMENT le JSON (tableau), sans texte avant ni après, sans markdown.
"""
    txt = call_ai(prompt, meta="Ressources humaines")

    try:
        import re
        # Nettoyer le texte retourné
        cleaned = txt.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*\n', '', cleaned)
            cleaned = re.sub(r'\n```\s*$', '', cleaned)

        # Extraire le JSON
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        if start != -1 and end > start:
            json_str = cleaned[start:end]
            personnel = json.loads(json_str)

            # Nettoyer les valeurs "[À compléter par le client]"
            for person in personnel:
                for key in list(person.keys()):
                    val = str(person.get(key, ""))
                    if "[" in val and "compléter" in val.lower():
                        person[key] = ""

            return personnel
        else:
            raise ValueError("Pas de tableau JSON trouvé")

    except Exception as e:
        print(f"[RH CIR] Erreur parsing JSON: {e}")
        print(f"[RH CIR] Contenu reçu: {txt[:200]}...")
        return [
            {
                "nom_prenom": "Erreur parsing",
                "diplome": "",
                "fonction": "",
                "contribution": str(e),
                "temps": "",
            }
        ]

def evaluateur_travaux(texte: str, *, type_dossier: str = "CIR") -> str:
    libelle = (
        "Description détaillée de la démarche scientifique suivie et des travaux réalisés"
        if type_dossier == "CIR"
        else "Description détaillée de la démarche expérimentale suivie et des travaux réalisés"
    )
    tpl = prompt_evaluateur_travaux()
    try:
        prompt = tpl.format(
            type_dossier=type_dossier,
            libelle=libelle,
            texte=texte,
        )
    except Exception as e:
        raise RuntimeError(f"Erreur formatage prompt evaluateur_travaux (CIR): {e}")

    return call_ai(prompt, meta="Travaux avec questions")

QUESTION_RE = re.compile(r"(?is)\b(?:Pouvez|Pourriez)[\-\u2011\u2013 ]vous[^?]*\?")

def wrap_questions_rouge(texte: str) -> str:
    if not texte:
        return texte
    if "[[ROUGE:" in texte:
        return texte

    pieces = []
    pos = 0
    for m in QUESTION_RE.finditer(texte):
        start, end = m.span()
        if start > pos:
            pieces.append(texte[pos:start])
        q = m.group(0)
        pieces.append(f"[[ROUGE: {q} ]]")
        pos = end

    pieces.append(texte[pos:])
    return "".join(pieces)

def enrich_manual_articles_iso(
    articles: List[dict],
    articles_texts: List[str],
    max_retries: int = 2,
) -> List[dict]:
    """
    Enrichit les articles manuels (sans auteurs/citations) avec les metadonnees ISO 690.
    Ajoute des logs de progression et retry en cas d'echec.
    """
    import time

    if not articles:
        print("[ISO690] Aucun article a enrichir")
        return []

    out: List[dict] = [dict(a) for a in articles]
    manual_idxs = [
        i for i, a in enumerate(out)
        if not (a.get("authors") or "").strip() and int(a.get("citations") or 0) == 0
    ]

    total_manual = len(manual_idxs)
    print(f"[ISO690] Debut enrichissement: {total_manual} articles manuels a traiter sur {len(articles)} total")

    if total_manual == 0:
        print("[ISO690] Aucun article manuel a enrichir (tous ont deja des auteurs/citations)")
        return out

    txt_idx = 0
    processed = 0
    errors = 0

    for i in manual_idxs:
        processed += 1
        article_title = out[i].get("title", "Sans titre")[:50]
        print(f"[ISO690] Traitement article {processed}/{total_manual}: '{article_title}...'")
        start_time = time.time()

        if txt_idx >= len(articles_texts):
            print(f"[ISO690] Plus de texte disponible pour l'article {processed}")
            break
        txt = (articles_texts[txt_idx] or "").strip()
        txt_idx += 1
        if not txt:
            print(f"[ISO690] Texte vide pour l'article {processed}, passage au suivant")
            continue

        snippet = txt[:4000]

        prompt = f"""
Tu es un expert en normalisation bibliographique (norme ISO 690).
On te fournit le début du texte d'un article scientifique.

Objectif:
1. Identifier les métadonnées principales:
   - auteurs (liste),
   - titre de l'article,
   - titre de la revue ou conférence,
   - année de publication,
   - volume,
   - pages.
2. Produire une référence complète au format ISO 690 (style francophone) pour un article de revue.

Texte (extrait) :
\"\"\"{snippet}\"\"\"

Format de réponse STRICT (JSON UTF-8) :

{{
  "authors": "ARSLAN Muhammad, GHANEM Hussam, MUNAWAR Saba, et al.",
  "title": "A Survey on RAG with LLMs",
  "journal": "Procedia Computer Science",
  "year": 2024,
  "volume": "246",
  "pages": "3781-3790",
  "iso_citation": "ARSLAN, Muhammad, GHANEM, Hussam, MUNAWAR, Saba, et al. A Survey on RAG with LLMs. Procedia Computer Science, 2024, vol. 246, p. 3781-3790."
}}

Ne renvoie que le JSON, sans texte avant ni après.
"""
        # Retry logic
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"[ISO690] Retry {attempt}/{max_retries} pour l'article {processed}")
                    time.sleep(2 * attempt)  # Backoff exponentiel

                raw = call_ai(prompt, meta="ISO690_manual_article")
                raw = (raw or "").strip()
                start = raw.find("{")
                end = raw.rfind("}")
                json_str = raw[start : end + 1] if start != -1 and end != -1 else "{}"
                data = json.loads(json_str)

                a = out[i]
                if data.get("authors"):
                    a["authors"] = data["authors"]
                if data.get("title"):
                    a["title"] = data["title"]
                if data.get("journal"):
                    a["journal"] = data["journal"]
                if data.get("year"):
                    try:
                        a["year"] = int(data["year"])
                    except Exception:
                        pass
                if data.get("volume"):
                    a["volume"] = str(data["volume"])
                if data.get("pages"):
                    a["pages"] = str(data["pages"])
                if data.get("iso_citation"):
                    a["iso_citation"] = data["iso_citation"]

                elapsed = time.time() - start_time
                print(f"[ISO690] Article {processed}/{total_manual} enrichi en {elapsed:.1f}s")
                last_error = None
                break  # Succes, sortir de la boucle retry

            except Exception as e:
                last_error = e
                print(f"[ISO690] Erreur article {processed} (tentative {attempt + 1}): {type(e).__name__}: {e}")

        if last_error:
            errors += 1
            print(f"[ISO690] Echec definitif pour l'article {processed} apres {max_retries + 1} tentatives")

    print(f"[ISO690] Enrichissement termine: {processed - errors}/{total_manual} articles enrichis, {errors} erreurs")
    return out
