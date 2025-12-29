# Core/rag.py
import os, json
from typing import Optional, Dict, Any, List, Callable
import re

from dotenv import load_dotenv
from openai import AzureOpenAI

from Core.embeddings import embed_texts
from app.services.prompts import fetch_cir, fetch_cii, prompt_evaluateur_travaux

load_dotenv()

# -------------------- Azure OpenAI --------------------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
GPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

TOKENS_SINK: Optional[Callable[[Dict[str, Any]], None]] = None
def set_tokens_sink(fn: Callable[[Dict[str, Any]], None]):
    """Instrumentation optionnelle pour récupérer la conso tokens."""
    global TOKENS_SINK
    TOKENS_SINK = fn

def _extract_usage(resp):
    try:
        if hasattr(resp, "model_dump"):
            return (resp.model_dump() or {}).get("usage", {}) or {}
        return dict(getattr(resp, "usage", {}) or {})
    except Exception:
        return {}

def call_ai(prompt: str, *, meta: Optional[str] = None) -> str:
    r = client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Tu es un expert du CIR/CII."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=30000,
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
    return txt
def call_ai_json(prompt: str, *, meta: Optional[str] = None) -> str:
    """
    Variante 'JSON strict' :
    - température 0 pour réduire les sorties non conformes
    - même modèle et même infra
    """
    r = client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Tu es un générateur JSON strict. Tu ne renvoies que du JSON valide."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=30000,
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
    return txt

# -------------------- RAG helpers --------------------
def search_similar_chunks(query: str, index, chunks: List[str], vectors, top_k: int = 3):
    """Retourne les chunks les plus proches de la requête (embedding + KNN)."""
    if not chunks or index is None:
        return []
    q = embed_texts([query])[0]
    import numpy as np  # local import pour éviter le coût si inutilisé
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

# -------------------- Prompt Manager (Azure Blob) --------------------
def _tmpl(name: str) -> str:
    """
    Mappe un identifiant logique -> nom de fichier dans le conteneur 'prompts'.
    Les fichiers existent déjà dans ton blob (capture Azure) :
      - bibliographie.txt
      - contribution.txt
      - entreprise.txt
      - footnotes.txt
      - gestion_recherche.txt
      - indicateurs.txt
      - objectifs.txt
      - partenariat.txt
      - prompt_contexte.txt
      - ressources_humaines.txt
      - resume.txt
      - travaux.txt
    """
    mapping = {
        "contexte": "prompt_contexte.txt",
        "indicateurs": "indicateurs.txt",
        "objectifs": "objectifs.txt",
        "travaux": "travaux.txt",
        "contribution": "contribution.txt",
        "partenariat": "partenariat.txt",
        "verrou": "verrou.txt",  # si tu ajoutes le fichier au blob
        "entreprise": "entreprise.txt",
        "gestion": "gestion_recherche.txt",
        "resume": "resume.txt",
        "ressources": "ressources_humaines.txt",
        "bibliographie": "bibliographie.txt",
        "footnotes": "footnotes.txt",
    }
    filename = mapping.get(name, f"{name}.txt")
    return fetch_cir(filename)  # lit depuis le conteneur Azure 'prompts'

def generate_section_with_rag(title, instruction, index, chunks, vectors) -> str:
    ctx = "\n".join(search_similar_chunks(title or "section", index, chunks, vectors)) if index else ""
    prompt = f"""Rédige la section "{title}".
Contexte:
\"\"\"{ctx}\"\"\"
Consignes:
{instruction}
"""
    return call_ai(prompt, meta=title)

# Core/rag.py (extrait)
from app.services.prompts import fetch_cii

def _tmpl_cii(key: str) -> str:
    mapping = {
        "style": "style_guide.txt",
        "presentation": "presentation_strat.txt",
        "resume": "resume_scientifique.txt",
        "contexte": "contexte_general.txt",
        "analyse": "concurrence.txt",
        "demarche": "demarche_experimentale.txt",
        "resultats": "resultats.txt",
        "rh_intro": "rh.txt",
        "biblio_intro": "biblio.txt",
    }
    return fetch_cii(mapping.get(key, f"{key}.txt"))


# -------------------- Génération de sections CIR --------------------
def prompt_objectifs_filtre(objectif, verrou, annee, societe, articles: List[dict]) -> str:
    tpl = _tmpl("objectifs")
    lst = "\n".join([f"- {a.get('authors','?')} ({a.get('year','?')}). {a.get('title','')}" for a in articles or []])
    return _build(
        tpl,
        objectif=objectif,
        verrou=verrou,
        annee=annee,
        societe=societe,
        liste_articles=lst,
        annee_debut=annee - 5,
        annee_fin=annee - 1,
    )

def generate_contexte_section(i, c, v, obj, ver, an, soc):
    return generate_section_with_rag(
        "Contexte de l’opération de R&D",
        _build(_tmpl("contexte"), objectif=obj, verrou=ver, annee=an, societe=soc),
        i, c, v,
    )

def generate_indicateurs_section(i, c, v, obj, ver, an, soc):
    return generate_section_with_rag(
        "Indicateurs de R&D",
        _build(_tmpl("indicateurs"), objectif=obj, verrou=ver, annee=an, societe=soc),
        i, c, v,
    )

def generate_objectifs_section(i, c, v, obj, ver, an, soc, articles: List[dict] = []):
    return generate_section_with_rag(
        "Objet de l’opération de R&D",
        prompt_objectifs_filtre(obj, ver, an, soc, articles),
        i, c, v,
    )

def generate_travaux_section(i, c, v, obj, ver, an, soc):
    return generate_section_with_rag(
        "Description de la démarche suivie et des travaux réalisés",
        _build(_tmpl("travaux"), objectif=obj, verrou=ver, annee=an, societe=soc),
        i, c, v,
    )

def generate_contribution_section(i, c, v, obj, ver, an, soc):
    return generate_section_with_rag(
        "Contribution scientifique, technique ou technologique",
        _build(_tmpl("contribution"), objectif=obj, verrou=ver, annee=an, societe=soc),
        i, c, v,
    )

def generate_partenariat_section(i, c, v, obj, ver, an, soc):
    return generate_section_with_rag(
        "Partenariat scientifique et recherche confiée",
        _build(_tmpl("partenariat"), objectif=obj, verrou=ver, annee=an, societe=soc),
        i, c, v,
    )

def generate_entreprise_section(i, c, v, obj, ver, an, soc, style=None,site_web: str = ""):
    return generate_section_with_rag(
        "L’entreprise",
        _build(_tmpl("entreprise"), objectif=obj, verrou=ver, annee=an, societe=soc, style=(style or ""), site_web=site_web or "(site non renseigné)"),
        i, c, v,
    )

def generate_gestion_recherche_section(i, c, v, obj, ver, an, soc):
    return generate_section_with_rag(
        "Gestion de la recherche",
        _build(_tmpl("gestion"), objectif=obj, verrou=ver, annee=an, societe=soc),
        i, c, v,
    )

def generate_resume_section(i, c, v, obj, ver, an, soc):
    return generate_section_with_rag(
        "Résumé scientifique de l’opération",
        _build(_tmpl("resume"), objectif=obj, verrou=ver, annee=an, societe=soc),
        i, c, v,
    )

def generate_verrou_section(i, c, v, obj, ver, an, soc):
    # Section "Verrou technique" (assure-toi d’avoir 'verrou.txt' dans le conteneur)
    return generate_section_with_rag(
        "Verrou technique",
        _build(_tmpl("verrou"), objectif=obj, objet=obj, verrou=ver, annee=an, societe=soc),
        i, c, v,
    )

# -------------------- Bibliographie & RH --------------------
import re

def _parse_authors(authors_raw: str):
    """
    Transforme 'M Arslan, H Ghanem, S Munawar' en liste [(NOM, Prénoms), ...].
    Heuristique suffisante avec les auteurs Serper.
    """
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

    # CAS 1 : article manuel "ISO 690 déjà prêt"
    # - pas d'auteurs ni de journal,
    # - mais un "titre" non vide (on l'interprète comme référence complète).
    if titre and not auteurs and not journal:
        return titre  # on renvoie la chaîne telle quelle

    # CAS 2 : article Serper / structuré -> on formate
    auteurs_iso = _format_authors_iso(auteurs)
    year = a.get("year") or "s.d."
    return f"{auteurs_iso}. {titre or 'Sans titre'}. {journal or 'Revue inconnue'}, {year}."


def generate_biblio_section(articles: List[dict] = []) -> str:
    if not articles:
        return "Aucune référence sélectionnée."

    uniq: List[dict] = []
    seen_keys = set()

    for a in articles:
        title = (a.get("title") or "").strip().lower()
        authors = (a.get("authors") or "").strip().lower()
        journal = (a.get("journal") or "").strip().lower()

        # articles manuels ISO : pas d'auteurs / journal -> pas de dédoublonnage
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
Tu reçois le contenu brut de CVs. Pour chaque personne, rends un JSON:

[{{"nom_prenom":"le nom en majuscule et le prenom","diplome":"son diplome le plus elevé","fonction":"son fonction dans l'operation ","contribution":"[À compléter par le client]","temps":"[À compléter par le client]"}}...]
CVs:
\"\"\"{ctx}\"\"\"
"""
    txt = call_ai(prompt, meta="Ressources humaines")
    try:
        return json.loads(txt)
    except Exception:
        return [
            {
                "nom_prenom": "Erreur parsing",
                "diplome": "",
                "fonction": "",
                "contribution": txt,
                "temps": "",
            }
        ]


def evaluateur_travaux(texte: str, *, type_dossier: str = "CIR") -> str:
    """
    Insère des questions encadrées [[ROUGE: ...]] dans la section Travaux,
    sans altérer le texte d’origine (sauf insertion des questions).

    Le prompt est chargé depuis le Blob (evaluateur_travaux.txt dans PROMPTS_CONTAINER_OTHERS).
    """
    # Libellé de la section en fonction du type de dossier
    libelle = (
        "Description détaillée de la démarche scientifique suivie et des travaux réalisés"
        if type_dossier == "CIR"
        else "Description détaillée de la démarche expérimentale suivie et des travaux réalisés"
    )

    # Récupération et formatage du template
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


# Détection des questions de type "Pouvez-vous / Pourriez-vous ... ?"
QUESTION_RE = re.compile(
    r"(?is)\b(?:Pouvez|Pourriez)[\-\u2011\u2013 ]vous[^?]*\?"
)

def wrap_questions_rouge(texte: str) -> str:
    """
    Encadre toutes les questions de type 'Pouvez-vous / Pourriez-vous ... ?'
    avec [[ROUGE: ...]] si le texte ne contient pas déjà [[ROUGE:.
    Sert de sécurité pour que les questions soient bien rouges même si le LLM
    oublie le bon format.
    """
    if not texte:
        return texte

    # Si le LLM a déjà utilisé [[ROUGE: ...]], on ne touche à rien
    if "[[ROUGE:" in texte:
        return texte

    pieces = []
    pos = 0
    for m in QUESTION_RE.finditer(texte):
        start, end = m.span()
        if start > pos:
            pieces.append(texte[pos:start])
        q = m.group(0)
        # On encadre la question telle quelle
        pieces.append(f"[[ROUGE: {q} ]]")
        pos = end

    pieces.append(texte[pos:])
    return "".join(pieces)



def enrich_manual_articles_iso(
    articles: List[dict],
    articles_texts: List[str],
) -> List[dict]:
    """
    Pour les articles ajoutés manuellement (ceux venant du front avec citations=0
    et sans auteurs/journal/année), utilise l’IA sur le texte PDF correspondant
    pour générer une référence ISO 690.

    On suppose que l’ordre des articles_texts correspond à l’ordre des articles
    manuels envoyés par le front.
    """
    if not articles:
        return []

    # On clone pour ne pas modifier l'entrée
    out: List[dict] = [dict(a) for a in articles]

    # Heuristique pour détecter un article "manuel" :
    # - pas d'auteurs
    # - citations == 0
    # - url éventuellement présente
    manual_idxs = [i for i, a in enumerate(out) if not (a.get("authors") or "").strip() and int(a.get("citations") or 0) == 0]

    # On parcourt les articles manuels et on mappe sur articles_texts par ordre
    txt_idx = 0
    for i in manual_idxs:
        if txt_idx >= len(articles_texts):
            break
        txt = (articles_texts[txt_idx] or "").strip()
        txt_idx += 1
        if not txt:
            continue

        # On limite le texte pour l'IA (début de l'article suffit)
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

        try:
            raw = call_ai(prompt, meta="ISO690_manual_article")
            raw = (raw or "").strip()
            # on sécurise: on cherche la plus grosse portion {...}
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

        except Exception as e:
            print(f"[ISO690] Erreur enrichissement article manuel: {e}")

    return out
