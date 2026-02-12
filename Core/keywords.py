# Core/keywords.py
import os
import re
from typing import List, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
GPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


def extract_keywords_with_rag(
    index,
    chunks: List[str],
    vectors,
    max_keywords: int = 10
) -> List[str]:
    """
    Extrait des mots-clés en utilisant le RAG pour trouver les passages pertinents.
    Moins coûteux car on n'envoie que les passages pertinents au LLM.
    """
    from Core.rag import search_similar_chunks

    # Requêtes de recherche pour trouver les passages les plus pertinents
    search_queries = [
        "objectif du projet innovation technologie",
        "méthode technique procédé développement",
        "problème scientifique verrou technique",
        "solution algorithme système architecture",
    ]

    # Collecter les passages pertinents (dédupliqués)
    relevant_passages = []
    seen_passages = set()

    for query in search_queries:
        passages = search_similar_chunks(query, index, chunks, vectors, top_k=3)
        for p in passages:
            # Dédupliquer par hash du contenu
            p_hash = hash(p[:200])  # Hash sur les premiers 200 caractères
            if p_hash not in seen_passages:
                seen_passages.add(p_hash)
                relevant_passages.append(p)

    # Limiter à ~8000 caractères pour le contexte
    context = ""
    for p in relevant_passages:
        if len(context) + len(p) > 8000:
            break
        context += "\n\n" + p

    if not context.strip():
        print("[KEYWORDS] Aucun passage pertinent trouvé via RAG")
        return []

    print(f"[KEYWORDS] RAG: {len(relevant_passages)} passages, {len(context)} caractères")

    return _call_llm_for_keywords(context, max_keywords)


def extract_keywords(text: str, max_keywords: int = 8, with_synonyms: bool = True) -> List[str]:
    """
    Extrait des mots-clés du texte complet (fallback si pas de RAG).
    Utilise tout le texte en le découpant intelligemment.
    """
    # Découper le texte en parties pour couvrir tout le document
    text_length = len(text)

    if text_length <= 6000:
        # Petit document : envoyer tout
        context = text
    else:
        # Grand document : prendre début + milieu + fin
        part_size = 2000
        beginning = text[:part_size]
        middle_start = (text_length // 2) - (part_size // 2)
        middle = text[middle_start:middle_start + part_size]
        end = text[-part_size:]
        context = f"{beginning}\n\n[...]\n\n{middle}\n\n[...]\n\n{end}"

    print(f"[KEYWORDS] Fallback sans RAG: {len(context)} caractères")
    return _call_llm_for_keywords(context, max_keywords)


def _call_llm_for_keywords(context: str, max_keywords: int) -> List[str]:
    """
    Appelle le LLM pour extraire les mots-clés du contexte donné.
    """
    prompt = f"""Tu es un consultant expert en Crédit d'Impôt Recherche (CIR) qui doit trouver des articles scientifiques sur Google Scholar pour justifier le caractère innovant d'un projet R&D.

Analyse les extraits du document technique ci-dessous et extrais {max_keywords} expressions de recherche que tu taperais dans Google Scholar pour trouver des articles scientifiques pertinents.

RÈGLES IMPORTANTES :
- Extrais des termes TECHNIQUES SPÉCIFIQUES au projet (technologies, méthodes, matériaux, procédés)
- Évite les termes génériques comme "innovation", "R&D", "CIR", "recherche"
- Utilise des expressions de 2-4 mots qui donneront des résultats précis
- Pense comme un chercheur : quels mots-clés scientifiques décrivent ce projet ?

Exemples de BONS mots-clés :
- "machine learning predictive maintenance"
- "CRISPR gene editing"
- "lithium-ion battery optimization"
- "computer vision defect detection"

Exemples de MAUVAIS mots-clés (trop génériques) :
- "innovation technologique"
- "projet de recherche"
- "amélioration des performances"

Extraits du document technique :
\"\"\"{context}\"\"\"

Donne exactement {max_keywords} expressions de recherche, une par ligne, sans numérotation ni tirets."""

    resp = client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Tu es un expert CIR qui aide à trouver des articles scientifiques sur Google Scholar. Tu réponds uniquement avec des mots-clés de recherche, un par ligne."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=600
    )
    raw = (resp.choices[0].message.content or "").strip()
    return _parse_keywords(raw, max_keywords)


def _parse_keywords(raw: str, max_keywords: int) -> List[str]:
    """
    Parse la réponse du LLM pour extraire les mots-clés.
    """
    kws = []
    seen = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Nettoyer : enlever numérotation, tirets, puces, guillemets
        line = re.sub(r'^[\d\.\)\-\*•]+\s*', '', line)  # "1. " ou "- " ou "• "
        line = line.strip('"\'')  # guillemets
        line = line.strip()

        if not line:
            continue

        # Garder uniquement si c'est une expression valide
        if line.lower() not in seen and len(line) > 3:
            seen.add(line.lower())
            kws.append(line)

    return kws[:max_keywords]
