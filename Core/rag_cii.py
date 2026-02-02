# Core/rag_cii.py
from typing import List, Dict, Any
import json
import re
from app.services.prompts import fetch_cii, prompt_evaluateur_travaux
from Core.rag import generate_section_with_rag, _build, call_ai


def _tmpl_cii(key: str) -> str:
    """
    Charge le prompt CII correspondant depuis le conteneur Blob (PROMPTS_CONTAINER_CII).
    Adapte les noms de fichiers si tu en utilises d'autres dans le blob.
    """
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
def normalize_newlines(s: str) -> str:
    """Normalise les fins de ligne CRLF/CR en LF."""
    if not s:
        return ""
    return s.replace("\r\n", "\n").replace("\r", "\n")


def parse_blocks(md_text: str):
    """
    Découpe un texte Markdown light en blocs ('p', 'ul', 'ol').

    - Puces : lignes débutant par '-', '/', ou '*', avec éventuelle indentation.
      Indentation (≥1 espace ou tab) => niveau 1, sinon niveau 0.
    - Listes numérotées : '1. ' ou '1)'.
    - Les lignes vides séparent les blocs.
    - Les paragraphes consécutifs (non-listes) sont fusionnés par des sauts de ligne.
    """
    text = normalize_newlines(md_text or "").strip()
    if not text:
        return []

    lines = text.split("\n")

    blocks = []
    buf_para = []   # accumulateur de lignes de paragraphe
    buf_list = None # {"type": "ul"/"ol", "items": [...]}

    def flush_para():
        nonlocal buf_para
        if buf_para:
            p = "\n".join(buf_para).strip()
            if p:
                blocks.append({"type": "p", "text": p})
        buf_para = []

    def flush_list():
        nonlocal buf_list
        if buf_list and buf_list.get("items"):
            blocks.append(buf_list)
        buf_list = None

    bullet_re = re.compile(r'^(?P<indent>[ \t]*)(?P<marker>[-/*])\s+(?P<txt>.+)$')
    order_re  = re.compile(r'^(?P<indent>[ \t]*)(?P<num>\d+)[\.)]\s+(?P<txt>.+)$')

    for raw in lines:
        line = raw.rstrip()

        # Ligne vide => séparation de blocs
        if not line.strip():
            flush_list()
            flush_para()
            continue

        m_bul = bullet_re.match(line)
        m_ord = order_re.match(line)

        if m_bul or m_ord:
            # On bascule en mode liste; on termine le paragraphe courant
            flush_para()

            if m_bul:
                linetype  = "ul"
                indent    = m_bul.group("indent") or ""
                item_text = (m_bul.group("txt") or "").strip()
                marker    = m_bul.group("marker")
            else:
                linetype  = "ol"
                indent    = m_ord.group("indent") or ""
                item_text = (m_ord.group("txt") or "").strip()
                marker    = None

            indent_norm = indent.replace("\t", " ")
            level = 1 if len(indent_norm) >= 1 else 0

            if buf_list is None or buf_list["type"] != linetype:
                flush_list()
                buf_list = {"type": linetype, "items": []}

            if linetype == "ul":
                buf_list["items"].append((level, item_text, marker))
            else:
                buf_list["items"].append((level, item_text))

        else:
            # Ligne de paragraphe
            flush_list()
            buf_para.append(line)

    flush_list()
    flush_para()

    return blocks


def _render_blocks_for_docx(blocks) -> str:
    """
    Recompose les blocs en texte :
    - 1 ligne par item de liste, sans lignes vides superflues,
    - 1 seule ligne vide entre blocs (p vs liste).
    """
    if not blocks:
        return ""

    out_lines = []
    first_block = True

    for block in blocks:
        if block["type"] == "p":
            # Séparation avec le bloc précédent
            if not first_block:
                out_lines.append("")
            # Le paragraphe peut déjà contenir des '\n' internes
            for l in block["text"].split("\n"):
                out_lines.append(l.strip())

        elif block["type"] == "ul":
            if not first_block:
                out_lines.append("")
            for level, txt, marker in block["items"]:
                prefix = ("  " * level) + f"{marker} "
                out_lines.append(prefix + txt)

        elif block["type"] == "ol":
            if not first_block:
                out_lines.append("")
            num = 1
            for level, txt in block["items"]:
                prefix = ("  " * level) + f"{num}. "
                out_lines.append(prefix + txt)
                num += 1

        first_block = False

    # On évite les espaces inutiles
    text = "\n".join(out_lines).rstrip()
    return text


def _normalize_md_lists(text: str) -> str:
    """
    Pipeline complet :
    - normalise les retours à la ligne,
    - découpe en blocs (p / ul / ol),
    - recombine en texte sans sauts de lignes superflus.
    """
    if not text:
        return ""
    blocks = parse_blocks(text)
    return _render_blocks_for_docx(blocks)


# --- Fonctions de génération (une par section) -------------------------


def gen_presentation(
    i,
    c,
    v,
    *,
    societe,
    annee,
    referent_nom,
    referent_titre,
    referent_tel,
    referent_email,
    site_web: str = "",
    contexte_societe: str = "",
    secteur: str = "",
    performance_type: str = "",
) -> str:
    instr = _build(
        _tmpl_cii("presentation"),
        societe=societe,
        annee=annee,
        referent_nom=referent_nom,
        referent_titre=referent_titre,
        referent_tel=referent_tel,
        referent_email=referent_email,
        site_web=site_web or "(site non renseigné)",
        contexte_societe=contexte_societe,
        secteur=secteur,
        performance_type=performance_type,
    )
    raw = generate_section_with_rag(
        "Présentation globale & stratégie d'innovation (CII)",
        instr,
        i, c, v,
    )
    return _normalize_md_lists(raw)




def gen_resume(
    i,
    c,
    v,
    *,
    projet,
    annee,
    period,
    performance_type: str = "",
) -> str:
    instr = _build(
        _tmpl_cii("resume"),
        projet=projet,
        annees_couvertes=str(annee),
        period=period,
        performance_type=performance_type,
    )
    raw = generate_section_with_rag(
        "Résumé scientifique de l'innovation",
        instr,
        i, c, v,
    )
    return _normalize_md_lists(raw)

def gen_resume_from_sections(
    sections: dict,
    *,
    projet: str,
    annee: int,
    period: str,
    performance_type: str = "",
) -> str:
    """
    Génère un résumé basé sur les sections déjà générées du document CII.
    Garantit que le résumé reflète exactement le contenu du document final.
    """
    from app.services.prompts import prompt_cii_resume

    # Construire le contenu des sections pour le prompt
    sections_content = ""

    if sections.get("presentation"):
        sections_content += f"\n\n=== PRÉSENTATION DE L'ENTREPRISE ===\n{sections['presentation']}"

    if sections.get("contexte"):
        sections_content += f"\n\n=== CONTEXTE DU PROJET ===\n{sections['contexte']}"

    if sections.get("analyse"):
        sections_content += f"\n\n=== ANALYSE CONCURRENTIELLE ===\n{sections['analyse']}"

    if sections.get("performances"):
        sections_content += f"\n\n=== PERFORMANCES VISÉES ===\n{sections['performances']}"

    if sections.get("demarche"):
        sections_content += f"\n\n=== DÉMARCHE ET TRAVAUX ===\n{sections['demarche']}"

    if sections.get("resultats"):
        sections_content += f"\n\n=== RÉSULTATS OBTENUS ===\n{sections['resultats']}"

    if sections.get("rh_intro"):
        sections_content += f"\n\n=== RESSOURCES HUMAINES ===\n{sections['rh_intro']}"

    # Récupérer le template du prompt depuis le blob
    prompt_template = prompt_cii_resume()

    # Remplacer les variables dans le prompt
    prompt = prompt_template.format(
        sections_content=sections_content,
        projet=projet,
        annee=annee,
        period=period,
        performance_type=performance_type
    )

    raw = call_ai(prompt, meta="resume_cii_from_sections", temperature=0.3)
    return _normalize_md_lists(raw)


def gen_contexte(
    i,
    c,
    v,
    *,
    annee,
    visee_generale: str = "",
    performance_type: str = "",
) -> str:
    instr = _build(
        _tmpl_cii("contexte"),
        annee=annee,
        visee_generale=visee_generale,
        performance_type=performance_type,
    )
    raw= generate_section_with_rag(
        "Description du contexte général",
        instr,
        i, c, v,
    )
    return _normalize_md_lists(raw)


def gen_analyse(
    i,
    c,
    v,
    *,
    concurrents,
    axes_cibles,
    performance_type: str = "",
    liste_concurrents: str = "",
) -> str:
    """
    Génère la section "État du marché & justification du caractère innovant".

    Paramètres :
    - concurrents : liste de dicts {name, website, axes, weakness, client_advantage}
    - axes_cibles : liste de libellés d'axes (fonctionnels / techniques / ...)
    - liste_concurrents : version texte lisible de la liste des concurrents (pour le prompt)
    """
    concurrents = concurrents or []
    axes_cibles = axes_cibles or []

    concurrents_json = json.dumps(concurrents, ensure_ascii=False)

    # Si la version texte n'a pas été fournie, on la reconstruit rapidement
    if not liste_concurrents:
        lines = []
        for cpt in concurrents:
            name = (cpt.get("name") or "").strip()
            if not name:
                continue
            website = (cpt.get("website") or "").strip()
            axes_str = ", ".join(cpt.get("axes") or [])
            line = f"- {name}"
            if website:
                line += f" ({website})"
            if axes_str:
                line += f" — axes visés: {axes_str}"
            lines.append(line)
        liste_concurrents = "\n".join(lines) if lines else "- Aucun concurrent renseigné."

    instr = _build(
        _tmpl_cii("analyse"),
        concurrents_selectionnes=concurrents_json,       # JSON complet
        liste_concurrents=liste_concurrents,             # chaîne lisible
        axes_cibles=", ".join(axes_cibles),
        performance_type=performance_type,
    )

    raw=generate_section_with_rag(
        "État du marché & justification du caractère innovant",
        instr,
        i, c, v,
    )
    return _normalize_md_lists(raw)


def gen_performances(
    i,
    c,
    v,
    *,
    projet,
    performance_type: str = "",
) -> str:
    instr = _build(
        _tmpl_cii("performances"),
        projet=projet,
        performance_type=performance_type,
    )
    raw= generate_section_with_rag(
        "Nouvelles performances visées",
        instr,
        i, c, v,
    )
    return _normalize_md_lists(raw)


def gen_demarche_annee(
    i,
    c,
    v,
    *,
    annee,
    performance_type: str = "",
) -> str:
    instr = _build(
        _tmpl_cii("demarche"),
        annee=annee,
        performance_type=performance_type,
    )
    raw=generate_section_with_rag(
        f"Démarche expérimentale {annee}",
        instr,
        i, c, v,
    )
    return _normalize_md_lists(raw)


def gen_resultats_annee(
    i,
    c,
    v,
    *,
    annee,
    performance_type: str = "",
) -> str:
    instr = _build(
        _tmpl_cii("resultats"),
        annee=annee,
        performance_type=performance_type,
    )
    raw= generate_section_with_rag(
        f"Résultats & supériorité {annee}",
        instr,
        i, c, v,
    )
    return _normalize_md_lists(raw)


def gen_rh_intro(
    i,
    c,
    v,
    *,
    annee,
    total_heures: str = "",
    performance_type: str = "",
) -> dict:
    instr = _build(
        _tmpl_cii("rh_intro"),
        annee=annee,
        total_heures=total_heures,
        performance_type=performance_type,
    )

    # Améliorer le prompt pour forcer un JSON valide
    enhanced_instr = instr + """

IMPORTANT: Extrait les informations sur le personnel UNIQUEMENT à partir des documents fournis.
Ne génère PAS de valeurs fictives ou de placeholder comme "[À compléter par le client]".
Si une information n'est PAS présente dans les documents, utilise une chaîne vide "".

Renvoie un JSON valide avec cette structure:
```json
{
  "introduction": "Texte d'introduction sur les ressources humaines...",
  "personnel": [
    {
      "nom_prenom": "NOM Prénom réel trouvé dans les docs",
      "diplome": "Diplôme réel ou vide",
      "fonction": "Fonction réelle ou vide",
      "contribution": "Contribution réelle liée à une activité d'innovation ou vide",
      "temps": "à compléter par le client ou vide"
    }
  ]
}
```

Renvoie UNIQUEMENT le JSON, sans texte avant ni après, sans markdown."""

    raw = generate_section_with_rag(
        "Introduction RH",
        enhanced_instr,
        i, c, v,
    )

    # Parser le JSON et extraire les données
    try:
        import json
        import re

        # Nettoyer et extraire le JSON
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*\n', '', cleaned)
            cleaned = re.sub(r'\n```\s*$', '', cleaned)

        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            json_str = cleaned[start:end]
            data = json.loads(json_str)

            intro = data.get("introduction", "").strip()
            personnel = data.get("personnel", [])

            # Nettoyer les valeurs "[À compléter par le client]"
            for person in personnel:
                for key in list(person.keys()):
                    val = str(person[key])
                    if "[" in val and "compléter" in val.lower():
                        person[key] = ""

            # Retourner la structure de données pour le template Jinja
            return {
                "introduction": intro,
                "personnel": personnel
            }

    except json.JSONDecodeError as e:
        print(f"[RH] Erreur parsing JSON: {e}")
        print(f"[RH] Contenu reçu: {raw[:200]}...")
    except Exception as e:
        print(f"[RH] Erreur inattendue: {e}")

    # Fallback: retourner une structure minimale
    return {
        "introduction": raw,
        "personnel": []
    }


def get_biblio_intro() -> str:
    """
    Intro générique de la partie bibliographie CII (si tu en as une).
    """
    return _tmpl_cii("biblio_intro") or ""
def evaluateur_travaux(texte: str, *, type_dossier: str = "CII") -> str:
    """
    Insère des questions encadrées [[ROUGE: ...]] dans la section Travaux (CII),
    sans altérer le texte d’origine (sauf insertion des questions).

    Le prompt est partagé avec la version CIR et est chargé depuis le Blob
    (evaluateur_travaux.txt dans PROMPTS_CONTAINER_OTHERS).
    """
    libelle = (
        "Description détaillée de la démarche expérimentale suivie et des travaux réalisés"
        if type_dossier == "CII"
        else "Description détaillée de la démarche scientifique suivie et des travaux réalisés"
    )

    tpl = prompt_evaluateur_travaux()
    try:
        prompt = tpl.format(
            type_dossier=type_dossier,
            libelle=libelle,
            texte=texte,
        )
    except Exception as e:
        raise RuntimeError(f"Erreur formatage prompt evaluateur_travaux (CII): {e}")

    return call_ai(prompt, meta="Travaux avec questions")


# Détection des questions de type "Pouvez-vous / Pourriez-vous ... ?"
QUESTION_RE = re.compile(
    r"(?is)\b(?:Pouvez|Pourriez)[\-\u2011\u2013 ]vous[^?]*\?"
)


def generate_tableau_comparatif(
    i, c, v,  # index RAG
    *,
    societe: str,
    projet: str,
    competitors: List[Dict[str, Any]],
    section_travaux: str,
    section_concurrence: str,
) -> Dict[str, Any]:
    """
    Génère les données JSON pour le tableau comparatif CII.
    Analyse les sections travaux et concurrence pour extraire les éléments
    différenciants et évaluer leur présence chez les concurrents.

    Retourne:
    {
        "elements": [
            {
                "nom": "Element 1",
                "client": "Oui",
                "concurrents": {"Concurrent A": "Non", "Concurrent B": "Partiel"}
            },
            ...
        ]
    }
    """
    competitors_names = [comp.get("name", "") for comp in competitors if comp.get("name")]

    if not competitors_names:
        print("[TABLEAU] Aucun concurrent fourni, tableau vide")
        return {"elements": []}

    competitors_list = ", ".join(competitors_names)

    prompt = f"""Tu es un expert en analyse concurrentielle pour dossiers CII (Crédit Impôt Innovation).

CONTEXTE:
- Société cliente: {societe}
- Projet innovant: {projet}
- Concurrents analysés: {competitors_list}

SECTION TRAVAUX DU DOSSIER:
{section_travaux[:8000]}

SECTION ANALYSE CONCURRENTIELLE:
{section_concurrence[:8000]}

TÂCHE:
1. Extrais les 5 à 8 éléments/fonctionnalités clés qui différencient le projet "{projet}" de la société "{societe}"
2. Pour chaque élément, évalue si le client et chaque concurrent le possède

RÈGLES IMPORTANTES:
- "Oui" = La fonctionnalité est présente et performante
- "Non" = La fonctionnalité est absente ou très limitée
- "Partiel" = La fonctionnalité existe mais est incomplète ou moins performante
- Le client ({societe}) doit avoir "Oui" pour TOUS ses éléments (c'est son innovation)
- Base-toi sur les informations des sections fournies pour évaluer les concurrents
- Si tu n'as pas d'information sur un concurrent pour un élément, mets "Non" par défaut

CONCURRENTS À ÉVALUER: {competitors_list}

Réponds UNIQUEMENT avec ce JSON valide, sans texte avant ni après, sans markdown:
{{
  "elements": [
    {{
      "nom": "Nom court de l'élément (max 50 caractères)",
      "client": "Oui",
      "concurrents": {{
        "{competitors_names[0]}": "Oui|Non|Partiel"{"".join([f',"{name}": "Oui|Non|Partiel"' for name in competitors_names[1:]])}
      }}
    }}
  ]
}}"""

    raw = generate_section_with_rag("Tableau comparatif", prompt, i, c, v)

    # Parser le JSON
    try:
        cleaned = raw.strip()
        # Supprimer les balises markdown si présentes
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)

        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            json_str = cleaned[start:end]
            data = json.loads(json_str)

            # Validation basique
            if "elements" in data and isinstance(data["elements"], list):
                print(f"[TABLEAU] {len(data['elements'])} éléments extraits")
                return data

    except json.JSONDecodeError as e:
        print(f"[TABLEAU] Erreur parsing JSON: {e}")
        print(f"[TABLEAU] Contenu reçu: {raw[:300]}...")
    except Exception as e:
        print(f"[TABLEAU] Erreur inattendue: {e}")

    # Fallback: structure vide
    return {"elements": []}