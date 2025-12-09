# app/services/figures_planner.py

import io
import json
import os
import re
import tempfile
from typing import List, Dict, Tuple, Optional

from PIL import Image

from Core import description_img as desc_img  # Core/description_img.py
from Core.images_figures import build_docx_from_images
from Core import rag
from app.services.prompts import prompt_figures_plan


def _parse_captions_raw(captions_raw: str) -> Dict[int, str]:
    """
    Transforme un texte du type:
      "Image 1: ...\nImage 2: ...\n..."
    en dict {1: "…", 2: "…"}.
    """
    out: Dict[int, str] = {}
    if not captions_raw:
        return out

    for line in captions_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^Image\s+(\d+)\s*[:\-–]\s*(.+)$", line, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        caption = m.group(2).strip()
        if caption:
            out[idx] = caption
    return out


def _plan_figures_with_llm(
    dossier_type: str,
    sections_payload: Dict[str, str],
    images_meta: List[Dict],
    max_figures: int = 5,
) -> Optional[Dict]:
    """
    Appelle l'IA pour :
      - choisir max_figures images,
      - insérer des mentions "Figure k" dans les sections,
      - renvoyer un JSON { sections: {...}, mapping: [...] }.

    Le prompt est chargé depuis le Blob (figures_plan.txt dans PROMPTS_CONTAINER_OTHERS).
    """
    dt = dossier_type.upper()

    # Préparation des JSON passés au prompt
    sections_json = json.dumps(sections_payload, ensure_ascii=False)
    images_json = json.dumps(
        [
            {
                "id": m["id"],
                "caption": m.get("caption", ""),
                "width": m.get("width", 0),
                "height": m.get("height", 0),
            }
            for m in images_meta
        ],
        ensure_ascii=False,
    )

    # Chargement du template de prompt depuis le Blob
    try:
        tpl = prompt_figures_plan()
    except RuntimeError as e:
        print(f"[figures_planner] {e}")
        return None

    # Formatage du prompt avec les placeholders
    try:
        prompt = tpl.format(
            dossier_type=dt,
            sections_json=sections_json,
            images_json=images_json,
            max_figures=max_figures,
        )
    except Exception as e:
        print(f"[figures_planner] Erreur formatage prompt figures_plan: {e}")
        return None

    raw = rag.call_ai(prompt, meta=f"Plan figures {dt}")
    if not raw:
        return None

    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        print("[figures_planner] JSON non détecté dans la réponse IA.")
        return None

    json_str = raw[start : end + 1]
    try:
        data = json.loads(json_str)
    except Exception as e:
        print(f"[figures_planner] Erreur parse JSON plan figures: {e}")
        return None

    if not isinstance(data, dict):
        return None

    return data


def _semantic_filter_images(images_meta: List[Dict], min_keep: int = 3) -> List[Dict]:
    """
    Filtre les images candidates à partir de leur caption.

    On garde en priorité celles qui semblent montrer :
      - schéma, diagramme, architecture,
      - interface, écran, tableau de bord,
      - produit, dispositif, prototype.

    On exclut les images qui semblent être :
      - photos de personnes, réunions, conférences,
      - photos de groupe, portraits,
      - paysages, skyline, bâtiments, chantiers.

    Si aucune image ne passe le filtre, on renvoie une liste vide
    (=> pas de figures insérées).
    """
    positive_keywords = [
        "schéma", "schema", "diagramme", "diagram",
        "architecture", "architectur",
        "flux", "pipeline", "processus", "workflow",
        "interface", "écran", "ecran", "dashboard", "tableau de bord",
        "graphique", "courbe", "chart",
        "prototype", "produit", "dispositif", "capteur", "module",
        "architecture logicielle", "architecture système",
        "schéma fonctionnel", "schéma technique",
        "organigramme", "bloc diagramme",
    ]

    negative_keywords = [
        "personne", "personnes", "gens", "public", "auditoire",
        "groupe", "équipe", "team",
        "salle", "conférence", "conference", "présentation", "presentation",
        "réunion", "meeting",
        "portrait", "selfie",
        "bâtiment", "immeuble", "building", "chantier",
        "paysage", "skyline", "ville", "city",
        "photo", "photographie",
    ]

    # On tient aussi compte des étiquettes éventuelles mises en début de caption
    # par le prompt vision (SCHÉMA, INTERFACE, PRODUIT, PHOTO CONTEXTE, etc.)
    def score_image(caption: str) -> int:
        txt = (caption or "").lower()
        if not txt:
            return 0

        # Tag explicite utile (schéma, interface, produit)
        tag_bonus = 0
        if "schéma" in txt or "schema" in txt or "diagramme" in txt:
            tag_bonus += 3
        if "interface" in txt or "écran" in txt or "ecran" in txt or "dashboard" in txt:
            tag_bonus += 2
        if "produit" in txt or "prototype" in txt or "dispositif" in txt or "capteur" in txt:
            tag_bonus += 2

        pos = sum(1 for k in positive_keywords if k in txt)
        neg = any(k in txt for k in negative_keywords)

        if neg:
            return 0

        return pos + tag_bonus

    scored: List[Tuple[int, Dict]] = []
    for m in images_meta:
        cap = m.get("caption", "") or ""
        s = score_image(cap)
        if s > 0:
            scored.append((s, m))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for (s, m) in scored]

    # Aucun schéma / interface / produit détecté -> pas de figures
    print(
        "[figures_planner] Aucun schéma/diagramme/interface/produit détecté dans les captions, "
        "aucune figure insérée."
    )
    return []


def _prepare_figures_for_dossier(
    dossier_type: str,
    docs_client_data: List[Dict[str, bytes]],
    sections: Dict[str, str],
    section_keys_for_figures: List[str],
    max_figures: int = 5,
    min_side: int = 400,
    max_images_candidates: int = 40,
) -> Tuple[Dict[str, str], Optional[str], Optional[str]]:
    """
    Pipeline générique CIR/CII :
    - écrit les docs techniques sur disque (temp),
    - extrait toutes les images (PDF/DOCX/PNG/JPG),
    - filtre les petites images, les réencode en PNG propre,
    - garde les max_images_candidates plus grandes,
    - appelle Azure Vision pour décrire ces images,
    - appelle l'IA pour planifier les figures,
    - si le plan IA est invalide -> on ne met PAS de figures.
    """
    if not docs_client_data:
        return sections, None, None

    if Image is None:
        print("[figures_planner] PIL non disponible, pas de figures.")
        return sections, None, None

    # 1) Écriture des docs clients + extraction brute des images
    with tempfile.TemporaryDirectory() as tmpdir:
        paths_all: List[str] = []
        for i, doc in enumerate(docs_client_data):
            filename = doc.get("filename") or f"doc_{i}"
            ext = os.path.splitext(filename)[1] or ".bin"
            out_path = os.path.join(tmpdir, f"src_{i}{ext}")
            with open(out_path, "wb") as fh:
                fh.write(doc["data"])
            paths_all.append(out_path)

        # 2) Extraction brute de toutes les images
        images_raw = desc_img._collect_images_from_paths(paths_all)

        # 3) Filtrage + réencodage PNG + meta (images "valables")
        candidates: List[Dict] = []
        for idx, (data, ext) in enumerate(images_raw, start=1):
            if not data:
                continue
            try:
                im = Image.open(io.BytesIO(data))
                im.load()
            except Exception:
                continue

            w, h = im.size
            if max(w, h) < min_side:
                continue

            area = w * h
            out = io.BytesIO()
            im.save(out, format="PNG")
            png_bytes = out.getvalue()

            candidates.append(
                {
                    "id": idx,          # id "global" (ordre brut)
                    "bytes": png_bytes, # PNG propre
                    "ext": "png",
                    "width": w,
                    "height": h,
                    "area": area,
                }
            )

        if not candidates:
            print("[figures_planner] Aucune image candidate retenue après filtrage dimensionnel.")
            return sections, None, None

        # 4) Tri par taille (aire) et limitation à max_images_candidates
        candidates.sort(key=lambda x: x["area"], reverse=True)
        selected = candidates[:max_images_candidates]

        # 5) Préparation de fichiers PNG temporaires pour Vision
        selected_paths: List[str] = []
        for i, cand in enumerate(selected):
            img_path = os.path.join(tmpdir, f"img_sel_{i+1}.png")
            with open(img_path, "wb") as fh:
                fh.write(cand["bytes"])
            selected_paths.append(img_path)

        # 6) Descriptions des images via description_img.summarise_with_azure
        try:
            captions_raw = desc_img.summarise_with_azure(
                selected_paths,
                consigne=None,
                model=None,
                token_limit_per_call=30000,
                prompt_overhead_tokens=1000,
                max_output_tokens=500,
                temperature=0.0,
            )
        except Exception as e:
            print(f"[figures_planner] Erreur description_img.summarise_with_azure: {e}")
            captions_raw = ""

        captions_by_idx = _parse_captions_raw(captions_raw)

        # 7) Construction images_meta pour le LLM (id = 1..N sur la sélection)
        images_meta: List[Dict] = []
        id_map: Dict[int, Dict] = {}
        for new_id, cand in enumerate(selected, start=1):
            meta = {
                "id": new_id,
                "bytes": cand["bytes"],
                "ext": "png",
                "width": cand["width"],
                "height": cand["height"],
                "area": cand["area"],
                "caption": captions_by_idx.get(new_id, ""),
            }
            images_meta.append(meta)
            id_map[new_id] = meta

    if not images_meta:
        print("[figures_planner] Aucune image meta.")
        return sections, None, None

    # 8) Filtre sémantique sur les images (à partir des captions)
    images_meta = _semantic_filter_images(images_meta, min_keep=min(3, max_figures))
    # et on coupe à max_figures pour le LLM
    images_meta = images_meta[:max_figures]

    if not images_meta:
        print("[figures_planner] Aucune image retenue après filtrage sémantique.")
        return sections, None, None

    # 9) Préparation des sections à envoyer au LLM
    sections_payload: Dict[str, str] = {}
    for key in section_keys_for_figures:
        sections_payload[key] = sections.get(key, "") or ""

    # 10) Planification des figures par l'IA
    plan = _plan_figures_with_llm(
        dossier_type,
        sections_payload,
        images_meta,
        max_figures=max_figures,
    )

    # Si le plan IA est vide ou invalide -> on ne met PAS d'images
    if not plan or not isinstance(plan, dict) or not plan.get("mapping"):
        print("[figures_planner] Plan de figures vide ou invalide -> AUCUNE figure insérée.")
        return sections, None, None

    print("[figures_planner] Plan de figures IA OK.")
    new_sections = dict(sections)

    sec_plan = plan.get("sections") or {}
    for key in section_keys_for_figures:
        v = sec_plan.get(key)
        if isinstance(v, str) and v.strip():
            new_sections[key] = v

    # 11) Construction de la liste d'images dans l'ordre des figures
    mapping = plan["mapping"]
    ordered_images: List[Tuple[bytes, str]] = []
    legends: List[str] = []

    try:
        mapping_sorted = sorted(
            mapping, key=lambda x: int(x.get("figure_number", 0))
        )
    except Exception:
        mapping_sorted = mapping

    # index par id local (1..n sur la sélection filtrée)
    id_map = {m["id"]: m for m in images_meta}

    for item in mapping_sorted:
        try:
            img_id = int(item.get("image_id"))
            fig_num = int(item.get("figure_number"))
        except Exception:
            continue
        if fig_num < 1:
            continue
        meta = id_map.get(img_id)
        if not meta:
            continue
        ordered_images.append((meta["bytes"], meta["ext"]))

        legend = (item.get("legend") or "").strip()
        if not legend:
            legend = (meta.get("caption") or "").strip()
        legends.append(legend)

    if not ordered_images:
        print("[figures_planner] Aucune image retenue après mapping IA.")
        return new_sections, None, None

    # 12) captions_text utilisable par insert_images_by_reference_live
    lines = []
    for i, leg in enumerate(legends, start=1):
        if leg:
            lines.append(f"Figure {i}: {leg}")
        else:
            lines.append(f"Figure {i}:")
    captions_text = "\n".join(lines)

    # 13) Construction du DOCX source pour les figures
    src_docx_path = build_docx_from_images(ordered_images)

    return new_sections, src_docx_path, captions_text


def prepare_figures_for_cir(
    docs_client_data: List[Dict[str, bytes]],
    sections_cir: Dict[str, str],
    max_figures: int = 5,
    min_side: int = 400,
) -> Tuple[Dict[str, str], Optional[str], Optional[str]]:
    """
    Spécifique CIR : on travaille sur les sections
    - contexte
    - travaux
    - contribution
    """
    section_keys = ["contexte", "travaux", "contribution"]
    return _prepare_figures_for_dossier(
        "CIR",
        docs_client_data,
        sections_cir,
        section_keys,
        max_figures=max_figures,
        min_side=min_side,
    )


def prepare_figures_for_cii(
    docs_client_data: List[Dict[str, bytes]],
    sections_cii: Dict[str, str],
    max_figures: int = 5,
    min_side: int = 400,
) -> Tuple[Dict[str, str], Optional[str], Optional[str]]:
    """
    Spécifique CII : on travaille sur les sections
    - presentation
    - resume
    - contexte
    - analyse
    - performances
    - demarche
    - resultats
    """
    section_keys = [
        "presentation",
        "resume",
        "contexte",
        "analyse",
        "performances",
        "demarche",
        "resultats",
    ]
    return _prepare_figures_for_dossier(
        "CII",
        docs_client_data,
        sections_cii,
        section_keys,
        max_figures=max_figures,
        min_side=min_side,
    )
