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
    max_figures: int = 10,
) -> Optional[Dict]:
    dt = dossier_type.upper()

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

    try:
        tpl = prompt_figures_plan()
    except RuntimeError as e:
        print(f"[figures_planner] {e}")
        return None

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

    # JSON strict
    raw = rag.call_ai(prompt, meta=f"Plan figures {dt}")

    raw = raw or ""
    raw_stripped = raw.strip()

    print("\n========== FIGURES PLANNER — RÉPONSE LLM ==========")
    print(f"len={len(raw_stripped)}")
    print(raw_stripped[:4000] if raw_stripped else "(vide)")
    print("========== FIN RÉPONSE LLM ==========\n")

    # Fallback propre si vide
    if not raw_stripped:
        return {"sections": dict(sections_payload), "mapping": []}

    # Extraction du bloc JSON
    start = raw_stripped.find("{")
    end = raw_stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        print("[figures_planner] JSON non détecté dans la réponse IA -> fallback mapping vide.")
        return {"sections": dict(sections_payload), "mapping": []}

    json_str = raw_stripped[start : end + 1]

    try:
        data = json.loads(json_str)
    except Exception as e:
        print(f"[figures_planner] Erreur parse JSON plan figures: {e} -> fallback mapping vide.")
        return {"sections": dict(sections_payload), "mapping": []}

    # Normalisation du format
    if not isinstance(data, dict):
        return {"sections": dict(sections_payload), "mapping": []}

    if "sections" not in data or not isinstance(data.get("sections"), dict):
        data["sections"] = dict(sections_payload)

    if "mapping" not in data or not isinstance(data.get("mapping"), list):
        data["mapping"] = []

    return data



def _semantic_filter_images(images_meta: List[Dict], *, target_keep: int = 10) -> List[Dict]:
    """
    Garde suffisamment d'images candidates pour permettre au planner
    d'en choisir plusieurs figures (ex: 4).
    - On score les captions (schéma/interface/produit).
    - On garde jusqu'à target_keep.
    - Si pas assez de “bonnes” images, on complète avec les plus grandes restantes.
    """
    positive_keywords = [
        "schéma", "schema", "diagramme", "diagram",
        "architecture", "architectur",
        "interface", "écran", "ecran", "dashboard", "tableau de bord",
        "graphique", "courbe", "chart",
        "prototype", "produit", "dispositif", "capteur",
    ]
    negative_keywords = [
        "photo contexte", "personne", "personnes", "réunion", "meeting",
        "bâtiment", "ville", "paysage", "portrait", "selfie",
    ]

    def score(caption: str) -> int:
        txt = (caption or "").lower().strip()
        if not txt:
            return 0
        if any(k in txt for k in negative_keywords):
            return 0
        s = sum(1 for k in positive_keywords if k in txt)
        # Bonus si le préfixe Vision est explicite
        if txt.startswith("schéma"):
            s += 3
        if txt.startswith("interface"):
            s += 2
        if txt.startswith("produit"):
            s += 2
        return s

    # Score
    scored = [(score(m.get("caption", "")), m) for m in images_meta]
    scored_pos = [m for s, m in sorted(scored, key=lambda x: x[0], reverse=True) if s > 0]

    keep = []
    used_ids = set()

    # 1) on prend les meilleures
    for m in scored_pos:
        if len(keep) >= target_keep:
            break
        keep.append(m)
        used_ids.add(m.get("id"))

    # 2) compléter avec les plus grandes restantes si besoin
    if len(keep) < target_keep:
        rest = [m for m in images_meta if m.get("id") not in used_ids]
        rest = sorted(rest, key=lambda m: int(m.get("area", 0)), reverse=True)
        for m in rest:
            if len(keep) >= target_keep:
                break
            keep.append(m)

    print(f"[figures_planner] Images retenues après filtre sémantique: {len(keep)} (cible={target_keep})")
    return keep





def _prepare_figures_for_dossier(
    dossier_type: str,
    docs_client_data: List[Dict[str, bytes]],
    sections: Dict[str, str],
    section_keys_for_figures: List[str],
    max_figures: int = 10,
    min_side: int = 400,
    max_images_candidates: int = 20,
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
    images_meta = _semantic_filter_images(images_meta, target_keep=max_figures)
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
    try:
        plan = _plan_figures_with_llm(
            dossier_type,
            sections_payload,
            images_meta,
            max_figures=max_figures,
        )
        print("\n========== PLAN FIGURES IA ==========\n")
        print(plan)
        print("\n========== FIN PLAN FIGURES IA ==========\n")
    except Exception as e:
        print(f"[figures_planner] Erreur lors de la planification des figures: {e}")
        import traceback
        traceback.print_exc()
        return sections, None, None

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
        print("[figures_planner] Aucune image retenue après mapping IA → sections originales conservées (sans références figures).")
        return sections, None, None

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
    max_figures: int = 10,
    min_side: int = 300,
    max_images_candidates: int = 20,
) -> Tuple[Dict[str, str], Optional[str], Optional[str]]:
    """
    Spécifique CIR : figures uniquement dans travaux et contribution (résultats).
    """
    section_keys = ["travaux", "contribution"]
    return _prepare_figures_for_dossier(
        "CIR",
        docs_client_data,
        sections_cir,
        section_keys,
        max_figures=max_figures,
        min_side=min_side,
        max_images_candidates=max_images_candidates,
    )


def prepare_figures_for_cii(
    docs_client_data: List[Dict[str, bytes]],
    sections_cii: Dict[str, str],
    max_figures: int = 10,
    min_side: int = 300,
    max_images_candidates: int = 20,
) -> Tuple[Dict[str, str], Optional[str], Optional[str]]:
    """
    Spécifique CII : figures uniquement dans démarche (travaux) et résultats.
    """
    section_keys = [
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
        max_images_candidates=max_images_candidates,
    )
def _fix_placeholder_figure_x(sections: Dict[str, str], mapping_sorted: List[Dict]) -> Dict[str, str]:
    """
    Remplace 'Figure X' par un numéro valide.
    Stratégie : remplacer par le plus petit numéro de figure non déjà présent.
    """
    all_nums = []
    for m in mapping_sorted:
        try:
            all_nums.append(int(m.get("figure_number", 0)))
        except Exception:
            pass
    all_nums = sorted(n for n in all_nums if n > 0)

    if not all_nums:
        return sections

    out = dict(sections)

    used = set()
    num_re = re.compile(r"(?i)\bfigure\s+(\d+)\b")
    for k, v in out.items():
        for mm in num_re.finditer(v or ""):
            try:
                used.add(int(mm.group(1)))
            except Exception:
                pass

    remaining = [n for n in all_nums if n not in used]
    replacement = remaining[0] if remaining else all_nums[-1]

    figx_re = re.compile(r"(?i)\bfigure\s+x\b")
    for k, v in out.items():
        if v:
            out[k] = figx_re.sub(f"Figure {replacement}", v)
    return out
