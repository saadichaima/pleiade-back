# Core/images_figures.py
# -*- coding: utf-8 -*-
import re
import io
import tempfile
from typing import List, Tuple, Dict, Optional

from docx import Document
from docx.text.paragraph import Paragraph
from docx.shared import Emu
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image  # pour valider / redimensionner les images

try:
    import fitz  # PyMuPDF pour PDF
except Exception:
    fitz = None  # type: ignore

# ==========================
# Utils espace / pattern
# ==========================

def _iter_all_paragraphs_doc(doc: Document):
    """
    Retourne tous les Paragraph du document, y compris ceux des tableaux,
    en-têtes et pieds de page.
    """
    def iter_container(container):
        for p in getattr(container, "paragraphs", []):
            yield p
        for table in getattr(container, "tables", []):
            for row in table.rows:
                for cell in row.cells:
                    yield from iter_container(cell)

    # Corps principal
    yield from iter_container(doc)

    # Headers / footers
    for section in doc.sections:
        yield from iter_container(section.header)
        yield from iter_container(section.footer)

FIG_PATTERN = re.compile(r"(?i)\b(?:cf\.\s*)?(?:fig(?:\.|ure)?|image)\s*[:\-–]?\s*(\d+)\b")

def normalize_and_filter_images(
    images: List[Tuple[bytes, str]],
    *,
    min_side: int = 400,
    max_images: int = 30,
) -> List[Tuple[bytes, str]]:
    """
    - Ouvre chaque image avec Pillow pour vérifier qu'elle est valide.
    - Ne garde que les images dont la plus grande dimension >= min_side.
    - Réencode en PNG (format sûr pour python-docx).
    - Trie par taille (aire w*h décroissante) et garde max_images.

    Retourne une liste [(bytes_png, "png"), ...].
    """
    cleaned: List[Tuple[bytes, str, int]] = []

    for idx, (data, ext) in enumerate(images, start=1):
        if not data:
            continue
        try:
            im = Image.open(io.BytesIO(data))
            im.load()  # force le décodage
            w, h = im.size
        except Exception:
            # image illisible -> on passe
            continue

        if max(w, h) < min_side:
            # trop petite (icône, logo minuscule, etc.) -> on ignore
            continue

        # réencodage en PNG propre
        try:
            out = io.BytesIO()
            im.save(out, format="PNG")
            png_bytes = out.getvalue()
        except Exception:
            continue

        area = w * h
        cleaned.append((png_bytes, "png", area))

    # tri par taille décroissante
    cleaned.sort(key=lambda x: x[2], reverse=True)
    # on garde au maximum max_images
    selected = cleaned[:max_images]

    print(
        f"[images_figures] normalize_and_filter_images: "
        f"{len(images)} image(s) en entrée, {len(selected)} retenue(s) "
        f"(min_side={min_side}, max_images={max_images})"
    )

    # on renvoie sans l'aire
    return [(b, ext) for (b, ext, _area) in selected]

def _norm_space(s: str) -> str:
    # normalise NBSP & NNBSP en espace simple
    return (s or "").replace("\u00A0", " ").replace("\u202F", " ")


# ==========================
# Extraction images DOCX source (ordre apparition) : a:blip + VML imagedata
# ==========================

def _attr_local(elm, local_name: str) -> Optional[str]:
    """Retourne la valeur d'un attribut dont le *local-name()* = local_name, sinon None."""
    for k, v in elm.attrib.items():
        if k.split("}")[-1] == local_name:
            return v
    return None


def _collect_embeds_live(parent_elm) -> List[str]:
    """
    Sur un élément OXML 'vivant' (doc._element, header._element...), collecte
    les rId d'images dans l'ordre d'apparition :
      - DrawingML:  //a:blip  -> @r:embed | @r:link
      - VML:       //w:pict//v:imagedata -> @r:id | @r:link
    """
    rids: List[str] = []
    # 1) a:blip
    for blip in parent_elm.xpath(".//*[local-name()='blip']"):
        rid = _attr_local(blip, "embed") or _attr_local(blip, "link")
        if rid:
            rids.append(rid)
    # 2) v:imagedata sous w:pict
    for imd in parent_elm.xpath(".//*[local-name()='pict']//*[local-name()='imagedata']"):
        rid = _attr_local(imd, "id") or _attr_local(imd, "link")
        if rid:
            rids.append(rid)
    return rids


def _resolve_rids(part, rids: List[str]) -> List[Tuple[bytes, str]]:
    """
    Convertit une liste de rId en [(bytes, ext)] en résolvant part.rels.
    """
    out: List[Tuple[bytes, str]] = []
    for rid in rids:
        rel = part.rels.get(rid)
        if not rel:
            continue
        target = rel._target
        data = getattr(target, "blob", None)
        if not data:
            continue
        partname = str(getattr(target, "partname", ""))  # e.g. /word/media/image1.png
        ext = partname.rsplit(".", 1)[-1].lower() if "." in partname else "png"
        out.append((data, ext))
    return out


def extract_images_from_docx_ordered_any(path: str, include_header_footer: bool = False) -> List[Tuple[bytes, str]]:
    """
    Extrait toutes les images d'un DOCX dans l'ordre d'apparition (a:blip + VML imagedata).
    Retourne [(data, ext)].
    """
    import docx
    doc = docx.Document(path)
    images: List[Tuple[bytes, str]] = []
    images.extend(_resolve_rids(doc.part, _collect_embeds_live(doc._element)))
    if include_header_footer:
        for section in doc.sections:
            if section.header:
                images.extend(_resolve_rids(section.header.part, _collect_embeds_live(section.header._element)))
            if section.footer:
                images.extend(_resolve_rids(section.footer.part, _collect_embeds_live(section.footer._element)))
    print(f"[images_figures] extract_images_from_docx_ordered_any: {len(images)} image(s) trouvée(s) dans {path}")
    return images


def extract_images_from_docx_bytes(data: bytes) -> List[Tuple[bytes, str]]:
    """
    Helper : écrit les bytes DOCX dans un fichier temp et appelle extract_images_from_docx_ordered_any.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    tmp.write(data)
    tmp.close()
    return extract_images_from_docx_ordered_any(tmp.name, include_header_footer=False)


def extract_images_from_pdf_bytes(data: bytes) -> List[Tuple[bytes, str]]:
    """
    Extrait les images d'un PDF (bytes) via PyMuPDF.
    Retourne [(bytes, ext)].
    """
    if not fitz:
        print("[images_figures] PyMuPDF (fitz) non disponible, impossible d'extraire les images PDF.")
        return []
    images: List[Tuple[bytes, str]] = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                for xref, *_ in page.get_images(full=True):
                    base = doc.extract_image(xref)
                    img = base.get("image")
                    ext = (base.get("ext") or "png").lower()
                    if img:
                        images.append((img, ext))
    except Exception as e:
        print(f"[images_figures] Erreur extract_images_from_pdf_bytes: {e}")
    print(f"[images_figures] extract_images_from_pdf_bytes: {len(images)} image(s) trouvée(s) dans le PDF")
    return images


# ==========================
# Construction d'un DOCX "source d'images"
# ==========================

def build_docx_from_images(images: List[Tuple[bytes, str]]) -> str:
    """
    Construit un DOCX temporaire qui contient toutes les images fournies (une par paragraphe).
    Les images sont supposées déjà filtrées/normalisées (ex: via normalize_and_filter_images).
    """
    doc = Document()
    count_ok = 0
    for i, (img_bytes, _ext) in enumerate(images, start=1):
        try:
            p = doc.add_paragraph()
            run = p.add_run()
            run.add_picture(io.BytesIO(img_bytes))
            count_ok += 1
        except Exception as e:
            # on log et on ignore l'image problématique
            print(f"[images_figures] build_docx_from_images: image {i} ignorée (erreur add_picture: {e})")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc.save(tmp.name)
    print(
        f"[images_figures] build_docx_from_images: DOCX temporaire {tmp.name} "
        f"avec {count_ok} image(s) insérée(s) sur {len(images)}"
    )
    return tmp.name



# ==========================
# Arbre vivant : trouver le w:p hôte dans le body, insérer un w:p, y ajouter une image
# ==========================

def _find_body_host_paragraph_live(w_t) -> Optional[object]:
    """
    Depuis un w:t (élément vivant), remonte au w:p qui appartient au w:body.
    Si le w:t est dans une textbox, on remonte au w:p externe hébergé par le body.
    """
    # remonter au w:p le plus proche
    el = w_t
    while el is not None and el.tag != qn("w:p"):
        el = el.getparent()
    if el is None:
        return None

    # si ce w:p appartient au body, ok
    anc = el
    while anc is not None and anc.tag != qn("w:body"):
        anc = anc.getparent()
    if anc is not None and anc.tag == qn("w:body"):
        return el

    # sinon, remonter plus haut jusqu'à trouver un w:p ayant un ancêtre w:body
    p_candidate = el.getparent()
    while p_candidate is not None:
        # chercher le prochain ancêtre w:p
        while p_candidate is not None and p_candidate.tag != qn("w:p"):
            p_candidate = p_candidate.getparent()
        if p_candidate is None:
            break
        anc = p_candidate
        while anc is not None and anc.tag != qn("w:body"):
            anc = anc.getparent()
        if anc is not None and anc.tag == qn("w:body"):
            return p_candidate
        p_candidate = p_candidate.getparent()
    return None


def _insert_paragraph_after_live(p_body):
    """
    Insère un nouveau <w:p> juste après p_body (w:p du corps vivant) et le retourne.
    """
    new_p = OxmlElement("w:p")
    p_body.addnext(new_p)
    return new_p


def _add_picture_to_paragraph_live(doc: Document, p_elm, image_bytes: bytes, width_emu: int):
    """
    Ajoute une image dans le w:p fourni (arbre vivant) via l'API python-docx.
    """
    para = Paragraph(p_elm, doc._body)  # doc._body est un BlockItemContainer valide
    run = para.add_run()
    run.add_picture(io.BytesIO(image_bytes), width=Emu(width_emu))


def _max_text_width_emu(doc: Document) -> int:
    sect = doc.sections[0]
    return int(sect.page_width - sect.left_margin - sect.right_margin)


# ==========================
# Helpers légendes
# ==========================

def _caption_for_num(captions_text, num: int):
    """
    Résout la légende pour l'image numéro ``num`` (1-indexé) à partir de ``captions_text``.
    """
    if captions_text is None:
        return None

    if isinstance(captions_text, dict):
        return captions_text.get(num)

    if isinstance(captions_text, (list, tuple)):
        return captions_text[num - 1] if 1 <= num <= len(captions_text) else None

    if isinstance(captions_text, str):
        block = captions_text.strip()

        pattern = re.compile(
            r'(?im)^\s*(?:fig(?:\.| |ure)?|image)\s*'
            r'(\d+)\s*'
            r'(?:[:\-–\.\)]+\s*)'
            r'(.+?)\s*$'
        )
        matches = pattern.findall(block)
        if matches:
            mapping: Dict[int, str] = {}
            for n_str, txt in matches:
                try:
                    idx = int(n_str)
                    if idx not in mapping:
                        mapping[idx] = txt.strip()
                except ValueError:
                    continue
            return mapping.get(num)

        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) >= num and len(lines) > 1:
            return lines[num - 1]

        try:
            return captions_text.format(n=num)
        except Exception:
            return None

    return None


# ==========================
# Fonction principale : insertion des images par référence
# ==========================

def insert_images_by_reference_live(
    src_docx: str,
    dst_docx: str,
    out_docx: str,
    include_header_footer_src: bool = False,
    captions_text=None,
    caption_label: str = "Figure",
    caption_style: str = "Caption",
    nbspace_before_colon: bool = True,
    renumber_references: bool = True,
    target_label: str = "Figure",
) -> str:
    """
    Nouvelle version basée sur python-docx :
    - lit les images du DOCX source (src_docx),
    - parcourt tous les Paragraph du DOCX cible (dst_docx),
    - renumérote les références 'Figure X' / 'Image X',
    - insère, à la première occurrence de chaque numéro, l'image correspondante
      sous le paragraphe, avec une légende optionnelle.
    """
    print(
        f"[images_figures] insert_images_by_reference_live: src={src_docx}, "
        f"dst={dst_docx}, out={out_docx}"
    )

    # 1) Index des images du DOCX source (numéro d'origine -> (bytes, ext))
    images = extract_images_from_docx_ordered_any(
        src_docx, include_header_footer=include_header_footer_src
    )
    num_to_img = {i + 1: im for i, im in enumerate(images)}
    print(
        f"[images_figures] {len(num_to_img)} image(s) indexée(s) "
        f"depuis le DOCX source"
    )

    # 2) Document cible
    docB = Document(dst_docx)
    width_emu = _max_text_width_emu(docB)
    nbsp = "\u202F" if nbspace_before_colon else " "

    # État
    ref_seen: set[int] = set()
    old2new: Dict[int, int] = {}
    inserted_for_old: set[int] = set()
    next_new_idx = 1
    inserted_count = 0
    missing_count = 0

    # Helper pour formater le texte remplacé
    def _replace_in_text(text: str):
        nonlocal next_new_idx
        if not text:
            return text, []

        new_text_parts = []
        idx_last = 0
        nums_in_this_para: List[int] = []

        for m in FIG_PATTERN.finditer(text):
            start, end = m.span()
            full = m.group(0)
            old_num = int(m.group(1))

            # texte avant la référence
            new_text_parts.append(text[idx_last:start])

            # renumérotation
            if old_num not in old2new:
                old2new[old_num] = next_new_idx
                next_new_idx += 1
            new_num = old2new[old_num]

            # préfixe cf. éventuel
            has_cf = re.search(r"(?i)\bcf\.\s*", full) is not None
            prefix = "cf. " if has_cf else ""

            if renumber_references:
                repl = f"{prefix}{target_label} {new_num}"
            else:
                repl = full

            new_text_parts.append(repl)
            idx_last = end

            ref_seen.add(old_num)
            if old_num not in nums_in_this_para:
                nums_in_this_para.append(old_num)

        new_text_parts.append(text[idx_last:])
        return "".join(new_text_parts), nums_in_this_para

    # 3) Parcours de tous les paragraphes (y compris tableaux, headers, footers)
    for p in _iter_all_paragraphs_doc(docB):
        original_txt = _norm_space(p.text or "")
        if not original_txt:
            continue

        new_txt, nums_here = _replace_in_text(original_txt)

        # Si pas de référence dans ce paragraphe, on passe
        if not nums_here:
            continue

        # On remplace le texte du paragraphe (perte de styles inline possible)
        if new_txt != original_txt:
            p.text = new_txt

        # Pour chaque numéro rencontré dans ce paragraphe, on insère l'image
        # uniquement à la première occurrence globale
        for old_num in nums_here:
            if old_num in inserted_for_old:
                continue

            if old_num not in num_to_img:
                # référence sans image correspondante
                missing_count += 1
                continue

            img_bytes, _ext = num_to_img[old_num]

            # insertion juste après ce paragraphe
            new_p_elm = _insert_paragraph_after_live(p._p)
            _add_picture_to_paragraph_live(docB, new_p_elm, img_bytes, width_emu)

            # Légende optionnelle
            cap_txt = _caption_for_num(captions_text, old_num)
            if cap_txt:
                cap_p_elm = _insert_paragraph_after_live(new_p_elm)
                para_cap = Paragraph(cap_p_elm, docB._body)
                try:
                    if caption_style:
                        para_cap.style = caption_style
                except Exception:
                    pass
                new_idx = old2new.get(old_num, old_num)
                para_cap.add_run().text = (
                    f"{caption_label} {new_idx}{nbsp}:{nbsp}{cap_txt}"
                )

            inserted_for_old.add(old_num)
            inserted_count += 1

    docB.save(out_docx)

    print(
        f"[images_figures] terminé. Références vues: "
        f"{sorted(ref_seen) if ref_seen else 'aucune'}, "
        f"images insérées: {inserted_count}, "
        f"références sans image: {missing_count}"
    )
    return out_docx
