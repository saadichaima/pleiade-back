# Core/writer_tpl.py
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Cm, Pt, RGBColor
from docx import Document as DocxDocument
import os, re

DEFAULT_FONT = "Times New Roman"

# On gère aussi les espaces insécables (U+00A0)
SPACE_RE = re.compile(r"[\s\u00A0]+")


def _collapse_spaces(text: str) -> str:
    """Remplace toutes les séquences d'espaces/tab (y compris NBSP) par un seul espace."""
    if not text:
        return ""
    return SPACE_RE.sub(" ", text).strip()


def generate_cir_docx(template_path: str, output_path: str, d: dict):
    if not os.path.exists(template_path):
        raise FileNotFoundError(template_path)
    doc = DocxTemplate(template_path)
    logo = d["info"].get("logo")
    if logo:
        d["info"]["logo"] = InlineImage(doc, logo, width=Cm(4))
    doc.render({"d": d, "cm": Cm})
    doc.save(output_path)
    clean_custom_tags(output_path)
    format_docx(output_path)
    return output_path


def _add_red(par, text):
    run = par.add_run(text)
    run.font.name = DEFAULT_FONT
    run.font.color.rgb = RGBColor(255, 0, 0)


def _add_black(par, text):
    run = par.add_run(text)
    run.font.name = DEFAULT_FONT
    run.font.color.rgb = RGBColor(0, 0, 0)


def _apply_markdown_style(par):
    """
    Transforme **texte** en gras et *texte* en italique dans un paragraphe.
    On reconstruit les runs en supprimant les astérisques.
    """
    raw = par.text
    if "*" not in raw:
        return

    par.clear()
    i = 0
    n = len(raw)

    while i < n:
        # priorité au gras (**...**)
        if i + 1 < n and raw[i] == "*" and raw[i + 1] == "*":
            end = raw.find("**", i + 2)
            if end == -1:
                # pas de fermeture -> texte normal
                run = par.add_run(raw[i:])
                run.font.name = DEFAULT_FONT
                break
            content = raw[i + 2:end]
            if content:
                run = par.add_run(content)
                run.bold = True
                run.font.name = DEFAULT_FONT
            i = end + 2
        # italique simple (*...*)
        elif raw[i] == "*":
            end = raw.find("*", i + 1)
            if end == -1:
                run = par.add_run(raw[i:])
                run.font.name = DEFAULT_FONT
                break
            content = raw[i + 1:end]
            if content:
                run = par.add_run(content)
                run.italic = True
                run.font.name = DEFAULT_FONT
            i = end + 1
        else:
            # texte normal jusqu’au prochain *
            j = i
            while j < n and raw[j] != "*":
                j += 1
            chunk = raw[i:j]
            if chunk:
                run = par.add_run(chunk)
                run.font.name = DEFAULT_FONT
            i = j


def clean_custom_tags(path: str):
    doc = DocxDocument(path)
    for p in doc.paragraphs:
        if "[[ROUGE:" in p.text:
            raw = p.text
            p.clear()
            parts = re.split(r"(\[\[ROUGE:.*?\]\])", raw)
            for part in parts:
                if part.startswith("[[ROUGE:") and part.endswith("]]"):
                    _add_red(p, part[len("[[ROUGE:"):-2].strip())
                elif part:
                    _add_black(p, part)
    doc.save(path)


def _iter_all_paragraphs(doc):
    """
    Retourne tous les paragraphes du document, y compris ceux contenus
    dans les tableaux (et sous-tableaux).
    """

    def iter_container(container):
        # paragraphes directs
        for p in container.paragraphs:
            yield p
        # tableaux éventuels
        for table in getattr(container, "tables", []):
            for row in table.rows:
                for cell in row.cells:
                    # récursif pour gérer les tableaux imbriqués
                    yield from iter_container(cell)

    # corps principal du document
    yield from iter_container(doc)

    # headers/footers éventuels
    for section in doc.sections:
        yield from iter_container(section.header)
        yield from iter_container(section.footer)


import re
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

# ... le reste de writer_tpl (DEFAULT_FONT, _collapse_spaces, _add_red/_add_black, _apply_markdown_style, _iter_all_paragraphs, etc.)

def format_docx(path: str):
    from docx import Document as DocxDocument

    doc = DocxDocument(path)

    # texte "à compléter par le client" en rouge
    pat_red = re.compile(r"(à\s*compléter\s*par\s*le\s*client\s*:?.*?)", re.IGNORECASE)
    # liste numérotée 1. 2. 3.
    order_re = re.compile(r"^(?P<num>\d+)[\.)]\s+(?P<txt>.+)$")

    # Nouveau : détection robuste des listes de niveau 1 et 2
    bullet_lvl1_re = re.compile(r"^[-*]\s+(?P<txt>.+)$")
    bullet_lvl2_re = re.compile(r"^/\s+(?P<txt>.+)$")

    # Suivi d'un bloc de listes de niveau 2 ("/") pour appliquer la règle ; / .
    in_lvl2_block = False
    last_lvl2_par = None

    def close_lvl2_block():
        """Remplace le dernier ';' du bloc de niveau 2 courant par '.'."""
        nonlocal in_lvl2_block, last_lvl2_par
        if in_lvl2_block and last_lvl2_par is not None:
            for r in reversed(last_lvl2_par.runs):
                t = r.text or ""
                idx = t.rfind(";")
                if idx != -1:
                    r.text = t[:idx] + "." + t[idx + 1:]
                    break
        in_lvl2_block = False
        last_lvl2_par = None

    # Parcours de tous les paragraphes (y compris tableaux)
    for p in _iter_all_paragraphs(doc):
        txt = p.text or ""

        # 1) Gestion "à compléter par le client" en rouge
        if pat_red.search(txt):
            raw = txt
            p.clear()
            last = 0
            for m in pat_red.finditer(raw):
                if m.start() > last:
                    _add_black(p, raw[last:m.start()])
                _add_red(p, m.group(1))
                last = m.end()
            if last < len(raw):
                _add_black(p, raw[last:])
        else:
            # 2) Application du markdown léger (*italique*, **gras**)
            _apply_markdown_style(p)

        # 3) Détection des listes
        txt_after = p.text or ""
        stripped = txt_after.lstrip("\u00A0 \t")  # espaces + NBSP

        m_lvl2 = bullet_lvl2_re.match(stripped)
        m_lvl1 = bullet_lvl1_re.match(stripped)
        m_ord = order_re.match(stripped)

        if m_lvl2:
            # ----- Liste niveau 2: lignes commençant par "/ ..." -----
            content = _collapse_spaces(m_lvl2.group("txt"))
            # on neutralise la ponctuation finale, on laissera close_lvl2_block faire le point final
            content = content.rstrip(" ;.")

            p.clear()
            run = p.add_run(f"• {content};")
            run.font.name = DEFAULT_FONT
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT

            in_lvl2_block = True
            last_lvl2_par = p

        elif m_lvl1:
            # ----- Liste niveau 1: lignes commençant par "- ..." ou "* ..." -----
            # Si on sort d'un bloc de niveau 2, on clôture le ';' -> '.'
            close_lvl2_block()

            content = _collapse_spaces(m_lvl1.group("txt"))

            p.clear()
            run = p.add_run(f"• {content}")
            run.font.name = DEFAULT_FONT
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT

        elif m_ord:
            # ----- Liste numérotée 1. 2. 3. -----
            close_lvl2_block()

            num = m_ord.group("num")
            content = _collapse_spaces(m_ord.group("txt"))
            p.clear()
            run = p.add_run(f"{num}. {content}")
            run.font.name = DEFAULT_FONT
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT

        else:
            # Paragraphe normal : si on sort d'un bloc de niveau 2, on ajuste le dernier ';'
            close_lvl2_block()

        # 4) Format commun (espacement + police)
        p.paragraph_format.space_after = Pt(11)
        for r in p.runs:
            r.font.name = DEFAULT_FONT

    # Si le document se termine sur un bloc de niveau 2, on corrige le dernier ';'
    close_lvl2_block()

    doc.save(path)
