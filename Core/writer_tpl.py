# Core/writer_tpl.py
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Cm, Pt, RGBColor
from docx import Document as DocxDocument
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os
import re

DEFAULT_FONT = "Times New Roman"

SPACE_RE = re.compile(r"[\s\u00A0]+")


def _collapse_spaces(text: str) -> str:
    if not text:
        return ""
    return SPACE_RE.sub(" ", text).strip()


def _apply_markdown_style(par):
    """
    Transforme **texte** en gras et *texte* en italique dans un paragraphe.
    On reconstruit les runs en supprimant les astérisques.
    """
    raw = par.text
    if not raw or "*" not in raw:
        return

    par.clear()
    i = 0
    n = len(raw)

    while i < n:
        # priorité au gras (**...**)
        if i + 1 < n and raw[i] == "*" and raw[i + 1] == "*":
            end = raw.find("**", i + 2)
            if end == -1:
                chunk = raw[i:]
                if chunk:
                    r = par.add_run(chunk)
                    r.font.name = DEFAULT_FONT
                break
            content = raw[i + 2:end]
            if content:
                r = par.add_run(content)
                r.bold = True
                r.font.name = DEFAULT_FONT
            i = end + 2

        # italique simple (*...*)
        elif raw[i] == "*":
            end = raw.find("*", i + 1)
            if end == -1:
                chunk = raw[i:]
                if chunk:
                    r = par.add_run(chunk)
                    r.font.name = DEFAULT_FONT
                break
            content = raw[i + 1:end]
            if content:
                r = par.add_run(content)
                r.italic = True
                r.font.name = DEFAULT_FONT
            i = end + 1

        else:
            j = i
            while j < n and raw[j] != "*":
                j += 1
            chunk = raw[i:j]
            if chunk:
                r = par.add_run(chunk)
                r.font.name = DEFAULT_FONT
            i = j


def _iter_all_paragraphs(doc: DocxDocument):
    """
    Tous les paragraphes du document, y compris tableaux, headers et footers.
    """

    def iter_container(container):
        for p in getattr(container, "paragraphs", []):
            yield p
        for table in getattr(container, "tables", []):
            for row in table.rows:
                for cell in row.cells:
                    yield from iter_container(cell)

    yield from iter_container(doc)

    for section in doc.sections:
        yield from iter_container(section.header)
        yield from iter_container(section.footer)


def format_docx(path: str):
    """
    Post-traitement global :
    - convertit 'À compléter par le client ...' en [[ROUGE: ...]],
    - normalise les listes (-, /, 1.) en puces / listes numérotées,
    - applique le markdown léger (*italique*, **gras**),
    - uniformise la police / espacement.
    """
    doc = DocxDocument(path)

    # "À compléter par le client" -> tag ROUGE, on laisse la couleur
    # pour clean_custom_tags
    pat_a_completer = re.compile(
        r"(?i)(à\s*compléter\s*par\s*le\s*client\s*:?.*?)"
    )

    order_re = re.compile(r"^(?P<num>\d+)[\.)]\s+(?P<txt>.+)$")
    bullet_lvl1_re = re.compile(r"^[-*]\s+(?P<txt>.+)$")
    bullet_lvl2_re = re.compile(r"^/\s+(?P<txt>.+)$")

    in_lvl2_block = False
    last_lvl2_par = None

    def close_lvl2_block():
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

    for p in _iter_all_paragraphs(doc):
        # 0) injecter [[ROUGE: ...]] pour "À compléter par le client"
        txt0 = p.text or ""
        if pat_a_completer.search(txt0):
            new_txt = pat_a_completer.sub(
                lambda m: f"[[ROUGE: {m.group(1)} ]]", txt0
            )
            p.text = new_txt

        txt = p.text or ""
        stripped = txt.lstrip("\u00A0 \t")

        m_lvl2 = bullet_lvl2_re.match(stripped)
        m_lvl1 = bullet_lvl1_re.match(stripped)
        m_ord = order_re.match(stripped)

        if m_lvl2:
            content = _collapse_spaces(m_lvl2.group("txt"))
            content = content.rstrip(" ;.")
            p.clear()
            run = p.add_run(f"• {content};")
            run.font.name = DEFAULT_FONT
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            in_lvl2_block = True
            last_lvl2_par = p

        elif m_lvl1:
            close_lvl2_block()
            content = _collapse_spaces(m_lvl1.group("txt"))
            p.clear()
            run = p.add_run(f"• {content}")
            run.font.name = DEFAULT_FONT
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT

        elif m_ord:
            close_lvl2_block()
            num = m_ord.group("num")
            content = _collapse_spaces(m_ord.group("txt"))
            p.clear()
            run = p.add_run(f"{num}. {content}")
            run.font.name = DEFAULT_FONT
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT

        else:
            close_lvl2_block()

        # Markdown sur le texte actuel (avec éventuellement [[ROUGE: ...]])
        _apply_markdown_style(p)

        # Formatage commun
        p.paragraph_format.space_after = Pt(11)
        for r in p.runs:
            r.font.name = DEFAULT_FONT

    close_lvl2_block()
    doc.save(path)


def clean_custom_tags(path: str):
    """
    Remplace, dans tout le document, les séquences [[ROUGE: ...]]
    par du texte rouge, en conservant le gras / italique des autres runs.
    """
    doc = DocxDocument(path)

    for p in _iter_all_paragraphs(doc):
        # Vérifier s'il y a au moins un [[ROUGE:
        if "[[ROUGE:" not in "".join(r.text or "" for r in p.runs):
            continue

        old_runs = list(p.runs)
        # On enlève tous les runs existants
        for r in old_runs:
            p._p.remove(r._r)

        for r in old_runs:
            text = r.text or ""
            parts = re.split(r"(\[\[ROUGE:.*?\]\])", text)
            for part in parts:
                if not part:
                    continue

                # On clone les propriétés de police du run d’origine
                def _new_run(txt, make_red=False):
                    nr = p.add_run(txt)
                    nr.font.name = r.font.name or DEFAULT_FONT
                    nr.bold = bool(r.bold)
                    nr.italic = bool(r.italic)
                    nr.underline = bool(r.underline)
                    if r.font.size is not None:
                        nr.font.size = r.font.size
                    if make_red:
                        nr.font.color.rgb = RGBColor(255, 0, 0)
                    else:
                        # conserver la couleur éventuelle d'origine
                        if r.font.color is not None and r.font.color.rgb is not None:
                            nr.font.color.rgb = r.font.color.rgb
                    return nr

                if part.startswith("[[ROUGE:") and part.endswith("]]"):
                    inner = part[len("[[ROUGE:") : -2].strip()
                    if inner:
                        _new_run(inner, make_red=True)
                else:
                    _new_run(part, make_red=False)

    doc.save(path)
