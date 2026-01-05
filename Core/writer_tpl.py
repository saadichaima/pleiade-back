# Core/writer_tpl.py
# Objectif : post-traiter le contenu généré (listes, markdown, tags ROUGE, retraits/espaces)
# SANS modifier la page de garde / titres du template, et SANS corrompre le DOCX.
#
# Principes de sécurité :
# - ne traite pas les textboxes (txbxContent) -> source fréquente de DOCX corrompu
# - ne supprime aucun paragraphe au niveau XML
# - n'insère pas de paragraphes pour les lignes vides lors du split "\n"
# - ne touche pas aux paragraphes vides (qui structurent souvent la couverture)
# - ne touche pas aux paragraphes dont le style ressemble à Title/Heading/Titre/etc.

from docx import Document as DocxDocument
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor
import re

DEFAULT_FONT = "Times New Roman"
SPACE_RE = re.compile(r"[\s\u00A0]+")  # espaces + NBSP


def _collapse_spaces(text: str) -> str:
    if not text:
        return ""
    return SPACE_RE.sub(" ", text).strip()


def _apply_markdown_style(par):
    """
    Transforme **texte** en gras et *texte* en italique dans un paragraphe.
    Reconstruit les runs en supprimant les astérisques.
    IMPORTANT : ne pas appeler sur les titres/couverture (on les skip).
    """
    raw = par.text
    if not raw or "*" not in raw:
        return

    par.clear()
    i = 0
    n = len(raw)

    while i < n:
        # gras **...**
        if i + 1 < n and raw[i] == "*" and raw[i + 1] == "*":
            end = raw.find("**", i + 2)
            if end == -1:
                chunk = raw[i:]
                if chunk:
                    r = par.add_run(chunk)
                    r.font.name = DEFAULT_FONT
                break
            content = raw[i + 2 : end]
            if content:
                r = par.add_run(content)
                r.bold = True
                r.font.name = DEFAULT_FONT
            i = end + 2

        # italique *...*
        elif raw[i] == "*":
            end = raw.find("*", i + 1)
            if end == -1:
                chunk = raw[i:]
                if chunk:
                    r = par.add_run(chunk)
                    r.font.name = DEFAULT_FONT
                break
            content = raw[i + 1 : end]
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
    Tous les paragraphes du document, y compris tableaux, headers/footers.
    IMPORTANT : on ne traverse PAS les textboxes (txbxContent) pour éviter la corruption.
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


def _apply_first_existing_style(paragraph, candidates) -> bool:
    """
    Applique le premier style existant parmi les candidats (EN/FR).
    """
    for name in candidates:
        try:
            paragraph.style = name
            return True
        except Exception:
            continue
    return False


def _normalize_inline_lists(text: str) -> str:
    """
    Transforme les items de niveau 2 collés sur une ligne :
      '; / ' -> ';\n/ '
      '. / ' -> '.\n/ '
    """
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s*;\s*/\s+", ";\n/ ", s)
    s = re.sub(r"\s*\.\s*/\s+", ".\n/ ", s)
    return s


def _style_name_lower(p) -> str:
    try:
        return (p.style.name or "").strip().lower()
    except Exception:
        return ""


# Styles qu'on ne doit PAS toucher (cover/titres/entêtes)
# Ajustez si vos templates utilisent d'autres noms.
_SKIP_STYLE_KEYWORDS = (
    "title", "titre",
    "heading", "en-tête", "entête",
    "subtitle", "sous-titre",
    "header", "footer",
    "toc", "table of contents", "table des matières",
    "caption",  # souvent utilisé pour figures
)


def _should_skip_paragraph(p) -> bool:
    """
    Retourne True si on doit préserver intégralement ce paragraphe (page de garde, titres, etc.)
    """
    # 1) Paragraphes vides : souvent utilisés pour la mise en page de la couverture
    if (p.text or "").strip() == "":
        return True

    # 2) Styles de titres/cover
    st = _style_name_lower(p)
    if any(k in st for k in _SKIP_STYLE_KEYWORDS):
        return True

    return False


def format_docx(path: str):
    """
    Post-traitement global :
    - tags rouge
    - éclatement multi-lignes en paragraphes (sans créer de paragraphes vides)
    - normalisation listes (-, /, 1.) en VRAIES listes Word
    - markdown léger
    - uniformisation police/espacement SUR LE CONTENU UNIQUEMENT
    - correction retraits parasites (paragraphes non-listes)
    """
    from docx.oxml import OxmlElement
    from docx.text.paragraph import Paragraph

    doc = DocxDocument(path)

    pat_a_completer = re.compile(r"(?i)(à\s*compléter\s*par\s*le\s*client\s*:?.*?)")
    pat_rien_declarer = re.compile(r"(?i)\brien\s+à\s+déclarer\b")

    # Détection listes
    order_re = re.compile(r"^(?P<num>\d+)[\.)]\s+(?P<txt>.+)$")
    bullet_lvl1_re = re.compile(r"^[\-\u2010\u2011\u2012\u2013\u2014\*]\s+(?P<txt>.+)$")
    bullet_lvl2_re = re.compile(r"^[/／]\s+(?P<txt>.+)$")

    def _insert_paragraph_after(par: Paragraph) -> Paragraph:
        """
        Insère un paragraphe après `par` en CONSERVANT le style du template,
        pour ne pas casser la mise en page.
        """
        new_p = OxmlElement("w:p")
        par._p.addnext(new_p)
        new_par = Paragraph(new_p, par._parent)
        try:
            new_par.style = par.style
        except Exception:
            pass
        try:
            new_par.alignment = par.alignment
        except Exception:
            pass
        return new_par

    # 1) Éclatement des paragraphes multi-lignes (CONTENU seulement)
    paras = list(_iter_all_paragraphs(doc))
    i = 0
    while i < len(paras):
        p = paras[i]

        if _should_skip_paragraph(p):
            i += 1
            continue

        raw = p.text or ""
        raw2 = _normalize_inline_lists(raw)
        if raw2 != raw:
            p.text = raw2

        if "\n" in (p.text or ""):
            lines = (p.text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")

            # 1ère ligne : reste dans p
            p.text = (lines[0] or "").strip()

            insert_after = p
            new_paras = []

            # lignes suivantes : nouveaux paragraphes (on ignore les vides)
            for ln in lines[1:]:
                ln2 = (ln or "").strip()
                if not ln2:
                    continue
                np = _insert_paragraph_after(insert_after)
                np.text = ln2
                new_paras.append(np)
                insert_after = np

            paras[i + 1 : i + 1] = new_paras

        i += 1

    # 2) Formatage contenu (tags rouge / listes / markdown / retraits / espaces)
    in_lvl2_block = False
    last_lvl2_par = None

    def close_lvl2_block():
        nonlocal in_lvl2_block, last_lvl2_par
        if in_lvl2_block and last_lvl2_par is not None:
            # dernier item lvl2 => ponctuation finale "."
            for r in reversed(last_lvl2_par.runs):
                t = r.text or ""
                idx = t.rfind(";")
                if idx != -1:
                    r.text = t[:idx] + "." + t[idx + 1 :]
                    break
        in_lvl2_block = False
        last_lvl2_par = None

    for p in paras:
        if _should_skip_paragraph(p):
            continue

        txt = p.text or ""

        # Tags ROUGE (sur contenu seulement)
        if pat_a_completer.search(txt):
            txt = pat_a_completer.sub(lambda m: f"[[ROUGE: {m.group(1)} ]]", txt)
        if pat_rien_declarer.search(txt):
            txt = pat_rien_declarer.sub(lambda m: f"[[ROUGE: {m.group(0)} ]]", txt)
        if txt != (p.text or ""):
            p.text = txt

        # Nettoyage indentation contenu (cause du “décalage”)
        # On ne touche pas aux titres/cover car ils sont déjà skip.
        p.text = (p.text or "").replace("\t", " ").lstrip("\u00A0\u202F \t")

        stripped = (p.text or "").lstrip("\u00A0\u202F \t")
        stripped = stripped.replace("‐", "-").replace("–", "-").replace("—", "-")

        m_lvl2 = bullet_lvl2_re.match(stripped)
        m_lvl1 = bullet_lvl1_re.match(stripped)
        m_ord = order_re.match(stripped)

        is_list_item = False
        pf = p.paragraph_format
        pf.line_spacing = 1.15


        if m_lvl2:
            content = _collapse_spaces(m_lvl2.group("txt")).rstrip(" ;.")
            p.clear()
            run = p.add_run(f"{content};")
            run.font.name = DEFAULT_FONT
            _apply_first_existing_style(
                p,
                ["List Bullet 2", "Liste à puces 2", "List Paragraph", "Paragraphe de liste"],
            )
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            in_lvl2_block = True
            last_lvl2_par = p
            is_list_item = True

        elif m_lvl1:
            close_lvl2_block()
            content = _collapse_spaces(m_lvl1.group("txt"))
            p.clear()
            run = p.add_run(content)
            run.font.name = DEFAULT_FONT
            _apply_first_existing_style(
                p,
                ["List Bullet", "Liste à puces", "List Paragraph", "Paragraphe de liste"],
            )
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            is_list_item = True

        elif m_ord:
            close_lvl2_block()
            content = _collapse_spaces(m_ord.group("txt"))
            p.clear()
            run = p.add_run(content)
            run.font.name = DEFAULT_FONT
            _apply_first_existing_style(
                p,
                ["List Number", "Liste numérotée", "List Paragraph", "Paragraphe de liste"],
            )
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            is_list_item = True

        else:
            close_lvl2_block()

        # Markdown léger (contenu seulement)
        _apply_markdown_style(p)

        # Espacements / retraits : contenu seulement
        pf = p.paragraph_format
        if is_list_item:
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
        else:
            # IMPORTANT : on neutralise le retrait qui provoque "ligne 2 décalée"
            pf.left_indent = Pt(0)
            pf.first_line_indent = Pt(0)
            pf.space_before = Pt(0)
            # Ajustez si vous voulez plus/moins d'espace entre paragraphes du contenu
            pf.space_after = Pt(6)

        # Police : uniquement sur contenu (pas titres/cover)
        for r in p.runs:
            r.font.name = DEFAULT_FONT

    close_lvl2_block()
    doc.save(path)


def clean_custom_tags(path: str):
    """
    Remplace, dans tout le document, les séquences [[ROUGE: ...]]
    par du texte rouge, en conservant le gras / italique / underline.
    IMPORTANT : ne supprime pas de paragraphes ; uniquement des runs du paragraphe.
    """
    doc = DocxDocument(path)

    for p in _iter_all_paragraphs(doc):
        # Ne pas toucher page de garde / titres
        if _should_skip_paragraph(p):
            continue

        if "[[ROUGE:" not in "".join(r.text or "" for r in p.runs):
            continue

        old_runs = list(p.runs)

        # supprimer les runs existants
        for r in old_runs:
            try:
                p._p.remove(r._r)
            except Exception:
                pass

        for r in old_runs:
            text = r.text or ""
            parts = re.split(r"(\[\[ROUGE:.*?\]\])", text)

            for part in parts:
                if not part:
                    continue

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