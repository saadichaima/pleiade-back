# Core/footnotes.py
"""
Injection de vraies notes de bas de page dans un DOCX :
- notes pour les termes techniques/scientifiques (glossaire généré par l’IA),
- notes pour les URLs présentes dans le texte.

Principe :
1) on lit word/document.xml avec lxml,
2) on injecte footnotes.xml, settings.xml, _rels/document.xml.rels, [Content_Types].xml si besoin,
3) on insère les références au bon endroit via un algorithme run-par-run,
4) la numérotation des notes est continue.

L’IA est appelée via Core.rag.call_ai.
"""

import json
import re
import zipfile
from collections import OrderedDict
from typing import Dict
from app.services.prompts import prompt_footnotes_glossary
from lxml import etree

from Core import rag  # même module que pour les autres appels LLM

# =======================
# Namespaces / constantes
# =======================
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
FOOT_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"
FOOT_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"


# =======================
# Extraction texte
# =======================
def extract_text_from_docx(docx_path: str) -> str:
    """Renvoie tous les w:t concaténés (séparés par des sauts de ligne)."""
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml")
    root = etree.fromstring(xml)
    texts = root.xpath("//w:t", namespaces=NS)
    return "\n".join(t.text for t in texts if t.text)


# =======================
# Terme -> définition via LLM
# =======================
def extract_terms_with_llm(text: str, max_terms: int = 15) -> OrderedDict:
    """
    Utilise rag.call_ai pour obtenir un JSON de la forme :
    {"items": [{"term":"…", "definition":"…"}, ...]}

    Le prompt est chargé depuis le Blob (footnotes_glossary.txt dans PROMPTS_CONTAINER_OTHERS).
    """
    if not text or not text.strip():
        return OrderedDict()

    try:
        tpl = prompt_footnotes_glossary()
    except RuntimeError as e:
        # Si le prompt n'est pas disponible, on n'interrompt pas la génération :
        print(f"[footnotes] {e}")
        return OrderedDict()

    # Construction du prompt à partir du template
    try:
        prompt = tpl.format(
            max_terms=max_terms,
            text=text,
        )
    except Exception as e:
        print(f"[footnotes] Erreur formatage prompt footnotes_glossary: {e}")
        return OrderedDict()

    raw = rag.call_ai(prompt, meta="DOCX Footnotes Glossary")
    raw = (raw or "").strip()

    # On cherche la plus grande portion {...}
    start = raw.find("{")
    end = raw.rfind("}")
    json_str = raw[start : end + 1] if start != -1 and end != -1 else "{}"

    try:
        data = json.loads(json_str)
    except Exception:
        return OrderedDict()

    glossary: OrderedDict[str, str] = OrderedDict()
    for item in data.get("items", []):
        term = (item.get("term") or "").strip()
        definition = (item.get("definition") or "").strip()
        if term and definition and term.lower() not in glossary:
            glossary[term] = definition

    return glossary



# =======================
# Numérotation continue
# =======================
def _ensure_continuous_footnote_numbering_settings(settings_root: etree._Element) -> None:
    if settings_root is None:
        return
    footnotePr = settings_root.find("w:footnotePr", namespaces=NS)
    if footnotePr is None:
        footnotePr = etree.SubElement(settings_root, f"{{{NS['w']}}}footnotePr")
    numRestart = footnotePr.find("w:numRestart", namespaces=NS)
    if numRestart is None:
        numRestart = etree.SubElement(footnotePr, f"{{{NS['w']}}}numRestart")
    numRestart.set(f"{{{NS['w']}}}val", "continuous")


def _ensure_continuous_footnote_numbering_document(doc_root: etree._Element) -> None:
    if doc_root is None:
        return
    for footnotePr in doc_root.xpath("//w:footnotePr", namespaces=NS):
        numRestart = footnotePr.find("w:numRestart", namespaces=NS)
        if numRestart is None:
            numRestart = etree.SubElement(footnotePr, f"{{{NS['w']}}}numRestart")
        numRestart.set(f"{{{NS['w']}}}val", "continuous")


# =======================
# Helpers package DOCX
# =======================
def _read_or_default_settings(z: zipfile.ZipFile) -> etree._Element:
    if "word/settings.xml" in z.namelist():
        return etree.fromstring(z.read("word/settings.xml"))
    return etree.fromstring(f'<w:settings xmlns:w="{NS["w"]}"/>')


def _read_or_default_footnotes(z: zipfile.ZipFile) -> etree._Element:
    if "word/footnotes.xml" in z.namelist():
        return etree.fromstring(z.read("word/footnotes.xml"))
    return etree.fromstring(f'<w:footnotes xmlns:w="{NS["w"]}"/>')


def _read_or_default_document_rels(z: zipfile.ZipFile) -> etree._Element:
    path = "word/_rels/document.xml.rels"
    if path in z.namelist():
        return etree.fromstring(z.read(path))
    return etree.fromstring(f'<Relationships xmlns="{RELS_NS}"/>')


def _ensure_footnotes_rel(rels_root: etree._Element) -> None:
    """Ajoute la relation footnotes si absente."""
    has = False
    for rel in rels_root.findall(f"{{{RELS_NS}}}Relationship"):
        if rel.get("Type") == FOOT_REL_TYPE:
            has = True
            break
    if has:
        return

    used = set()
    for rel in rels_root.findall(f"{{{RELS_NS}}}Relationship"):
        rid = rel.get("Id") or ""
        if rid.startswith("rId"):
            try:
                used.add(int(rid[3:]))
            except Exception:
                continue
    n = 1
    while n in used:
        n += 1
    rid = f"rId{n}"
    etree.SubElement(
        rels_root,
        f"{{{RELS_NS}}}Relationship",
        Id=rid,
        Type=FOOT_REL_TYPE,
        Target="footnotes.xml",
    )


def _read_or_default_content_types(z: zipfile.ZipFile) -> etree._Element:
    path = "[Content_Types].xml"
    if path in z.namelist():
        return etree.fromstring(z.read(path))
    return etree.fromstring(f'<Types xmlns="{CT_NS}"/>')


def _ensure_footnotes_override(ct_root: etree._Element) -> None:
    has = False
    for ov in ct_root.findall(f"{{{CT_NS}}}Override"):
        if ov.get("PartName") == "/word/footnotes.xml":
            has = True
            break
    if has:
        return
    etree.SubElement(
        ct_root,
        f"{{{CT_NS}}}Override",
        PartName="/word/footnotes.xml",
        ContentType=FOOT_CT,
    )


def _write_back_docx(
    src_path: str,
    out_path: str,
    doc_root: etree._Element,
    foot_root: etree._Element,
    settings_root: etree._Element,
    rels_root: etree._Element,
    ct_root: etree._Element,
) -> str:
    """Réécrit le DOCX avec les XML modifiés."""
    with zipfile.ZipFile(src_path, "r") as zin, zipfile.ZipFile(out_path, "w") as zout:
        names = {i.filename for i in zin.infolist()}
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = etree.tostring(doc_root, encoding="UTF-8", standalone="yes")
            elif item.filename == "word/footnotes.xml":
                data = etree.tostring(foot_root, encoding="UTF-8", standalone="yes")
            elif item.filename == "word/settings.xml":
                data = etree.tostring(settings_root, encoding="UTF-8", standalone="yes")
            elif item.filename == "word/_rels/document.xml.rels":
                data = etree.tostring(rels_root, encoding="UTF-8", standalone="yes")
            elif item.filename == "[Content_Types].xml":
                data = etree.tostring(ct_root, encoding="UTF-8", standalone="yes")
            zout.writestr(item, data)

        if "word/footnotes.xml" not in names:
            zout.writestr("word/footnotes.xml", etree.tostring(foot_root, encoding="UTF-8", standalone="yes"))
        if "word/settings.xml" not in names:
            zout.writestr("word/settings.xml", etree.tostring(settings_root, encoding="UTF-8", standalone="yes"))
        if "word/_rels/document.xml.rels" not in names:
            zout.writestr("word/_rels/document.xml.rels", etree.tostring(rels_root, encoding="UTF-8", standalone="yes"))
        if "[Content_Types].xml" not in names:
            zout.writestr("[Content_Types].xml", etree.tostring(ct_root, encoding="UTF-8", standalone="yes"))
    return out_path


# =======================
# Création footnotes / runs
# =======================
def _make_footnote_reference_run(note_id: int) -> etree._Element:
    run_ref = etree.Element(f"{{{NS['w']}}}r")
    pr_ref = etree.SubElement(run_ref, f"{{{NS['w']}}}rPr")
    etree.SubElement(pr_ref, f"{{{NS['w']}}}rStyle", attrib={f"{{{NS['w']}}}val": "FootnoteReference"})
    etree.SubElement(pr_ref, f"{{{NS['w']}}}vertAlign", attrib={f"{{{NS['w']}}}val": "superscript"})
    fn_ref = etree.SubElement(run_ref, f"{{{NS['w']}}}footnoteReference")
    fn_ref.set(f"{{{NS['w']}}}id", str(note_id))
    return run_ref


def _append_footnote(foot_root: etree._Element, note_id: int, text: str) -> None:
    """Ajoute une note de bas de page simple 'numéro + texte'."""
    footnote = etree.SubElement(foot_root, f"{{{NS['w']}}}footnote")
    footnote.set(f"{{{NS['w']}}}id", str(note_id))
    p = etree.SubElement(footnote, f"{{{NS['w']}}}p")

    # numéro
    run_num = etree.SubElement(p, f"{{{NS['w']}}}r")
    pr_num = etree.SubElement(run_num, f"{{{NS['w']}}}rPr")
    etree.SubElement(pr_num, f"{{{NS['w']}}}rStyle", attrib={f"{{{NS['w']}}}val": "FootnoteReference"})
    etree.SubElement(pr_num, f"{{{NS['w']}}}vertAlign", attrib={f"{{{NS['w']}}}val": "superscript"})
    etree.SubElement(run_num, f"{{{NS['w']}}}footnoteRef")

    # contenu
    r = etree.SubElement(p, f"{{{NS['w']}}}r")
    rPr = etree.SubElement(r, f"{{{NS['w']}}}rPr")
    etree.SubElement(rPr, f"{{{NS['w']}}}rFonts", attrib={
        f"{{{NS['w']}}}ascii": "Times New Roman",
        f"{{{NS['w']}}}hAnsi": "Times New Roman",
        f"{{{NS['w']}}}cs": "Times New Roman",
    })
    etree.SubElement(rPr, f"{{{NS['w']}}}sz", attrib={f"{{{NS['w']}}}val": "20"})
    etree.SubElement(rPr, f"{{{NS['w']}}}szCs", attrib={f"{{{NS['w']}}}val": "20"})
    t_el = etree.SubElement(r, f"{{{NS['w']}}}t")
    t_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t_el.text = text


# =======================
# Utils runs / wrappers
# =======================
def _container_and_top_anchor_for_run(run_el: etree._Element):
    anchor = run_el
    parent = run_el.getparent()
    while parent is not None and parent.tag != f"{{{NS['w']}}}p":
        anchor = parent
        parent = parent.getparent()
    return parent, anchor


def _direct_child_under(parent, node):
    cur = node
    while cur is not None and cur.getparent() is not parent:
        cur = cur.getparent()
    return cur


def _copy_rPr(dst_run, src_run):
    src_pr = src_run.find("w:rPr", namespaces=NS)
    if src_pr is None:
        return
    dst_pr = etree.SubElement(dst_run, f"{{{NS['w']}}}rPr")
    for child in src_pr:
        dst_pr.append(etree.fromstring(etree.tostring(child)))


def _insert_sequence_at(parent_run: etree._Element, run_seq) -> None:
    """
    Insère des runs juste après parent_run, en gérant les wrappers (hyperlink, smartTag, etc.).
    """
    p, anchor = _container_and_top_anchor_for_run(parent_run)

    if p is None or anchor is None:
        par = parent_run.getparent()
        if par is not None:
            pos = par.index(parent_run) + 1
            for r in run_seq:
                par.insert(pos, r)
                pos += 1
        return

    if anchor.tag == f"{{{NS['w']}}}r":
       parent = parent_run.getparent()
       holder = parent if parent is not None else p
       pos = holder.index(parent_run) + 1
       for r in run_seq:
         holder.insert(pos, r)
         pos += 1
       return


    child_top = _direct_child_under(anchor, parent_run) or parent_run
    children = list(anchor)
    try:
        idx_child = children.index(child_top)
    except ValueError:
        idx_child = len(children) - 1

    tail_nodes = children[idx_child + 1 :]
    for node in tail_nodes:
        anchor.remove(node)

    pos = p.index(anchor) + 1
    for r in run_seq:
        p.insert(pos, r)
        pos += 1

    for node in tail_nodes:
        p.insert(pos, node)
        pos += 1


# =======================
# Insertion des termes (glossaire)
# =======================
def insert_footnotes(docx_path: str, glossary: Dict[str, str], out_path: str = None) -> str:
    """
    Pour chaque (terme -> définition), insère une note à la première occurrence du terme.
    """
    from pathlib import Path

    out_path = out_path or docx_path
    tmp_out = str(Path(out_path).with_suffix(".footnotes.tmp.docx"))

    with zipfile.ZipFile(docx_path) as z:
        doc_xml = z.read("word/document.xml")
        settings_root = _read_or_default_settings(z)
        foot_root = _read_or_default_footnotes(z)
        rels_root = _read_or_default_document_rels(z)
        ct_root = _read_or_default_content_types(z)

    doc_root = etree.fromstring(doc_xml)

    _ensure_continuous_footnote_numbering_document(doc_root)
    _ensure_continuous_footnote_numbering_settings(settings_root)

    existing_ids = [
        int(fn.get(f"{{{NS['w']}}}id"))
        for fn in foot_root.xpath("//w:footnote", namespaces=NS)
        if fn.get(f"{{{NS['w']}}}id") and fn.get(f"{{{NS['w']}}}id").isdigit()
    ]
    next_id = max(existing_ids) + 1 if existing_ids else 1

    for term, definition in (glossary or {}).items():
        for t in doc_root.xpath("//w:t", namespaces=NS):
            if not t.text:
                continue
            idx = t.text.lower().find(term.lower())
            if idx == -1:
                continue

            parent_run = t.getparent()
            before = t.text[:idx]
            word = t.text[idx : idx + len(term)]
            after = t.text[idx + len(term) :]

            t.text = before

            run_word = etree.Element(f"{{{NS['w']}}}r")
            _copy_rPr(run_word, parent_run)
            run_word_t = etree.SubElement(run_word, f"{{{NS['w']}}}t")
            run_word_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            run_word_t.text = word

            run_ref = _make_footnote_reference_run(next_id)

            run_after = etree.Element(f"{{{NS['w']}}}r")
            _copy_rPr(run_after, parent_run)
            run_after_t = etree.SubElement(run_after, f"{{{NS['w']}}}t")
            run_after_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            run_after_t.text = after

            _insert_sequence_at(parent_run, [run_word, run_ref, run_after])

            # note : "Terme : Définition"
            p = etree.SubElement(foot_root, f"{{{NS['w']}}}footnote")
            p.set(f"{{{NS['w']}}}id", str(next_id))
            pp = etree.SubElement(p, f"{{{NS['w']}}}p")

            def _add_run(text: str, bold: bool = False):
                r = etree.SubElement(pp, f"{{{NS['w']}}}r")
                rPr = etree.SubElement(r, f"{{{NS['w']}}}rPr")
                etree.SubElement(rPr, f"{{{NS['w']}}}rFonts", attrib={
                    f"{{{NS['w']}}}ascii": "Times New Roman",
                    f"{{{NS['w']}}}hAnsi": "Times New Roman",
                    f"{{{NS['w']}}}cs": "Times New Roman",
                })
                etree.SubElement(rPr, f"{{{NS['w']}}}sz", attrib={f"{{{NS['w']}}}val": "20"})
                etree.SubElement(rPr, f"{{{NS['w']}}}szCs", attrib={f"{{{NS['w']}}}val": "20"})
                if bold:
                    etree.SubElement(rPr, f"{{{NS['w']}}}b")
                t_el = etree.SubElement(r, f"{{{NS['w']}}}t")
                t_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                t_el.text = text

            _add_run(f"{term} : ", bold=True)
            _add_run(definition)

            next_id += 1
            break  # 1re occurrence seulement

    _ensure_footnotes_rel(rels_root)
    _ensure_footnotes_override(ct_root)

    _write_back_docx(docx_path, tmp_out, doc_root, foot_root, settings_root, rels_root, ct_root)

    # on remplace le fichier d'origine
    import os
    os.replace(tmp_out, out_path)
    return out_path


# =======================
# URLs -> notes de bas de page
# =======================
_URL_RE = re.compile(
    r"(?:<(?P<angle>https?://[^>\s]+)>)"          # <https://...>
    r"|(?:\((?P<paren>https?://[^)\s]+)\))"       # (https://...)
    r"|(?:\[(?P<brack>https?://[^\]\s]+)\])"      # [https://...]
    r"|(?P<bare>https?://[^\s\]\)\>]+?)(?="
    r"(?:[.,;:!?\u2026\u00BB](?:\s|$))"           # ponctuation terminale après l’URL
    r"|(?:[\s\]\)\>]|$)"                          # ou espace / ] / ) / > / fin de texte
    r")",
    re.IGNORECASE,
)

_TAIL_PUNCT = ",.;:!?…»"


def _first_url_in_text(s: str):
    if not s:
        return None
    m = _URL_RE.search(s)
    if not m:
        return None
    grp = (
        "angle"
        if m.group("angle")
        else "paren"
        if m.group("paren")
        else "brack"
        if m.group("brack")
        else "bare"
    )
    url = m.group(grp)
    s_idx, e_idx = m.span(grp)
    if grp == "bare" and url and url[-1] in _TAIL_PUNCT:
        k = 0
        while k < len(url) and url[-(k + 1)] in _TAIL_PUNCT:
            k += 1
        if k:
            url = url[:-k]
            e_idx -= k
    return grp, s_idx, e_idx, url


def _process_single_url_in_run(t_node: etree._Element, foot_root: etree._Element, next_id: int) -> int:
    text = t_node.text or ""
    found = _first_url_in_text(text)
    if not found:
        return next_id

    _, s_idx, e_idx, url = found
    parent_run = t_node.getparent()

    left = text[:s_idx]
    right = text[e_idx:]

    t_node.text = left

    run_ref = _make_footnote_reference_run(next_id)

    run_after = None
    if right:
        run_after = etree.Element(f"{{{NS['w']}}}r")
        _copy_rPr(run_after, parent_run)
        t_after = etree.SubElement(run_after, f"{{{NS['w']}}}t")
        t_after.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t_after.text = right

    seq = [run_ref] + ([run_after] if run_after is not None else [])
    _insert_sequence_at(parent_run, seq)

    _append_footnote(foot_root, next_id, url)

    return next_id + 1


def insert_url_footnotes(docx_path: str, out_path: str = None) -> str:
    from pathlib import Path

    out_path = out_path or docx_path
    tmp_out = str(Path(out_path).with_suffix(".urls.tmp.docx"))

    with zipfile.ZipFile(docx_path) as z:
        doc_xml = z.read("word/document.xml")
        settings_root = _read_or_default_settings(z)
        foot_root = _read_or_default_footnotes(z)
        rels_root = _read_or_default_document_rels(z)
        ct_root = _read_or_default_content_types(z)

    doc_root = etree.fromstring(doc_xml)

    _ensure_continuous_footnote_numbering_document(doc_root)
    _ensure_continuous_footnote_numbering_settings(settings_root)

    existing_ids = [
        int(fn.get(f"{{{NS['w']}}}id"))
        for fn in foot_root.xpath("//w:footnote", namespaces=NS)
        if fn.get(f"{{{NS['w']}}}id") and fn.get(f"{{{NS['w']}}}id").isdigit()
    ]
    next_id = max(existing_ids) + 1 if existing_ids else 1

    for p in doc_root.xpath("//w:body//w:p", namespaces=NS):
        changed = True
        while changed:
            changed = False
            t_nodes = p.xpath(".//w:r/w:t", namespaces=NS)
            for t in t_nodes:
                if not t.text:
                    continue
                if _first_url_in_text(t.text):
                    next_id = _process_single_url_in_run(t, foot_root, next_id)
                    changed = True
                    break

    _ensure_footnotes_rel(rels_root)
    _ensure_footnotes_override(ct_root)

    _write_back_docx(docx_path, tmp_out, doc_root, foot_root, settings_root, rels_root, ct_root)

    import os
    os.replace(tmp_out, out_path)
    return out_path


# =======================
# Pipeline complet pour ton dossier
# =======================
def auto_annotate_docx_with_footnotes(
    docx_path: str,
    use_llm_terms: bool = True,
    max_terms: int = 15,
    add_url_footnotes: bool = True,
) -> str:
    """
    1) extrait le texte du docx,
    2) appelle l’IA pour obtenir un glossaire (terme -> définition),
    3) insère des notes pour ces termes,
    4) insère des notes pour toutes les URLs.
    """
    glossary = OrderedDict()  # type: ignore[var-annotated]
    if use_llm_terms:
        text = extract_text_from_docx(docx_path)
        glossary = extract_terms_with_llm(text, max_terms=max_terms)

    if glossary:
        insert_footnotes(docx_path, glossary, out_path=docx_path)

    if add_url_footnotes:
        insert_url_footnotes(docx_path, out_path=docx_path)

    return docx_path
