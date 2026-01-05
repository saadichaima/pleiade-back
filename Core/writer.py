# Core/writer.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from docxtpl import DocxTemplate, InlineImage
from docx.shared import Cm

from Core.writer_tpl import format_docx, clean_custom_tags
from Core.footnotes import auto_annotate_docx_with_footnotes


# Caractères interdits dans XML 1.0 (Word)
_XML_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def _sanitize_text_for_docx(s: str) -> str:
    """
    Rend une chaîne sûre pour injection dans docxtpl (XML Word).
    - retire UNIQUEMENT les caractères de contrôle illégaux

    NOTE IMPORTANTE : DocxTpl gère déjà l'échappement XML automatiquement.
    Ne pas échapper manuellement &, <, > car cela crée un double échappement
    qui corrompt les caractères accentués (é → Ã©, ' → â€™, etc.)
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    # Retire uniquement les caractères de contrôle XML illégaux
    s = _XML_ILLEGAL_RE.sub("", s)

    return s


def _sanitize_context(obj: Any) -> Any:
    """
    Sanitization récursive du contexte Jinja/DocxTpl :
    - str -> sanitize
    - dict/list/tuple -> recurse
    - autres -> inchangé
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return _sanitize_text_for_docx(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_context(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_context(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_context(v) for v in obj)
    return obj


def generate_docx(
    template_path: str,
    output_path: str,
    d: Dict[str, Any],
    branding_tokens: Optional[Dict[str, Any]] = None,
    logo_bytes: Optional[bytes] = None,
) -> str:
    """
    Génère un DOCX via docxtpl puis post-traite (listes/markdown/rouge/footnotes).
    Correction critique : sanitization du texte avant doc.render() pour éviter la
    corruption XML (ex: '<https://...>' ou tout '<...>').
    """
    branding_tokens = branding_tokens or {}

    doc = DocxTemplate(template_path)

    # Construction du contexte attendu par le template
    context: Dict[str, Any] = {
        "d": d,
        **(branding_tokens or {}),
    }

    # Logo (si ton template l'utilise)
    if logo_bytes:
        # InlineImage attend un chemin ou un file-like; DocxTpl accepte BytesIO.
        from io import BytesIO
        context["logo"] = InlineImage(doc, BytesIO(logo_bytes), width=Cm(3))

    # SANITIZE: éviter DOCX corrompu par du texte non échappé
    context = _sanitize_context(context)

    # Render & save
    doc.render(context)
    doc.save(output_path)

    # Post-traitement (listes, markdown, tags rouge, footnotes)
    format_docx(output_path)
    clean_custom_tags(output_path)
    auto_annotate_docx_with_footnotes(
        output_path,
        use_llm_terms=True,
        max_terms=15,
        add_url_footnotes=True,
    )

    return output_path