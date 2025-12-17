# Core/writer.py
from io import BytesIO
from typing import Optional, Dict, Any

from docxtpl import DocxTemplate, InlineImage
from docx.shared import Cm

from .writer_tpl import clean_custom_tags, format_docx
from Core import footnotes


def generate_docx(
    template_path: str,
    output_path: str,
    d: Dict[str, Any],
    branding_tokens: Optional[Dict[str, str]] = None,
    logo_bytes: Optional[bytes] = None,
) -> str:
    """
    Rend la template docx avec le contexte 'd' + tokens de branding.
    Après rendu :
    - formate le document (Markdown, listes, 'À compléter...'),
    - remplace les tags [[ROUGE:...]] par du texte rouge,
    - puis insère des notes de bas de page (termes techniques + URLs).
    """
    doc = DocxTemplate(template_path)

    branding = dict(branding_tokens or {})
    if logo_bytes:
        try:
            branding["LOGO"] = InlineImage(doc, BytesIO(logo_bytes), width=Cm(4))
        except Exception:
            pass

    context = {
        "d": d,
        "cm": Cm,
        **branding,
        "BRANDING": branding,
    }
    doc.render(context)
    doc.save(output_path)

    # 1) Mise en forme (markdown, listes, tagging 'À compléter...')
    format_docx(output_path)

    # 2) Conversion de tous les [[ROUGE: ...]] en rouge
    clean_custom_tags(output_path)

    # 3) Notes de bas de page (glossaire + URLs)
    footnotes.auto_annotate_docx_with_footnotes(
        output_path,
        use_llm_terms=True,
        max_terms=20,
        add_url_footnotes=True,
    )

    return output_path
