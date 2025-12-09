# Core/writer.py
from io import BytesIO
from typing import Optional, Dict, Any

from docxtpl import DocxTemplate, InlineImage
from docx.shared import Cm
from docx import Document as DocxDocument

from .writer_tpl import clean_custom_tags, format_docx
from Core import footnotes  # <-- ajout


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
    - nettoie les tags [[ROUGE:...]],
    - applique la mise en forme (Times, interligne...),
    - puis insère des notes de bas de page (termes techniques + URLs) via docx_footnotes.
    """
    doc = DocxTemplate(template_path)

    # Préparer les tokens de branding (page 1)
    branding = dict(branding_tokens or {})

    # Support d'un placeholder {{ LOGO }} si tu veux l'insérer sur la 1re page
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

    # Post-traitements : couleurs / mise en forme
    clean_custom_tags(output_path)
    format_docx(output_path)

    # Ajout des VRAIES notes de bas de page
    # - use_llm_terms=True : termes techniques/scientifiques + concurrents détectés par l’IA
    # - add_url_footnotes=True : chaque URL dans le texte devient une note contenant l’URL
    footnotes.auto_annotate_docx_with_footnotes(
        output_path,
        use_llm_terms=True,
        max_terms=20,
        add_url_footnotes=True,
    )

    return output_path
