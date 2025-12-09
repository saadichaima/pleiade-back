# Core/description_img.py
"""
Module léger pour :
- extraire les images d'une liste de fichiers (PDF, DOCX, JPG/PNG),
- les réduire/compresser raisonnablement,
- appeler Azure OpenAI Vision pour produire des descriptions "Image N: ...".

Fonctions publiques utilisées par le backend :
- main(doc) -> str : doc = chemin ou liste de chemins; renvoie un texte "Image 1: ...".
- _collect_images_from_paths(paths) -> List[(bytes, ext)] : images extraites (ordre d'apparition).

Configuration Azure lue dans l'environnement :
- AZURE_OPENAI_KEY
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_VERSION (optionnel, défaut "2024-02-15-preview")
- AZURE_OPENAI_VISION_DEPLOYMENT (nom du déploiement Vision, ex: "gpt-4.1")
"""

from __future__ import annotations

import os
import io
import base64
import re
import hashlib
from typing import List, Tuple, Optional, Dict, Iterable
import importlib

from openai import AzureOpenAI

from app.services.prompts import prompt_vision_describe_images

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None  # type: ignore
    ImageOps = None  # type: ignore


# ======================= Extraction basse couche =======================

def _import_fitz():
    try:
        return importlib.import_module("fitz")  # PyMuPDF
    except ImportError as e:
        raise ImportError("PyMuPDF requis : pip install pymupdf") from e


def _extract_images_from_pdf(path: str) -> List[Tuple[bytes, str]]:
    """Extrait les images d'un PDF sous forme [(bytes, ext)]."""
    fitz = _import_fitz()
    out: List[Tuple[bytes, str]] = []
    with fitz.open(path) as doc:
        for page in doc:
            for xref, *_ in page.get_images(full=True):
                base = doc.extract_image(xref)
                data = base.get("image")
                ext = (base.get("ext") or "png").lower()
                if data:
                    out.append((data, ext))
    return out


def _extract_images_from_docx(path: str) -> List[Tuple[bytes, str]]:
    """
    Extrait les images d'un DOCX dans l'ordre d'apparition sous forme [(bytes, ext)].
    On s'appuie sur python-docx + lxml.
    """
    try:
        import docx
        from lxml import etree
    except ImportError as e:
        raise ImportError(
            "python-docx et lxml requis : pip install python-docx lxml"
        ) from e

    doc = docx.Document(path)

    def _collect_rids(xml: str) -> List[str]:
        NS = {
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        }
        root = etree.fromstring(xml.encode("utf-8"))
        blips = root.xpath(".//a:blip", namespaces=NS)
        rids: List[str] = []
        for blip in blips:
            rid = (
                blip.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                )
                or blip.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}link"
                )
            )
            if rid:
                rids.append(rid)
        return rids

    images: List[Tuple[bytes, str]] = []

    # corps
    for rid in _collect_rids(doc.element.xml):
        rel = doc.part.rels.get(rid)
        if not rel:
            continue
        tgt = rel._target
        data = getattr(tgt, "blob", None)
        if not data:
            continue
        partname = str(getattr(tgt, "partname", ""))
        ext = os.path.splitext(partname)[1].lstrip(".").lower() or "png"
        images.append((data, ext))

    # en-têtes / pieds
    for section in doc.sections:
        if section.header is not None:
            for rid in _collect_rids(section.header._element.xml):
                rel = section.header.part.rels.get(rid)
                if rel and getattr(rel._target, "blob", None):
                    data = rel._target.blob
                    pname = str(getattr(rel._target, "partname", ""))
                    ext = os.path.splitext(pname)[1].lstrip(".").lower() or "png"
                    images.append((data, ext))
        if section.footer is not None:
            for rid in _collect_rids(section.footer._element.xml):
                rel = section.footer.part.rels.get(rid)
                if rel and getattr(rel._target, "blob", None):
                    data = rel._target.blob
                    pname = str(getattr(rel._target, "partname", ""))
                    ext = os.path.splitext(pname)[1].lstrip(".").lower() or "png"
                    images.append((data, ext))

    return images


def _collect_images_from_paths(paths: Iterable[str]) -> List[Tuple[bytes, str]]:
    """
    Accepte une liste de chemins vers .pdf, .docx, .jpg/.jpeg/.png.
    Retourne [(bytes, ext)] dans l'ordre des fichiers fournis.
    (API utilisée par figures_planner)
    """
    items: List[Tuple[bytes, str]] = []
    for p in (paths or []):
        if p is None:
            continue
        path = str(p)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Fichier introuvable: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            items.extend(_extract_images_from_pdf(path))
        elif ext == ".docx":
            items.extend(_extract_images_from_docx(path))
        elif ext in {".jpg", ".jpeg", ".png"}:
            with open(path, "rb") as f:
                items.append((f.read(), ext.lstrip(".")))
        else:
            continue
    return items


# ======================= Data URI / compression =======================

def _prepare_image_payloads(
    items: List[Tuple[bytes, str]],
    *,
    max_side: int = 1400,
    quality: int = 80,
    deduplicate: bool = True,
) -> List[Dict[str, object]]:
    """
    - Ouvre chaque image avec Pillow.
    - La convertit en JPEG RGB, max dimension max_side, qualité 80.
    - Crée une data URI.
    - Renvoie une liste {"uri": str, "tokens": int, "sha": str}.
    Images illisibles -> ignorées.
    """
    if Image is None:
        raise RuntimeError("Pillow (PIL) est requis pour description_img.")

    seen: set = set()
    payloads: List[Dict[str, object]] = []

    for data, _ext in items:
        try:
            im = Image.open(io.BytesIO(data))
            im = ImageOps.exif_transpose(im) if ImageOps else im
            im.load()
        except Exception:
            continue

        w, h = im.size
        longest = max(w, h)
        if longest > max_side:
            r = float(max_side) / float(longest)
            im = im.resize((int(w * r), max(1, int(h * r))), Image.LANCZOS)

        out = io.BytesIO()
        im.convert("RGB").save(
            out,
            format="JPEG",
            quality=quality,
            optimize=True,
            dpi=(96, 96),
        )
        jpeg_bytes = out.getvalue()

        sha = hashlib.sha256(jpeg_bytes).hexdigest()
        if deduplicate and sha in seen:
            continue
        seen.add(sha)

        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        uri = f"data:image/jpeg;base64,{b64}"
        approx_tokens = max(1, len(uri) // 4)

        payloads.append({"uri": uri, "tokens": approx_tokens, "sha": sha})

    return payloads


def _split_payloads_by_budget(
    payloads: List[Dict[str, object]],
    *,
    token_limit_per_call: int = 30000,
    prompt_overhead_tokens: int = 1000,
) -> List[List[Dict[str, object]]]:
    """
    Regroupe les payloads en veillant à ne pas dépasser token_limit_per_call.
    """
    if token_limit_per_call <= 0:
        raise ValueError("token_limit_per_call doit être > 0")
    budget_tokens = max(1, token_limit_per_call - max(0, prompt_overhead_tokens))

    groups: List[List[Dict[str, object]]] = []
    cur: List[Dict[str, object]] = []
    cur_tokens = 0

    for p in payloads:
        t = int(p["tokens"])
        next_tokens = cur_tokens + t

        if cur and next_tokens > budget_tokens:
            groups.append(cur)
            cur = []
            cur_tokens = 0

        cur.append(p)
        cur_tokens += t

    if cur:
        groups.append(cur)
    return groups


# ======================= Appel Azure OpenAI Vision =======================

_VISION_PROMPT_CACHE: Optional[str] = None


def _get_azure_client(
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
) -> AzureOpenAI:
    """
    Construit un client AzureOpenAI depuis paramètres ou variables d'env.
    """
    key = api_key or os.environ.get("AZURE_OPENAI_KEY")
    endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    ver = api_version or os.environ.get(
        "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
    )
    if not key or not endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_KEY et AZURE_OPENAI_ENDPOINT doivent être définis."
        )
    return AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=ver)


def _get_default_vision_prompt() -> str:
    """
    Charge et met en cache le prompt Vision par défaut depuis le Blob
    (vision_describe_images.txt dans le conteneur PROMPTS_CONTAINER_OTHERS).
    """
    global _VISION_PROMPT_CACHE
    if _VISION_PROMPT_CACHE is None:
        _VISION_PROMPT_CACHE = prompt_vision_describe_images()
    return _VISION_PROMPT_CACHE


def _call_azure_vision(
    client: AzureOpenAI,
    *,
    model: str,
    parts: List[dict],
    max_output_tokens: int = 500,
    temperature: float = 0.0,
) -> str:
    """
    Appelle Azure Vision une fois.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": parts}],
        max_tokens=max_output_tokens,
        temperature=temperature,
    )
    try:
        print("Tokens utilisés :", resp.usage)
    except Exception:
        pass

    out_lines: List[str] = []
    for ch in resp.choices:
        msg = getattr(ch, "message", None)
        if not msg:
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            out_lines.extend(content.split("\n"))
        elif isinstance(content, list):
            for c in content:
                if c.get("type") == "text":
                    out_lines.extend(str(c.get("text", "")).split("\n"))
    return "\n".join(out_lines).strip()


def _prepare_parts_from_payloads(
    payloads: List[Dict[str, object]],
    *,
    consigne: Optional[str],
    start_index: int = 1,
) -> List[dict]:
    """
    Construit un message multi-part pour l'API Vision.
    - consigne : prompt explicite si fourni,
    - sinon : prompt par défaut chargé depuis le Blob (vision_describe_images.txt).
    """
    parts: List[dict] = []

    if consigne and consigne.strip():
        _consigne = consigne.strip()
    else:
        _consigne = _get_default_vision_prompt()

    parts.append({"type": "text", "text": _consigne})
    idx = start_index
    for p in payloads:
        parts.append({"type": "text", "text": f"Image {idx}:"})
        parts.append({"type": "image_url", "image_url": {"url": str(p["uri"])}})
        idx += 1
    return parts


def _renumber_images(text: str) -> str:
    """
    Renumérote 'Image X' -> 'Image 1..n' pour corriger d'éventuels décalages.
    """
    lines = (text or "").split("\n")
    counter = 1
    out: List[str] = []
    pat = re.compile(r"^\s*Image\s+\d+\s*(:?.*)$", flags=re.IGNORECASE)
    for ln in lines:
        m = pat.match(ln)
        if m:
            suffix = re.sub(r"^\s*Image\s+\d+\s*", "", ln, flags=re.IGNORECASE)
            out.append(f"Image {counter}{suffix}")
            counter += 1
        else:
            out.append(ln)
    return "\n".join(out)


def summarise_with_azure(
    file_paths: Iterable[str],
    *,
    consigne: Optional[str] = None,
    model: Optional[str] = None,
    token_limit_per_call: int = 30000,
    prompt_overhead_tokens: int = 1000,
    max_output_tokens: int = 500,
    temperature: float = 0.0,
) -> str:
    """
    1) Extrait les images des fichiers,
    2) les compresse/encode en JPEG standard,
    3) groupe en lots sous budget de tokens,
    4) envoie chaque lot à Azure Vision,
    5) renvoie un texte concaténé "Image N: ...".
    """
    items = _collect_images_from_paths(file_paths)
    if not items:
        return ""

    payloads = _prepare_image_payloads(
        items,
        max_side=1400,
        quality=80,
        deduplicate=True,
    )

    if not payloads:
        return ""

    groups = _split_payloads_by_budget(
        payloads,
        token_limit_per_call=token_limit_per_call,
        prompt_overhead_tokens=prompt_overhead_tokens,
    )

    client = _get_azure_client()
    deployment = model or os.environ.get(
        "AZURE_OPENAI_VISION_DEPLOYMENT", "gpt-4.1"
    )

    results: List[str] = []
    start_idx = 1
    for gi, group in enumerate(groups, start=1):
        parts = _prepare_parts_from_payloads(
            group,
            consigne=consigne,
            start_index=start_idx,
        )
        text = _call_azure_vision(
            client,
            model=deployment,
            parts=parts,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        results.append(text)
        start_idx += len(group)
        print(f"[description_img] lot {gi}/{len(groups)} traité")

    final_text = "\n\n".join(s for s in results if s).strip()
    final_text = _renumber_images(final_text)
    return final_text


def main(doc) -> str:
    """
    Point d'entrée simple compatible avec figures_planner.
    `doc` peut être un chemin ou une liste de chemins (PDF/DOCX/JPG/PNG).
    Retourne un texte concaténé "Image N: ...".
    """
    if isinstance(doc, (list, tuple)):
        paths = [str(x) for x in doc]
    elif isinstance(doc, str):
        paths = [doc]
    else:
        paths = [str(doc)]

    return summarise_with_azure(
        paths,
        consigne=None,
        model=None,
        token_limit_per_call=30000,
        prompt_overhead_tokens=1000,
        max_output_tokens=500,
        temperature=0.0,
    )


__all__ = ["main", "_collect_images_from_paths"]
