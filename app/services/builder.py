# app/services/builder.py
import re
from typing import Tuple
from Core import document, embeddings, rag
from sklearn.neighbors import NearestNeighbors
import numpy as np

_ROUGE_TAG_RE = re.compile(r"\[\[ROUGE:.*?\]\]", re.DOTALL)


def _strip_rouge_tags(text: str) -> str:
    """Supprime les tags [[ROUGE: ...]] pour éviter la cascade dans les prompts suivants."""
    return _ROUGE_TAG_RE.sub("", text).strip()

RagPack = Tuple[object, list, list]


def build_rag_indexes(text_client: str, text_admin: str):
    chunks_client = [c for c in document.chunk_text(text_client) if c.strip()]
    index_client, vectors_client = embeddings.build_index(chunks_client) if chunks_client else (None, [])

    chunks_admin = [c for c in document.chunk_text(text_admin) if c.strip()]
    index_admin, vectors_admin = embeddings.build_index(chunks_admin) if chunks_admin else (None, [])

    mix = text_client + ("\n" + text_admin if text_admin else "")
    chunks_mix = [c for c in document.chunk_text(mix) if c.strip()]
    index_mix, vectors_mix = embeddings.build_index(chunks_mix) if chunks_mix else (None, [])

    return (
        (index_client, chunks_client, vectors_client),
        (index_admin, chunks_admin, vectors_admin),
        (index_mix, chunks_mix, vectors_mix),
    )


def build_sections_cir(
    index_client_pack: RagPack,
    index_mix_pack: RagPack,
    index_admin_pack: RagPack,
    index_articles_pack: RagPack,
    objectif,        # peut venir du front (optionnel)
    verrou,          # peut venir du front (optionnel)
    annee,
    societe,
    site_web: str,
    articles,
    doc_complete: bool,
    externalises: bool,
):
    import time
    total_start = time.time()
    print(f"[CIR-BUILD] Debut generation des sections CIR pour {societe} ({annee})")

    (index_client, chunks_client, vectors_client) = index_client_pack
    (index_mix, chunks_mix, vectors_mix) = index_mix_pack
    (index_adm, chunks_adm, vectors_adm) = index_admin_pack

    # Pack "base" (utile pour objectif/verrou): docs client + docs admin + articles
    base_pack = merge_rag_packs(index_client_pack, index_admin_pack, index_articles_pack)
    (idx_base, chunks_base, vecs_base) = base_pack

    # Pack "objet": docs client + articles (comme vous aviez)
    idx_obj, chunks_obj, vecs_obj = merge_rag_packs(index_client_pack, index_articles_pack)

    # 1) OBJECTIF UNIQUE (canonique)
    print("[CIR-BUILD] 1/12 Generation objectif_unique...")
    step_start = time.time()
    objectif_unique = (objectif or "").strip()
    if not objectif_unique:
        objectif_unique = rag.generate_objectif_unique(
            idx_base, chunks_base, vecs_base,
            annee=annee,
            societe=societe,
            articles=articles,
        )
    print(f"[CIR-BUILD] 1/12 objectif_unique OK en {time.time() - step_start:.1f}s")

    # Nettoyer les tags [[ROUGE:...]] de l'objectif pour le verrou et les sections suivantes
    obj_clean = _strip_rouge_tags(objectif_unique)

    # 2) VERROU UNIQUE (canonique) - utilise l'objectif nettoyé
    print("[CIR-BUILD] 2/12 Generation verrou_unique...")
    step_start = time.time()
    verrou_unique = (verrou or "").strip()
    if not verrou_unique:
        verrou_unique = rag.generate_verrou_unique(
            idx_base, chunks_base, vecs_base,
            objectif_unique=obj_clean,
            annee=annee,
            societe=societe,
            articles=articles,
        )
    print(f"[CIR-BUILD] 2/12 verrou_unique OK en {time.time() - step_start:.1f}s")

    # Nettoyer les tags [[ROUGE:...]] du verrou aussi
    # Les versions originales (avec ROUGE) restent dans le dict final pour le template
    ver_clean = _strip_rouge_tags(verrou_unique)

    # 3) OBJET (section détaillée) – ancrée par objectif_unique/verrou_unique
    print("[CIR-BUILD] 3/12 Generation section objet...")
    step_start = time.time()
    objet = rag.generate_objectifs_section(
        idx_obj, chunks_obj, vecs_obj,
        objectif_unique=obj_clean,
        verrou_unique=ver_clean,
        annee=annee,
        societe=societe,
        articles=articles,
    )
    print(f"[CIR-BUILD] 3/12 objet OK en {time.time() - step_start:.1f}s")

    # 4) SECTION VERROU (description longue) – MAIS elle doit reprendre la question canonique à l'identique
    print("[CIR-BUILD] 4/12 Generation section verrou...")
    step_start = time.time()
    section_verrou = rag.generate_verrou_section(
        idx_obj, chunks_obj, vecs_obj,
        objectif_unique=obj_clean,
        verrou_unique=ver_clean,
        annee=annee,
        societe=societe,
        articles=articles,
    )
    print(f"[CIR-BUILD] 4/12 verrou OK en {time.time() - step_start:.1f}s")

    # 5) Les autres sections utilisent les versions nettoyées (sans tags ROUGE)
    print("[CIR-BUILD] 5/12 Generation section contexte...")
    step_start = time.time()
    contexte = rag.generate_contexte_section(
        idx_obj, chunks_obj, vecs_obj,
        objectif_unique=obj_clean,
        verrou_unique=ver_clean,
        annee=annee,
        societe=societe,
        articles=articles,
    )
    print(f"[CIR-BUILD] 5/12 contexte OK en {time.time() - step_start:.1f}s")

    print("[CIR-BUILD] 6/12 Generation section indicateurs...")
    step_start = time.time()
    indicateurs = rag.generate_indicateurs_section(
        index_mix, chunks_mix, vectors_mix,
        objectif_unique=obj_clean,
        verrou_unique=ver_clean,
        annee=annee,
        societe=societe,
    )
    print(f"[CIR-BUILD] 6/12 indicateurs OK en {time.time() - step_start:.1f}s")

    print("[CIR-BUILD] 7/12 Generation section travaux...")
    step_start = time.time()
    travaux = rag.generate_travaux_section(
        index_client, chunks_client, vectors_client,
        objectif_unique=obj_clean,
        verrou_unique=ver_clean,
        annee=annee,
        societe=societe,
    )
    if not doc_complete:
        print("[CIR-BUILD] 7/12 Evaluation travaux (doc incomplete)...")
        travaux = rag.evaluateur_travaux(travaux)
        travaux = rag.wrap_questions_rouge(travaux)
    print(f"[CIR-BUILD] 7/12 travaux OK en {time.time() - step_start:.1f}s")

    print("[CIR-BUILD] 8/12 Generation section contribution...")
    step_start = time.time()
    contribution = rag.generate_contribution_section(
        index_client, chunks_client, vectors_client,
        objectif_unique=obj_clean,
        verrou_unique=ver_clean,
        annee=annee,
        societe=societe,
    )
    print(f"[CIR-BUILD] 8/12 contribution OK en {time.time() - step_start:.1f}s")

    print("[CIR-BUILD] 9/12 Generation section biblio...")
    step_start = time.time()
    biblio = rag.generate_biblio_section(articles)
    print(f"[CIR-BUILD] 9/12 biblio OK en {time.time() - step_start:.1f}s")

    print("[CIR-BUILD] 10/12 Generation section partenariat...")
    step_start = time.time()
    partenariat = "Rien à déclarer." if not externalises else rag.generate_partenariat_section(
        index_mix, chunks_mix, vectors_mix,
        objectif_unique=obj_clean,
        verrou_unique=ver_clean,
        annee=annee,
        societe=societe,
    )
    print(f"[CIR-BUILD] 10/12 partenariat OK en {time.time() - step_start:.1f}s")

    # entreprise: priorité docs admin
    ent_index  = index_adm or index_mix
    ent_chunks = chunks_adm or chunks_mix
    ent_vecs   = vectors_adm or vectors_mix

    print("[CIR-BUILD] 11/12 Generation section entreprise...")
    step_start = time.time()
    entreprise = rag.generate_entreprise_section(
        ent_index, ent_chunks, ent_vecs,
        objectif_unique=obj_clean,
        verrou_unique=ver_clean,
        annee=annee,
        societe=societe,
        style=None,
        site_web=site_web,
    )
    print(f"[CIR-BUILD] 11/12 entreprise OK en {time.time() - step_start:.1f}s")

    print("[CIR-BUILD] 12/12 Generation section gestion...")
    step_start = time.time()
    gestion = rag.generate_gestion_recherche_section(
        index_mix, chunks_mix, vectors_mix,
        objectif_unique=obj_clean,
        verrou_unique=ver_clean,
        annee=annee,
        societe=societe,
    )
    print(f"[CIR-BUILD] 12/12 gestion OK en {time.time() - step_start:.1f}s")

    # Générer le résumé en se basant sur les sections déjà générées
    print("[CIR-BUILD] FINAL Generation resume...")
    step_start = time.time()
    resume = rag.generate_resume_from_sections(
        sections={
            "entreprise": entreprise,
            "contexte": contexte,
            "objectifs": objet,
            "verrous": section_verrou,
            "travaux": travaux,
            "contribution": contribution,
            "indicateurs": indicateurs,
            "partenariat": partenariat,
            "gestion": gestion,
        },
        objectif_unique=obj_clean,
        verrou_unique=ver_clean,
        annee=annee,
        societe=societe,
        articles=articles,
    )
    print(f"[CIR-BUILD] FINAL resume OK en {time.time() - step_start:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"[CIR-BUILD] TERMINE - Toutes les sections generees en {total_elapsed:.1f}s")

    return {
        # variables canoniques (à injecter partout, et éventuellement dans le template)
        "objectif_unique": objectif_unique,
        "verrou_unique": verrou_unique,

        # sections
        "objet": objet,
        "verrou": section_verrou,  # section longue "description du verrou"
        "contexte": contexte,
        "indicateurs": indicateurs,
        "travaux": travaux,
        "contribution": contribution,
        "biblio": biblio,
        "partenariat": partenariat,
        "entreprise": entreprise,
        "gestion": gestion,
        "resume": resume,
    }


def merge_rag_packs(*packs: RagPack) -> RagPack:
    all_chunks: list[str] = []
    all_vecs: list[np.ndarray] = []

    for (idx, chunks, vecs) in packs:
        if not vecs:
            continue
        all_chunks.extend(chunks)
        all_vecs.extend(vecs)

    if not all_vecs:
        return (None, [], [])

    mat = np.stack(all_vecs, axis=0)
    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn.fit(mat)
    return (nn, all_chunks, all_vecs)
