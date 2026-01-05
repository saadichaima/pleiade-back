# app/services/builder.py
from typing import Tuple
from Core import document, embeddings, rag
from sklearn.neighbors import NearestNeighbors
import numpy as np

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
    (index_client, chunks_client, vectors_client) = index_client_pack
    (index_mix, chunks_mix, vectors_mix) = index_mix_pack
    (index_adm, chunks_adm, vectors_adm) = index_admin_pack

    # Pack "base" (utile pour objectif/verrou): docs client + docs admin + articles
    base_pack = merge_rag_packs(index_client_pack, index_admin_pack, index_articles_pack)
    (idx_base, chunks_base, vecs_base) = base_pack

    # Pack "objet": docs client + articles (comme vous aviez)
    idx_obj, chunks_obj, vecs_obj = merge_rag_packs(index_client_pack, index_articles_pack)

    # 1) OBJECTIF UNIQUE (canonique)
    objectif_unique = (objectif or "").strip()
    if not objectif_unique:
        objectif_unique = rag.generate_objectif_unique(
            idx_base, chunks_base, vecs_base,
            annee=annee,
            societe=societe,
            articles=articles,
        )

    # 2) VERROU UNIQUE (canonique)
    verrou_unique = (verrou or "").strip()
    if not verrou_unique:
        verrou_unique = rag.generate_verrou_unique(
            idx_base, chunks_base, vecs_base,
            objectif_unique=objectif_unique,
            annee=annee,
            societe=societe,
            articles=articles,
        )

    # 3) OBJET (section détaillée) – ancrée par objectif_unique/verrou_unique
    objet = rag.generate_objectifs_section(
        idx_obj, chunks_obj, vecs_obj,
        objectif_unique=objectif_unique,
        verrou_unique=verrou_unique,
        annee=annee,
        societe=societe,
        articles=articles,
    )

    # 4) SECTION VERROU (description longue) – MAIS elle doit reprendre la question canonique à l'identique
    section_verrou = rag.generate_verrou_section(
        idx_obj, chunks_obj, vecs_obj,
        objectif_unique=objectif_unique,
        verrou_unique=verrou_unique,
        annee=annee,
        societe=societe,
        articles=articles,
    )

    # 5) Les autres sections utilisent exactement objectif_unique + verrou_unique
    contexte = rag.generate_contexte_section(
        idx_obj, chunks_obj, vecs_obj,
        objectif_unique=objectif_unique,
        verrou_unique=verrou_unique,
        annee=annee,
        societe=societe,
        articles=articles,
    )

    indicateurs = rag.generate_indicateurs_section(
        index_mix, chunks_mix, vectors_mix,
        objectif_unique=objectif_unique,
        verrou_unique=verrou_unique,
        annee=annee,
        societe=societe,
    )

    travaux = rag.generate_travaux_section(
        index_client, chunks_client, vectors_client,
        objectif_unique=objectif_unique,
        verrou_unique=verrou_unique,
        annee=annee,
        societe=societe,
    )
    if not doc_complete:
        travaux = rag.evaluateur_travaux(travaux)

    contribution = rag.generate_contribution_section(
        index_client, chunks_client, vectors_client,
        objectif_unique=objectif_unique,
        verrou_unique=verrou_unique,
        annee=annee,
        societe=societe,
    )

    biblio = rag.generate_biblio_section(articles)

    partenariat = "Rien à déclarer." if not externalises else rag.generate_partenariat_section(
        index_mix, chunks_mix, vectors_mix,
        objectif_unique=objectif_unique,
        verrou_unique=verrou_unique,
        annee=annee,
        societe=societe,
    )

    # entreprise: priorité docs admin
    ent_index  = index_adm or index_mix
    ent_chunks = chunks_adm or chunks_mix
    ent_vecs   = vectors_adm or vectors_mix

    entreprise = rag.generate_entreprise_section(
        ent_index, ent_chunks, ent_vecs,
        objectif_unique=objectif_unique,
        verrou_unique=verrou_unique,
        annee=annee,
        societe=societe,
        style=None,
        site_web=site_web,
    )

    gestion = rag.generate_gestion_recherche_section(
        index_mix, chunks_mix, vectors_mix,
        objectif_unique=objectif_unique,
        verrou_unique=verrou_unique,
        annee=annee,
        societe=societe,
    )

    resume = rag.generate_resume_section(
        index_mix, chunks_mix, vectors_mix,
        objectif_unique=objectif_unique,
        verrou_unique=verrou_unique,
        annee=annee,
        societe=societe,
        articles=articles,
    )

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
