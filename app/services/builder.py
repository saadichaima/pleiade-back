# app/services/builder.py
from typing import List, Optional, Dict, Any
from Core import document, embeddings, rag, writer, writer_tpl
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
import numpy as np
RagPack = Tuple[object, list, list]
def build_rag_indexes(text_client: str, text_admin: str):
    # Index dédié documents techniques (docs_client)
    chunks_client = [c for c in document.chunk_text(text_client) if c.strip()]
    index_client, vectors_client = embeddings.build_index(chunks_client) if chunks_client else (None, [])

    # Index dédié documents administratifs (docs_admin)
    chunks_admin = [c for c in document.chunk_text(text_admin) if c.strip()]
    index_admin, vectors_admin = embeddings.build_index(chunks_admin) if chunks_admin else (None, [])

    # Index mixé client + admin (pour les sections globales)
    mix = text_client + ("\n" + text_admin if text_admin else "")
    chunks_mix = [c for c in document.chunk_text(mix) if c.strip()]
    index_mix, vectors_mix = embeddings.build_index(chunks_mix) if chunks_mix else (None, [])

    # On renvoie les 3 packs
    return (
        (index_client, chunks_client, vectors_client),
        (index_admin, chunks_admin, vectors_admin),
        (index_mix,   chunks_mix,   vectors_mix),
    )

# app/services/builder.py (suite)

def build_sections_cir(
    index_client_pack:RagPack,
    index_mix_pack:RagPack,
    index_admin_pack:RagPack,
    index_articles_pack: RagPack,
    objectif,
    verrou,
    annee,
    societe,
    site_web: str, 
    articles,
    doc_complete: bool,
    externalises: bool,
):
    (index,     chunks,     vectors)     = index_client_pack
    (index_mix, chunks_mix, vectors_mix) = index_mix_pack
    (index_adm, chunks_adm, vectors_adm) = index_admin_pack
    (index_art, chunks_art, vectors_art) = index_articles_pack
    pack_objet = merge_rag_packs(index_client_pack, index_articles_pack)
    (idx_obj, chunks_obj, vecs_obj) = pack_objet


    objet = rag.generate_objectifs_section(idx_obj, chunks_obj, vecs_obj, objectif, "", annee, societe, articles)
    verrou = rag.generate_verrou_section(index, chunks, vectors, objet, "", annee, societe)

    contexte     = rag.generate_contexte_section(idx_obj, chunks_obj, vecs_obj, objectif, verrou, annee, societe)
    indicateurs  = rag.generate_indicateurs_section(index_mix, chunks_mix, vectors_mix, objectif, verrou, annee, societe)
    travaux      = rag.generate_travaux_section(index, chunks, vectors, objectif, verrou, annee, societe)
    if not doc_complete:
        travaux = rag.evaluateur_travaux(travaux)
    contribution = rag.generate_contribution_section(index, chunks, vectors, objectif, verrou, annee, societe)
    biblio       = rag.generate_biblio_section(articles)
    partenariat  = "Rien à déclarer." if not externalises else rag.generate_partenariat_section(index_mix, chunks_mix, vectors_mix, objectif, verrou, annee, societe)

    # 🆕 L’entreprise basée en priorité sur les docs administratifs
    ent_index  = index_adm or index_mix
    ent_chunks = chunks_adm or chunks_mix
    ent_vecs   = vectors_adm or vectors_mix

    entreprise   = rag.generate_entreprise_section(ent_index, ent_chunks, ent_vecs, objectif, verrou, annee, societe, style=None, site_web=site_web )
    gestion      = rag.generate_gestion_recherche_section(index_mix, chunks_mix, vectors_mix, objectif, verrou, annee, societe)
    resume       = rag.generate_resume_section(index_mix, chunks_mix, vectors_mix, objectif, verrou, annee, societe)

    return {
        "objet": objet,
        "verrou": verrou,
        "contexte": contexte,
        "indicateurs": indicateurs,
        "travaux": travaux,
        "contribution": contribution,
        "biblio": biblio,
        "partenariat": partenariat,
        "entreprise": entreprise,   # 👉 utilisé dans d.entreprise dans generate_docx
        "gestion": gestion,
        "resume": resume,
    }



def merge_rag_packs(*packs: RagPack) -> RagPack:
    """
    Combine plusieurs packs RAG (index, chunks, vectors) en un seul :
    - concatène les chunks,
    - concatène les vecteurs,
    - reconstruit un NearestNeighbors global.

    Si aucun vecteur, renvoie (None, [], []).
    """
    all_chunks: list[str] = []
    all_vecs: list[np.ndarray] = []

    for (idx, chunks, vecs) in packs:
        if not vecs:
            continue
        all_chunks.extend(chunks)
        all_vecs.extend(vecs)

    if not all_vecs:
        return (None, [], [])

    # on empile les vecteurs en un seul array
    mat = np.stack(all_vecs, axis=0)
    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn.fit(mat)
    return (nn, all_chunks, all_vecs)
