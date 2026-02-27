# Core/embeddings.py
import os
import numpy as np
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors

load_dotenv()

from Core.model_fallback import build_embedding_configs, EmbeddingModelConfig
from typing import List

# Liste ordonnée des configs embedding : [primaire, fallback_1, ...]
EMBEDDING_CONFIGS: List[EmbeddingModelConfig] = build_embedding_configs()


def embed_texts(texts):
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []

    if not EMBEDDING_CONFIGS:
        print("[embeddings] Aucun modèle embedding configuré, RAG désactivé.")
        return []

    last_exc = None
    for i, cfg in enumerate(EMBEDDING_CONFIGS):
        try:
            r = cfg.client.embeddings.create(input=texts, model=cfg.deployment)
            if i > 0:
                print(f"[embeddings] Succès via modèle backup '{cfg.name}'")
            return [np.array(e.embedding, dtype=np.float32) for e in r.data]
        except Exception as e:
            last_exc = e
            if i < len(EMBEDDING_CONFIGS) - 1:
                print(f"[embeddings] Modèle '{cfg.name}' échoué ({type(e).__name__}: {e}), tentative sur fallback...")
            else:
                print(f"[embeddings] Tous les modèles échoués, RAG désactivé: {type(e).__name__}: {e}")

    return []

def build_index(chunks):
    vecs = embed_texts(chunks)
    if not vecs: return None, []
    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn.fit(vecs)
    return nn, vecs
