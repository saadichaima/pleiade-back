# Core/embeddings.py
import os, numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI, RateLimitError
from sklearn.neighbors import NearestNeighbors

load_dotenv()
EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_KEY") or os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=EMBEDDING_API_VERSION
)
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

def embed_texts(texts):
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    try:
        r = client.embeddings.create(input=texts, model=EMBEDDING_DEPLOYMENT)
    except RateLimitError as e:
        # 🔁 Quand Azure renvoie 429, on désactive le RAG pour cette requête
        print(f"[embeddings] Rate limit Azure OpenAI, RAG désactivé pour cette génération: {e}")
        return []
    except Exception as e:
        # pour ne pas casser la génération sur d'autres erreurs
        print(f"[embeddings] Erreur embeddings, RAG désactivé: {e}")
        return []

    return [np.array(e.embedding, dtype=np.float32) for e in r.data]

def embed_texts_batch(texts, batch_size=100, delay_between=0):
    """
    Embed une grande liste de textes par batches avec retry sur rate limit.
    delay_between: délai en secondes entre chaque batch (utile pour S0 tier).
    Retourne la liste complète de vecteurs numpy, ou [] si échec total.
    """
    import time
    all_vectors = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"[embeddings] Batch {batch_num}/{total_batches} ({len(batch)} textes)...")

        if i > 0 and delay_between > 0:
            time.sleep(delay_between)

        max_retries = 3
        for attempt in range(max_retries):
            vecs = embed_texts(batch)
            if vecs:
                all_vectors.extend(vecs)
                break
            elif attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"[embeddings] Retry dans {wait}s (tentative {attempt + 2}/{max_retries})...")
                time.sleep(wait)
            else:
                print(f"[embeddings] Échec batch {batch_num} après {max_retries} tentatives")
                return []

    print(f"[embeddings] {len(all_vectors)}/{total} textes embeddés avec succès")
    return all_vectors


def build_index(chunks):
    vecs = embed_texts(chunks)
    if not vecs: return None, []
    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn.fit(vecs)
    return nn, vecs
