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
    api_version=EMBEDDING_API_VERSION,
    max_retries=3,  # Retry auto sur erreurs 429 (rate limit)
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

def build_index(chunks):
    vecs = embed_texts(chunks)
    if not vecs: return None, []
    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn.fit(vecs)
    return nn, vecs
