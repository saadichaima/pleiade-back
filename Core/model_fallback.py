# Core/model_fallback.py
"""
Utilitaire de fallback multi-modèles pour Azure OpenAI.

Construit une liste ordonnée de configurations (primaire + fallbacks) pour :
- les modèles GPT (chat/completions/responses)
- les modèles d'embedding

Variables d'environnement reconnues :

GPT primaire :
  AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION

GPT fallback(s) :
  AZURE_OPENAI_FALLBACK_KEY, AZURE_OPENAI_FALLBACK_ENDPOINT,
  AZURE_OPENAI_FALLBACK_DEPLOYMENT, AZURE_OPENAI_FALLBACK_API_VERSION
  (pour un 2e fallback : AZURE_OPENAI_FALLBACK_2_KEY, ..._FALLBACK_2_ENDPOINT, etc.)

Embedding primaire :
  AZURE_OPENAI_EMBEDDING_KEY, AZURE_OPENAI_EMBEDDING_ENDPOINT,
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT, AZURE_OPENAI_EMBEDDING_API_VERSION

Embedding fallback(s) :
  AZURE_OPENAI_EMBEDDING_FALLBACK_KEY, AZURE_OPENAI_EMBEDDING_FALLBACK_ENDPOINT,
  AZURE_OPENAI_EMBEDDING_FALLBACK_DEPLOYMENT, AZURE_OPENAI_EMBEDDING_FALLBACK_API_VERSION
  (pour un 2e fallback : AZURE_OPENAI_EMBEDDING_FALLBACK_2_KEY, etc.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from openai import AzureOpenAI, OpenAI


# ---------------------------------------------------------------------------
# Dataclasses de configuration
# ---------------------------------------------------------------------------

@dataclass
class GptModelConfig:
    """Configuration d'un modèle GPT (primaire ou fallback)."""
    name: str                  # "primary", "fallback_1", "fallback_2"…
    deployment: str            # Nom du déploiement Azure
    use_responses_api: bool    # True = Responses API (GPT-5+), False = Chat Completions
    client: object = field(default=None, repr=False)  # OpenAI ou AzureOpenAI


@dataclass
class EmbeddingModelConfig:
    """Configuration d'un modèle d'embedding (primaire ou fallback)."""
    name: str              # "primary", "fallback_1"…
    deployment: str        # Nom du déploiement Azure
    client: AzureOpenAI = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Fonctions de construction des clients
# ---------------------------------------------------------------------------

def _build_gpt_client(key: str, endpoint: str, api_version: str,
                      use_responses_api: bool, timeout: float) -> object:
    """Crée le client OpenAI approprié selon le type d'API."""
    if use_responses_api:
        base_url = endpoint.rstrip("/") + "/openai/v1/"
        return OpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout,
            max_retries=2,  # Moins de retries car on a un fallback
        )
    else:
        return AzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=api_version,
            timeout=timeout,
            max_retries=2,
        )


def _build_embedding_client(key: str, endpoint: str, api_version: str) -> AzureOpenAI:
    """Crée le client AzureOpenAI pour les embeddings."""
    return AzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=api_version,
        max_retries=2,
    )


def _is_responses_api(api_version: str) -> bool:
    return "2025" in api_version


# ---------------------------------------------------------------------------
# Lecture des variables d'environnement et construction des listes
# ---------------------------------------------------------------------------

def build_gpt_configs(timeout: float) -> List[GptModelConfig]:
    """
    Construit la liste ordonnée des configs GPT (primaire + fallbacks).

    Conventions de nommage dans .env :
      - Primaire  : AZURE_OPENAI_KEY / ENDPOINT / DEPLOYMENT / API_VERSION
      - Fallback 1: AZURE_OPENAI_FALLBACK_KEY / FALLBACK_ENDPOINT / ...
      - Fallback 2: AZURE_OPENAI_FALLBACK_2_KEY / FALLBACK_2_ENDPOINT / ...
      - Fallback N: AZURE_OPENAI_FALLBACK_N_KEY / FALLBACK_N_ENDPOINT / ...
    """
    configs: List[GptModelConfig] = []

    # --- Modèle primaire ---
    pk = os.getenv("AZURE_OPENAI_KEY", "").strip()
    pe = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    pd = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    pv = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()

    if pk and pe and pd:
        use_resp = _is_responses_api(pv)
        cfg = GptModelConfig(name="primary", deployment=pd, use_responses_api=use_resp)
        cfg.client = _build_gpt_client(pk, pe, pv, use_resp, timeout)
        configs.append(cfg)
        print(f"[model_fallback] GPT primary: deployment={pd!r}, responses_api={use_resp}")
    else:
        print("[model_fallback] AVERTISSEMENT: Aucune configuration GPT primaire trouvée.")

    # --- Fallbacks ---
    i = 1
    while True:
        suffix = "FALLBACK" if i == 1 else f"FALLBACK_{i}"
        key = os.getenv(f"AZURE_OPENAI_{suffix}_KEY", "").strip()
        endpoint = os.getenv(f"AZURE_OPENAI_{suffix}_ENDPOINT", "").strip()
        deployment = os.getenv(f"AZURE_OPENAI_{suffix}_DEPLOYMENT", "").strip()
        api_version = os.getenv(f"AZURE_OPENAI_{suffix}_API_VERSION", "").strip()

        if not (key and endpoint and deployment):
            break  # Plus de fallbacks définis

        use_resp = _is_responses_api(api_version)
        name = f"fallback_{i}"
        cfg = GptModelConfig(name=name, deployment=deployment, use_responses_api=use_resp)
        cfg.client = _build_gpt_client(key, endpoint, api_version, use_resp, timeout)
        configs.append(cfg)
        print(f"[model_fallback] GPT {name}: deployment={deployment!r}, responses_api={use_resp}")
        i += 1

    if len(configs) > 1:
        print(f"[model_fallback] {len(configs)} modèles GPT configurés (1 primaire + {len(configs)-1} fallback(s))")
    return configs


def build_embedding_configs() -> List[EmbeddingModelConfig]:
    """
    Construit la liste ordonnée des configs embedding (primaire + fallbacks).

    Conventions :
      - Primaire  : AZURE_OPENAI_EMBEDDING_KEY / EMBEDDING_ENDPOINT / EMBEDDING_DEPLOYMENT / EMBEDDING_API_VERSION
      - Fallback 1: AZURE_OPENAI_EMBEDDING_FALLBACK_KEY / EMBEDDING_FALLBACK_ENDPOINT / ...
      - Fallback N: AZURE_OPENAI_EMBEDDING_FALLBACK_N_KEY / ...

    Si AZURE_OPENAI_EMBEDDING_KEY est absent, utilise AZURE_OPENAI_KEY comme clé par défaut.
    """
    configs: List[EmbeddingModelConfig] = []

    # --- Modèle primaire ---
    pk = (os.getenv("AZURE_OPENAI_EMBEDDING_KEY") or os.getenv("AZURE_OPENAI_KEY", "")).strip()
    pe = (os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT", "")).strip()
    pd = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()
    pv = (os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION", "")).strip()

    if pk and pe and pd:
        cfg = EmbeddingModelConfig(name="primary", deployment=pd)
        cfg.client = _build_embedding_client(pk, pe, pv)
        configs.append(cfg)
        print(f"[model_fallback] Embedding primary: deployment={pd!r}")
    else:
        print("[model_fallback] AVERTISSEMENT: Aucune configuration embedding primaire trouvée.")

    # --- Fallbacks ---
    i = 1
    while True:
        suffix = "FALLBACK" if i == 1 else f"FALLBACK_{i}"
        key = (
            os.getenv(f"AZURE_OPENAI_EMBEDDING_{suffix}_KEY")
            or os.getenv(f"AZURE_OPENAI_{suffix}_KEY", "")
        ).strip()
        endpoint = (
            os.getenv(f"AZURE_OPENAI_EMBEDDING_{suffix}_ENDPOINT")
            or os.getenv(f"AZURE_OPENAI_{suffix}_ENDPOINT", "")
        ).strip()
        deployment = os.getenv(f"AZURE_OPENAI_EMBEDDING_{suffix}_DEPLOYMENT", "").strip()
        api_version = (
            os.getenv(f"AZURE_OPENAI_EMBEDDING_{suffix}_API_VERSION")
            or os.getenv(f"AZURE_OPENAI_{suffix}_API_VERSION", "")
        ).strip()

        if not (key and endpoint and deployment):
            break

        name = f"fallback_{i}"
        cfg = EmbeddingModelConfig(name=name, deployment=deployment)
        cfg.client = _build_embedding_client(key, endpoint, api_version)
        configs.append(cfg)
        print(f"[model_fallback] Embedding {name}: deployment={deployment!r}")
        i += 1

    if len(configs) > 1:
        print(f"[model_fallback] {len(configs)} modèles embedding configurés (1 primaire + {len(configs)-1} fallback(s))")
    return configs
