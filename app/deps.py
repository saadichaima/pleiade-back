# app/deps.py
from fastapi import Depends, HTTPException
from app.config import settings

def ensure_openai():
    if not (settings.AZURE_OPENAI_KEY and settings.AZURE_OPENAI_ENDPOINT and settings.AZURE_OPENAI_DEPLOYMENT):
        raise HTTPException(status_code=500, detail="Azure OpenAI non configuré.")
    return True

def ensure_serper():
    if not settings.SERPER_API_KEY:
        raise HTTPException(status_code=500, detail="SERPER_API_KEY manquante.")
    return True
