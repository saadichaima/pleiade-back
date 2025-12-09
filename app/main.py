# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import health, keywords, articles, generate, cii, auth_ms, history
from fastapi.openapi.utils import get_openapi
import json, os
from app.routers import auth_ms, history


openapi_tags = [
    {"name": "nlp", "description": "Extraction de mots-clés et recherche d’articles (Serper/Scholar)."},
    {"name": "cii", "description": "Analyses concurrentielles CII (prompt + LLM)."},
    {"name": "generate", "description": "Génération de documents DOCX (CIR/CII)."},
]

app = FastAPI(
    title="Pleiades API",
    version="1.0.0",
    description="API interne pour CIR/CII : RAG, génération DOCX, recherche biblio.",
    contact={"name": "Équipe Pleiades", "email": "support@exemple.com"},
    license_info={"name": "Propriétaire"},
    openapi_tags=openapi_tags,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://zealous-sea-0b0962f03.3.azurestaticapps.net",

    ],
    allow_credentials=False,   # important si tu utilises "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(keywords.router, prefix="/nlp", tags=["nlp"])
app.include_router(articles.router, prefix="/nlp", tags=["nlp"])
app.include_router(cii.router,      prefix="/cii", tags=["cii"])
app.include_router(generate.router, prefix="/generate", tags=["generate"])
app.include_router(auth_ms.router,  tags=["auth"])
app.include_router(history.router,  tags=["history"])

# debug: afficher les routes au démarrage
@app.on_event("startup")
async def _show_routes():
    for r in app.routes:
        methods = ",".join(getattr(r, "methods", []) or [])
        print(f"ROUTE: {methods:10s} {r.path}")
async def _dump_openapi():
    schema = get_openapi(
        title=app.title, version=app.version,
        description=app.description, routes=app.routes
    )
    os.makedirs("Doc", exist_ok=True)
    with open("Doc/openapi.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print("OpenAPI écrit dans Doc/openapi.json")