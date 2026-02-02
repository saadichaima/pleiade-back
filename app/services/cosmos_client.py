# app/services/cosmos_client.py
from azure.cosmos import CosmosClient
from app.config import settings

_cosmos_client: CosmosClient | None = None

def get_cosmos_client() -> CosmosClient:
    global _cosmos_client
    if _cosmos_client is None:
        if not (settings.COSMOS_URI and settings.COSMOS_KEY):
            raise RuntimeError("COSMOS_URI ou COSMOS_KEY non configuré.")
        _cosmos_client = CosmosClient(settings.COSMOS_URI, credential=settings.COSMOS_KEY)
    return _cosmos_client

def get_db():
    client = get_cosmos_client()
    return client.get_database_client(settings.COSMOS_DB_NAME)

def get_projects_container():
    db = get_db()
    return db.get_container_client(settings.COSMOS_CONTAINER_PROJECTS)

def get_documents_container():
    db = get_db()
    return db.get_container_client(settings.COSMOS_CONTAINER_DOCUMENTS)

def get_outputs_container():
    db = get_db()
    return db.get_container_client(settings.COSMOS_CONTAINER_OUTPUTS)
def get_users_container():
    db = get_db()
    return db.get_container_client(settings.COSMOS_CONTAINER_USERS)


def get_jobs_container():
    """Conteneur pour la persistance des jobs de génération."""
    db = get_db()
    return db.get_container_client(settings.COSMOS_CONTAINER_JOBS)
