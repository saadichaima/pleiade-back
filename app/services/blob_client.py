# app/services/blob_client.py
from typing import Optional
from azure.storage.blob import BlobServiceClient, ContentSettings
from app.config import settings

_blob_service_client: BlobServiceClient | None = None

def get_blob_service_client() -> BlobServiceClient:
    global _blob_service_client
    if _blob_service_client is not None:
        return _blob_service_client

    if settings.AZURE_STORAGE_CONNECTION_STRING:
        _blob_service_client = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
        return _blob_service_client

    if settings.AZURE_STORAGE_ACCOUNT_URL and settings.AZURE_STORAGE_ACCOUNT_KEY:
        _blob_service_client = BlobServiceClient(
            account_url=settings.AZURE_STORAGE_ACCOUNT_URL,
            credential=settings.AZURE_STORAGE_ACCOUNT_KEY,
        )
        return _blob_service_client

    raise RuntimeError(
        "Configuration Azure Blob incomplète. "
        "Définis AZURE_STORAGE_CONNECTION_STRING ou AZURE_STORAGE_ACCOUNT_URL + AZURE_STORAGE_ACCOUNT_KEY."
    )


def upload_bytes_to_blob(
    container_name: str,
    blob_name: str,
    data: bytes,
    content_type: Optional[str] = None,
) -> str:
    """
    Upload de bytes dans un conteneur Blob.
    Retourne l'URL publique (ou accessible via ton compte selon la config).
    """
    if not data:
        raise ValueError("Aucune donnée à uploader.")

    service = get_blob_service_client()
    container = service.get_container_client(container_name)

    # On crée le conteneur si nécessaire (idempotent)
    try:
        container.create_container()
    except Exception:
        pass  # existe déjà

    content_settings = ContentSettings(content_type=content_type or "application/octet-stream")

    blob_client = container.get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=True, content_settings=content_settings)

    return blob_client.url
