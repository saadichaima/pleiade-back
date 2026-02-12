# app/services/prompts_admin_service.py
"""
Service pour gérer les prompts stockés sur Azure Blob Storage.
Permet de lister, récupérer et mettre à jour les prompts pour l'admin.
"""
from typing import List, Optional
from datetime import datetime
from azure.storage.blob import BlobServiceClient, BlobProperties, ContentSettings
from azure.core.exceptions import ResourceNotFoundError
from app.config import settings
from app.services.blob_client import get_blob_service_client


# Mapping des types de dossiers vers leurs conteneurs
TYPE_TO_CONTAINER = {
    "cir": lambda: settings.PROMPTS_CONTAINER_CIR,
    "cii": lambda: settings.PROMPTS_CONTAINER_CII,
    "others": lambda: settings.PROMPTS_CONTAINER_OTHERS,
}


class PromptConfig:
    """Représente un prompt avec ses métadonnées."""

    def __init__(
        self,
        section: str,
        type_dossier: str,
        content: str,
        blob_url: Optional[str] = None,
        last_modified: Optional[datetime] = None,
    ):
        self.section = section
        self.type_dossier = type_dossier
        self.content = content
        self.blob_url = blob_url
        self.last_modified = last_modified

    def to_dict(self):
        return {
            "section": self.section,
            "type_dossier": self.type_dossier,
            "content": self.content,
            "blob_url": self.blob_url,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
        }


def _get_container_for_type(type_dossier: str) -> str:
    """Retourne le nom du conteneur Blob selon le type de dossier."""
    if type_dossier.lower() == "cir":
        return settings.PROMPTS_CONTAINER_CIR
    elif type_dossier.lower() == "cii":
        return settings.PROMPTS_CONTAINER_CII
    elif type_dossier.lower() == "others":
        return settings.PROMPTS_CONTAINER_OTHERS
    else:
        raise ValueError(f"Type de dossier inconnu: {type_dossier}")


def _list_blobs_in_container(container_name: str) -> List[str]:
    """Liste tous les fichiers .txt dans un conteneur Blob."""
    service = get_blob_service_client()
    container = service.get_container_client(container_name)

    blob_names = []
    try:
        # Lister tous les blobs du conteneur
        blob_list = container.list_blobs()
        for blob in blob_list:
            # Ne garder que les fichiers .txt
            if blob.name.endswith('.txt'):
                blob_names.append(blob.name)
    except Exception as e:
        print(f"[prompts_admin] Erreur lors du listage du conteneur {container_name}: {e}")

    return blob_names


def _extract_section_name(filename: str) -> str:
    """Extrait le nom de la section à partir du nom de fichier (sans .txt)."""
    return filename.replace(".txt", "")


def _get_blob_url(container: str, blob_name: str) -> str:
    """Construit l'URL publique ou avec SAS pour un blob."""
    base = f"https://{settings.PROMPTS_ACCOUNT}.blob.core.windows.net/{container}"

    if settings.PROMPTS_PUBLIC:
        return f"{base}/{blob_name}"

    # Sélectionne le SAS spécifique au conteneur
    if container == settings.PROMPTS_CONTAINER_CIR and settings.PROMPTS_SAS_CIR:
        sas = settings.PROMPTS_SAS_CIR.lstrip("?")
    elif container == settings.PROMPTS_CONTAINER_CII and settings.PROMPTS_SAS_CII:
        sas = settings.PROMPTS_SAS_CII.lstrip("?")
    elif container == settings.PROMPTS_CONTAINER_OTHERS and settings.PROMPTS_SAS_OTHERS:
        sas = settings.PROMPTS_SAS_OTHERS.lstrip("?")
    else:
        sas = settings.PROMPTS_SAS.lstrip("?") if settings.PROMPTS_SAS else ""

    return f"{base}/{blob_name}?{sas}" if sas else f"{base}/{blob_name}"


def list_all_prompts() -> List[PromptConfig]:
    """Liste tous les prompts disponibles (CIR, CII, OTHERS)."""
    all_prompts = []

    # CIR
    for type_dossier in ["cir", "cii", "others"]:
        prompts = _list_prompts_for_type(type_dossier)
        all_prompts.extend(prompts)

    return all_prompts


def _list_prompts_for_type(type_dossier: str) -> List[PromptConfig]:
    """Liste les prompts pour un type de dossier spécifique en scannant dynamiquement le conteneur."""
    container_name = _get_container_for_type(type_dossier)

    # Lister dynamiquement tous les fichiers .txt du conteneur
    prompt_files = _list_blobs_in_container(container_name)

    service = get_blob_service_client()
    container = service.get_container_client(container_name)

    prompts = []

    for filename in prompt_files:
        blob_client = container.get_blob_client(filename)

        try:
            # Récupérer les métadonnées du blob
            properties: BlobProperties = blob_client.get_blob_properties()

            # Télécharger le contenu
            download_stream = blob_client.download_blob()
            content = download_stream.readall().decode("utf-8")

            section = _extract_section_name(filename)
            blob_url = _get_blob_url(container_name, filename)

            prompt = PromptConfig(
                section=section,
                type_dossier=type_dossier,
                content=content,
                blob_url=blob_url,
                last_modified=properties.last_modified,
            )
            prompts.append(prompt)

        except ResourceNotFoundError:
            # Le fichier n'existe pas encore sur le blob, on le skip
            print(f"[prompts_admin] Blob non trouvé: {filename} dans {container_name}")
            continue
        except Exception as e:
            print(f"[prompts_admin] Erreur lors de la lecture de {filename}: {e}")
            continue

    return prompts


def get_prompt(section: str, type_dossier: str) -> Optional[PromptConfig]:
    """Récupère un prompt spécifique."""
    container_name = _get_container_for_type(type_dossier)
    filename = f"{section}.txt"

    service = get_blob_service_client()
    container = service.get_container_client(container_name)
    blob_client = container.get_blob_client(filename)

    try:
        properties: BlobProperties = blob_client.get_blob_properties()
        download_stream = blob_client.download_blob()
        content = download_stream.readall().decode("utf-8")

        blob_url = _get_blob_url(container_name, filename)

        return PromptConfig(
            section=section,
            type_dossier=type_dossier,
            content=content,
            blob_url=blob_url,
            last_modified=properties.last_modified,
        )
    except ResourceNotFoundError:
        return None
    except Exception as e:
        print(f"[prompts_admin] Erreur lors de la récupération de {filename}: {e}")
        return None


def update_prompt(section: str, type_dossier: str, content: str) -> Optional[PromptConfig]:
    """Met à jour le contenu d'un prompt sur Azure Blob Storage."""
    container_name = _get_container_for_type(type_dossier)
    filename = f"{section}.txt"

    service = get_blob_service_client()
    container = service.get_container_client(container_name)

    # Créer le conteneur s'il n'existe pas
    try:
        container.create_container()
    except Exception:
        pass  # existe déjà

    blob_client = container.get_blob_client(filename)

    try:
        # Upload du nouveau contenu (overwrite)
        blob_client.upload_blob(
            content.encode("utf-8"),
            overwrite=True,
            content_settings=ContentSettings(content_type="text/plain; charset=utf-8")
        )

        # Récupérer les nouvelles propriétés
        properties: BlobProperties = blob_client.get_blob_properties()
        blob_url = _get_blob_url(container_name, filename)

        return PromptConfig(
            section=section,
            type_dossier=type_dossier,
            content=content,
            blob_url=blob_url,
            last_modified=properties.last_modified,
        )
    except Exception as e:
        print(f"[prompts_admin] Erreur lors de la mise à jour de {filename}: {e}")
        return None
