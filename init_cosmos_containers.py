#!/usr/bin/env python3
"""
Script d'initialisation des conteneurs Cosmos DB.
À exécuter une seule fois pour créer les conteneurs manquants.

Usage:
    python init_cosmos_containers.py
"""

from azure.cosmos import CosmosClient, PartitionKey, exceptions
from app.config import settings

def init_containers():
    """Crée les conteneurs Cosmos DB s'ils n'existent pas."""

    print(f"Connexion à Cosmos DB : {settings.COSMOS_URI}")
    client = CosmosClient(settings.COSMOS_URI, credential=settings.COSMOS_KEY)

    database_name = settings.COSMOS_DB_NAME
    print(f"Base de données : {database_name}")

    # Créer la base de données si elle n'existe pas
    try:
        database = client.create_database_if_not_exists(id=database_name)
        print(f"✅ Base de données '{database_name}' OK")
    except exceptions.CosmosHttpResponseError as e:
        print(f"❌ Erreur création base de données : {e}")
        return

    # Définition des conteneurs avec leurs partition keys
    containers = [
        {
            "id": settings.COSMOS_CONTAINER_PROJECTS,
            "partition_key": "/project_id",
            "description": "Projets de génération CIR/CII"
        },
        {
            "id": settings.COSMOS_CONTAINER_OUTPUTS,
            "partition_key": "/project_id",
            "description": "Fichiers DOCX générés"
        },
        {
            "id": settings.COSMOS_CONTAINER_DOCUMENTS,
            "partition_key": "/project_id",
            "description": "Documents uploadés par les utilisateurs"
        },
        {
            "id": settings.COSMOS_CONTAINER_USERS,
            "partition_key": "/id",
            "description": "Utilisateurs de l'application"
        },
    ]

    # Créer chaque conteneur
    for container_def in containers:
        container_id = container_def["id"]
        partition_key = container_def["partition_key"]
        description = container_def["description"]

        try:
            container = database.create_container_if_not_exists(
                id=container_id,
                partition_key=PartitionKey(path=partition_key),
                offer_throughput=400  # 400 RU/s (minimum)
            )
            print(f"✅ Conteneur '{container_id}' OK ({description})")
            print(f"   Partition key: {partition_key}")
        except exceptions.CosmosHttpResponseError as e:
            print(f"❌ Erreur création conteneur '{container_id}' : {e}")

    print("\n✅ Initialisation terminée !")
    print("\nNOTE: Si vous voyez 'Conflict (409)', c'est normal - le conteneur existe déjà.")

if __name__ == "__main__":
    try:
        init_containers()
    except Exception as e:
        print(f"\n❌ Erreur fatale : {e}")
        import traceback
        traceback.print_exc()
