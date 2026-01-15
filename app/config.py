# app/config.py
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    # Azure OpenAI
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_KEY52: str = os.getenv("AZURE_OPENAI_KEY52", "")

    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")             # chat
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")

    # Serper
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")

    # Docs templates
    TEMPLATE_CIR: str = os.getenv("TEMPLATE_CIR", "./Doc/MEMOIRE_CIR2.docx")
    TEMPLATE_CII: str = os.getenv("TEMPLATE_CII", "./Doc/MEMOIRE_CII.docx")

    # Misc
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    PROMPTS_PROVIDER: str = os.getenv("PROMPTS_PROVIDER", "blob")  # blob | http
    PROMPTS_ACCOUNT: str = os.getenv("PROMPTS_ACCOUNT", "")
    PROMPTS_CONTAINER_CIR: str = os.getenv("PROMPTS_CONTAINER_CIR", "prompts")
    PROMPTS_CONTAINER_CII: str = os.getenv("PROMPTS_CONTAINER_CII", "prompts-cii")
    PROMPTS_CONTAINER_OTHERS: str = os.getenv("PROMPTS_CONTAINER_OTHERS", "prompts-autres")

    PROMPTS_PUBLIC: bool = os.getenv("PROMPTS_PUBLIC", "true").lower() == "true"

    PROMPTS_SAS: str = os.getenv("PROMPTS_SAS", "")
    PROMPTS_SAS_CIR: str = os.getenv("PROMPTS_SAS_CIR", "")
    PROMPTS_SAS_CII: str = os.getenv("PROMPTS_SAS_CII", "")
    PROMPTS_SAS_OTHERS: str = os.getenv("PROMPTS_SAS_OTHERS", "")
    STORAGE_OUTPUTS_SAS: str = os.getenv("STORAGE_OUTPUTS_SAS", "")

    PROMPTS_BASE_URL: str = os.getenv("PROMPTS_BASE_URL", "")



    # ===== Cosmos DB =====
    COSMOS_URI: str = os.getenv("COSMOS_URI", "")
    COSMOS_KEY: str = os.getenv("COSMOS_KEY", "")
    # ex: pleiades-db (dev) / pleiades-db-preprod / pleiades-db-prod
    COSMOS_DB_NAME: str = os.getenv("COSMOS_DB_NAME", "pleiades-db")
    COSMOS_CONTAINER_PROJECTS: str = os.getenv("COSMOS_CONTAINER_PROJECTS", "projects")
    COSMOS_CONTAINER_DOCUMENTS: str = os.getenv("COSMOS_CONTAINER_DOCUMENTS", "documents")
    COSMOS_CONTAINER_OUTPUTS: str = os.getenv("COSMOS_CONTAINER_OUTPUTS", "outputs")
    COSMOS_CONTAINER_USERS: str = os.getenv("COSMOS_CONTAINER_USERS", "users")
    COSMOS_CONTAINER_TEAMS: str = os.getenv("COSMOS_CONTAINER_TEAMS", "teams")

    # ===== Azure Blob Storage pour les fichiers client & sorties =====
    # Option 1 : connection string complète (recommandé)
    AZURE_STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    # Option 2 : URL + KEY (fallback si pas de connection string)
    AZURE_STORAGE_ACCOUNT_URL: str = os.getenv("AZURE_STORAGE_ACCOUNT_URL", "")
    AZURE_STORAGE_ACCOUNT_KEY: str = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "")

    # Conteneurs blob pour les uploads et les sorties générées
    STORAGE_CONTAINER_UPLOADS: str = os.getenv("STORAGE_CONTAINER_UPLOADS", "uploads")
    STORAGE_CONTAINER_OUTPUTS: str = os.getenv("STORAGE_CONTAINER_OUTPUTS", "outputs")
    AAD_TENANT_ID: str = os.getenv("AAD_TENANT_ID", "")
    AAD_CLIENT_ID: str = os.getenv("AAD_CLIENT_ID", "")
    AAD_CLIENT_ID: str = os.getenv("AAD_CLIENT_ID", "")
    AAD_AUDIENCE: str = os.getenv("AAD_AUDIENCE", "")


    

    @property
    def AAD_ISSUER(self) -> str:
        if not self.AAD_TENANT_ID:
            return ""
        return f"https://login.microsoftonline.com/{self.AAD_TENANT_ID}/v2.0"
    @property
    def AAD_ISSUER_V1(self) -> str:
     if not self.AAD_TENANT_ID:
        return ""
     return f"https://sts.windows.net/{self.AAD_TENANT_ID}/"

    @property
    def AAD_JWKS_URL(self) -> str:
        if not self.AAD_TENANT_ID:
            return ""
        return f"https://login.microsoftonline.com/{self.AAD_TENANT_ID}/discovery/v2.0/keys"

    # ===== Cosmos Users =====
    COSMOS_CONTAINER_USERS: str = os.getenv("COSMOS_CONTAINER_USERS", "users")

settings = Settings()
