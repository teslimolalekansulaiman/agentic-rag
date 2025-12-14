from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Project root = 2 levels above this file
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    # Data directories
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    ARTIFACTS_DIR: Path = DATA_DIR / "artifacts"
    EVALUATION_DATASET_DIR: Path = PROJECT_ROOT / "src/evaluation/datasets"
    RAIL_CONFIGURATION_DIR: Path = PROJECT_ROOT / "src/guardrail/rails"

    # Database (no secrets in code)
    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DB: str = "crew_agent"
    PG_USER: str        # required
    PG_PASSWORD: str    # required
    PG_VECTOR_TABLE: str = "procurement_chunks"
    PG_DOC_TABLE: str = "procurement_docs"

    # Embeddings
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_DIM: int = 1024

    # Chunking
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50

    # Providers
    EMBEDDING_MODEL_NAME: str = "nomic-embed-text:v1.5"

    # LLMs
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "gemma3:4b"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OPENAI_API_BASE_URL: str = "http://localhost:8000/v1"
    CONTEXT_LLM_MODEL: str = "llama3:8b-instruct"

    PHOENIX_HOST: str = "http://localhost:6006"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
