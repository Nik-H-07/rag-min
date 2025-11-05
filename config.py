import os
from dotenv import load_dotenv

load_dotenv()

CFG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", None),
    "LLM_MODEL": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
    "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 800)),
    "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 120)),
    "TOP_K": int(os.getenv("TOP_K", 6)),
    "USE_RERANKER": os.getenv("USE_RERANKER", "false").lower() == "true",
}
