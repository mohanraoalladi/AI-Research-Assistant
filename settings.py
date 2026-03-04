import os
from dotenv import load_dotenv

# load environment variables from .env file if present
load_dotenv()

# Directory where FAISS vector store will be persisted
DATA_DIR = os.environ.get("DATA_DIR", "data")

# LLM configuration: llama (default), openai, gemini, ollama
LLM_TYPE = os.environ.get("LLM_TYPE", "llama").lower()

# For local LLaMA models, specify path or name
LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "")

# Ollama settings (runs a local HTTP server by default)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama2")

# OpenAI settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

# Google Gemini settings - placeholder - requires additional setup if using Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Memory settings
CONVERSATION_MEMORY = os.environ.get("CONVERSATION_MEMORY", "buffer")

# Retrieval settings - reduced from 4 to 2 to avoid duplicate/noisy context
TOP_K = int(os.environ.get("TOP_K", "2"))

# Make sure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
