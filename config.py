import os

# 以config.py自身位置为基准，解决Windows相对路径问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 路径配置
RAW_TXT_PATH = os.path.join(BASE_DIR, "data")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "db")

# Ollama配置
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "qllama/bge-small-zh-v1.5:latest"
CHAT_MODEL = "qwen3:8b"

CHUNK_MAX_CHAR = 250
CHUNK_STEP = 100

# 入库、检索参数
BATCH_INSERT_SIZE = 20
TOP_N = 4
SIMILAR_THRESHOLD = 0.65

# 缓存文件路径
BM25_CACHE_PATH = os.path.join(BASE_DIR, "data", "bm25_cache.pkl")
FILE_MD5_CACHE_PATH = os.path.join(BASE_DIR, "data", "file_md5_cache.pkl")
