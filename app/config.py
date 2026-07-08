"""
Cấu hình server - đọc từ biến môi trường.

Tất cả cấu hình tập trung tại đây, các module khác import từ file này.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("veso-server")

# ==============================================================================
# LLM BACKEND
# ==============================================================================
LLM_BACKEND   = os.getenv("LLM_BACKEND", "vllm")           # vllm | ollama | nvidia

# ==============================================================================
# OCR & LLM URLS
# ==============================================================================
OCR_BASE_URL  = os.getenv("OCR_BASE_URL", "http://103.9.156.145:8000/v1")
LLM_BASE_URL  = os.getenv("LLM_BASE_URL", "http://10.225.0.28:8000/v1")

# ==============================================================================
# MODEL NAMES
# ==============================================================================
OCR_MODEL     = os.getenv("OCR_MODEL", "aisingapore/Qwen-SEA-LION-v4-8B-VL")
LLM_MODEL     = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

# ==============================================================================
# API KEYS
# ==============================================================================
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
XOSOAPI_KEY   = os.getenv("XOSOAPI_KEY", "sk_live_f4d14e2c0a820dfdd70f203fb58bf4f5f9288765c3be37a109bf5cd140301a5b")

# ==============================================================================
# SERVER
# ==============================================================================
SERVER_PORT   = int(os.getenv("SERVER_PORT", "8888"))

# ==============================================================================
# THREAD POOL (dùng chung cho toàn bộ app)
# ==============================================================================
executor = ThreadPoolExecutor(max_workers=8)
