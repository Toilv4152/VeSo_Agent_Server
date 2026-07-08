"""
FastAPI Server - Dò Vé Số Kiến Thiết
=====================================
Nhận ảnh vé số → OCR trích xuất thông tin → Tra cứu KQXS → Dò vé → Streaming kết quả.

Chạy server:
    python server.py

Hoặc:
    uvicorn server:app --host 0.0.0.0 --port 8888

Cấu hình qua biến môi trường:
    LLM_BACKEND   : vllm | ollama | nvidia  (mặc định: vllm)
    OCR_BASE_URL  : URL server OCR           (mặc định: http://103.9.156.145:8000/v1)
    LLM_BASE_URL  : URL server LLM text      (mặc định: http://10.225.0.28:8000/v1)
    OCR_MODEL     : Tên model OCR            (mặc định: aisingapore/Qwen-SEA-LION-v4-8B-VL)
    LLM_MODEL     : Tên model LLM text       (mặc định: openai/gpt-oss-20b)
    NVIDIA_API_KEY: API key cho NVIDIA NIM
    XOSOAPI_KEY   : API key cho XoSoAPI
    SERVER_PORT   : Port server              (mặc định: 8888)
"""

import socket
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import SERVER_PORT, LLM_BACKEND, OCR_BASE_URL, LLM_BASE_URL, OCR_MODEL, LLM_MODEL, logger
from app.routes import router

# ==============================================================================
# FASTAPI APPLICATION
# ==============================================================================
app = FastAPI(
    title="Vé Số Kiến Thiết - Dò Vé Số API",
    description="Upload ảnh vé số → OCR → Tra cứu KQXS → Dò vé. Kết quả streaming qua SSE.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký routes
app.include_router(router)

def get_local_ip():
    try:
        # Tạo socket kết nối ra ngoài để lấy IP LAN thực sự
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"

# ==============================================================================
# ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    local_ip = get_local_ip()
    
    logger.info(f"🚀 Starting Vé Số Server")
    logger.info(f"   Network URL : http://{local_ip}:{SERVER_PORT}")
    logger.info(f"   Local URL   : http://localhost:{SERVER_PORT}")
    logger.info(f"   Backend     : {LLM_BACKEND}")
    logger.info(f"   OCR URL     : {OCR_BASE_URL}")
    logger.info(f"   LLM URL     : {LLM_BASE_URL}")
    logger.info(f"   OCR Model   : {OCR_MODEL}")
    logger.info(f"   LLM Model   : {LLM_MODEL}")

    uvicorn.run(
        "server:app",
        host=local_ip,  # Lắng nghe 0.0.0.0 để máy khác trong LAN có thể truy cập qua local_ip
        port=SERVER_PORT,
        log_level="info",
        workers=1,  # Dùng 1 worker vì đã dùng asyncio + thread pool
    )
