"""
Factory — Tạo LLM backend dựa trên cấu hình biến môi trường.
"""

import os

from app.config import (
    LLM_BACKEND, OCR_BASE_URL, LLM_BASE_URL,
    OCR_MODEL, LLM_MODEL, NVIDIA_API_KEY, logger,
)
from app.backends.base import LLMBackend


def create_backend() -> LLMBackend:
    """Factory: tạo LLM backend dựa trên biến môi trường LLM_BACKEND."""
    backend_type = LLM_BACKEND.lower()
    logger.info(f"Khởi tạo LLM backend: {backend_type}")

    if backend_type == "vllm":
        from app.backends.vllm_backend import VLLMBackend
        return VLLMBackend(
            ocr_base_url=OCR_BASE_URL,
            llm_base_url=LLM_BASE_URL,
            ocr_model=OCR_MODEL,
            llm_model=LLM_MODEL,
        )
    elif backend_type == "ollama":
        from app.backends.ollama_backend import OllamaBackend
        ollama_ocr_url = os.getenv("OLLAMA_OCR_URL", "http://10.225.0.28:11434/api/generate")
        ollama_ocr_model = os.getenv("OCR_MODEL", "qwen3-vl:2b-instruct")
        ollama_llm_model = os.getenv("LLM_MODEL", "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT")
        return OllamaBackend(
            ocr_base_url=ollama_ocr_url,
            llm_base_url=ollama_ocr_url,
            ocr_model=ollama_ocr_model,
            llm_model=ollama_llm_model,
        )
    elif backend_type == "nvidia":
        from app.backends.nvidia_backend import NvidiaAPIBackend
        nvidia_ocr_model = os.getenv("OCR_MODEL", "qwen/qwen3.5-122b-a10b")
        nvidia_llm_model = os.getenv("LLM_MODEL", "qwen/qwen3.5-122b-a10b")
        return NvidiaAPIBackend(
            api_key=NVIDIA_API_KEY,
            ocr_model=nvidia_ocr_model,
            llm_model=nvidia_llm_model,
        )
    else:
        raise ValueError(f"Backend không hợp lệ: {backend_type}. Chọn: vllm, ollama, nvidia")
