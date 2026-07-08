"""
Ollama Backend — Gọi LLM qua Ollama API (ollama_ocr cho OCR, OpenAI client cho text).
"""

import os
import json
import base64
import tempfile

from app.backends.base import LLMBackend


class OllamaBackend(LLMBackend):
    """Backend gọi qua Ollama API (dùng ollama_ocr cho OCR, OpenAI client cho text)."""

    def __init__(self, ocr_base_url: str, llm_base_url: str, ocr_model: str, llm_model: str):
        from openai import OpenAI
        from ollama_ocr import OCRProcessor
        self.ocr = OCRProcessor(model_name=ocr_model, base_url=ocr_base_url)
        # Ollama cũng hỗ trợ OpenAI-compatible API tại /v1
        ollama_openai_url = llm_base_url.replace("/api/generate", "/v1")
        if not ollama_openai_url.endswith("/v1"):
            ollama_openai_url = ollama_openai_url.rstrip("/") + "/v1"
        self.llm_client = OpenAI(base_url=ollama_openai_url, api_key="EMPTY")
        self.llm_model = llm_model

    def generate_ocr(self, base64_image: str, prompt: str, schema: dict = None) -> str:
        # ollama_ocr cần đường dẫn file, không phải base64
        # Workaround: lưu ảnh tạm rồi gọi
        if "," in base64_image:
            b64_data = base64_image.split(",", 1)[1]
        else:
            b64_data = base64_image
        img_bytes = base64.b64decode(b64_data)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        try:
            result = self.ocr.process_image(
                image_path=tmp_path,
                format_type="json",
                custom_prompt=prompt,
                language="Vietnamese"
            )
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        finally:
            os.unlink(tmp_path)

    def generate_text(self, messages: list, model: str = None, schema: dict = None) -> str:
        kwargs = {
            "model": model or self.llm_model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.1,
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "gcn_schema", "schema": schema, "strict": True}
            }
        response = self.llm_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
