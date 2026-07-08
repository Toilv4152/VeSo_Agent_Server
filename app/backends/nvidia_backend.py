"""
NVIDIA NIM API Backend — Gọi LLM qua NVIDIA NIM REST API.
"""

import requests

from app.backends.base import LLMBackend


class NvidiaAPIBackend(LLMBackend):
    """Backend gọi qua NVIDIA NIM API."""

    INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self, api_key: str, ocr_model: str, llm_model: str):
        self.api_key = api_key
        self.ocr_model = ocr_model
        self.llm_model = llm_model

    def _call_api(self, payload: dict) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        response = requests.post(self.INVOKE_URL, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"NVIDIA API error {response.status_code}: {response.text}")

    def generate_ocr(self, base64_image: str, prompt: str, schema: dict = None) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": base64_image}},
                {"type": "text", "text": prompt},
            ],
        }]
        payload = {
            "model": self.ocr_model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": True}
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "gcn_schema", "schema": schema, "strict": True}
            }
        return self._call_api(payload)

    def generate_text(self, messages: list, model: str = None, schema: dict = None) -> str:
        payload = {
            "model": model or self.llm_model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "gcn_schema", "schema": schema, "strict": True}
            }
        return self._call_api(payload)
