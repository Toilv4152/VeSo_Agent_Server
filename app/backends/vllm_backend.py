"""
vLLM Backend — Gọi LLM qua OpenAI-compatible API (vLLM server).
"""

from app.backends.base import LLMBackend


class VLLMBackend(LLMBackend):
    """Backend gọi qua OpenAI-compatible API (vLLM server)."""

    def __init__(self, ocr_base_url: str, llm_base_url: str, ocr_model: str, llm_model: str):
        from openai import OpenAI
        self.ocr_client = OpenAI(base_url=ocr_base_url, api_key="EMPTY")
        self.llm_client = OpenAI(base_url=llm_base_url, api_key="EMPTY")
        self.ocr_model = ocr_model
        self.llm_model = llm_model

    def generate_ocr(self, base64_image: str, prompt: str, schema: dict = None) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": base64_image}},
                {"type": "text", "text": prompt},
            ],
        }]
        kwargs = {
            "model": self.ocr_model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "extra_body": {
                "top_k": 1,
                "min_p": 0.1,
                "repetition_penalty": 1.06,
            }
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "gcn_schema", "schema": schema, "strict": True}
            }
        response = self.ocr_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

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
