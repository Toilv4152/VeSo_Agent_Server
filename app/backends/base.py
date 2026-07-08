"""
Abstract Base Class cho tất cả LLM backends.

Mỗi backend mới chỉ cần kế thừa class này và implement 2 method:
- generate_ocr(): OCR ảnh vé số
- generate_text(): Gọi LLM text-only (dò vé số)
"""

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Interface chung cho tất cả các LLM backend."""

    @abstractmethod
    def generate_ocr(self, base64_image: str, prompt: str, schema: dict = None) -> str:
        """OCR ảnh vé số, trả về text kết quả."""
        ...

    @abstractmethod
    def generate_text(self, messages: list, model: str = None, schema: dict = None) -> str:
        """Gọi LLM text-only (không có ảnh), trả về text kết quả."""
        ...
