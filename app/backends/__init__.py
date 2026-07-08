"""
LLM Backends package — chứa các implementation cho từng nền tảng LLM.
"""

from app.backends.base import LLMBackend
from app.backends.factory import create_backend

__all__ = ["LLMBackend", "create_backend"]
