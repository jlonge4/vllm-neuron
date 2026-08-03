# SPDX-License-Identifier: Apache-2.0
from .config import MiniMaxM2Config
from . import model_bf16  # noqa: F401
from .factory import MiniMaxM2ForCausalLM

__all__ = [
    "MiniMaxM2Config",
    "MiniMaxM2ForCausalLM",
]
