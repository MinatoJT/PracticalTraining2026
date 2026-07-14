from __future__ import annotations

import threading
from typing import Optional

from PIL import Image


class LocalQwen3VLBackend:
    """Lazy single-model BF16 backend; deliberately does not depend on vLLM."""

    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def shared(cls, model_name: str, max_image_edge: int) -> "LocalQwen3VLBackend":
        key = (model_name, max_image_edge)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(model_name, max_image_edge)
            return cls._instances[key]

    def __init__(self, model_name: str, max_image_edge: int):
        self.model_name = model_name
        self.max_image_edge = max_image_edge
        self.model = None
        self.processor = None
        self.error: Optional[str] = None
        self._generate_lock = threading.Lock()

    def _load(self) -> None:
        if self.model is not None or self.error:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.model.eval()
        except Exception as exc:
            self.error = f"local_model_load_failed:{type(exc).__name__}"

    @property
    def available(self) -> bool:
        self._load()
        return self.model is not None and self.processor is not None

    def generate(self, image: Image.Image, prompt: str, max_tokens: int) -> str:
        self._load()
        if not self.available:
            raise RuntimeError(self.error or "local_model_unavailable")
        import torch

        prepared = image.convert("RGB").copy()
        prepared.thumbnail((self.max_image_edge, self.max_image_edge))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": prepared},
                {"type": "text", "text": prompt},
            ],
        }]
        with self._generate_lock, torch.inference_mode():
            formatted = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[formatted], images=[prepared], padding=True, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
            )
            prompt_length = inputs["input_ids"].shape[1]
            return self.processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

