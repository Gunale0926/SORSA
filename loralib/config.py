from typing import List, Optional
from transformers import PretrainedConfig


class LoRAConfig(PretrainedConfig):
    model_type = "lora"

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        rank: int = 4,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        target_modules: List[str] = ["query", "key", "value"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
