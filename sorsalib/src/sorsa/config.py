#  ------------------------------------------------------------------------------------------
#  Copyright 2024 Yang Cao.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import List, Optional
from dataclasses import dataclass, field
from transformers import PretrainedConfig, TrainingArguments


class SORSAConfig(PretrainedConfig):
    model_type = "sorsa"

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


@dataclass
class SORSATrainingArguments(TrainingArguments):
    gamma: float = field(
        default=0.0, metadata={"help": "SORSA-specific gamma hyperparameter"}
    )

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False
