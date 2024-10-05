#  ------------------------------------------------------------------------------------------
#  SORSA: Singular Values and Orthonormal Regularized Singular Vectors Adaptation of Large Language Models
#  arXiv: https://arxiv.org/abs/2409.00055
#  Copyright (c) 2024 Yang Cao
#  Licensed under the Apache License, Version 2.0.
#  ------------------------------------------------------------------------------------------

"""
SORSA configuration for SORSAModel and TrainingArguments for SORSATrainer.
"""
from typing import List, Optional
from dataclasses import dataclass, field
from transformers import PretrainedConfig, TrainingArguments


class SORSAConfig(PretrainedConfig):
    """
    Configuration class for SORSAModel.

    This class extends PretrainedConfig to include SORSA-specific parameters.

    Attributes:
        base_model_name_or_path (str): Name or path of the model to be wrapped.
        rank (int): Rank of the SORSA adapters.
        alpha (Optional[float]): Scaling factor for the SORSA update. Not mentioned in the paper, works same with LoRA.
        dropout (float): Dropout probability for SORSA matrices. Not mentioned in the paper, works same with LoRA.
        target_modules (List[str]): Names of modules to be replaced with SORSA versions.
    """

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
    """
    TrainingArguments for SORSATrainer.

    Attributes:
        gamma (float): SORSA's gamma hyperparameter.
    """

    gamma: float = field(default=0.0, metadata={"help": "SORSA's gamma hyperparameter"})

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False
