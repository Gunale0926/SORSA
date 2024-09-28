import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, Union
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForCausalLM,
    TrainingArguments,
)
from safetensors.torch import save_file, load_file
from sorsalib.layer import Linear as SORSALinear


class SORSAConfig(PretrainedConfig):
    model_type = "sorsa"

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        rank: int = 4,
        alpha: Optional[float] = None,
        sorsa_dropout: float = 0.0,
        target_modules: List[str] = ["query", "key", "value"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.rank = rank
        self.alpha = alpha
        self.sorsa_dropout = sorsa_dropout
        self.target_modules = target_modules


class SORSAModel(PreTrainedModel):
    config_class = SORSAConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
        self._replace_modules()

    def set_submodule(self, target: str, module: torch.nn.Module):
        atoms: List[str] = target.split(".")
        name = atoms.pop(-1)
        mod = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not an nn.Module")

        setattr(mod, name, module)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _replace_modules(self):
        if isinstance(self.config.target_modules, list):
            target_module_names = self.config.target_modules
        elif isinstance(self.config.target_modules, dict):
            target_module_names = list(self.config.target_modules.values())
        else:
            raise ValueError("target_modules must be a list or dict")
        for name, module in self.named_modules():
            if any(t in name for t in target_module_names) and isinstance(
                module, nn.Linear
            ):
                sorsa_module = SORSALinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        r=self.config.rank,
                        alpha=self.config.alpha,
                        bias=module.bias is not None,
                        sorsa_dropout=self.config.sorsa_dropout,
                    )
                with torch.no_grad():
                    sorsa_module.weight.copy_(module.weight)
                    if module.bias is not None:
                        sorsa_module.bias.copy_(module.bias)
                self.set_submodule(f"{name}", sorsa_module)

    def sorsa_init(self):
        print("Initializing SORSA Adapters...")
        for module in self.modules():
            if isinstance(module, SORSALinear):
                with torch.no_grad():
                    module.sorsa_init()

    def get_sorsa_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "sorsa_" in n]


@dataclass
class SORSATrainingArguments(TrainingArguments):
    gamma: float = field(
        default=0.0, metadata={"help": "SORSA-specific gamma parameter"}
    )

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False