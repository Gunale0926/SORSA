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
from loralib import Linear as LoRALinear


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


class LoRAModel(PreTrainedModel):
    config_class = LoRAConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, trust_remote_code=True
        )
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
                lora_module = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=self.config.rank,
                    alpha=self.config.alpha,
                    bias=module.bias is not None,
                    lora_dropout=self.config.lora_dropout,
                )
                with torch.no_grad():
                    lora_module.weight.copy_(module.weight)
                    if module.bias is not None:
                        lora_module.bias.copy_(module.bias)
                self.set_submodule(f"{name}", lora_module)

    def lora_init(self, pissa=False):
        print("Initializing LoRA Adapters...")
        for module in self.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    if not pissa:
                        module.lora_init()
                    else:
                        module.pissa_init()

    def lora_merge(self, mode=True):
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module._merge(mode)

    def get_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "lora_" in n]
