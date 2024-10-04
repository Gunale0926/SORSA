#  ------------------------------------------------------------------------------------------
#  Copyright 2024 Yang Cao.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from typing import List, Optional
from transformers import PreTrainedModel, AutoModelForCausalLM

from .layer import Linear as SORSALinear
from .config import SORSAConfig


class SORSAModel(PreTrainedModel):
    config_class = SORSAConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, trust_remote_code=True
        )
        self._replace_modules()

    def _set_submodule(self, target: str, module: torch.nn.Module):
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
                    dropout=self.config.dropout,
                )
                sorsa_module.weight.data = module.weight.data
                if sorsa_module.bias is not None:
                    sorsa_module.bias.data = module.bias.data
                self._set_submodule(f"{name}", sorsa_module)

    def sorsa_init(
        self,
        weight_dtype: Optional[torch.dtype] = None,
        adapter_dtype: Optional[torch.dtype] = None,
    ):
        print("Initializing SORSA Adapters...")
        for module in self.modules():
            if isinstance(module, SORSALinear):
                module.sorsa_init(weight_dtype, adapter_dtype)

    def merge(self, mode=True):
        for module in self.modules():
            if isinstance(module, SORSALinear):
                module._merge(mode)

    def get_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "sorsa_" in n]

    def set_trainable(self, mode=True):
        for name, param in self.named_parameters():
            if "sorsa_" in name:
                param.requires_grad = mode
