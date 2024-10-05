#  ------------------------------------------------------------------------------------------
#  SORSA: Singular Values and Orthonormal Regularized Singular Vectors Adaptation of Large Language Models
#  arXiv: https://arxiv.org/abs/2409.00055
#  Copyright (c) 2024 Yang Cao
#  Licensed under the Apache License, Version 2.0.
#  ------------------------------------------------------------------------------------------

"""
SORSA model intergrated with Hugging Face transformers.
"""
import torch
import torch.nn as nn
from typing import List, Optional
from transformers import PreTrainedModel, AutoModelForCausalLM

from .layer import Linear as SORSALinear
from .config import SORSAConfig


class SORSAModel(PreTrainedModel):
    """
    A wrapper model that applies SORSA to huggingface PreTrainedModel.

    Attributes:
        config (SORSAConfig): Configuration instance for this model.
        model (PreTrainedModel): The wrapped PreTrainedModel.
    """

    config_class = SORSAConfig

    def __init__(self, config):
        """
        Initialize the SORSAModel.

        Args:
            config (SORSAConfig): Configuration for the SORSA model.
        """
        super().__init__(config)
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, trust_remote_code=True
        )
        self._replace_modules()

    def _set_submodule(self, target: str, module: torch.nn.Module):
        """
        Set the submodule given by ``target`` if it exists, otherwise throw an error.
        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:
        .. code-block:: text
            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )
        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)
        To overide the ``Conv2d`` with a new submodule ``Linear``, you
        would call
        ``set_submodule("net_b.net_c.conv", nn.Linear(33, 16))``.
        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)
            module: The module to set the submodule to.
        Raises:
            ValueError: If the target string is empty
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
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
        """
        Forward pass of the model.

        Returns:
            The output of the wrapped model's forward pass.
        """
        return self.model(*args, **kwargs)

    def _replace_modules(self):
        """
        Replace linear layers in target_modules with SORSA enabled Linear.
        """
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
        """
        Initialize SORSA adapters for all SORSA enabled Linear layers in the model.

        Args:
            weight_dtype (Optional[torch.dtype]): Data type for the weight matrix.
            adapter_dtype (Optional[torch.dtype]): Data type for the SORSA matrices.
        """
        print("Initializing SORSA Adapters...")
        for module in self.modules():
            if isinstance(module, SORSALinear):
                module.sorsa_init(weight_dtype, adapter_dtype)

    def merge(self, mode=True):
        """
        Merge or unmerge all SORSA adapters in the model.

        Args:
            mode (bool): If True, merge the weights. If False, unmerge the weights.
        """
        for module in self.modules():
            if isinstance(module, SORSALinear):
                module._merge(mode)

    def get_parameters(self) -> List[nn.Parameter]:
        """
        Get all SORSA adapters in the model.

        Returns:
            List[nn.Parameter]: List of all parameters with 'sorsa_' in their name.
        """
        return [p for n, p in self.named_parameters() if "sorsa_" in n]

    def set_trainable(self, mode=True):
        """
        Set the trainable state of all SORSA adapters.

        Args:
            mode (bool): If True, make SORSA adapters trainable. If False, freeze them.
        """
        for name, param in self.named_parameters():
            if "sorsa_" in name:
                param.requires_grad = mode
            else:
                param.requires_grad = False
