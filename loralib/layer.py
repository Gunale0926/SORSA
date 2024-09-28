#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Module, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def init_pissa(self):
        if hasattr(self, "lora_A"):
            self.merged = False
            u, s, vt = torch.linalg.svd(self.weight.T, full_matrices=False)
            s_r = s[: self.r]
            self.lora_A.data = (u[:, : self.r] @ torch.diag(s_r**0.5)).T.contiguous()
            self.lora_B.data = (torch.diag(s_r**0.5) @ vt[: self.r, :]).T.contiguous()
            merge = self.lora_B @ self.lora_A
            self.weight.data = (self.weight - merge).to(torch.bfloat16)

    def init_lora(self):
        if hasattr(self, "lora_A"):
            self.merged = False
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self, mode: bool):
        if mode:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += self.lora_B @ self.lora_A * self.scaling
                self.merged = True
        else:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= self.lora_B @ self.lora_A * self.scaling
                self.merged = False

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)
            result += (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)
