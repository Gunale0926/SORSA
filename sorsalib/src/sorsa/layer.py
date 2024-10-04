#  ------------------------------------------------------------------------------------------
#  Copyright 2024 Yang Cao.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SORSALayer:
    def __init__(
        self,
        r: int,
        alpha: Optional[float],
        dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        if alpha == None:
            self.scale = 1
        else:
            self.scale = alpha / r
        # Optional dropout
        if dropout > 0.0:
            self.sorsa_dropout = nn.Dropout(p=dropout)
        else:
            self.sorsa_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(SORSALayer, nn.Module):
    # SORSA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        merge_weights: bool = True,
        bias=False,
    ):
        nn.Module.__init__(self)
        SORSALayer.__init__(
            self,
            r=r,
            alpha=alpha,
            dropout=dropout,
            merge_weights=merge_weights,
        )

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias == True:
            self.bias = nn.Parameter(torch.empty((out_features)))
        else:
            self.bias = None
        # Actual trainable parameters
        if r > 0:
            self.sorsa_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.sorsa_S = nn.Parameter(self.weight.new_zeros(r))
            self.sorsa_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

    def sorsa_init(
        self,
        weight_dtype: Optional[torch.dtype] = None,
        adapter_dtype: Optional[torch.dtype] = None,
    ):
        if weight_dtype is None:
            weight_dtype = self.weight.dtype
        if adapter_dtype is None:
            adapter_dtype = weight_dtype
        if hasattr(self, "sorsa_A"):
            self.merged = False
            self.weight.to(torch.float32)  # Convert to FP32 for SVD
            u, s, vt = torch.linalg.svd(self.weight.T, full_matrices=False)
            self.sorsa_A.data = u[:, : self.r].T.contiguous().to(adapter_dtype)
            self.sorsa_S.data = s[: self.r].to(adapter_dtype)
            self.sorsa_B.data = vt[: self.r, :].T.contiguous().to(adapter_dtype)
            merge = (self.sorsa_B * self.sorsa_S) @ self.sorsa_A
            self.weight.data = (self.weight - merge * self.scale).to(weight_dtype)

    def _merge(self, mode: bool):
        if mode:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    merge = (self.sorsa_B * self.sorsa_S) @ self.sorsa_A
                    self.weight.data += merge * self.scale
                self.merged = True
        else:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    merge = (self.sorsa_B * self.sorsa_S) @ self.sorsa_A
                    self.weight.data -= merge * self.scale
                self.merged = False

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.weight)
            result += (
                F.linear(
                    self.sorsa_dropout(x),
                    (self.sorsa_B * self.sorsa_S) @ self.sorsa_A,
                )
                * self.scale
            )
            if self.bias is not None:
                result += self.bias
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)


# Largely adpot idea from AdaLoRA
def calc_ortho(model):
    ortho_loss = 0.0
    den = 0
    for name, param in model.named_parameters():
        if "sorsa_A" in name:
            a = param
            ia = torch.eye(a.shape[0], device=a.device)
            ia.requires_grad = False
            a = a @ a.T - ia
            ortho_loss += torch.norm(a, p="fro")
            den += 1
        elif "sorsa_B" in name:
            b = param
            ib = torch.eye(b.shape[1], device=b.device)
            ib.requires_grad = False
            b = b.T @ b - ib
            ortho_loss += torch.norm(b, p="fro")
            den += 1
    if den != 0:
        return ortho_loss / den
    else:
        return None
