from typing import List
import torch
from transformers import LlamaForCausalLM
from tqdm import tqdm


class LoRALlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, target, rank, lora_dropout, lora_alpha=None):
        super().__init__(config=config)
        self.target = target
        self.replaced = []
        if lora_alpha is None:
            lora_alpha = rank
        self._lora_replace(rank, lora_dropout, lora_alpha)

    def set_submodule(self, target: str, module: torch.nn.Module):
        atoms: List[str] = target.split(".")
        name = atoms.pop(-1)
        mod: torch.nn.Module = self

        for item in atoms:

            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no " "attribute `" + item + "`"
                )

            mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not " "an nn.Module")

        setattr(mod, name, module)

    def _lora_replace(self, rank, lora_dropout, lora_alpha):
        for t in self.target:
            for name, module in self.named_modules():
                if t in name and isinstance(module, torch.nn.Linear):
                    self.replaced.append(name)

        for name in self.replaced:
            self.set_submodule(
                name,
                lora(
                    in_features=self.get_submodule(name).in_features,
                    out_features=self.get_submodule(name).out_features,
                    r=rank,
                    bias=False if self.get_submodule(name).bias is None else True,
                    lora_dropout=lora_dropout,
                    lora_alpha=lora_alpha,
                ),
            )

    def pissa_init(self):
        print(f"Start Initializing PiSSA Layers...")
        progress_bar = tqdm(range(1, len(self.replaced) + 1))
        for name in self.replaced:
            self.get_submodule(name).init_pissa()
            progress_bar.set_description(f"Initializing {name}")
            progress_bar.update(1)
        progress_bar.close()

    def lora_init(self):
        print(f"Start Initializing LoRA Layers...")
        progress_bar = tqdm(range(1, len(self.replaced) + 1))
        for name in self.replaced:
            self.get_submodule(name).init_lora()
            progress_bar.set_description(f"Initializing {name}")
            progress_bar.update(1)
        progress_bar.close()
