from typing import List
from dataclasses import dataclass, field
import torch
import transformers
from sorsalib.layers import Linear as sorsa
from loralib.layers import Linear as lora
from transformers import (
    LlamaForCausalLM,
    GemmaForCausalLM,
    RobertaForSequenceClassification,
    DebertaV2ForSequenceClassification,
    MistralForCausalLM,
    PhiForCausalLM,
    RwkvForCausalLM,
)
from transformers import (
    LlamaConfig,
    GemmaConfig,
    RobertaConfig,
    DebertaConfig,
    MistralConfig,
    PhiConfig,
    RwkvConfig,
)
from tqdm import tqdm


class BaseSORSAModel:
    def __init__(self, target, rank, alpha, sorsa_dropout):
        self.target = target
        self.replaced = []
        self.rank = rank
        if alpha is None:
            alpha = rank
        self.alpha = alpha
        self.sorsa_dropout = sorsa_dropout

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

    def _sorsa_replace(self):
        for t in self.target:
            for name, module in self.named_modules():
                if t in name and isinstance(module, torch.nn.Linear):
                    self.replaced.append(name)

        for name in self.replaced:
            self.set_submodule(
                name,
                sorsa(
                    in_features=self.get_submodule(name).in_features,
                    out_features=self.get_submodule(name).out_features,
                    r=self.rank,
                    alpha=self.alpha,
                    bias=False if self.get_submodule(name).bias is None else True,
                    sorsa_dropout=self.sorsa_dropout,
                ),
            )

    def _sorsa_init(self):
        print(f"Start Initializing SORSA Layers...")
        progress_bar = tqdm(range(1, len(self.replaced) + 1))
        for name in self.replaced:
            self.get_submodule(name).init_sorsa()
            progress_bar.set_description(f"Initializing {name}")
            progress_bar.update(1)
        progress_bar.close()

    def sorsa_merge(self, mode: bool):
        for name in self.replaced:
            self.get_submodule(name).merge(mode)


class SORSADebertaForSequenceClassification(
    DebertaV2ForSequenceClassification, BaseSORSAModel
):
    def __init__(self, config):
        DebertaV2ForSequenceClassification.__init__(self, config=config)
        BaseSORSAModel.__init__(
            self, config.target, config.rank, config.alpha, config.sorsa_dropout
        )
        self._sorsa_replace()


class SORSADebertaConfig(DebertaConfig):
    model_type = "deberta-v2"

    def __init__(self, rank=None, alpha=None, sorsa_dropout=0.1, target=None, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.alpha = alpha
        self.sorsa_dropout = sorsa_dropout
        self.target = target


class SORSARobertaForSequenceClassification(
    RobertaForSequenceClassification, BaseSORSAModel
):
    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config=config)
        BaseSORSAModel.__init__(
            self, config.target, config.rank, config.alpha, config.sorsa_dropout
        )
        self._sorsa_replace()


class SORSARobertaConfig(RobertaConfig):
    model_type = "roberta"

    def __init__(self, rank=None, alpha=None, sorsa_dropout=0.1, target=None, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.alpha = alpha
        self.sorsa_dropout = sorsa_dropout
        self.target = target


class SORSALlamaForCausalLM(LlamaForCausalLM, BaseSORSAModel):
    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config=config)
        BaseSORSAModel.__init__(
            self, config.target, config.rank, config.alpha, config.sorsa_dropout
        )
        self._sorsa_replace()


class SORSALlamaConfig(LlamaConfig):
    model_type = "llama"

    def __init__(self, rank=None, alpha=None, sorsa_dropout=0.1, target=None, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.alpha = alpha
        self.sorsa_dropout = sorsa_dropout
        self.target = target


class SORSAGemmaForCausalLM(GemmaForCausalLM, BaseSORSAModel):
    def __init__(self, config):
        GemmaForCausalLM.__init__(self, config=config)
        BaseSORSAModel.__init__(
            self, config.target, config.rank, config.alpha, config.sorsa_dropout
        )
        self._sorsa_replace()


class SORSAGemmaConfig(GemmaConfig):
    model_type = "gemma"

    def __init__(self, rank=None, alpha=None, sorsa_dropout=0.1, target=None, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.alpha = alpha
        self.sorsa_dropout = sorsa_dropout
        self.target = target


class SORSAMistralForCausalLM(MistralForCausalLM, BaseSORSAModel):
    def __init__(self, config):
        MistralForCausalLM.__init__(self, config=config)
        BaseSORSAModel.__init__(
            self, config.target, config.rank, config.alpha, config.sorsa_dropout
        )
        self._sorsa_replace()


class SORSAMistralConfig(MistralConfig):
    model_type = "mistral"

    def __init__(self, rank=None, alpha=None, sorsa_dropout=0.1, target=None, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.alpha = alpha
        self.sorsa_dropout = sorsa_dropout
        self.target = target


class SORSAPhiForCausalLM(PhiForCausalLM, BaseSORSAModel):
    def __init__(self, config):
        PhiForCausalLM.__init__(self, config=config)
        BaseSORSAModel.__init__(
            self, config.target, config.rank, config.alpha, config.sorsa_dropout
        )
        self._sorsa_replace()


class SORSAPhiConfig(PhiConfig):
    model_type = "phi"

    def __init__(self, rank=None, alpha=None, sorsa_dropout=0.1, target=None, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.alpha = alpha
        self.sorsa_dropout = sorsa_dropout
        self.target = target


class SORSARwkvForCausalLM(RwkvForCausalLM, BaseSORSAModel):
    def __init__(self, config):
        RwkvForCausalLM.__init__(self, config=config)
        BaseSORSAModel.__init__(
            self, config.target, config.rank, config.alpha, config.sorsa_dropout
        )
        self._sorsa_replace()


class SORSARwkvConfig(RwkvConfig):
    model_type = "rwkv"

    def __init__(self, rank=None, alpha=None, sorsa_dropout=0.1, target=None, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.alpha = alpha
        self.sorsa_dropout = sorsa_dropout
        self.target = target


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


@dataclass
class SORSATrainingArguments(transformers.TrainingArguments):
    gamma: float = 0.0
