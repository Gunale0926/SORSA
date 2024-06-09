from transformers import AutoConfig
from models import (
    SORSALlamaForCausalLM,
    SORSAGemmaForCausalLM,
    SORSAMistralForCausalLM,
    SORSARobertaForSequenceClassification,
    SORSADebertaForSequenceClassification,
    SORSAPhiForCausalLM,
)
from models import (
    SORSALlamaConfig,
    SORSAGemmaConfig,
    SORSAMistralConfig,
    SORSARobertaConfig,
    SORSADebertaConfig,
    SORSAPhiConfig,
)

SORSA_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = {
    "llama": SORSALlamaForCausalLM,
    "gemma": SORSAGemmaForCausalLM,
    "mistral": SORSAMistralForCausalLM,
    "phi": SORSAPhiForCausalLM,
}

SORSA_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = {
    "roberta": SORSARobertaForSequenceClassification,
    "deberta-v2": SORSADebertaForSequenceClassification,
}


SORSA_CONFIG_MAPPING = {
    "llama": SORSALlamaConfig,
    "gemma": SORSAGemmaConfig,
    "mistral": SORSAMistralConfig,
    "roberta": SORSARobertaConfig,
    "deberta-v2": SORSADebertaConfig,
    "phi": SORSAPhiConfig,
}


class SORSAAutoModelForSequenceClassification:
    _model_mapping = SORSA_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs,
    ):
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        model_type = config.model_type
        if model_type not in cls._model_mapping.keys():
            raise ValueError(f"Model type {model_type} not supported for SORSA")

        model_class = cls._model_mapping[model_type]
        # Initialize the model using the class
        model = model_class.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            *model_args,
            **kwargs,
        )

        return model


class SORSAAutoModelForCausalLM:
    _model_mapping = SORSA_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs,
    ):
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        model_type = config.model_type
        if model_type not in cls._model_mapping.keys():
            raise ValueError(f"Model type {model_type} not supported for SORSA")

        model_class = cls._model_mapping[model_type]
        # Initialize the model using the class
        model = model_class.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            *model_args,
            **kwargs,
        )

        return model


class SORSAAutoConfig:
    _config_mapping = SORSA_CONFIG_MAPPING

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config_class = cls._config_mapping.get(config.model_type)
        if config_class is None:
            raise ValueError(f"Unrecognized model type {config.model_type} for SORSA")

        return config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
