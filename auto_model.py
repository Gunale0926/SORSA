from transformers import AutoConfig
from models import (
    SORSALlamaForCausalLM,
    SORSAGemmaForCausalLM,
    SORSARobertaForSequenceClassification,
)
from models import SORSALlamaConfig, SORSAGemmaConfig, SORSARobertaConfig

SORSA_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = {
    "llama": SORSALlamaForCausalLM,
    "gemma": SORSAGemmaForCausalLM,
}

SORSA_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = {
    "roberta": SORSARobertaForSequenceClassification,
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


SORSA_CONFIG_MAPPING = {
    "llama": SORSALlamaConfig,
    "gemma": SORSAGemmaConfig,
    "roberta": SORSARobertaConfig,
}


class SORSAAutoConfig:
    _config_mapping = SORSA_CONFIG_MAPPING

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config_class = cls._config_mapping.get(config.model_type)
        if config_class is None:
            raise ValueError(f"Unrecognized model type {config.model_type} for SORSA")

        return config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
