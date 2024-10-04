# SORSA Python Package

## Initialize SORSA model:
```python
from sorsa import SORSAConfig, SORSAModel, SORSATrainer, SORSATrainingArguments
config = SORSAConfig(
    base_model_name_or_path="meta-llama/Llama-2-7b-hf",
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    rank=16,
    dropout=0,
)
sorsaModel = SORSAModel(config)
self.model.to("cuda")
self.model.sorsa_init() # Initialize SORSA adapters.
```

## Train SORSA model:
```python
trainingArguments = SORSATrainingArguments(
    # ...
    gamma=4e-4,
)
trainer = SORSATrainer(
        model=sorsaModel,
        args=trainingArguments,
        train_dataset=train_dataset,
)
trainer.train()
```
