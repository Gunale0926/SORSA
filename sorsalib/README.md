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
)
sorsaModel = SORSAModel(config)
sorsaModel.to("cuda") # Note: using CUDA to perform SVD will be less accurate than CPU
sorsaModel.sorsa_init() # Initialize SORSA adapters.
sorsaModel.set_trainable(True) # Set all adapters to trainable
```

## Train SORSA model:
```python
trainingArguments = SORSATrainingArguments(
    # ...
    gamma=4e-4, # Learning rate for Orthonormal Regularizer
)
trainer = SORSATrainer(
        model=sorsaModel,
        args=trainingArguments,
        train_dataset=train_dataset, # Your dataset
)
trainer.train()
```
