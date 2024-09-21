from datasets import load_dataset
import torch
from auto_model import SORSAAutoModelForCausalLM, SORSAAutoConfig
from models import SORSATrainingArguments
from transformers import AutoTokenizer, AutoConfig
from dataset import (
    preprocess_metamathqa,
    preprocess_codefeedback,
    preprocess_codefeedback_instructed,
    collate_fn,
)
from test import test_gsm, test_math
import argparse
import os
from dotenv import dotenv_values
from huggingface_hub import login
from trainer import Trainer

parser = argparse.ArgumentParser(
    prog="SORSA Runner",
    description="Train and Test SORSA Layers",
    epilog="--help for more information",
)

parser.add_argument("-n", "--name", type=str, default="SORSA")
parser.add_argument("-r", "--rank", type=int, default=4)
parser.add_argument("-", "--length", type=int, default=512)
parser.add_argument("-a", "--alpha", type=float, default=None)
parser.add_argument("-g", "--gamma", type=float, default=0.1)
parser.add_argument("-d", "--dropout", type=float, default=0.0)
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-T", "--test", action="store_true")
parser.add_argument("-m", "--merge", action="store_true")
parser.add_argument("--metamath", action="store_true")
parser.add_argument("--code", action="store_true")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--tf32", action="store_true")
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--svd-cache-path", type=str, default="./svd_cache")
parser.add_argument("--run-path", type=str, default="./runs")
parser.add_argument("--cache-path", type=str, default="./cache")
parser.add_argument("--local", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--valid-batch-size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--accum-steps", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--wd", type=float, default=0.01)
parser.add_argument("--gsm-8k", action="store_true")
parser.add_argument("--math", action="store_true")


args = parser.parse_args()


class TrainerConfig:
    def __init__(self, args: argparse.Namespace):
        self.seed = args.seed
        torch.manual_seed(args.seed)
        self.rank = args.rank
        self.dropout = args.dropout
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.valid_batch_size = args.valid_batch_size
        self.num_epochs = args.epochs
        self.accum_steps = args.accum_steps
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Device: {self.device}")
        self.run_path = f"{args.run_path}/{args.name}"
        self.metadata_path = f"{self.run_path}/metadata.pt"
        self.checkpoint_path = f"{self.run_path}/checkpoint"
        if args.local is False:
            env = dotenv_values(".env")
            login(token=env["hf"])
        if args.tokenizer is None:
            args.tokenizer = args.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, cache_dir=args.cache_path, use_fast=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if args.metamath:
            self.train_dataset = load_dataset(
                "meta-math/MetaMathQA", split="train[:100000]"
            )
            self.train_dataset = self.train_dataset.map(
                lambda x: preprocess_metamathqa(x, self.tokenizer, args.length)
            )

        elif args.code:
            self.train_dataset = load_dataset(
                "m-a-p/CodeFeedback-Filtered-Instruction", split="train[:100000]"
            )
            self.train_dataset = self.train_dataset.map(
                lambda x: preprocess_codefeedback_instructed(
                    x, self.tokenizer, args.length
                )
            )
        self.train_dataset.set_format(
            type="torch", columns=["input_ids", "labels", "attention_mask"]
        )
        target = [
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        config = SORSAAutoConfig.from_pretrained(
            args.model,
            target=target,
            rank=self.rank,
            alpha=args.alpha,
            sorsa_dropout=self.dropout,
        )
        self.loaded = False
        # Load model in ALL float32 (residual is quantized to bf16)
        if os.path.isdir(args.svd_cache_path):
            print("Loading SVDed Model...")
            self.model = SORSAAutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.svd_cache_path,
                config=config,
            )
        else:
            print("Start from Pretrained Model...")
            self.model = SORSAAutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model,
                cache_dir=args.cache_path,
                config=config,
            )
            self.model.to("cuda")
            self.model._sorsa_init()
            self.model.save_pretrained(args.svd_cache_path)
        for name, param in self.model.named_parameters():
            if "sorsa_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.training_arguments = SORSATrainingArguments(
            run_name=args.name,
            output_dir=self.run_path,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.accum_steps,
            logging_dir=self.run_path,
            logging_steps=1,
            save_steps=1000,
            save_total_limit=1,
            report_to=["wandb"],
            save_strategy="steps",
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            seed=self.seed,
            weight_decay=args.wd,
            learning_rate=args.lr,
            gamma=self.gamma,
            bf16=True if args.bf16 else False,
            fp16=True if args.fp16 else False,
            tf32=True if args.tf32 else False,
        )


if args.test:
    if args.local is False:
        env = dotenv_values(".env")
        login(token=env["hf"])
    if args.tokenizer is None:
        args.tokenizer = args.model
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, cache_dir=args.cache_path, use_fast=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.math:
        dataset = "datasets/MATH_test.jsonl"
    else:
        dataset = "datasets/test-00000-of-00001.parquet"
    run_path = f"{args.run_path}/{args.name}"
    checkpoint_path = f"{run_path}/checkpoint"
    if args.math:
        test_math(
            model=checkpoint_path,
            tokenizer=args.tokenizer,
            dataset=dataset,
            precision=(
                "bfloat16" if args.bf16 else ("float16" if args.fp16 else "float32")
            ),
        )
    elif args.gsm_8k:
        test_gsm(
            model=checkpoint_path,
            tokenizer=args.tokenizer,
            dataset=dataset,
            precision=(
                "bfloat16" if args.bf16 else ("float16" if args.fp16 else "float32")
            ),
        )

elif args.train:
    config = TrainerConfig(args)
    print(config.training_arguments)
    trainer = Trainer(
        model=config.model,
        args=config.training_arguments,
        data_collator=lambda x: (
            collate_fn(x, config.tokenizer)
        ),
        train_dataset=config.train_dataset,
    )
    print(f"Trainable Parameters: {trainer.get_num_trainable_parameters()}")
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()
    trainer.save_model(config.checkpoint_path)

elif args.merge:
    run_path = f"{args.run_path}/{args.name}"
    checkpoint_path = f"{run_path}/checkpoint"
    model = SORSAAutoModelForCausalLM.from_pretrained(checkpoint_path)
    model.to("cuda")
    model.sorsa_merge(True)
    for name, param in model.named_parameters():
        if "sorsa_" in name:
            model.set_submodule(name, None)
    del model.replaced
    model.save_pretrained(checkpoint_path, save_config=False)
    if args.local is False:
        env = dotenv_values(".env")
        login(token=env["hf"])
    config = AutoConfig.from_pretrained(args.model)
    config.save_pretrained(checkpoint_path)
