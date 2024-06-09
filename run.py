import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, Subset
from auto_model import SORSAAutoModelForCausalLM, SORSAAutoConfig
from models import SORSATrainingArguments
from transformers import AutoTokenizer
from dataset import MATHDataset, MetaMathQADataset, GSM8KDataset
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
parser.add_argument("-g", "--gamma", type=float, default=0.1)
parser.add_argument("-d", "--dropout", type=float, default=0.0)
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-T", "--test", action="store_true")
parser.add_argument("-l", "--local", action="store_true")
parser.add_argument("--cache-path", type=str, default="./cache")
parser.add_argument("--tokenizer-path", type=str, default="")
parser.add_argument("--model-path", type=str, default="")
parser.add_argument("--svd-cache-path", type=str, default="./svd_cache")
parser.add_argument("--run-path", type=str, default="./runs")
parser.add_argument("--llama3", action="store_true")
parser.add_argument("--gemma", action="store_true")
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
        model_name = args.name
        cache_path = args.cache_path
        self.run_path = f"{args.run_path}/{model_name}"
        self.metadata_path = f"{self.run_path}/metadata.pt"
        self.checkpoint_path = f"{self.run_path}/checkpoint"
        self.writer = SummaryWriter(self.run_path)
        if args.local is False:
            env = dotenv_values(".env")
            login(token=env["hf"])
            if args.llama3:
                llama_path = "meta-llama/Meta-Llama-3-8B"
            elif args.gemma:
                llama_path = "google/gemma-7b"
            else:
                llama_path = "meta-llama/Llama-2-7b-hf"
            tokenizer_path = llama_path
        else:
            llama_path = args.model_path
            tokenizer_path = args.tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, cache_dir=cache_path, use_fast=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # self.train_dataset = GSM8KDataset(
        #     file_path="llama/datasets/train-00000-of-00001.parquet",
        #     tokenizer=self.tokenizer,
        #     # max_length=512,
        # )
        self.train_dataset = MetaMathQADataset(
            file_path="llama/datasets/MetaMathQA-395K.json",
            tokenizer=self.tokenizer,
            max_length=512,
        )
        self.train_subset = Subset(self.train_dataset, range(0, 100000))
        if args.gsm_8k:
            self.valid_dataset = GSM8KDataset(
                file_path="llama/datasets/test-00000-of-00001.parquet",
                tokenizer=self.tokenizer,
                max_length=1024,
            )
        elif args.math:
            self.valid_dataset = MATHDataset(
                file_path="llama/datasets/MATH_test.jsonl",
                tokenizer=self.tokenizer,
                max_length=2048,
            )
        else:
            self.valid_dataset = None
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
            llama_path,
            target=target,
            rank=self.rank,
            sorsa_dropout=self.dropout,
        )
        self.loaded = False
        if os.path.isdir(self.checkpoint_path):
            print("Loading Tuned Model...")
            self.model = SORSAAutoModelForCausalLM.from_pretrained(
                self.checkpoint_path,
                config=config,
            )
            self.loaded = True
        elif os.path.isdir(args.svd_cache_path):
            print("Loading SVDed Model...")
            self.model = SORSAAutoModelForCausalLM.from_pretrained(
                args.svd_cache_path,
                config=config,
            )
        else:
            print("Start from Pretrained Model...")
            self.model = SORSAAutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=llama_path,
                cache_dir=cache_path,
                config=config,
            )
            self.model._sorsa_init()
            self.model.save_pretrained(args.svd_cache_path)
        self.sorsa_params = []
        for name, param in self.model.named_parameters():
            if "sorsa_" in name:
                param.requires_grad = True
                self.sorsa_params.append(param)
            else:
                param.requires_grad = False

        self.train_param = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.train_param += param.numel()

        self.training_arguments = SORSATrainingArguments(
            output_dir=self.run_path,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.accum_steps,
            logging_dir=self.run_path,
            logging_steps=1,
            save_steps=1000,
            save_total_limit=1,
            report_to=["tensorboard"],
            save_strategy="steps",
            seed=self.seed,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=args.wd,
            learning_rate=args.lr,
            gamma=self.gamma,
            bf16=True,
            tf32=True,
        )


config = TrainerConfig(args)
print(f"Trainable Parameters: {config.train_param / 1000**2}M")


def do_test():
    test_loader = DataLoader(
        config.valid_dataset,
        batch_size=6,
        shuffle=False,
        num_workers=0,
        collate_fn=config.valid_dataset.collate_fn,
    )
    if args.math:
        test_math(
            model=config.model,
            tokenizer=config.tokenizer,
            device=config.device,
            dataset=test_loader,
        )
    elif args.gsm_8k:
        test_gsm(
            model=config.model,
            tokenizer=config.tokenizer,
            device=config.device,
            dataset=test_loader,
        )


if args.test:
    config.model.to(config.device)
    if config.loaded:
        config.model.sorsa_merge(True)
        do_test()
    else:
        raise RuntimeError("No Tuned Model Found!")

elif args.train:
    trainer = Trainer(
        model=config.model,
        args=config.training_arguments,
        data_collator=config.train_dataset.collate_fn,
        train_dataset=config.train_subset,
    )
    # trainer.train(resume_from_checkpoint=True)
    print(trainer.get_num_trainable_parameters())
    trainer.train()
    trainer.save_model(config.checkpoint_path)
    trainer.save_state()
