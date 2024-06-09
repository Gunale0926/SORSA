import torch
from transformers import (
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
    LlamaTokenizerFast,
)
from torch.utils.data import DataLoader, Subset
import argparse
from dotenv import dotenv_values
from huggingface_hub import login
import sys
import os
from transformers import set_seed

sys.path.append(os.path.expanduser('../'))

from models import LoRALlamaForCausalLM, SORSALlamaForCausalLM
from auto_model import SORSAAutoModelForCausalLM, SORSAAutoConfig
from dataset import MetaMathQADataset
from analysis import analysis

parser = argparse.ArgumentParser(
    prog="SORSA Analysis Runner",
    description="Train and Analysis SORSA Layers",
    epilog="--help for more information",
)

parser.add_argument("-n", "--name", type=str, default="SORSA")
parser.add_argument("-r", "--rank", type=int, default=4)
parser.add_argument("-a", "--alpha", type=float, default=None)
parser.add_argument("-g", "--gamma", type=float, default=0.005)
parser.add_argument("-d", "--dropout", type=float, default=0.0)
parser.add_argument("-l", "--local", action="store_true")
parser.add_argument("-f", "--freq", type=int, default=100)
parser.add_argument("--cache-path", type=str, default="./cache")
parser.add_argument("--tokenizer-path", type=str, default="")
parser.add_argument("--model-path", type=str, default="")
parser.add_argument("--svd-cache-path", type=str, default="./svd_cache")
parser.add_argument("--run-path", type=str, default="./runs")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--accum-steps", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.00002)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--lora", action="store_true")
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--pissa", action="store_true")
parser.add_argument("--sorsa", action="store_true")
parser.add_argument("--ft", action="store_true")


args = parser.parse_args()


class TrainerConfig:
    def __init__(self, args: argparse.Namespace):
        set_seed(args.seed)
        self.rank = args.rank
        self.dropout = args.dropout
        self.gamma = args.gamma
        self.batch_size = args.batch_size
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
        if args.local is False:
            env = dotenv_values(".env")
            login(token=env["hf"])
            llama_path = "meta-llama/Meta-Llama-3-8B"
            tokenizer_path = llama_path
        else:
            tokenizer_path = args.tokenizer_path
            llama_path = args.model_path
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            tokenizer_path,
            cache_dir=cache_path,
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.train_dataset = MetaMathQADataset(
            file_path="../llama/datasets/MetaMathQA-395K.json",
            tokenizer=self.tokenizer,
            max_length=512,
        )
        self.train_subset = Subset(self.train_dataset, range(0, 100000))
        self.train_loader = DataLoader(
            self.train_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.train_dataset.collate_fn,
        )
        self.num_batches = len(self.train_loader)
        self.total_steps = self.num_batches * self.num_epochs
        target = [
            "q_proj",
            "v_proj",
        ]

        save_target = [
            "q_proj",
            "v_proj",
        ]

        # Init W_0 (Model same as FT)
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=llama_path,
            cache_dir=cache_path,
        )
        self.W_0 = {}
        self.weights = []
        for t in save_target:
            for name, module in self.model.named_modules():
                if t in name and isinstance(module, torch.nn.Linear):
                    weight = f"{name}.weight"
                    self.W_0[weight] = module.weight.clone().to(self.device)
                    self.weights.append(weight)

        if args.sorsa:
            config = SORSAAutoConfig.from_pretrained(
                llama_path,
                target=target,
                rank=self.rank,
                alpha=self.alpha,
                sorsa_dropout=self.dropout,
            )
            if os.path.isdir(self.checkpoint_path):
                print("Loading Tuned Model...")
                self.model = SORSALlamaForCausalLM.from_pretrained(
                    self.checkpoint_path,
                    config=config
                )
            elif os.path.isdir(args.svd_cache_path):
                print("Loading SVDed Model...")
                self.model = SORSALlamaForCausalLM.from_pretrained(
                    args.svd_cache_path,
                    config=config
                )
            else:
                print("Start from Pretrained Model...")
                self.model = SORSALlamaForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=llama_path,
                    config=config
                )
                self.model._sorsa_init()
                # self.model.save_pretrained(args.svd_cache_path)
            self.model.to(self.device)
            self.sorsa_params = []
            for name, param in self.model.named_parameters():
                if "sorsa_" in name:
                    param.requires_grad = True
                    self.sorsa_params.append(param)
                else:
                    param.requires_grad = False
        elif args.lora or args.pissa:
            if os.path.isdir(self.checkpoint_path):
                print("Loading Tuned Model...")
                self.model = LoRALlamaForCausalLM.from_pretrained(
                    self.checkpoint_path,
                    target=target,
                    rank=self.rank,
                    lora_dropout=self.dropout,
                    lora_alpha=args.alpha
                )
            elif args.pissa and os.path.isdir(args.svd_cache_path):
                print("Loading SVDed Model...")
                self.model = LoRALlamaForCausalLM.from_pretrained(
                    args.svd_cache_path,
                    target=target,
                    rank=self.rank,
                    lora_dropout=self.dropout,
                    lora_alpha=args.alpha
                )
            else:
                print("Start from Pretrained Model...")
                self.model = LoRALlamaForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=llama_path,
                    cache_dir=cache_path,
                    target=target,
                    rank=self.rank,
                    lora_dropout=self.dropout,
                    lora_alpha=args.alpha
                )
                if args.pissa:
                    self.model.pissa_init()
                    self.model.save_pretrained(args.svd_cache_path)
                else:
                    self.model.lora_init()
            self.model.to(self.device)
            self.lora_params = []
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                    self.lora_params.append(param)
                else:
                    param.requires_grad = False
        elif args.ft:
            if os.path.isdir(self.checkpoint_path):
                print("Loading Tuned Model...")
                self.model = LlamaForCausalLM.from_pretrained(
                    self.checkpoint_path,
                )
            else:
                print("Start from Pretrained Model...")
            self.model.to(self.device)
            self.ft_params = []

            # Set require_grad
            for name, param in self.model.named_parameters():
                param.requires_grad = False

            for t in target:
                for name, module in self.model.named_modules():
                    if t in name and isinstance(module, torch.nn.Linear):
                        weight = f"{name}.weight"
                        param = self.model.get_parameter(weight)
                        param.requires_grad = True
                        self.ft_params.append(param)

        self.model.weights = self.weights

        self.train_param = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.train_param += param.numel()

        params = []
        if args.sorsa:
            params = self.sorsa_params
        elif args.lora or args.pissa:
            params = self.lora_params
        elif args.ft:
            params = self.ft_params
        self.optimizer = torch.optim.AdamW(
            [{"params": params, "lr": args.lr, "weight_decay": args.wd}]
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0.03 * self.total_steps / self.accum_steps,
            num_training_steps=self.total_steps / self.accum_steps,
        )


config = TrainerConfig(args)
print(f"Trainable Parameters: {config.train_param / 1000**2}M")

if os.path.isfile(config.metadata_path):
    print(f"=> loaded optimizer state from '{config.metadata_path}'")
    metadata = torch.load(config.metadata_path)
    config.optimizer.load_state_dict(metadata["optimizer"])
    if config.scheduler is not None:
        config.scheduler.load_state_dict(metadata["scheduler"])
    start_epoch = metadata["epoch"]
    step = metadata["step"]
    best_val_loss = metadata["best_val_loss"]
    loaded = True
else:
    print(f"=> no optimizer found")
    start_epoch = 0
    step = 0
    best_val_loss = float("inf")
    loaded = False


analysis(
    model=config.model,
    train_loader=config.train_loader,
    s_gamma=(args.gamma / args.lr),
    optimizer=config.optimizer,
    scheduler=config.scheduler,
    save_freq=args.freq,
    device=config.device,
    start_epoch=start_epoch,
    num_epochs=config.num_epochs,
    accum_steps=config.accum_steps,
    checkpoint_path=config.checkpoint_path,
    metadata_path=config.metadata_path,
    step=step,
    best_val_loss=best_val_loss,
    sorsa=args.sorsa
)
