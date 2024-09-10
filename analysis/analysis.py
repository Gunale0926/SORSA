import math
from typing import Optional
import torch
from torch._C import device
from torch.utils.data import DataLoader
from utils import save_checkpoint
from tqdm import tqdm
from accelerate import Accelerator
import sys
import os

sys.path.append(os.path.expanduser("../"))
from sorsalib import calc_ortho


def save_weight(model, step, metadata_path):
    weights = {}
    with torch.no_grad():
        model.sorsa_merge(True)
        progress_bar = tqdm(range(len(model.weights)))
        for name in model.weights:
            wt = model.get_parameter(name)
            weights[name] = wt
            progress_bar.update(1)
        torch.save(weights, f"{metadata_path}_{str(step)}.pt")
        progress_bar.close()
        model.sorsa_merge(False)


def analysis(
    model,
    train_loader: DataLoader,
    optimizer,
    s_gamma=0.005,
    scheduler=None,
    save_freq: int = 100,
    start_epoch: int = 0,
    num_epochs: int = 5,
    accum_steps: int = 1,
    step: int = 0,
    best_val_loss: float = float("inf"),
    checkpoint_path: str = "checkpoint.safetensors",
    metadata_path: str = "metadata.pt",
    device: device = torch.device("cpu"),
    sorsa: bool = False,
) -> object:
    print("Start training")
    accelerator = Accelerator(mixed_precision="bf16")
    model, train_loader = accelerator.prepare(model, train_loader)
    if scheduler is not None:
        scheduler.step()
    for epoch in range(start_epoch, num_epochs):
        model.train()
        step_loss = 0.0
        if sorsa:
            ortho_loss = 0.0
        epoch_step = math.ceil(len(train_loader) / accum_steps)
        skip = step % epoch_step
        print(f"Epoch Step: {epoch_step}\nSkip: {skip}")
        progress_bar = tqdm(enumerate(train_loader), total=epoch_step, initial=skip)
        if skip % save_freq == 0:
            save_weight(model, skip, metadata_path)
        try:
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx < skip * accum_steps:
                    continue
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    labels=batch["labels"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                del batch
                loss = outputs.loss / accum_steps
                if not torch.isnan(loss):
                    accelerator.backward(loss)
                    step_loss += loss.item()
                del loss

                # Gradient descending here
                if ((batch_idx + 1) % accum_steps) == 0 or (
                    (batch_idx + 1) == len(train_loader)
                    and not (batch_idx + 1) % accum_steps == 0
                ):
                    step += 1
                    if sorsa:
                        ortho_loss = calc_ortho(model)
                        accelerator.backward(s_gamma * ortho_loss)
                        del ortho_loss
                    optimizer.step()
                    progress_bar.set_description(
                        f"Epoch: {epoch + 1}/{num_epochs}, "
                        f"Batch Loss: {step_loss:.4f}, "
                        f"lr: {scheduler.get_last_lr()[0]:.10f}"
                        if scheduler is not None
                        else f"Epoch: {epoch + 1}/{num_epochs}, "
                        f"Batch Loss: {step_loss:.4f}, "
                    )
                    if scheduler is not None:
                        scheduler.step()

                    # Clean
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    step_loss = 0.0

                    if step % save_freq == 0:
                        print(f"Step: {step}, Saving...")
                        save_weight(model, step, metadata_path)

                    progress_bar.update(1)
            progress_bar.close()
            save_weight(model, step, metadata_path)
        except KeyboardInterrupt or Exception:
            print("Training interrupted, Saving...")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "best_val_loss": best_val_loss,
                    "step": step,
                },
                filename=f"{checkpoint_path}",
                metadata=f"{metadata_path}",
            )
            return
    return
