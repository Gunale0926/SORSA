import math
from typing import Optional
import torch
from torch._C import device
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from utils import save_checkpoint
from tqdm import tqdm
from accelerate import Accelerator
import sys
import os

sys.path.append(os.path.expanduser("../"))
from sorsalib.layers import calc_ortho


def save_weight(model, step, metadata_path):
    weights = {}
    with torch.no_grad():
        model.eval()  # Set model to eval mode in order to merge LoRA and SORSA
        progress_bar = tqdm(range(len(model.weights)))
        for name in model.weights:
            wt = model.get_parameter(name)
            weights[name] = wt
            progress_bar.update(1)
        torch.save(weights, f"{metadata_path}_{str(step)}.pt")
        progress_bar.close()
        model.train()


def analysis(
    model,
    W_0,
    train_loader: DataLoader,
    optimizer,
    gamma=0.1,
    writer: Optional[SummaryWriter] = None,
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
    accelerator.prepare(model, train_loader)
    if scheduler is not None:
        scheduler.step()
    for epoch in range(start_epoch, num_epochs):
        model.train()
        # optimizer.train()
        step_loss = 0.0
        if sorsa:
            ortho_loss = 0.0
        epoch_step = math.ceil(len(train_loader) / accum_steps)
        skip = step % epoch_step
        print(f"Epoch Step: {epoch_step}\nSkip: {skip}")
        progress_bar = tqdm(enumerate(train_loader), total=epoch_step, initial=skip)
        if skip % save_freq == 0:
            # log_metrics(writer, model, W_0, step)
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
                        accelerator.backward(gamma * ortho_loss)
                        del ortho_loss
                    optimizer.step()
                    total_norm = 0.0
                    for param in model.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm**0.5
                    progress_bar.set_description(
                        f"Epoch: {epoch + 1}/{num_epochs}, "
                        f"Batch Loss: {step_loss:.4f}, "
                        f"lr: {scheduler.get_last_lr()[0]:.10f}"
                        if scheduler is not None
                        else f"Epoch: {epoch + 1}/{num_epochs}, "
                        f"Batch Loss: {step_loss:.4f}, "
                    )
                    if writer is not None:
                        if scheduler is not None:
                            writer.add_scalar(
                                "Learning_Rate",
                                scheduler.get_last_lr()[0],
                                step,
                            )
                        writer.add_scalar(f"Loss/Train", step_loss, step)
                        if sorsa:
                            writer.add_scalar(f"Loss/Ortho", ortho_loss, step)
                        writer.add_scalar(f"Gradient Norm", total_norm, step)
                    if scheduler is not None:
                        scheduler.step()

                    # Clean
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    step_loss = 0.0

                    if step % save_freq == 0:
                        print(f"Step: {step}, Saving and Analyzing...")
                        # save_checkpoint(
                        #     {
                        #         "epoch": epoch,
                        #         "model": model,
                        #         "optimizer": optimizer,
                        #         "scheduler": scheduler,
                        #         "best_val_loss": best_val_loss,
                        #         "step": step,
                        #     },
                        #     filename=f"{checkpoint_path}",
                        #     metadata=f"{metadata_path}",
                        # )
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
