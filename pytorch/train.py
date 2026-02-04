"""BSRoformer training loop for NVIDIA GH200.

Usage:
    python -m training.train --config training/phase_1/config.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import yaml

from .factory import create_model, model_summary, save_checkpoint
from .dataset import BSRoformerDataset
from .losses import MultiResolutionSTFTLoss, combined_loss


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup."""
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        target_lr = min_lr + (base_lr - min_lr) * cosine_decay
        return target_lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Seed
    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = create_model(config)
    print(model_summary(model))

    gc = config.get("hardware", {}).get("gradient_checkpointing", True)
    model.gradient_checkpointing = gc

    model = model.to(device)

    precision = config.get("hardware", {}).get("precision", "bf16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    # Dataset
    data_cfg = config["data"]
    train_dataset = BSRoformerDataset(
        root=data_cfg["train_dir"],
        segment_samples=data_cfg.get("segment_samples", 588800),
        sample_rate=data_cfg.get("sample_rate", 44100),
        augment=True,
    )
    print(f"Training songs: {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
    )

    # Validation (optional)
    val_loader = None
    if data_cfg.get("val_dir") and os.path.isdir(data_cfg["val_dir"]):
        val_dataset = BSRoformerDataset(
            root=data_cfg["val_dir"],
            segment_samples=data_cfg.get("segment_samples", 588800),
            sample_rate=data_cfg.get("sample_rate", 44100),
            augment=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=data_cfg.get("pin_memory", True),
        )
        print(f"Validation songs: {len(val_dataset)}")

    # Optimizer
    opt_cfg = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
        eps=opt_cfg.get("eps", 1e-8),
    )

    # Loss
    loss_cfg = config["training"]["loss"]
    mrstft = MultiResolutionSTFTLoss(
        resolutions=[tuple(r) for r in loss_cfg["mrstft_resolutions"]]
    ).to(device)

    # Scheduler
    sched_cfg = config["training"]["scheduler"]
    grad_accum = config["training"].get("gradient_accumulation", 1)
    steps_per_epoch = len(train_loader) // grad_accum
    max_epochs = config["training"]["max_epochs"]
    total_steps = steps_per_epoch * max_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=sched_cfg.get("warmup_steps", 1000),
        total_steps=total_steps,
        min_lr=sched_cfg.get("min_lr", 1e-6),
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    # Checkpoint dir
    ckpt_cfg = config.get("checkpoint", {})
    save_dir = ckpt_cfg.get("save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            mixture = batch["mixture"].to(device)      # [B, 2, T]
            targets = batch["stems"].to(device)         # [B, 6, 2, T]

            with torch.amp.autocast("cuda", dtype=dtype):
                pred = model(mixture)                   # [B, 6, 2, T]
                losses = combined_loss(
                    pred, targets, mrstft,
                    l1_weight=loss_cfg["l1_weight"],
                    mrstft_weight=loss_cfg["mrstft_weight"],
                )
                loss = losses["total"] / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            epoch_loss += losses["total"].item()

            if (batch_idx + 1) % 50 == 0:
                avg = epoch_loss / (batch_idx + 1)
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [{epoch+1}] step {batch_idx+1}/{len(train_loader)} "
                    f"loss={avg:.4f} l1={losses['l1'].item():.4f} "
                    f"mrstft={losses['mrstft'].item():.4f} lr={lr:.2e}"
                )

        epoch_loss /= len(train_loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{max_epochs} — train_loss={epoch_loss:.4f} ({elapsed:.0f}s)")

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    mixture = batch["mixture"].to(device)
                    targets = batch["stems"].to(device)
                    with torch.amp.autocast("cuda", dtype=dtype):
                        pred = model(mixture)
                        losses = combined_loss(
                            pred, targets, mrstft,
                            l1_weight=loss_cfg["l1_weight"],
                            mrstft_weight=loss_cfg["mrstft_weight"],
                        )
                    val_loss += losses["total"].item()
            val_loss /= len(val_loader)
            print(f"  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, global_step, val_loss,
                    os.path.join(save_dir, "best.pt"),
                )
                print(f"  Saved best model (val_loss={val_loss:.4f})")

        # Periodic save
        save_every = ckpt_cfg.get("save_every_n_epochs", 5)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, global_step, epoch_loss,
                os.path.join(save_dir, f"epoch_{epoch+1:04d}.pt"),
            )

    # Final save
    save_checkpoint(
        model, optimizer, max_epochs - 1, global_step, epoch_loss,
        os.path.join(save_dir, "final.pt"),
    )
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    train(args.config)
