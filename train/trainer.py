"""
trainer.py
==========
Base trainer yang dipakai SEMUA model secara seragam.
Protokol identik = fair benchmark.

WandB integration: aktifkan dengan use_wandb=True di config.
"""

import csv
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.losses import SegmentationLoss

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False


# ─────────────────────────────────────────────
#  KONSTANTA KELAS & VISUALISASI
# ─────────────────────────────────────────────

_CLASS_NAMES = [
    "crack", "alligator_crack", "efflorescence", "exposed_rebars",
    "graffiti", "hollowareas", "joint_tape", "restformwork",
    "rockpocket", "rust", "spalling", "washouts", "weathering",
    "wetspot", "cavity", "bearing", "drainage", "expansion_joint",
    "protective_equipment",
]

# Warna unik per kelas untuk visualisasi segmentasi
_PALETTE = np.array([
    [220,  20,  60], [119,  11,  32], [  0,   0, 142], [  0,   0, 230],
    [106,   0, 228], [  0,  60, 100], [  0,  80, 100], [  0,   0,  70],
    [  0,   0, 192], [250, 170,  30], [100, 170,  30], [220, 220,   0],
    [175, 116, 175], [250,   0,  30], [165,  42,  42], [255,  77, 255],
    [  0, 226, 252], [182, 182, 255], [  0,  82,   0],
], dtype=np.uint8)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ─────────────────────────────────────────────
#  CONFIG DEFAULT
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "epochs"          : 50,
    "lr"              : 1e-4,
    "weight_decay"    : 1e-4,
    "batch_size"      : 8,
    "loss_alpha"      : 0.5,
    "use_focal"       : False,
    "scheduler"       : "cosine",
    "warmup_epochs"   : 3,
    "patience"        : 10,
    "use_amp"         : True,
    "num_classes"     : 19,
    "ignore_index"    : 255,
    "img_size"        : 512,
    "output_dir"      : "outputs",
    "model_name"      : "model",
    # ── WandB ──────────────────────────────────
    "use_wandb"       : False,
    "wandb_project"   : "bridge-benchmark-cv",
    "wandb_entity"    : None,   # username/org WandB, None = default
    "wandb_log_images": True,   # log sample predictions sebagai gambar
    "wandb_image_freq": 5,      # log gambar setiap N epoch
    "wandb_n_samples" : 4,      # jumlah sampel yang di-log per sesi
}


# ─────────────────────────────────────────────
#  TRAINER CLASS
# ─────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model         : nn.Module,
        train_loader  : DataLoader,
        val_loader    : DataLoader,
        config        : dict = None,
        device        : str  = "cuda",
        class_weights : Optional[torch.Tensor] = None,
    ):
        self.config       = {**DEFAULT_CONFIG, **(config or {})}
        self.device       = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.out_dir = Path(self.config["output_dir"]) / self.config["model_name"]
        self.out_dir.mkdir(parents=True, exist_ok=True)

        weights = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = SegmentationLoss(
            num_classes  = self.config["num_classes"],
            ignore_index = self.config["ignore_index"],
            alpha        = self.config["loss_alpha"],
            class_weights= weights,
            use_focal    = self.config["use_focal"],
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = self.config["lr"],
            weight_decay = self.config["weight_decay"],
        )

        self.scheduler = self._build_scheduler()
        self.scaler    = GradScaler("cuda", enabled=self.config["use_amp"])

        self.metric = JaccardIndex(
            task         = "multiclass",
            num_classes  = self.config["num_classes"],
            average      = "none",
            ignore_index = self.config["ignore_index"],
        ).to(self.device)

        self.best_miou    = 0.0
        self.patience_ctr = 0
        self.history      = []

        self.csv_path = self.out_dir / "training_log.csv"
        self._init_csv()

        # ── WandB init ──────────────────────────
        self._wandb_on = self.config["use_wandb"] and _WANDB_OK
        if self.config["use_wandb"] and not _WANDB_OK:
            print("[WandB] wandb tidak terinstall — jalankan: pip install wandb")
        if self._wandb_on:
            wandb.init(
                project = self.config["wandb_project"],
                entity  = self.config.get("wandb_entity"),
                name    = self.config["model_name"],
                config  = self.config,
                resume  = "allow",
                dir     = str(self.out_dir),
            )
            print(f"[WandB] Dashboard: {wandb.run.url}")

        print(f"\n{'='*55}")
        print(f"  Trainer : {self.config['model_name']}")
        print(f"  Device  : {self.device}")
        print(f"  Epochs  : {self.config['epochs']}")
        print(f"  LR      : {self.config['lr']}")
        print(f"  AMP     : {self.config['use_amp']}")
        print(f"  WandB   : {self._wandb_on}")
        print(f"  Output  : {self.out_dir}")
        print(f"{'='*55}\n")

    # ─────────────────────────────────────────────
    #  HELPER: BUILD SCHEDULER
    # ─────────────────────────────────────────────

    def _build_scheduler(self):
        sched  = self.config["scheduler"]
        epochs = self.config["epochs"]
        if sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6)
        elif sched == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=15, gamma=0.5)
        elif sched == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.5)
        else:
            raise ValueError(f"Scheduler tidak dikenal: {sched}")

    # ─────────────────────────────────────────────
    #  HELPER: CSV LOGGING
    # ─────────────────────────────────────────────

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_ce", "train_dice",
                "val_loss", "val_miou", "lr", "time_s"
            ])

    def _log_csv(self, row: dict):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                row["epoch"],
                f"{row['train_loss']:.4f}",
                f"{row['train_ce']:.4f}",
                f"{row['train_dice']:.4f}",
                f"{row['val_loss']:.4f}",
                f"{row['val_miou']:.4f}",
                f"{row['lr']:.2e}",
                f"{row['time_s']:.1f}",
            ])

    # ─────────────────────────────────────────────
    #  HELPER: WANDB LOGGING
    # ─────────────────────────────────────────────

    def _wandb_log_metrics(self, epoch: int, row: dict,
                           miou_per_class: List[float]):
        """Log scalar metrics + per-class IoU ke WandB."""
        if not self._wandb_on:
            return

        log_dict = {
            "train/loss"      : row["train_loss"],
            "train/loss_ce"   : row["train_ce"],
            "train/loss_dice" : row["train_dice"],
            "val/loss"        : row["val_loss"],
            "val/mIoU"        : row["val_miou"],
            "lr"              : row["lr"],
            "epoch_time_s"    : row["time_s"],
        }

        # IoU per kelas — bisa diplot sebagai grouped bar di WandB
        for cls_name, iou in zip(_CLASS_NAMES, miou_per_class):
            log_dict[f"val/iou/{cls_name}"] = float(iou)

        wandb.log(log_dict, step=epoch)

    def _mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        """Mask (H,W) → RGB (H,W,3) menggunakan color palette."""
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls_idx, color in enumerate(_PALETTE):
            rgb[mask == cls_idx] = color
        return rgb

    def _denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Tensor (3,H,W) → numpy uint8 (H,W,3), denormalisasi ImageNet."""
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = (img * _IMAGENET_STD + _IMAGENET_MEAN).clip(0, 1)
        return (img * 255).astype(np.uint8)

    @torch.no_grad()
    def _wandb_log_predictions(self, epoch: int):
        """
        Log N sampel prediksi segmentasi ke WandB.
        Setiap gambar: Image | Ground Truth | Prediction (side-by-side).
        """
        if not self._wandb_on or not self.config["wandb_log_images"]:
            return

        self.model.eval()
        n = self.config["wandb_n_samples"]

        batch   = next(iter(self.val_loader))
        images  = batch["image"][:n].to(self.device)
        targets = batch["mask"][:n].cpu().numpy()

        with autocast(device_type="cuda", enabled=self.config["use_amp"]):
            logits = self.model(images)
            if isinstance(logits, dict):
                logits = logits.get("logits", logits.get("out"))
            if logits.shape[-2:] != images.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=images.shape[-2:],
                    mode="bilinear", align_corners=False)

        preds = logits.argmax(dim=1).cpu().numpy()

        panels = []
        for i in range(min(n, len(images))):
            img_rgb  = self._denormalize(images[i])
            gt_rgb   = self._mask_to_rgb(
                np.where(targets[i] == self.config["ignore_index"], 0, targets[i])
            )
            pred_rgb = self._mask_to_rgb(preds[i])

            # Tiga panel side-by-side
            panel = np.concatenate([img_rgb, gt_rgb, pred_rgb], axis=1)
            panels.append(
                wandb.Image(panel, caption=f"Input | GT | Pred  [ep {epoch}]")
            )

        wandb.log({"val/predictions": panels}, step=epoch)

    def _wandb_log_artifact(self, epoch: int, miou: float):
        """Upload best_model.pth ke WandB Artifacts (model registry)."""
        if not self._wandb_on:
            return
        artifact = wandb.Artifact(
            name     = f"{self.config['model_name']}-best",
            type     = "model",
            metadata = {
                "epoch"   : epoch,
                "val_mIoU": round(miou, 4),
                "dataset" : "dacl10k",
            },
        )
        artifact.add_file(str(self.out_dir / "best_model.pth"))
        wandb.log_artifact(artifact)
        print(f"  [WandB] Artifact uploaded: {artifact.name} (mIoU={miou:.4f})")

    # ─────────────────────────────────────────────
    #  TRAINING & VALIDATION EPOCH
    # ─────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = total_ce = total_dice = 0.0
        n_batches  = len(self.train_loader)

        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch:3d} [Train]", leave=False)

        for batch in pbar:
            images  = batch["image"].to(self.device)
            targets = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=self.config["use_amp"]):
                logits = self.model(images)
                if isinstance(logits, dict):
                    logits = logits.get("logits", logits.get("out"))
                if logits.shape[-2:] != targets.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=targets.shape[-2:],
                        mode="bilinear", align_corners=False)
                losses = self.criterion(logits, targets)

            self.scaler.scale(losses["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["loss"].item()
            total_ce   += losses["loss_ce"].item()
            total_dice += losses["loss_dice"].item()

            pbar.set_postfix({
                "loss": f"{losses['loss'].item():.3f}",
                "dice": f"{losses['loss_dice'].item():.3f}",
            })

        return {
            "train_loss" : total_loss / n_batches,
            "train_ce"   : total_ce   / n_batches,
            "train_dice" : total_dice / n_batches,
        }

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.model.eval()
        self.metric.reset()
        total_loss = 0.0
        n_batches  = len(self.val_loader)

        pbar = tqdm(self.val_loader,
                    desc="           [Val]  ", leave=False)

        for batch in pbar:
            images  = batch["image"].to(self.device)
            targets = batch["mask"].to(self.device)

            with autocast(device_type="cuda", enabled=self.config["use_amp"]):
                logits = self.model(images)
                if isinstance(logits, dict):
                    logits = logits.get("logits", logits.get("out"))
                if logits.shape[-2:] != targets.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=targets.shape[-2:],
                        mode="bilinear", align_corners=False)
                losses = self.criterion(logits, targets)

            total_loss += losses["loss"].item()
            preds = logits.argmax(dim=1)
            self.metric.update(preds, targets)

        miou_per_class = self.metric.compute()
        miou_mean      = miou_per_class.mean().item()
        self.metric.reset()

        return {
            "val_loss"          : total_loss / n_batches,
            "val_miou"          : miou_mean,
            "val_miou_per_class": miou_per_class.cpu().tolist(),
        }

    # ─────────────────────────────────────────────
    #  CHECKPOINT
    # ─────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, miou: float, is_best: bool):
        state = {
            "epoch"      : epoch,
            "model_name" : self.config["model_name"],
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "best_miou"  : miou,
            "config"     : self.config,
        }
        torch.save(state, self.out_dir / "last_checkpoint.pth")
        if is_best:
            torch.save(state, self.out_dir / "best_model.pth")
            print(f"  Best model saved! mIoU={miou:.4f}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.best_miou = ckpt.get("best_miou", 0.0)
        start_epoch    = ckpt.get("epoch", 0) + 1
        print(f"Checkpoint loaded: epoch={ckpt['epoch']}, mIoU={self.best_miou:.4f}")
        return start_epoch

    # ─────────────────────────────────────────────
    #  MAIN TRAINING LOOP
    # ─────────────────────────────────────────────

    def fit(self, start_epoch: int = 1):
        print(f"Mulai training: {self.config['model_name']}")
        print(f"Train batches : {len(self.train_loader)}")
        print(f"Val batches   : {len(self.val_loader)}\n")

        for epoch in range(start_epoch, self.config["epochs"] + 1):
            t0 = time.time()

            # Warmup LR
            if epoch <= self.config["warmup_epochs"]:
                warmup_factor = epoch / self.config["warmup_epochs"]
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.config["lr"] * warmup_factor

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch()

            if self.config["scheduler"] == "plateau":
                self.scheduler.step(val_metrics["val_miou"])
            elif epoch > self.config["warmup_epochs"]:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed    = time.time() - t0

            row = {
                "epoch": epoch, "lr": current_lr, "time_s": elapsed,
                **train_metrics, **val_metrics,
            }
            self.history.append(row)
            self._log_csv(row)

            # ── WandB: log metrics setiap epoch ──
            self._wandb_log_metrics(epoch, row,
                                    val_metrics["val_miou_per_class"])

            # ── WandB: log prediksi visual setiap N epoch ──
            if epoch % self.config["wandb_image_freq"] == 0:
                self._wandb_log_predictions(epoch)

            is_best = val_metrics["val_miou"] > self.best_miou
            marker  = " ⭐" if is_best else ""
            print(
                f"Epoch {epoch:3d}/{self.config['epochs']} | "
                f"Loss={train_metrics['train_loss']:.3f} | "
                f"Dice={train_metrics['train_dice']:.3f} | "
                f"Val_mIoU={val_metrics['val_miou']:.4f} | "
                f"LR={current_lr:.1e} | "
                f"{elapsed:.0f}s{marker}"
            )

            if is_best:
                self.best_miou    = val_metrics["val_miou"]
                self.patience_ctr = 0
                self._save_checkpoint(epoch, val_metrics["val_miou"], is_best=True)
                # ── WandB: upload artifact saat best model baru ──
                self._wandb_log_artifact(epoch, val_metrics["val_miou"])
            else:
                self.patience_ctr += 1
                self._save_checkpoint(epoch, val_metrics["val_miou"], is_best=False)

            if self.patience_ctr >= self.config["patience"]:
                print(f"\nEarly stopping di epoch {epoch}.")
                break

        print(f"\n{'='*55}")
        print(f"  Training selesai : {self.config['model_name']}")
        print(f"  Best val mIoU   : {self.best_miou:.4f}")
        print(f"  Log             : {self.csv_path}")
        print(f"  Checkpoint      : {self.out_dir / 'best_model.pth'}")
        print(f"{'='*55}\n")

        return self.best_miou

    def wandb_finish(self):
        """Tutup WandB run. Panggil setelah semua evaluasi (test + robustness) selesai."""
        if self._wandb_on and wandb.run is not None:
            wandb.finish()
            print("[WandB] Run selesai.")
