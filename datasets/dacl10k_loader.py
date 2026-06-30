"""
dacl10k_loader.py  (Supervisely format — Dataset Ninja)
========================================================
DataLoader untuk dacl10k yang didownload via datasetninja.com
Format: Supervisely (bukan COCO)

Struktur folder yang diharapkan:
    data/dacl10k/
    ├── meta.json          ← class definitions
    ├── train/
    │   ├── ann/           ← 1 JSON per gambar (nama: img.jpg.json)
    │   └── img/           ← file gambar
    ├── val/
    │   ├── ann/
    │   └── img/
    └── test/
        ├── ann/
        └── img/

19 kelas total (13 damage + 6 komponen), sesuai meta.json
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────
#  KONSTANTA KELAS (sesuai meta.json)
# ─────────────────────────────────────────────

# 13 kelas damage
DAMAGE_CLASSES = [
    "crack",                      # 0
    "alligator crack",            # 1
    "efflorescence",              # 2
    "exposed rebars",             # 3
    "graffiti",                   # 4
    "hollowareas",                # 5
    "joint tape",                 # 6
    "restformwork",               # 7
    "rockpocket",                 # 8
    "rust",                       # 9
    "spalling",                   # 10
    "washouts/concrete corrosion",# 11
    "weathering",                 # 12
    "wetspot",                    # 13
    "cavity",                     # 14
]

# 6 kelas komponen/objek jembatan
COMPONENT_CLASSES = [
    "bearing",                    # 15
    "drainage",                   # 16
    "expansion joint",            # 17
    "protective equipment",       # 18
]

# Semua kelas (urutan sesuai index training)
ALL_CLASSES = DAMAGE_CLASSES + COMPONENT_CLASSES

# Map nama kelas → index (dari meta.json, diurutkan alfabetis sesuai Dataset Ninja)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(ALL_CLASSES)}

# Map class_id Supervisely → index training kita
# (diambil dari meta.json)
SUPERVISELY_ID_TO_IDX = {
    6510463: CLASS_TO_IDX["alligator crack"],
    6510470: CLASS_TO_IDX["bearing"],
    6510460: CLASS_TO_IDX["cavity"],
    6510473: CLASS_TO_IDX["crack"],
    6510461: CLASS_TO_IDX["drainage"],
    6510462: CLASS_TO_IDX["efflorescence"],
    6510471: CLASS_TO_IDX["expansion joint"],
    6510469: CLASS_TO_IDX["exposed rebars"],
    6510466: CLASS_TO_IDX["graffiti"],
    6510472: CLASS_TO_IDX["hollowareas"],
    6510464: CLASS_TO_IDX["joint tape"],
    6510458: CLASS_TO_IDX["protective equipment"],
    6510467: CLASS_TO_IDX["restformwork"],
    6510475: CLASS_TO_IDX["rockpocket"],
    6510459: CLASS_TO_IDX["rust"],
    6510465: CLASS_TO_IDX["spalling"],
    6510474: CLASS_TO_IDX["washouts/concrete corrosion"],
    6510457: CLASS_TO_IDX["weathering"],
    6510468: CLASS_TO_IDX["wetspot"],
}

NUM_CLASSES  = len(ALL_CLASSES)   # 19
IGNORE_INDEX = 255

# ImageNet normalisasi
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────

def get_train_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.2),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1, p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(),
        ], p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_robust_transforms(
    img_size: int = 512,
    corruption: str = "blur"
) -> A.Compose:
    """Simulasi kondisi lapangan: blur | noise | brightness."""
    assert corruption in ("blur", "noise", "brightness")
    corruption_aug = {
        "blur"      : A.GaussianBlur(blur_limit=(7, 11), p=1.0),
        "noise"     : A.GaussNoise(var_limit=(80.0, 150.0), p=1.0),
        "brightness": A.RandomBrightnessContrast(
                        brightness_limit=(-0.5, 0.5),
                        contrast_limit=0.4, p=1.0),
    }[corruption]

    return A.Compose([
        A.Resize(img_size, img_size),
        corruption_aug,
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
#  DATASET CLASS
# ─────────────────────────────────────────────

class Dacl10kDataset(Dataset):
    """
    PyTorch Dataset untuk dacl10k (Supervisely format dari Dataset Ninja).

    Args:
        root_dir  : path ke folder dacl10k/ (yang berisi meta.json)
        split     : "train" | "val" | "test"
        transform : albumentations Compose object
        img_size  : ukuran input model
        debug     : jika True, hanya load 50 sampel pertama
    """

    def __init__(
        self,
        root_dir  : str,
        split     : str = "train",
        transform : Optional[Callable] = None,
        img_size  : int = 512,
        debug     : bool = False,
    ):
        assert split in ("train", "val", "test"), \
            "split harus: 'train', 'val', atau 'test'"

        self.root_dir = Path(root_dir)
        self.split    = split
        self.img_size = img_size
        self.debug    = debug

        self.transform = transform or (
            get_train_transforms(img_size) if split == "train"
            else get_val_transforms(img_size)
        )

        # Supervisely: gambar di img/, annotation di ann/
        self.img_dir = self.root_dir / split / "img"
        self.ann_dir = self.root_dir / split / "ann"

        self._validate_dirs()
        self.samples = self._load_samples()

        if self.debug:
            self.samples = self.samples[:50]

        print(f"[Dacl10kDataset] split={split} | "
              f"samples={len(self.samples)} | "
              f"img_size={img_size} | "
              f"num_classes={NUM_CLASSES}")

    def _validate_dirs(self):
        """Pastikan folder img/ dan ann/ ada."""
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Folder gambar tidak ditemukan: {self.img_dir}\n"
                f"Struktur yang diharapkan: {self.root_dir}/{self.split}/img/"
            )
        if not self.ann_dir.exists():
            raise FileNotFoundError(
                f"Folder annotasi tidak ditemukan: {self.ann_dir}\n"
                f"Struktur yang diharapkan: {self.root_dir}/{self.split}/ann/"
            )

    def _load_samples(self) -> List[dict]:
        """
        Scan folder img/ dan pasangkan dengan annotation di ann/.
        Supervisely: annotation bernama <namagambar>.json
        Contoh: img001.jpg → ann/img001.jpg.json
        """
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        samples  = []

        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in img_exts:
                continue

            # Supervisely: ann file = img_filename + ".json"
            ann_path = self.ann_dir / (img_path.name + ".json")
            if not ann_path.exists():
                # Fallback: coba tanpa ekstensi gambar
                ann_path_alt = self.ann_dir / (img_path.stem + ".json")
                if ann_path_alt.exists():
                    ann_path = ann_path_alt
                else:
                    # Skip jika tidak ada annotation
                    continue

            samples.append({
                "img_path" : str(img_path),
                "ann_path" : str(ann_path),
                "image_id" : img_path.stem,
            })

        if len(samples) == 0:
            raise FileNotFoundError(
                f"Tidak ada gambar+annotation di {self.img_dir}\n"
                f"Pastikan struktur folder sudah benar:\n"
                f"  {self.root_dir}/{self.split}/img/  ← gambar\n"
                f"  {self.root_dir}/{self.split}/ann/  ← JSON annotation"
            )
        return samples

    def _build_mask(self, ann_data: dict) -> np.ndarray:
        """
        Bangun mask segmentasi dari Supervisely annotation.

        Format Supervisely:
          ann_data["size"]    = {"height": H, "width": W}
          ann_data["objects"] = [{"classTitle": ..., "labelsMap": ...,
                                  "points": {"exterior": [[x,y],...],
                                             "interior": []}}]

        Returns:
            mask: np.ndarray (H, W) uint8, nilai = class index, 255 = ignore
        """
        h = ann_data["size"]["height"]
        w = ann_data["size"]["width"]
        mask = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)

        for obj in ann_data.get("objects", []):
            class_title = obj.get("classTitle", "")

            # Cari index kelas
            if class_title not in CLASS_TO_IDX:
                continue  # skip kelas tidak dikenal
            class_idx = CLASS_TO_IDX[class_title]

            # Gambar polygon exterior
            exterior = obj.get("points", {}).get("exterior", [])
            if len(exterior) < 3:
                continue  # polygon tidak valid

            pts = np.array(exterior, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], color=int(class_idx))

            # Handle interior (lubang dalam polygon)
            for interior in obj.get("points", {}).get("interior", []):
                if len(interior) >= 3:
                    pts_in = np.array(interior, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(mask, [pts_in], color=IGNORE_INDEX)

        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load gambar
        image = cv2.imread(sample["img_path"])
        if image is None:
            raise IOError(f"Gagal load gambar: {sample['img_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load & parse annotation
        with open(sample["ann_path"], "r") as f:
            ann_data = json.load(f)

        # Build mask
        mask = self._build_mask(ann_data)

        # Simpan ignore pixels sebelum augmentasi
        ignore_pixels = (mask == IGNORE_INDEX)
        mask_aug = mask.copy()
        mask_aug[ignore_pixels] = 0  # sementara isi 0

        # Apply transforms
        augmented = self.transform(image=image, mask=mask_aug)
        image_t   = augmented["image"]   # Tensor (3, H, W)
        mask_t    = augmented["mask"]    # Tensor (H, W)

        # Restore ignore pixels setelah resize
        ignore_resized = cv2.resize(
            ignore_pixels.astype(np.uint8),
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        mask_t[torch.from_numpy(ignore_resized)] = IGNORE_INDEX

        return {
            "image"    : image_t,           # FloatTensor (3, H, W)
            "mask"     : mask_t.long(),     # LongTensor  (H, W)
            "image_id" : sample["image_id"],
            "img_path" : sample["img_path"],
        }


# ─────────────────────────────────────────────
#  DATALOADER FACTORY
# ─────────────────────────────────────────────

def build_dataloaders(
    root_dir    : str,
    img_size    : int = 512,
    batch_size  : int = 8,
    num_workers : int = 4,
    pin_memory  : bool = True,
    debug       : bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Buat train / val / test DataLoader sekaligus.

    Returns:
        train_loader, val_loader, test_loader
    """
    splits = {
        "train": get_train_transforms(img_size),
        "val"  : get_val_transforms(img_size),
        "test" : get_val_transforms(img_size),
    }
    loaders = {}
    for split, transform in splits.items():
        ds = Dacl10kDataset(
            root_dir  = root_dir,
            split     = split,
            transform = transform,
            img_size  = img_size,
            debug     = debug,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = (split == "train"),
        )

    return loaders["train"], loaders["val"], loaders["test"]


def build_robust_loader(
    root_dir    : str,
    corruption  : str = "blur",
    img_size    : int = 512,
    batch_size  : int = 8,
    num_workers : int = 4,
) -> DataLoader:
    """DataLoader khusus robustness evaluation pada test split."""
    ds = Dacl10kDataset(
        root_dir  = root_dir,
        split     = "test",
        transform = get_robust_transforms(img_size, corruption),
        img_size  = img_size,
    )
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )


# ─────────────────────────────────────────────
#  CLASS WEIGHTS
# ─────────────────────────────────────────────

def compute_class_weights(
    train_loader : DataLoader,
    device       : str = "cpu",
) -> torch.Tensor:
    """Hitung inverse-frequency class weights untuk loss function."""
    print("Menghitung class weights...")
    counts = torch.zeros(NUM_CLASSES)

    for batch in train_loader:
        masks = batch["mask"]
        for c in range(NUM_CLASSES):
            counts[c] += (masks == c).sum().item()

    total   = counts.sum()
    weights = total / (NUM_CLASSES * counts.clamp(min=1))
    weights = weights / weights.sum() * NUM_CLASSES

    print(f"\n{'Idx':<4} {'Class':<32} {'Count':>10} {'Weight':>8}")
    print("-" * 58)
    for i, (cls, w) in enumerate(zip(ALL_CLASSES, weights)):
        print(f"[{i:2d}] {cls:<32} {counts[i]:>10.0f} {w:>8.4f}")

    return weights.to(device)


# ─────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/dacl10k")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("=" * 55)
    print("  Dacl10k DataLoader — Quick Test (Supervisely format)")
    print("=" * 55)
    print(f"\nKonfigurasi:")
    print(f"  root  : {args.root}")
    print(f"  split : {args.split}")
    print(f"  debug : {args.debug}")
    print(f"  kelas : {NUM_CLASSES} total "
          f"({len(DAMAGE_CLASSES)} damage + {len(COMPONENT_CLASSES)} komponen)")

    # Test dataset
    dataset = Dacl10kDataset(
        root_dir = args.root,
        split    = args.split,
        debug    = args.debug,
    )

    sample = dataset[0]
    print(f"\nSample [0]:")
    print(f"  image  : {sample['image'].shape}  {sample['image'].dtype}")
    print(f"  mask   : {sample['mask'].shape}   {sample['mask'].dtype}")
    print(f"  labels : {sample['mask'].unique().tolist()}")
    print(f"  id     : {sample['image_id']}")

    # Test DataLoader
    print(f"\nMembangun DataLoader (batch_size={args.batch_size})...")
    train_l, val_l, test_l = build_dataloaders(
        root_dir   = args.root,
        batch_size = args.batch_size,
        num_workers= 0,     # 0 untuk Windows compatibility
        debug      = args.debug,
    )
    print(f"  train batches : {len(train_l)}")
    print(f"  val batches   : {len(val_l)}")
    print(f"  test batches  : {len(test_l)}")

    batch = next(iter(train_l))
    print(f"\nBatch pertama:")
    print(f"  images : {batch['image'].shape}")
    print(f"  masks  : {batch['mask'].shape}")
    print(f"  range  : [{batch['image'].min():.2f}, {batch['image'].max():.2f}]")

    print("\n✅ Semua test PASSED!")
    print("\nLangkah berikutnya:")
    print("  python datasets/dacl10k_loader.py --root data/dacl10k --split train --debug")