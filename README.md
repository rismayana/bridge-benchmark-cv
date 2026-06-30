# Bridge Benchmark CV

Benchmark perbandingan model **semantic segmentation** untuk deteksi kerusakan jembatan menggunakan dataset **DACL10k**.

## Dataset

[DACL10k](https://github.com/phiyodr/dacl10k-toolkit) — 9.920 gambar jembatan nyata dengan 19 kelas anotasi:

- **15 kelas kerusakan:** crack, alligator crack, efflorescence, exposed rebars, graffiti, hollowareas, joint tape, restformwork, rockpocket, rust, spalling, washouts, weathering, wetspot, cavity
- **4 kelas komponen:** bearing, drainage, expansion joint, protective equipment

| Split | Jumlah |
|-------|--------|
| Train | 6.935  |
| Val   | 975    |
| Test  | 2.010  |

## Model yang Dibandingkan

| Model | Backbone | Library |
|-------|----------|---------|
| SegFormer-B2 | Mix Transformer | HuggingFace Transformers |
| DeepLabV3+ | ResNet-50 | torchvision |
| Swin-UNet | Swin Transformer | timm |
| UNet + EfficientNet-B4 | EfficientNet | segmentation-models-pytorch |
| YOLOv11-seg | CSPNet | ultralytics |
| YOLOv8-seg | CSPNet | ultralytics |

## Struktur Project

```
bridge-benchmark-cv/
├── configs/           # Konfigurasi YAML per model
├── datasets/          # DataLoader DACL10k (Supervisely format)
├── models/            # Wrapper tiap model
├── train/             # Trainer, loss functions, scheduler
├── eval/              # Metrics, robustness, visualisasi, perbandingan
├── notebooks/         # Eksplorasi & analisis hasil
├── outputs/           # Checkpoint, log, figure (tidak di-commit)
├── data/              # Dataset DACL10k (tidak di-commit)
├── train_all.py       # Jalankan training semua model sekaligus
└── eval_all.py        # Evaluasi semua model sekaligus
```

## MLOps: WandB

Semua eksperimen di-track otomatis via [Weights & Biases](https://wandb.ai).

Yang di-log per training run:
- Loss curve (train/val) per epoch
- mIoU per epoch + per-class IoU (19 kelas)
- Sample prediksi segmentasi visual (setiap 5 epoch)
- Best model checkpoint sebagai WandB Artifact

Aktifkan dengan menambahkan ke config:
```python
config = {
    "use_wandb"    : True,
    "wandb_project": "bridge-benchmark-cv",
    "wandb_entity" : "<username-wandb>",
}
```

## Instalasi

```bash
pip install -r requirements.txt
```

Untuk WandB:
```bash
pip install wandb
wandb login
```

## Download Dataset

```bash
python download_dacl10k.py
```

## Training

```bash
# Satu model
python train/trainer.py --config configs/segformer.yaml

# Semua model sekaligus
python train_all.py
```

## Evaluasi

```bash
python eval_all.py
```

## Metrik

- **mIoU** (mean Intersection over Union) — metrik utama
- Per-class IoU untuk 19 kelas
- Robustness test: blur / noise / brightness corruption
