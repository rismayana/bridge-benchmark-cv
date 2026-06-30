# Bridge Benchmark CV

> **Studi Komparatif Model Semantic Segmentation untuk Deteksi Kerusakan Jembatan**

---

## Latar Belakang

Jembatan merupakan infrastruktur kritis yang memerlukan inspeksi berkala. Inspeksi manual bersifat mahal, lambat, dan berisiko bagi petugas. Computer vision berbasis deep learning membuka peluang otomatisasi deteksi kerusakan dari foto, namun belum ada studi yang secara sistematis membandingkan performa berbagai arsitektur modern pada domain ini.

Penelitian ini melakukan **benchmark komprehensif** terhadap enam model semantic segmentation—mulai dari transformer-based hingga YOLO-based—pada dataset kerusakan beton jembatan dunia nyata (**DACL10k**). Tujuannya adalah menentukan model mana yang paling efektif untuk keperluan inspeksi jembatan otomatis, dilihat dari akurasi segmentasi, kemampuan mendeteksi kelas minoritas, dan ketahanan terhadap kondisi gambar yang buruk.

---

## Task & Pendekatan

**Semantic Segmentation** — setiap piksel pada foto jembatan diklasifikasikan ke salah satu dari 19 kelas (kerusakan atau komponen struktural). Model memproduksi *dense prediction map* berukuran sama dengan gambar input.

Pipeline penelitian:

```
Foto Jembatan → Pre-processing → Model → Mask Segmentasi (19 kelas) → Evaluasi
```

Seluruh model dilatih dan dievaluasi dengan **protokol yang identik** (dataset, loss function, optimizer, scheduler, metrik) untuk memastikan perbandingan yang adil (*fair benchmark*).

---

## Dataset: DACL10k

[DACL10k v2](https://github.com/phiyodr/dacl10k-toolkit) adalah dataset inspeksi jembatan berskala besar yang terdiri dari foto-foto struktur beton nyata dengan anotasi segmentasi piksel-per-piksel.

| Split | Jumlah Gambar |
|-------|--------------|
| Train | 6.935        |
| Val   | 975          |
| Test  | 2.010        |
| **Total** | **9.920** |

### 19 Kelas Anotasi

**15 Kelas Kerusakan (Damage):**

| # | Kelas | Keterangan |
|---|-------|------------|
| 0 | crack | Retak halus memanjang |
| 1 | alligator crack | Retak pola kulit buaya |
| 2 | efflorescence | Endapan garam pada beton |
| 3 | exposed rebars | Tulangan besi terekspos |
| 4 | graffiti | Coretan vandalisme |
| 5 | hollowareas | Area berongga di dalam beton |
| 6 | joint tape | Selotip sambungan rusak |
| 7 | restformwork | Sisa bekisting |
| 8 | rockpocket | Beton sarang lebah |
| 9 | rust | Karat pada komponen besi |
| 10 | spalling | Beton mengelupas |
| 11 | washouts / concrete corrosion | Korosi beton |
| 12 | weathering | Pelapukan permukaan |
| 13 | wetspot | Bercak basah |
| 14 | cavity | Rongga / lubang |

**4 Kelas Komponen Struktural:**

| # | Kelas | Keterangan |
|---|-------|------------|
| 15 | bearing | Bantalan jembatan |
| 16 | drainage | Sistem drainase |
| 17 | expansion joint | Sambungan ekspansi |
| 18 | protective equipment | Peralatan proteksi |

Format anotasi: **Supervisely JSON** (diunduh via Dataset Ninja). Dataset tidak disertakan dalam repo — lihat bagian [Download Dataset](#download-dataset).

---

## Model yang Dibandingkan

| Model | Arsitektur | Backbone | Library | Kategori |
|-------|------------|----------|---------|----------|
| **SegFormer-B2** | Encoder-Decoder | Mix Transformer | HuggingFace Transformers | Transformer |
| **DeepLabV3+** | ASPP + Decoder | ResNet-50 | torchvision | CNN |
| **Swin-UNet** | U-shaped Transformer | Swin Transformer | timm | Transformer |
| **UNet + EfficientNet-B4** | U-Net | EfficientNet-B4 | segmentation-models-pytorch | CNN |
| **YOLOv11-seg** | Single-stage | CSPNet | ultralytics | Real-time |
| **YOLOv8-seg** | Single-stage | CSPNet | ultralytics | Real-time |

Model dipilih untuk merepresentasikan berbagai paradigma arsitektur: **transformer encoder-decoder**, **CNN klasik**, **hybrid U-Net**, dan **single-stage real-time detector**.

---

## Protokol Training

Semua model menggunakan konfigurasi yang seragam:

| Parameter | Nilai |
|-----------|-------|
| Input size | 512 × 512 px |
| Optimizer | AdamW |
| Loss | CrossEntropy + Dice (α = 0.5) |
| Scheduler | Cosine Annealing |
| Warmup | 3 epoch |
| Early stopping | patience = 10 |
| Mixed precision | AMP (FP16) |
| Max epochs | 50 |

**Augmentasi training:** horizontal flip, vertical flip, random rotate 90°, brightness/contrast jitter, color jitter, Gaussian noise.

---

## Metrik Evaluasi

- **mIoU** (mean Intersection over Union) — metrik utama perbandingan
- **Per-class IoU** — untuk 19 kelas secara individual (penting untuk kelas minoritas)
- **Robustness score** — mIoU pada gambar yang dikondisikan dengan:
  - Gaussian blur (simulasi foto buram)
  - Gaussian noise (simulasi sensor noise)
  - Brightness corruption (simulasi pencahayaan buruk)

---

## MLOps: Weights & Biases

Seluruh eksperimen di-track otomatis via [Weights & Biases](https://wandb.ai) untuk reprodusibilitas dan kemudahan perbandingan antar model.

**Yang di-log per run:**

| Sinyal | Frekuensi |
|--------|-----------|
| Train loss (total, CE, Dice) | Setiap epoch |
| Val loss + val mIoU | Setiap epoch |
| Per-class IoU (19 kelas) | Setiap epoch |
| Learning rate | Setiap epoch |
| Prediksi segmentasi visual | Setiap 5 epoch |
| Best model checkpoint | Saat model terbaik baru |

Semua 6 model berada dalam satu WandB **project** sehingga kurva training dan hasil evaluasi bisa dibandingkan langsung di dashboard.

---

## Struktur Repositori

```
bridge-benchmark-cv/
├── datasets/
│   └── dacl10k_loader.py      # Dataset class + DataLoader (Supervisely format)
├── models/
│   ├── segformer.py           # SegFormer-B2 wrapper
│   ├── deeplabv3.py           # DeepLabV3+ wrapper
│   ├── swinunet.py            # Swin-UNet wrapper
│   ├── unet_effnet.py         # UNet + EfficientNet-B4 wrapper
│   ├── yolov11.py             # YOLOv11-seg wrapper
│   └── yolo_seg.py            # YOLOv8-seg wrapper
├── train/
│   ├── trainer.py             # Base trainer (identik untuk semua model)
│   └── losses.py              # CrossEntropy + Dice + Focal Loss
├── eval/
│   ├── metrics.py             # mIoU, per-class IoU, confusion matrix
│   ├── robustness.py          # Evaluasi ketahanan terhadap korups gambar
│   ├── visualize.py           # Visualisasi prediksi segmentasi
│   └── compare_models.py      # Tabel perbandingan semua model
├── configs/
│   ├── base.yaml              # Config dasar (shared)
│   ├── segformer.yaml         # Config spesifik SegFormer
│   ├── deeplabv3.yaml         # Config spesifik DeepLabV3+
│   └── ...                    # Config model lainnya
├── notebooks/                 # Eksplorasi data & analisis hasil
├── outputs/                   # Checkpoint, log, figure (di-gitignore)
├── data/                      # Dataset DACL10k (di-gitignore)
├── train_all.py               # Training semua model sekaligus
├── eval_all.py                # Evaluasi semua model sekaligus
└── download_dacl10k.py        # Script download dataset
```

---

## Instalasi

```bash
# Clone repo
git clone https://github.com/rismayana/bridge-benchmark-cv.git
cd bridge-benchmark-cv

# Install dependencies (Python 3.10+, CUDA 11.8+)
pip install -r requirements.txt

# Login WandB
pip install wandb
wandb login
```

---

## Download Dataset

```bash
python download_dacl10k.py
```

Dataset (~5.5 GB) akan tersimpan di `data/dacl10k/` dengan struktur:

```
data/dacl10k/
├── meta.json
├── train/  ├── img/  └── ann/
├── val/    ├── img/  └── ann/
└── test/   ├── img/  └── ann/
```

---

## Training

```bash
# Training satu model
python train_all.py --model segformer

# Training semua model secara berurutan
python train_all.py

# Aktifkan WandB tracking
python train_all.py --wandb
```

---

## Evaluasi

```bash
# Evaluasi semua model yang sudah ditraining
python eval_all.py

# Evaluasi robustness
python eval_all.py --robustness
```

---

## Hasil (Work in Progress)

Tabel ini akan diisi setelah semua model selesai ditraining.

| Model | mIoU (val) | mIoU (test) | Params (M) |
|-------|-----------|------------|------------|
| SegFormer-B2 | - | - | ~25 |
| DeepLabV3+ | - | - | ~39 |
| Swin-UNet | - | - | ~27 |
| UNet + EfficientNet-B4 | - | - | ~19 |
| YOLOv11-seg | - | - | ~10 |
| YOLOv8-seg | - | - | ~12 |

---

## Dependencies Utama

| Library | Versi | Kegunaan |
|---------|-------|---------|
| PyTorch | 2.2.0 | Framework utama |
| HuggingFace Transformers | 4.40.0 | SegFormer |
| torchvision | 0.17.0 | DeepLabV3+ |
| timm | latest | Swin-UNet backbone |
| segmentation-models-pytorch | latest | UNet + EfficientNet |
| ultralytics | 8.2.0 | YOLOv8 & YOLOv11 |
| albumentations | 1.4.3 | Augmentasi |
| wandb | 0.17.0 | Experiment tracking |
