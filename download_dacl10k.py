"""
download_dacl10k.py
===================
Download dacl10k dataset via HuggingFace Hub.
Jalankan: python download_dacl10k.py
"""
import os
from pathlib import Path

def download_via_toolkit():
    """Metode 1: via dacl10k-toolkit resmi."""
    try:
        from dacl10k.data import download_dataset
        print("Menggunakan dacl10k-toolkit...")
        download_dataset(target_dir="data/dacl10k")
        return True
    except ImportError:
        print("dacl10k-toolkit tidak terinstall, coba metode HuggingFace...")
        return False
    except Exception as e:
        print(f"dacl10k-toolkit error: {e}")
        return False

def download_via_huggingface():
    """Metode 2: via HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
        print("Menggunakan HuggingFace Hub...")
        snapshot_download(
            repo_id    = "phiyodr/dacl10k",
            repo_type  = "dataset",
            local_dir  = "data/dacl10k",
        )
        return True
    except ImportError:
        print("Install dulu: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"HuggingFace error: {e}")
        return False

def check_structure():
    """Verifikasi struktur folder setelah download."""
    base = Path("data/dacl10k")
    print("\n=== Struktur folder dacl10k ===")

    total_json = 0
    total_img  = 0

    for split in ["train", "val", "testdev"]:
        split_dir = base / split
        if not split_dir.exists():
            print(f"  [{split}] TIDAK ADA — cek hasil download")
            continue

        # Cari semua JSON dan gambar rekursif
        jsons = list(split_dir.rglob("*.json"))
        imgs  = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.png"))

        total_json += len(jsons)
        total_img  += len(imgs)
        print(f"  [{split}] {len(imgs):5d} gambar | {len(jsons):5d} JSON")

        # Tampilkan contoh struktur subfolder
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        for d in subdirs[:3]:
            print(f"    └── {d.name}/")

    print(f"\nTotal: {total_img} gambar | {total_json} JSON")

    if total_img == 0:
        print("\nWARNING: Tidak ada gambar ditemukan!")
        print("Kemungkinan struktur folder berbeda dari yang diharapkan loader.")
        print("Jalankan: python datasets/dacl10k_loader.py --inspect")
    else:
        print("\nDataset siap digunakan!")

if __name__ == "__main__":
    print("=" * 50)
    print("  dacl10k Dataset Downloader")
    print("=" * 50)

    # Coba toolkit dulu, fallback ke HuggingFace
    success = download_via_toolkit()
    if not success:
        success = download_via_huggingface()

    if success:
        check_structure()
    else:
        print("\nGagal download. Coba manual:")
        print("  pip install git+https://github.com/phiyodr/dacl10k-toolkit")
        print("  pip install huggingface_hub")