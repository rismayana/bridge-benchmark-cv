"""
Jalankan dari root project:
python create_init.py
"""
from pathlib import Path

folders = [
    "train",
    "datasets", 
    "models",
    "eval",
]

for folder in folders:
    init_file = Path(folder) / "__init__.py"
    init_file.parent.mkdir(parents=True, exist_ok=True)
    if not init_file.exists():
        init_file.write_text("# auto-generated\n")
        print(f"Created: {init_file}")
    else:
        print(f"Exists : {init_file}")

print("\nDone! Semua __init__.py sudah ada.")