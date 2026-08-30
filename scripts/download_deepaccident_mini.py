import zipfile
from pathlib import Path

import gdown

MINI_FILE_ID = "1NXC_-zTWFdHj-30g3zSUfNFMp4Mk_7Hh"
MINI_EXPECTED_SIZE_GB = 9.0

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "DeepAccident_mini.zip"
EXTRACT_DIR = RAW_DIR / "DeepAccident_mini"


def download_mini() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"[skip] {ZIP_PATH} already exists, skipping download.")
        return

    print(f"[download] Fetching DeepAccident mini sample (~{MINI_EXPECTED_SIZE_GB} GB)...")
    url = f"https://drive.google.com/uc?id={MINI_FILE_ID}"
    gdown.download(url, str(ZIP_PATH), quiet=False)


def extract_mini() -> None:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print(f"[skip] {EXTRACT_DIR} already populated, skipping extraction.")
        return

    print(f"[extract] Extracting {ZIP_PATH.name} to {EXTRACT_DIR} ...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print("[extract] Done.")


def verify_structure(max_depth: int = 3) -> None:
    print("\n[verify] Top-level structure of extracted mini sample:\n")

    def walk(path: Path, depth: int, prefix: str = ""):
        if depth > max_depth:
            return
        entries = sorted(path.iterdir())
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]

        for d in dirs:
            print(f"{prefix}{d.name}/")
            walk(d, depth + 1, prefix + "  ")

        for f in files[:2]:
            print(f"{prefix}{f.name}")
        if len(files) > 2:
            print(f"{prefix}... ({len(files)} files total)")

    walk(EXTRACT_DIR, 0)

    scenario_type_dirs = [d for d in EXTRACT_DIR.iterdir() if d.is_dir()]
    print(f"\n[verify] Found {len(scenario_type_dirs)} top-level scenario-type folders.")
    print("[verify] Compare this output against data/README.md's documented structure.")
    print("[verify] Update data/README.md's verification checklist once confirmed.")


if __name__ == "__main__":
    download_mini()
    extract_mini()
    verify_structure()