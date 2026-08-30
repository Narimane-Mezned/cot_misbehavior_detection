import zipfile
from pathlib import Path

import gdown

FULL_DATASET_FILES = [
    {"name": "train01", "id": "1Nhzsw1HXmus8xKjSP5pu68G277Vj1xdU", "size_gb": 25.3, "split": "train"},
    {"name": "train02", "id": "1lovJv7x07bXCgL3lZOb8wLEiAFZxRXLo", "size_gb": 24.5, "split": "train"},
    {"name": "train03", "id": "1cVbmMoMrBqMFr0R1VOhtsyxOZL4HQEha", "size_gb": 24.7, "split": "train"},
    {"name": "train04", "id": "1MaLY3JBJIXZANwRJoLbhSr95eb0dAy6o", "size_gb": 21.7, "split": "train"},
    {"name": "train05", "id": "1ZcfI7sQoo9zmxYFLrujStwtzc_8NPAcf", "size_gb": 24.3, "split": "train"},
    {"name": "train06", "id": "1UQQAFwkyk4JmWJ2DbRb__vCfRscgsWE0", "size_gb": 22.2, "split": "train"},
    {"name": "train07", "id": "1Mm4iHEqtjrZSREX8RddSr-JisJuKMhLI", "size_gb": 23.0, "split": "train"},
    {"name": "train08", "id": "1Gl4NIAT14_m8Sj8iIuq2hWlDR4cf73qM", "size_gb": 25.2, "split": "train"},
    {"name": "train09", "id": "1cYHN6KiHzBiRChiUo_hoF8SZyVCXiV6z", "size_gb": 23.9, "split": "train"},
    {"name": "train10", "id": "1hLgDv3bM_LP7eITmEZKQYZ2KNxVrkNZ4", "size_gb": 7.3, "split": "train"},
    {"name": "val01", "id": "1r1nUm_DMu-7-AAe6eelCquHx8fku-dth", "size_gb": 22.8, "split": "val"},
    {"name": "val02", "id": "1tSR90QIgLLc2-Ycc2J-eAp1Ocq5FbzeJ", "size_gb": 22.3, "split": "val"},
    {"name": "test01", "id": "1C6-O0gAVZM_4chzTRjPgeSSzmTI3k3OD", "size_gb": 22.9, "split": "test"},
    {"name": "test02", "id": "19TK77uMnMdkFaRG5plln4GZBe-I0VURz", "size_gb": 23.5, "split": "test"},
]

SPLIT_TXT_FILE = {"name": "split_txt_files", "id": "1n_EG6mS18s4EfXhBPf3XM1aEr6LRjgi7", "size_gb": 0.0001}

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"


def download_file(file_id: str, output_path: Path, size_gb: float, name: str) -> bool:
    if output_path.exists():
        print(f"[skip] {name}: {output_path.name} already exists")
        return True

    print(f"[download] {name} (~{size_gb} GB): {output_path.name}")
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(output_path), quiet=False)
        return output_path.exists()
    except Exception as e:
        print(f"[error] {name} failed: {e}")
        print(f"[fallback] Try manually: https://drive.google.com/file/d/{file_id}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    if not zip_path.exists():
        return False

    print(f"[extract] {zip_path.name} -> {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        return True
    except Exception as e:
        print(f"[error] Extraction failed for {zip_path.name}: {e}")
        return False


def download_and_extract_all(delete_zips_after: bool = False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_files = FULL_DATASET_FILES + [SPLIT_TXT_FILE]
    failed = []

    for file_info in all_files:
        zip_path = RAW_DIR / f"DeepAccident_{file_info['name']}.zip"
        success = download_file(file_info["id"], zip_path, file_info["size_gb"], file_info["name"])

        if not success:
            failed.append(file_info["name"])
            continue

        extracted = extract_zip(zip_path, RAW_DIR)
        if extracted and delete_zips_after:
            zip_path.unlink()
            print(f"[cleanup] Deleted {zip_path.name} after extraction")

    if failed:
        print(f"\n[summary] {len(failed)} file(s) failed to download: {failed}")
        print("[summary] Retry this script to resume -- already-downloaded files are skipped.")
    else:
        print(f"\n[summary] All {len(all_files)} files downloaded and extracted successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-zips-after", action="store_true", help="Delete zip files after successful extraction to save disk space")
    args = parser.parse_args()

    download_and_extract_all(delete_zips_after=args.delete_zips_after)