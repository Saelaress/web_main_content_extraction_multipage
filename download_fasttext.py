# -*- coding: utf-8 -*-
"""
Скачивание FastText-модели cc.en.300.bin в директорию models/.

Размер архива ~4.2 ГБ (.gz), распакованный .bin ~7.2 ГБ.
URL: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz

Запуск:
    python download_fasttext.py

После завершения файл будет в models/cc.en.300.bin и автоматически найден
через config.FASTTEXT_MODEL_PATH (см. src/config.py).
"""

from __future__ import annotations

import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
GZ_PATH = MODELS_DIR / "cc.en.300.bin.gz"
BIN_PATH = MODELS_DIR / "cc.en.300.bin"


def _format_size(n: int) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} ТБ"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        sys.stderr.write(
            f"\r  {percent:3d}%  {_format_size(downloaded)} / {_format_size(total_size)}"
        )
        sys.stderr.flush()


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if BIN_PATH.is_file():
        print(f"[OK] FastText уже скачан: {BIN_PATH} ({_format_size(BIN_PATH.stat().st_size)})")
        return 0

    if not GZ_PATH.is_file():
        print(f"Скачиваю {URL}")
        print(f"  → {GZ_PATH}")
        print("  (~4.2 ГБ, может занять 10–40 минут в зависимости от скорости сети)")
        try:
            urllib.request.urlretrieve(URL, GZ_PATH, reporthook=_progress)
            sys.stderr.write("\n")
        except Exception as e:
            print(f"\n[ERROR] Не удалось скачать: {e}", file=sys.stderr)
            print(
                "Проверьте подключение к интернету. "
                "Альтернатива — скачать вручную с https://fasttext.cc/docs/en/crawl-vectors.html "
                f"и положить файл в {BIN_PATH}",
                file=sys.stderr,
            )
            return 1
    else:
        print(f"[OK] Архив уже скачан: {GZ_PATH}")

    print(f"Распаковываю в {BIN_PATH} (~7.2 ГБ)...")
    try:
        with gzip.open(GZ_PATH, "rb") as fin, open(BIN_PATH, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
    except Exception as e:
        print(f"[ERROR] Распаковка не удалась: {e}", file=sys.stderr)
        if BIN_PATH.is_file():
            BIN_PATH.unlink()
        return 1

    print(f"[OK] Готово: {BIN_PATH} ({_format_size(BIN_PATH.stat().st_size)})")
    print("Архив .gz можно удалить вручную, если место на диске критично.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
