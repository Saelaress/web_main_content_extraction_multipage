# -*- coding: utf-8 -*-
"""
Инференс лучшей модели (эксп. 3-v3-weak): BiDirSAGEStabV3 + per-group threshold tuning.

Расширение stability_inference.py:
  - Использует архитектуру BiDirSAGEStabV3 (DropEdge + Feature Dropout-stab)
  - По умолчанию загружает stability_v3_weak_gnn.pt (test_f1_main = 0.7696)
  - По умолчанию применяет лучшие per-group пороги 0.64/0.75.
    --thresholds W,N — переопределить вручную; --argmax — baseline 0.5/0.5.

Использование:
    python stability_v3_inference.py page.html [page2.html ...] [options]

    --siblings-dir DIR    Директория с .html-файлами соседей (sibling-страницы того же домена)
    --model PATH          Чекпоинт модели (по умолчанию: MODELS_DIR/stability_v3_weak_gnn.pt)
    --thresholds W,N      Явные пороги {with_signal}/{no_signal} для main-класса.
                          По умолчанию — лучшие 0.64/0.75 (оптимум для v3-weak на val).
    --argmax              Использовать argmax (0.5/0.5) вместо лучших порогов.
    --output FILE         Путь к выходному файлу (по умолчанию: stdout)
    --encoding ENC        Кодировка HTML-файлов (по умолчанию: utf-8)
    --device DEVICE       Устройство для инференса: cpu/cuda (по умолчанию: авто)

Замечания:
  * stab-фичи вычисляются через htmldiff между основной страницей и sibling-страницами.
  * Без --siblings-dir модель работает в "деградированном" режиме: stab = 0 (no_signal группа).
    Это эквивалентно прогону v3 на узлах без stab-сигнала.
  * scaler_stab.pkl ожидается в той же папке, что и precomputed_stability_by1 (1-сосед),
    либо передаётся через --scaler-stab.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_PROJECT_DIR))

try:
    from bs4 import BeautifulSoup  # noqa: E402
except ImportError as _e:
    raise RuntimeError(
        "Требуется bs4: pip install beautifulsoup4 html5lib"
    ) from _e

from config import (  # noqa: E402
    MODELS_DIR,
    STAB_PRECOMPUTED_DIR,
    STAB_STABILITY_DIM,
    STAB_EMBED_DIM,
    STAB_HIDDEN_DIM,
    STAB_NUM_LAYERS,
    STAB_V3_DROP_EDGE_P,
    STAB_V3_FEAT_DROP_P,
    SAGE_PREPROCESSORS_FILE,
)
from gnn_model import build_stability_bidir_v3_model  # noqa: E402
from sage_features import compute_page_feature_tensors  # noqa: E402
from sage_graph import html_to_graph_tensors  # noqa: E402
from xpath_utils import build_node_order  # noqa: E402

# Дефолтная лучшая модель v3-weak (test_f1_main = 0.7696 с per-group thr 0.64/0.75)
V3_WEAK_MODEL_FILE = "stability_v3_weak_gnn.pt"
# Дефолтные пороги — те, что найдены sweep'ом на val у v3-weak
V3_WEAK_THR_WITH_SIGNAL = 0.64
V3_WEAK_THR_NO_SIGNAL   = 0.75


# ---------------------------------------------------------------------------
# Кэшированная загрузка ресурсов
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_fasttext_cached(path: str):
    import fasttext
    return fasttext.load_model(path)


@lru_cache(maxsize=4)
def _load_bundle_cpu(model_path_str: str):
    """Загружает препроцессоры (sage_preprocessors.pt) и чекпоинт модели."""
    prep_path = MODELS_DIR / SAGE_PREPROCESSORS_FILE
    model_path = Path(model_path_str)
    if not prep_path.is_file():
        raise FileNotFoundError(
            f"Препроцессоры не найдены: {prep_path}\n"
            "Запустите precompute_features.py для создания sage_preprocessors.pt"
        )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Чекпоинт модели не найден: {model_path}\n"
            f"Ожидался stability_v3_weak_gnn.pt — запустите train_stability_v3.py"
        )
    try:
        prep = torch.load(prep_path, map_location="cpu", weights_only=False)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        prep = torch.load(prep_path, map_location="cpu")
        ckpt = torch.load(model_path, map_location="cpu")
    return prep, ckpt


@lru_cache(maxsize=1)
def _load_scaler_stab(scaler_path_str: str):
    """Загружает scaler_stab.pkl для масштабирования stability-признаков."""
    scaler_path = Path(scaler_path_str)
    if not scaler_path.is_file():
        raise FileNotFoundError(
            f"scaler_stab.pkl не найден: {scaler_path}\n"
            "Запустите precompute_stability_features.py для создания scaler_stab.pkl"
        )
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def _get_model_and_preprocessors(model_path: Path, device: torch.device):
    """Инициализирует BiDirSAGEStabV3 из чекпоинта и возвращает (model, prep, ft_path_str)."""
    prep, ckpt = _load_bundle_cpu(str(model_path))

    ft_dim = int(prep["ft_dim"])
    num_numeric = int(prep["num_numeric"])
    embed_dim = int(ckpt.get("embed_dim", STAB_EMBED_DIM))
    hidden_dim = int(ckpt.get("hidden_dim", STAB_HIDDEN_DIM))
    num_layers = int(ckpt.get("num_layers", STAB_NUM_LAYERS))
    stability_dim = int(ckpt.get("stability_dim", STAB_STABILITY_DIM))
    drop_edge_p = float(ckpt.get("drop_edge_p", STAB_V3_DROP_EDGE_P))
    feat_drop_p = float(ckpt.get("feat_drop_p", STAB_V3_FEAT_DROP_P))
    num_tag_embeddings = int(prep.get("num_tag_embeddings", 0))

    if num_tag_embeddings <= 0 or "tag_stoi" not in prep:
        raise ValueError(
            "sage_preprocessors.pt не содержит tag_stoi/num_tag_embeddings — "
            "запустите precompute_features.py заново"
        )

    model = build_stability_bidir_v3_model(
        ft_dim=ft_dim,
        num_numeric=num_numeric,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_tag_embeddings=num_tag_embeddings,
        stability_dim=stability_dim,
        drop_edge_p=drop_edge_p,
        feat_drop_p=feat_drop_p,
    )
    state = ckpt.get("state_dict") or ckpt.get("best_state") or ckpt.get("model_state_dict")
    if state is None and isinstance(ckpt, dict):
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
    if state is None:
        raise ValueError(f"В чекпоинте не найден state_dict: {model_path}")

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    meta = {
        "stability_dim": stability_dim,
        "drop_edge_p": drop_edge_p,
        "feat_drop_p": feat_drop_p,
        "num_numeric_base": num_numeric,
        "num_numeric_total": num_numeric + stability_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    # FastText путь: сначала пробуем то, что было записано в preprocessors при обучении;
    # если такого файла нет на текущей машине — берём из локального config.FASTTEXT_MODEL_PATH
    # (см. download_fasttext.py). Это позволяет переносить test_diplom между машинами.
    ft_path_str = prep.get("fasttext_path", "")
    if not ft_path_str or not Path(ft_path_str).is_file():
        from config import FASTTEXT_MODEL_PATH  # noqa: PLC0415
        if FASTTEXT_MODEL_PATH and Path(FASTTEXT_MODEL_PATH).is_file():
            ft_path_str = str(FASTTEXT_MODEL_PATH)
    return model, prep, ft_path_str, meta


# ---------------------------------------------------------------------------
# Вычисление stability-признаков
# ---------------------------------------------------------------------------

def _compute_stability_for_page(
    html_path: Path,
    elements: list,
    sibling_htmls: list[str],
    stability_dim: int,
    encoding: str = "utf-8",
) -> np.ndarray:
    """
    Вычисляет матрицу stability-признаков (N, stability_dim) для одной страницы.
    Если sibling_htmls пуст или htmldiff падает — возвращает нули.
    """
    n_nodes = len(elements)
    zeros = np.zeros((n_nodes, stability_dim), dtype=np.float32)

    if stability_dim == 0:
        return np.zeros((n_nodes, 0), dtype=np.float32)
    if not sibling_htmls:
        return zeros

    try:
        base_html = html_path.read_text(encoding=encoding, errors="replace")
    except Exception as e:
        print(f"[stability_v3_inference] WARNING: не удалось прочитать {html_path}: {e}", file=sys.stderr)
        return zeros

    node_order = build_node_order(elements)

    try:
        from htmldiff import HtmlDiffConfig, compute_stability_features  # noqa: PLC0415
        from htmldiff.exceptions import InsufficientPagesWarning  # noqa: PLC0415

        config = HtmlDiffConfig(
            n_workers=1,
            diff_timeout=float("inf"),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsufficientPagesWarning)
            result = compute_stability_features(
                pages=[base_html] + sibling_htmls,
                node_order=node_order,
                config=config,
                normalize=False,
            )

        matrix = result.feature_matrix.astype(np.float32)
        if matrix.shape[0] != n_nodes:
            print(
                f"[stability_v3_inference] WARNING: XPATH_MISMATCH для {html_path.name}: "
                f"feature_matrix={matrix.shape[0]} != n_nodes={n_nodes} — используются нули",
                file=sys.stderr,
            )
            return zeros

        return matrix

    except Exception as e:
        print(
            f"[stability_v3_inference] WARNING: htmldiff ошибка для {html_path.name}: {e} "
            "— используются нули для stability",
            file=sys.stderr,
        )
        return zeros


def _load_sibling_htmls(siblings_dir: Path | None, encoding: str = "utf-8") -> list[str]:
    if siblings_dir is None or not siblings_dir.is_dir():
        return []
    htmls = []
    for p in sorted(siblings_dir.glob("*.html")):
        try:
            htmls.append(p.read_text(encoding=encoding, errors="replace"))
        except Exception as e:
            print(f"[stability_v3_inference] WARNING: не удалось прочитать sibling {p}: {e}", file=sys.stderr)
    return htmls


# ---------------------------------------------------------------------------
# Постобработка предсказаний: разметка HTML / выделение main-текста / labeler
# ---------------------------------------------------------------------------

_MAIN_CLASS     = "sure-main-content"
_TEMPLATE_CLASS = "sure-template-content"


def _apply_labels_to_elements(elements: list, predictions: np.ndarray) -> None:
    """
    In-place: добавляет CSS-классы (sure-main-content / sure-template-content)
    к BeautifulSoup Tag-узлам по предсказаниям модели. Узлы ссылаются на
    исходный soup, поэтому потом достаточно сериализовать корень.
    """
    for elem, label in zip(elements, predictions.tolist()):
        target_class = _MAIN_CLASS if int(label) == 1 else _TEMPLATE_CLASS
        existing = elem.get("class") or []
        if isinstance(existing, str):
            existing = existing.split()
        # уже размечен — пропускаем (не дублируем)
        if target_class in existing:
            continue
        # снимаем противоположный класс если был (например после reset)
        opposite = _TEMPLATE_CLASS if target_class == _MAIN_CLASS else _MAIN_CLASS
        if opposite in existing:
            existing = [c for c in existing if c != opposite]
        existing.append(target_class)
        elem["class"] = existing


def _get_root_soup(elements: list):
    """
    Возвращает корневой узел soup (BeautifulSoup object) из любого Tag.
    Идём по .parent до тех пор, пока он есть.
    """
    if not elements:
        return None
    node = elements[0]
    while node.parent is not None:
        node = node.parent
    return node


_TEXT_EXCLUDE_TAGS = {
    "script", "style", "noscript", "iframe", "svg", "path",
    "object", "embed", "template", "meta", "link",
}
_TEXT_EXCLUDE_CLASSES = {
    "content-labeler-ui", "content-labeler-hover",
    "content-labeler-highlight", "js-disabled-indicator",
}


def _visible_text(node) -> str:
    """
    Извлекает видимый текст из узла, пропуская содержимое <script>, <style>,
    <noscript>, <iframe>, <svg>, а также элементов UI разметчика
    (.content-labeler-ui и т.п.). Это исключает CSS/JS и кодовые блоки, которые
    BS4 get_text() по умолчанию включает.
    """
    from bs4 import NavigableString, Comment

    parts: list[str] = []
    for child in node.descendants:
        if isinstance(child, Comment):
            continue
        if not isinstance(child, NavigableString):
            continue
        # Поднимаемся по предкам — если встретили исключаемый тег/класс, пропускаем
        skip = False
        parent = child.parent
        while parent is not None and parent is not node.parent:
            name = getattr(parent, "name", None)
            if name in _TEXT_EXCLUDE_TAGS:
                skip = True
                break
            cls = parent.get("class") if hasattr(parent, "get") else None
            if cls:
                if isinstance(cls, str):
                    cls = cls.split()
                if any(c in _TEXT_EXCLUDE_CLASSES for c in cls):
                    skip = True
                    break
            parent = parent.parent
        if skip:
            continue
        s = str(child).strip()
        if s:
            parts.append(s)
    # Сворачиваем пробельные runs в одиночные пробелы
    import re as _re
    joined = " ".join(parts)
    joined = _re.sub(r"\s+", " ", joined).strip()
    return joined


def _extract_main_text(elements: list, predictions: np.ndarray) -> str:
    """
    Собирает основной текст из узлов с label=1.
    Чтобы не дублировать (родитель + его дети размечены main), берём только
    «корневые» main-узлы — те, у которых нет main-предка среди elements.
    Скрипты, стили, UI разметчика и прочий невидимый контент исключаются
    через _visible_text.
    """
    main_node_ids = {
        id(elem) for elem, label in zip(elements, predictions.tolist()) if int(label) == 1
    }
    texts: list[str] = []
    for elem, label in zip(elements, predictions.tolist()):
        if int(label) != 1:
            continue
        # пропускаем если есть main-предок
        has_main_ancestor = False
        for ancestor in elem.parents:
            if id(ancestor) in main_node_ids:
                has_main_ancestor = True
                break
        if has_main_ancestor:
            continue
        t = _visible_text(elem)
        if t:
            texts.append(t)
    return "\n\n".join(texts)


def _apply_thresholds(probs_main: np.ndarray, hs_mask: np.ndarray,
                      thr_with: float | None, thr_no: float | None) -> np.ndarray:
    """
    Применяет per-group thresholds.
    Если thr_with и thr_no = None — возвращает argmax-эквивалент (>= 0.5).
    """
    if thr_with is None or thr_no is None:
        return (probs_main >= 0.5).astype(int)
    pred = np.zeros_like(probs_main, dtype=int)
    pred[hs_mask]  = (probs_main[hs_mask]  >= thr_with).astype(int)
    pred[~hs_mask] = (probs_main[~hs_mask] >= thr_no).astype(int)
    return pred


# ---------------------------------------------------------------------------
# Основная функция инференса
# ---------------------------------------------------------------------------

def predict_labels_for_html_path(
    html_path: Path,
    siblings_dir: Path | None = None,
    sibling_htmls: list[str] | None = None,
    model_path: Path | None = None,
    scaler_stab_path: Path | None = None,
    encoding: str = "utf-8",
    device: torch.device | None = None,
    thr_with_signal: float | None = None,
    thr_no_signal: float | None = None,
    return_elements: bool = False,
) -> list[dict] | tuple[list[dict], list, np.ndarray]:
    """
    Выполняет инференс для одного HTML-файла.

    По умолчанию (return_elements=False) возвращает список словарей по узлам.

    Если return_elements=True — возвращает (nodes, elements, predictions),
    где elements — список BeautifulSoup Tag-узлов (для post-processing:
    разметка HTML, выделение main-текста), а predictions — np.ndarray (N,)
    с финальными метками после применения thresholds.

    Параметры:
        thr_with_signal / thr_no_signal — пороги для main-класса (по группам).
                                          Если оба None — argmax (thr=0.5/0.5).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model_path = MODELS_DIR / V3_WEAK_MODEL_FILE
    if scaler_stab_path is None:
        scaler_stab_path = STAB_PRECOMPUTED_DIR / "scaler_stab.pkl"

    model, prep, ft_path_str, meta = _get_model_and_preprocessors(model_path, device)
    stability_dim = meta["stability_dim"]

    if not ft_path_str or not Path(ft_path_str).is_file():
        raise FileNotFoundError(f"FastText модель не найдена: {ft_path_str}")

    ft_model = _load_fasttext_cached(str(Path(ft_path_str).resolve()))
    scaler_base = prep["scaler"]
    label_encoders = prep["label_encoders"]
    tag_stoi = prep["tag_stoi"]

    # scaler_stab нужен только если модель использует stab (stability_dim > 0)
    scaler_stab = _load_scaler_stab(str(scaler_stab_path)) if stability_dim > 0 else None

    edge_index, _y, elements, data_mls, _ = html_to_graph_tensors(
        html_path, encoding=encoding
    )
    if not elements:
        if return_elements:
            return [], [], np.zeros(0, dtype=int)
        return []

    if sibling_htmls is None:
        sibling_htmls = _load_sibling_htmls(siblings_dir, encoding=encoding)

    if stability_dim > 0 and len(sibling_htmls) < 1:
        print(
            f"[stability_v3_inference] WARNING: нет sibling-страниц для {html_path.name}. "
            "Модель работает в деградированном режиме (stab = 0, эквивалентно no_signal). "
            "Используйте --siblings-dir для указания соседних страниц того же домена.",
            file=sys.stderr,
        )

    # Базовые числовые признаки
    x_tag, x_text, x_class, x_num = compute_page_feature_tensors(
        elements, data_mls, ft_model, label_encoders, tag_stoi
    )
    x_num_np = x_num.numpy().astype(np.float64)
    x_num_scaled = scaler_base.transform(x_num_np).astype(np.float32)

    # Stability-признаки
    if stability_dim > 0:
        x_stab = _compute_stability_for_page(
            html_path, elements, sibling_htmls, stability_dim, encoding=encoding
        )
        # has_signal маска: n_pages_compared > 0 (последний raw столбец stab)
        hs_mask = (x_stab[:, -1] > 0)
        # Когда соседа нет (degraded), x_stab — нули. НЕ масштабируем их: scaler.transform(0)
        # даёт ~-6.7σ выброс по n_pages_compared (mean≈0.98, scale≈0.15), которого модель
        # на всей странице не видела при обучении → коллапс recall. Подаём масштабно-нейтральный 0.
        if sibling_htmls:
            x_stab_scaled = scaler_stab.transform(x_stab.astype(np.float64)).astype(np.float32)
        else:
            x_stab_scaled = np.zeros((len(elements), stability_dim), dtype=np.float32)
        x_num_combined = np.concatenate([x_num_scaled, x_stab_scaled], axis=1)
    else:
        x_num_combined = x_num_scaled
        hs_mask = np.zeros(len(elements), dtype=bool)

    x_num_t = torch.from_numpy(x_num_combined).float().to(device)
    x_tag = x_tag.long().to(device)
    x_text = x_text.float().to(device)
    x_class = x_class.float().to(device)
    edge_index = edge_index.to(device)

    # Инференс
    with torch.no_grad():
        logits = model(x_tag, x_text, x_class, x_num_t, edge_index)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    probs_main = probs[:, 1]
    pred = _apply_thresholds(probs_main, hs_mask, thr_with_signal, thr_no_signal)

    label_names = {0: "template", 1: "main-content"}
    results = []
    for i, elem in enumerate(elements):
        try:
            from xpath_utils import build_xpath_from_bs4_element  # noqa: PLC0415
            xpath = build_xpath_from_bs4_element(elem)
        except Exception:
            xpath = ""

        results.append({
            "node_index": i,
            "xpath": xpath,
            "label": int(pred[i]),
            "label_name": label_names.get(int(pred[i]), str(pred[i])),
            "prob_main": float(probs_main[i]),
            "prob_template": float(probs[i, 0]),
            "has_signal": bool(hs_mask[i]),
        })

    if return_elements:
        return results, elements, pred
    return results


def clear_inference_cache() -> None:
    _load_fasttext_cached.cache_clear()
    _load_bundle_cpu.cache_clear()
    _load_scaler_stab.cache_clear()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_thresholds(s: str | None) -> tuple[float | None, float | None]:
    if s is None:
        return None, None
    try:
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError("expected W,N")
        thr_w = float(parts[0])
        thr_n = float(parts[1])
        if not (0.0 <= thr_w <= 1.0 and 0.0 <= thr_n <= 1.0):
            raise ValueError("thresholds must be in [0, 1]")
        return thr_w, thr_n
    except Exception as e:
        raise argparse.ArgumentTypeError(f"--thresholds: {e}. Пример: --thresholds 0.64,0.75")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Инференс лучшей модели v3-weak (BiDirSAGEStabV3 + DropEdge + FeatDropout-stab). "
            "test_f1_main = 0.7696 с per-group thr 0.64/0.75 на 1-соседской конфигурации."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "html_files",
        nargs="+",
        metavar="HTML_FILE",
        help="Один или несколько HTML-файлов для классификации",
    )
    parser.add_argument(
        "--siblings-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Директория с .html-файлами соседних страниц того же домена. "
            "Используется для вычисления stability-признаков через htmldiff. "
            "Без этого аргумента stab = 0 (деградированный режим, эквивалент no_signal)."
        ),
    )
    parser.add_argument(
        "--max-siblings",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Ограничить число используемых соседей (берутся первые N по имени файла). "
            "По умолчанию — без ограничения (все .html из --siblings-dir). "
            "Для эталонной by1-конфигурации (test_f1_main = 0.7696) укажите --max-siblings 1."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Чекпоинт модели (по умолчанию: {MODELS_DIR / V3_WEAK_MODEL_FILE})",
    )
    parser.add_argument(
        "--scaler-stab",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Путь к scaler_stab.pkl (по умолчанию: {STAB_PRECOMPUTED_DIR / 'scaler_stab.pkl'})",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        metavar="W,N",
        help=(
            "Явно задать per-group пороги для main-класса: --thresholds W,N "
            "(W = для узлов с stab-сигналом, N = для узлов без). "
            f"По умолчанию используются лучшие пороги {V3_WEAK_THR_WITH_SIGNAL},{V3_WEAK_THR_NO_SIGNAL} "
            "(test_f1_main = 0.7696)."
        ),
    )
    parser.add_argument(
        "--argmax",
        action="store_true",
        help=(
            "Использовать argmax (пороги 0.5/0.5) вместо лучших по умолчанию. "
            "Baseline: test_f1_main = 0.7247."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["html", "text"],
        default="text",
        help=(
            "Формат вывода (по умолчанию: text). "
            "text — только основной текст (тексты узлов с label=main). "
            "html — размеченный HTML с классами sure-main-content/sure-template-content."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Путь к выходу. Для одного HTML-файла — путь к файлу. "
            "Для нескольких — путь к директории (внутри будут сохранены "
            "<original_stem>_labeled.html / _main.txt). "
            "Без флага — stdout (только для одного входного файла)."
        ),
    )
    parser.add_argument("--encoding", default="utf-8", metavar="ENC",
                        help="Кодировка HTML-файлов (по умолчанию: utf-8)")
    parser.add_argument("--device", default=None, metavar="DEVICE",
                        help="Устройство: cpu / cuda (по умолчанию: авто)")
    return parser


def _output_path_for(args, html_path: Path, fmt: str, multi: bool) -> Path | None:
    """Определяет путь сохранения для одного входного файла."""
    if args.output is None:
        return None
    suffix_map = {"html": "_labeled.html", "text": "_main.txt"}
    suffix = suffix_map[fmt]
    if multi:
        # output должен быть директорией
        args.output.mkdir(parents=True, exist_ok=True)
        return args.output / (html_path.stem + suffix)
    # single file mode — args.output это файл; создаём родительскую папку при необходимости
    if args.output.parent and not args.output.parent.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
    return args.output


def _force_utf8_streams() -> None:
    """
    Переключаем stdout/stderr на utf-8 с errors='replace',
    чтобы вывод в консоль работал независимо от локали ОС.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main() -> None:
    _force_utf8_streams()
    parser = _build_parser()
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    model_path = args.model
    scaler_stab_path = args.scaler_stab
    fmt = args.format
    multi = len(args.html_files) > 1

    # Проверка существования входных файлов выполняется ПЕРВОЙ — до загрузки
    missing = [f for f in args.html_files if not Path(f).is_file()]
    if missing:
        for f in missing:
            print(f"[stability_v3_inference] ERROR: файл не найден: {f}", file=sys.stderr)
        sys.exit(1)

    if multi and args.output is None:
        print(
            "[stability_v3_inference] ERROR: при нескольких входных файлах требуется "
            "--output PATH (директория). Stdout-режим работает только для одного файла.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.argmax:
        # Явный baseline: argmax (0.5/0.5)
        thr_with, thr_no = None, None
    elif args.thresholds is not None:
        # Явное переопределение порогов пользователем
        thr_with, thr_no = _parse_thresholds(args.thresholds)
    else:
        # По умолчанию — лучшие пороги v3-weak
        thr_with, thr_no = V3_WEAK_THR_WITH_SIGNAL, V3_WEAK_THR_NO_SIGNAL
    if thr_with is not None:
        print(
            f"[stability_v3_inference] per-group thresholds: "
            f"with_signal={thr_with} no_signal={thr_no}",
            file=sys.stderr,
        )
    else:
        print(
            "[stability_v3_inference] thresholds: argmax (0.5/0.5, baseline). "
            "Уберите --argmax для лучших порогов (test_f1_main = 0.7696).",
            file=sys.stderr,
        )

    sibling_htmls = _load_sibling_htmls(args.siblings_dir, encoding=args.encoding)
    if args.siblings_dir is not None and not args.siblings_dir.is_dir():
        print(
            f"[stability_v3_inference] WARNING: --siblings-dir не найден: {args.siblings_dir}",
            file=sys.stderr,
        )
    # Ограничение числа соседей (например, --max-siblings 1 для by1-конфигурации).
    if args.max_siblings is not None and len(sibling_htmls) > args.max_siblings:
        print(
            f"[stability_v3_inference] соседей ограничено до {args.max_siblings} "
            f"(из {len(sibling_htmls)} найденных).",
            file=sys.stderr,
        )
        sibling_htmls = sibling_htmls[:args.max_siblings]

    for html_file_str in args.html_files:
        html_path = Path(html_file_str)
        if not html_path.is_file():
            print(f"[stability_v3_inference] ERROR: файл не найден: {html_path}", file=sys.stderr)
            continue

        print(f"[stability_v3_inference] обработка: {html_path.name}", file=sys.stderr)

        try:
            nodes, elements, preds = predict_labels_for_html_path(
                html_path=html_path,
                sibling_htmls=sibling_htmls,
                model_path=model_path,
                scaler_stab_path=scaler_stab_path,
                encoding=args.encoding,
                device=device,
                thr_with_signal=thr_with,
                thr_no_signal=thr_no,
                return_elements=True,
            )
        except FileNotFoundError as e:
            print(f"[stability_v3_inference] ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[stability_v3_inference] ERROR для {html_path.name}: {e}", file=sys.stderr)
            continue

        n_main = int(sum(1 for n in nodes if n["label"] == 1))
        n_tmpl = int(sum(1 for n in nodes if n["label"] == 0))
        n_with_signal = int(sum(1 for n in nodes if n["has_signal"]))
        print(
            f"[stability_v3_inference]   узлов: {len(nodes)}, "
            f"main: {n_main}, template: {n_tmpl}, "
            f"with_signal: {n_with_signal} / no_signal: {len(nodes) - n_with_signal}",
            file=sys.stderr,
        )

        out_path = _output_path_for(args, html_path, fmt, multi)

        # ---- HTML output ----
        if fmt == "html":
            _apply_labels_to_elements(elements, preds)
            root = _get_root_soup(elements)
            html_out = str(root) if root is not None else ""
            if out_path is not None:
                out_path.write_text(html_out, encoding="utf-8")
                print(f"[stability_v3_inference] сохранён HTML → {out_path}", file=sys.stderr)
            else:
                sys.stdout.write(html_out)
                if not html_out.endswith("\n"):
                    sys.stdout.write("\n")

        # ---- text output ----
        elif fmt == "text":
            text_out = _extract_main_text(elements, preds)
            if out_path is not None:
                out_path.write_text(text_out, encoding="utf-8")
                print(f"[stability_v3_inference] сохранён text → {out_path}", file=sys.stderr)
            else:
                sys.stdout.write(text_out)
                if not text_out.endswith("\n"):
                    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
