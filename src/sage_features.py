# -*- coding: utf-8 -*-
"""
Признаки для PyG: индекс тега (словарь по train + UNK), FastText только текст и class,
санитизация class, числовой вектор как в sklearn (FEATURE_SPEC + LabelEncoder).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LABEL_MAIN, LABEL_TEMPLATE  # noqa: E402
from text_extraction import get_node_visible_text  # noqa: E402
from vectorize import data_ml_to_flat_row, prepare_for_sklearn  # noqa: E402

try:
    from bs4.element import Tag
except ImportError:
    Tag = None

# Метки разметки в class — убираем из эмбеддинга (утечка).
# x-nc-sel* — служебные классы внешнего бенчмарка L3S-GN1; на обычных
# страницах не встречаются, но фильтруются для согласованности с обучением.
_L3S_CONTENT_CLASSES = frozenset({"x-nc-sel1", "x-nc-sel2", "x-nc-sel3"})
_LEAKAGE_CLASSES = frozenset({LABEL_MAIN, LABEL_TEMPLATE}) | _L3S_CONTENT_CLASSES

# Индекс 0 в nn.Embedding — UNK / пустой тег; известные теги с 1
TAG_UNK_INDEX = 0


def sanitize_class_for_embedding(class_attr: str | None) -> list[str]:
    """
    Слова из class-атрибута без меток разметки.
    Разбиваем по пробелам (несколько классов), затем каждый класс —
    по дефисам и подчёркиваниям (BEM, snake_case, kebab-case).
    Пример: "article__body content-main" → ["article", "body", "content", "main"]
    Если class отсутствует — пустой список → нулевой эмбеддинг.
    """
    if not class_attr or not str(class_attr).strip():
        return []
    if isinstance(class_attr, list):
        class_tokens = [str(t) for t in class_attr if str(t).strip()]
    else:
        class_tokens = str(class_attr).split()
    # удаляем метки разметки до разбивки
    class_tokens = [t for t in class_tokens if t.strip() not in _LEAKAGE_CLASSES]
    # разбиваем каждый class-токен по дефисам и подчёркиваниям → отдельные слова
    words: list[str] = []
    for token in class_tokens:
        parts = re.split(r"[-_]+", token)
        words.extend(p.lower() for p in parts if p)
    return words


def tokenize_words(text: str) -> list[str]:
    """Простая токенизация для FastText."""
    if not text or not text.strip():
        return []
    # буквы/цифры и дефисы внутри слов
    return re.findall(r"[^\W\d_]+|\d+", text.lower(), flags=re.UNICODE)


def _avg_word_vectors(ft_model: Any, words: list[str]) -> np.ndarray:
    """Усреднение векторов слов; пустой список → нули."""
    dim = int(ft_model.get_dimension())
    if not words:
        return np.zeros(dim, dtype=np.float32)
    vecs = []
    for w in words:
        try:
            vecs.append(np.asarray(ft_model.get_word_vector(w), dtype=np.float32))
        except Exception:
            continue
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


def tag_as_word(el: Tag) -> str:
    if el is None or not isinstance(el, Tag):
        return ""
    return str(el.name or "").lower()


def tag_to_index(el: Tag, tag_stoi: dict[str, int], unk: int = TAG_UNK_INDEX) -> int:
    t = tag_as_word(el)
    if not t:
        return unk
    return int(tag_stoi.get(t, unk))


def compute_fasttext_text_class(
    el: Tag,
    ft_model: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Два сырых FastText-вектора (dim,): видимый текст узла и class (санитизированный)."""
    vis = get_node_visible_text(el)
    text_vec = _avg_word_vectors(ft_model, tokenize_words(vis))

    cls_raw = el.get("class")
    cls_tokens = sanitize_class_for_embedding(cls_raw)
    class_vec = _avg_word_vectors(ft_model, cls_tokens)

    return text_vec, class_vec


def flat_row_for_node(data_ml: dict[str, Any]) -> dict[str, Any]:
    """Плоская строка для vectorize + bool→int."""
    row = data_ml_to_flat_row(data_ml)
    out = dict(row)
    for k, v in list(out.items()):
        if isinstance(v, bool):
            out[k] = 1 if v else 0
    return out


def rows_to_numeric_matrix(
    flat_rows: list[dict[str, Any]],
    label_encoders: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Та же матрица X, что для sklearn (категории через LabelEncoder)."""
    X, _, encoders = prepare_for_sklearn(flat_rows, label_encoders=label_encoders)
    return X.astype(np.float64), encoders or {}


def stack_numeric_for_page(
    data_mls: list[dict[str, Any]],
    label_encoders: dict[str, Any],
) -> np.ndarray:
    flat = [flat_row_for_node(d) for d in data_mls]
    X, _ = rows_to_numeric_matrix(flat, label_encoders=label_encoders)
    return X


def numpy_text_class_to_tensors(
    text_list: list[np.ndarray],
    class_list: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Списки векторов одной страницы → FloatTensor [N, dim]."""
    if not text_list:
        z = torch.zeros(0, 1, dtype=torch.float32)
        return z, z
    return (
        torch.from_numpy(np.stack(text_list, axis=0)),
        torch.from_numpy(np.stack(class_list, axis=0)),
    )


def compute_page_feature_tensors(
    elements: list[Tag],
    data_mls: list[dict[str, Any]],
    ft_model: Any,
    label_encoders: dict[str, Any],
    tag_stoi: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Одна страница: x_tag [N] long (индексы эмбеддинга тега), x_text/x_class [N, ft_dim], x_num [N, F].
    """
    tag_idx: list[int] = []
    text_l, class_l = [], []
    for el in elements:
        tag_idx.append(tag_to_index(el, tag_stoi))
        xv, cv = compute_fasttext_text_class(el, ft_model)
        text_l.append(xv)
        class_l.append(cv)

    x_tag = torch.tensor(tag_idx, dtype=torch.long)
    x_text, x_class = numpy_text_class_to_tensors(text_l, class_l)
    X_num = stack_numeric_for_page(data_mls, label_encoders)
    x_num = torch.from_numpy(X_num.astype(np.float32))
    return x_tag, x_text, x_class, x_num
