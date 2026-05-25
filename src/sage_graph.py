# -*- coding: utf-8 -*-
"""
Построение графа PyTorch Geometric из HTML: узлы с data-ml, рёбра child→parent.
Порядок узлов совпадает с dataset.iter_nodes_from_html и text_extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (  # noqa: E402
    MERGED_PAGES_DIR,
    _extract_data_ml,
    _get_tag_name,
    _inherit_label,
    read_html_for_ml_inference,
)

try:
    from bs4 import BeautifulSoup
    from bs4.element import Tag
except ImportError:
    BeautifulSoup = None
    Tag = None


def iter_elements_labels_data_ml(
    html_path: Path,
    merged_pages_dir: Path | None = None,
    encoding: str = "utf-8",
    label_fn=None,
    page_id_override: str | None = None,
) -> tuple[list[Tag], list[dict[str, Any]], list[int], str]:
    """
    Обходит HTML в том же порядке, что iter_nodes_from_html:
    возвращает списки элементов, data-ml dict, меток и page_id.

    label_fn: callable(el, soup) -> int — альтернативный источник меток
              (для внешних датасетов L3S, Google Trends).
              Если None, используется _inherit_label (метки из CSS-классов).
    page_id_override: если задан, используется вместо вычисленного page_id.
    """
    if BeautifulSoup is None:
        raise RuntimeError(
            "Требуется bs4 + html5lib. Установите: pip install beautifulsoup4 html5lib"
        )

    base = merged_pages_dir if merged_pages_dir is not None else MERGED_PAGES_DIR
    content = read_html_for_ml_inference(html_path, encoding=encoding)

    soup = BeautifulSoup(content, "html5lib")

    if page_id_override is not None:
        page_id = page_id_override
    else:
        try:
            page_id = str(html_path.relative_to(base))
        except ValueError:
            page_id = html_path.name

    elements: list[Tag] = []
    data_mls: list[dict[str, Any]] = []
    labels: list[int] = []

    for el in soup.descendants:
        if not isinstance(el, Tag):
            continue
        if _get_tag_name(el) == "":
            continue
        data_ml = _extract_data_ml(el)
        if data_ml is None:
            continue
        if label_fn is not None:
            label = label_fn(el, soup)
        else:
            label = _inherit_label(el, soup)
        elements.append(el)
        data_mls.append(data_ml)
        labels.append(label)

    return elements, data_mls, labels, page_id


def build_child_to_parent_edge_index(
    elements: list[Tag],
) -> torch.Tensor:
    """
    Для каждого узла — ближайший предок с data-ml: ребро child_i -> parent_j.
    Формат: edge_index [2, E], строка 0 = источник (ребёнок), строка 1 = цель (родитель).
    """
    id_to_idx = {id(el): i for i, el in enumerate(elements)}
    src: list[int] = []
    dst: list[int] = []

    for i, el in enumerate(elements):
        p = el.parent if isinstance(el.parent, Tag) else None
        while p is not None:
            pid = id(p)
            if pid in id_to_idx:
                src.append(i)
                dst.append(id_to_idx[pid])
                break
            p = p.parent if isinstance(p.parent, Tag) else None

    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def html_to_graph_tensors(
    html_path: Path,
    merged_pages_dir: Path | None = None,
    encoding: str = "utf-8",
    label_fn=None,
    page_id_override: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[Tag], list[dict[str, Any]], str]:
    """
    Возвращает edge_index, y [N], элементы, data-ml список, page_id.

    label_fn: callable(el, soup) -> int — для внешних датасетов.
    page_id_override: явно задать page_id вместо вычисленного из пути.
    """
    elements, data_mls, labels, page_id = iter_elements_labels_data_ml(
        html_path,
        merged_pages_dir=merged_pages_dir,
        encoding=encoding,
        label_fn=label_fn,
        page_id_override=page_id_override,
    )
    if not elements:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            [],
            [],
            page_id,
        )

    edge_index = build_child_to_parent_edge_index(elements)
    y = torch.tensor(labels, dtype=torch.long)
    return edge_index, y, elements, data_mls, page_id
