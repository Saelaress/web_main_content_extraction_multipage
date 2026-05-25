# -*- coding: utf-8 -*-
"""
Чтение HTML для инференса: загрузка файла, при необходимости разметка data-ml
через annotate_dom, наследование меток из CSS-классов (sure-main/template-content).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

MAX_PATH_WIN = 259

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MERGED_PAGES_DIR,
    LABEL_MAIN,
    LABEL_TEMPLATE,
    CLASS_MAIN,
    CLASS_TEMPLATE,
)

try:
    from bs4 import BeautifulSoup
    from bs4.element import Tag
except ImportError:
    BeautifulSoup = None
    Tag = None


def _path_for_open(path: Path) -> str:
    path = path.resolve()
    s = str(path)
    if sys.platform == "win32" and len(s) >= MAX_PATH_WIN:
        if not s.startswith("\\\\?\\"):
            return "\\\\?\\" + s
    return s


def _iter_tags(root):
    for el in getattr(root, "descendants", []):
        if isinstance(el, Tag):
            yield el


def _get_tag_name(el: Tag | None) -> str:
    if not isinstance(el, Tag):
        return ""
    return (el.name or "").lower()


def _get_label_from_class(el: Tag) -> int | None:
    cls_attr = el.get("class")
    if not cls_attr:
        return None
    if isinstance(cls_attr, list):
        raw_classes = cls_attr
    else:
        raw_classes = str(cls_attr).split()
    classes = [c.strip() for c in raw_classes if c.strip()]
    if LABEL_MAIN in classes:
        return CLASS_MAIN
    if LABEL_TEMPLATE in classes:
        return CLASS_TEMPLATE
    return None


def _inherit_label(el: Tag, root) -> int:
    explicit = _get_label_from_class(el)
    if explicit is not None:
        return explicit
    tag = _get_tag_name(el)
    if tag in ("script", "style", "noscript"):
        return CLASS_TEMPLATE
    parent = el.parent if isinstance(el.parent, Tag) else None
    while parent is not None and parent != root:
        parent_label = _get_label_from_class(parent)
        if parent_label is not None:
            return parent_label
        parent = parent.parent if isinstance(parent.parent, Tag) else None
    # Нет ни явной метки, ни помеченного предка — sentinel для двухпроходной разметки.
    return -1  # sentinel: явно не помечен


def _extract_data_ml(el: Tag) -> dict[str, Any] | None:
    raw = el.get("data-ml")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def ensure_ml_annotated_html_content(html: str) -> str:
    """
    Если в дереве нет ни одного валидного data-ml, прогоняет HTML через annotate_dom
    """
    if BeautifulSoup is None:
        return html
    soup = BeautifulSoup(html, "html5lib")
    for el in _iter_tags(soup):
        if _get_tag_name(el) == "":
            continue
        if _extract_data_ml(el) is not None:
            return html
    try:
        # annotate_dom.py лежит рядом с dataset.py в test_diplom/src/
        ar = str(Path(__file__).resolve().parent)
        if ar not in sys.path:
            sys.path.insert(0, ar)
        from annotate_dom import annotate_html  # noqa: WPS433

        return annotate_html(html)
    except Exception:
        return html


def read_html_for_ml_inference(html_path: Path, encoding: str = "utf-8") -> str:
    """Читает HTML и при необходимости добавляет data-ml для инференса."""
    path_to_open = _path_for_open(html_path)
    with open(path_to_open, "rb") as f:
        content = f.read().decode(encoding, errors="replace")
    return ensure_ml_annotated_html_content(content)
