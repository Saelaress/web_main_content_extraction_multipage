# -*- coding: utf-8 -*-
"""
Извлечение видимого текста узла DOM (для FastText-эмбеддингов в sage_features).
Сохраняет блочную структуру абзацев и пропускает технические поддеревья
(script/style/noscript/iframe/svg).
"""

from __future__ import annotations

import re

try:
    from bs4.element import NavigableString, Tag
except ImportError:
    NavigableString = None
    Tag = None


def _get_tag_name(el: Tag | None) -> str:
    if not isinstance(el, Tag):
        return ""
    return (el.name or "").lower()


_BLOCK_TAGS = frozenset({
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "dt", "dd",
    "tr", "td", "th", "caption",
    "blockquote", "pre", "address",
    "article", "section", "header", "footer", "main", "nav",
    "aside", "figure", "figcaption",
    "details", "summary",
})


def _collect_text_recursive(el: Tag, parts: list) -> None:
    """Рекурсивно собирает текст, вставляя \n на блочных тегах и пропуская технические поддеревья."""
    tag = _get_tag_name(el)
    if tag == "":
        return
    if tag in ("script", "style", "noscript", "iframe", "svg"):
        return
    if tag == "br":
        parts.append("\n")
        return
    is_block = tag in _BLOCK_TAGS
    if is_block:
        parts.append("\n")
    for child in el.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            _collect_text_recursive(child, parts)
    if is_block:
        parts.append("\n")


def _get_text_content_clean(el: Tag) -> str:
    """Извлекает текст узла с сохранением блочной структуры абзацев."""
    parts: list[str] = []
    _collect_text_recursive(el, parts)
    text = "".join(parts)
    text = re.sub(r"[^\S\n]+", " ", text)   # схлопывает горизонтальные пробелы
    text = re.sub(r" *\n *", "\n", text)     # убирает пробелы у переносов
    text = re.sub(r"\n{2,}", "\n", text)     # схлопывает многократные переносы
    return text.strip()


def get_node_visible_text(el: Tag) -> str:
    """
    Видимый текст узла (как для ground truth / эмбеддингов).
    Обертка над _get_text_content_clean для использования из sage_features.
    """
    return _get_text_content_clean(el)
