# -*- coding: utf-8 -*-
"""
Утилиты для построения XPath-строк из BeautifulSoup-элементов.

NOTE (2026-05-15): htmldiff/parser.py now uses html5lib with treebuilder="lxml"
and namespaceHTMLElements=False, which is the same algorithm BS4 uses internally.
XPaths from tree.getpath() in the diff pipeline therefore match BS4 element
positions natively.  This module's manual XPath construction is likely no longer
needed for the diff pipeline, but is kept for reference and backward compatibility
in case it is still called from GNN feature extraction code.

bs4 использует html5lib-парсер, lxml — свой. Для некорректного HTML порядок
узлов может различаться. Чтобы строки feature_matrix из htmldiff совпадали
с узлами GNN-тензора, мы строим XPath вручную, обходя tag.parents.
"""

from __future__ import annotations

try:
    from bs4.element import Tag
except ImportError:
    Tag = None  # type: ignore[misc,assignment]


def build_xpath_from_bs4_element(tag) -> str:
    """
    Возвращает XPath-строку для BS4-тега в стиле lxml getpath().

    lxml добавляет [n] только когда среди siblings есть несколько элементов
    с одинаковым именем. Уникальные элементы пишутся без индекса: /html/body/div[2].
    BS4's [document]-нода игнорируется — она не является настоящим XML-элементом.
    """
    from bs4.element import Tag as BsTag

    segments: list[str] = []
    current = tag
    while current is not None and isinstance(current, BsTag):
        if current.name == '[document]':
            break
        parent = current.parent
        if parent is None or not isinstance(parent, BsTag) or parent.name == '[document]':
            # Корневой элемент (html) — всегда уникален
            segments.append(f"/{current.name}")
            break
        # Все siblings с тем же именем
        same = [s for s in parent.children if isinstance(s, BsTag) and s.name == current.name]
        if len(same) > 1:
            pos = same.index(current) + 1
            segments.append(f"/{current.name}[{pos}]")
        else:
            segments.append(f"/{current.name}")
        current = parent

    segments.reverse()
    return "".join(segments)


def build_node_order(elements) -> list[str]:
    """
    Возвращает список XPath-строк для каждого элемента в DFS-порядке BS4.

    Передаётся как node_order в compute_stability_features().
    """
    return [build_xpath_from_bs4_element(el) for el in elements]
