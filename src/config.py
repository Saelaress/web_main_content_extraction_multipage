# -*- coding: utf-8 -*-
"""
Конфиг для запуска модели
Все пути относительны корню репозитория
"""

from __future__ import annotations

import os
from pathlib import Path

# Корень репозитория
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Артефакты модели ---
MODELS_DIR           = PROJECT_ROOT / "models"
STAB_PRECOMPUTED_DIR = PROJECT_ROOT / "models"
SAGE_PREPROCESSORS_FILE = "sage_preprocessors.pt"

# --- Классы разметки ---
CLASS_MAIN     = 1
CLASS_TEMPLATE = 0
LABEL_MAIN     = "sure-main-content"
LABEL_TEMPLATE = "sure-template-content"

# --- Категориальные признаки (для совместимости с vectorize.py) ---
CATEGORICAL_COLUMNS = [
    "node__tag_name",
    "node__parent_tag",
    "node__grandparent_tag",
    "language__language_code",
]

# --- Схема плоского вектора (data-ml) — нужна для vectorize.py ---
FEATURE_SPEC = [
    # node
    "node.dom_depth",
    "node.dom_depth_norm",
    "node.tag_name",
    "node.tag_is_a",
    "node.tag_is_div",
    "node.tag_is_p",
    "node.tag_is_heading",
    "node.tag_is_article",
    "node.tag_is_nav",
    "node.tag_is_footer",
    "node.tag_is_header",
    "node.parent_tag",
    "node.grandparent_tag",
    "node.has_parent_article",
    "node.is_excluded_tag",
    "node.num_children",
    # subtree
    "subtree.tag_count",
    "subtree.num_leaves",
    "subtree.text_length_chars",
    "subtree.word_count",
    "subtree.link_count",
    "subtree.link_text_length",
    "subtree.link_text_ratio",
    "subtree.text_without_links_ratio",
    "subtree.words_per_tag",
    "subtree.words_per_leaf",
    "subtree.chars_per_descendant",
    "subtree.links_per_descendant",
    "subtree.children_ratio",
    # text
    "text.has_visible_text",
    "text.is_whitespace_only",
    "text.has_only_links",
    "text.digit_ratio",
    "text.r_punctuation",
    "text.ends_with_punctuation",
    "text.num_lines",
    "text.avg_word_length",
    "text.avg_sentence_length",
    "text.nlp_comma_density",
    # meta
    "meta.has_email",
    "meta.has_microdata_article",
    "meta.image_caption_ratio",
    "meta.list_internal_link_ratio",
    # language
    "language.language_code",
    "language.language_confidence",
    # def31
    "def31.word_ratio",
    "def31.hyperlink_ratio",
    "def31.children_ratio_binary",
    "def31.position_ratio",
]

# --- FastText ---
def _resolve_fasttext_model_path() -> Path:
    """
    Приоритет: переменная окружения FASTTEXT_MODEL_PATH → models/cc.xx.300.bin →
               models/cc.en.300.bin → PROJECT_ROOT/cc.en.300.bin.
    Если ничего не найдено, возвращается пустой Path() — тогда инференс
    выбросит понятную ошибку с указанием запустить download_fasttext.py.
    """
    env = os.environ.get("FASTTEXT_MODEL_PATH", "").strip()
    if env:
        return Path(env).expanduser()
    for base in (MODELS_DIR, PROJECT_ROOT):
        for fname in ("cc.xx.300.bin", "cc.en.300.bin"):
            p = (base / fname).resolve()
            if p.is_file():
                return p
    return Path()


FASTTEXT_MODEL_PATH = _resolve_fasttext_model_path()

# --- Гиперпараметры BiDirSAGEStabV3 (для построения архитектуры под чекпоинт) ---
STAB_EMBED_DIM       = 64
STAB_HIDDEN_DIM      = 128
STAB_NUM_LAYERS      = 3
STAB_STABILITY_DIM   = 5      # [n_structural, n_content, n_attr, n_total, n_pages_compared]
STAB_V3_DROP_EDGE_P  = 0.1
STAB_V3_FEAT_DROP_P  = 0.2

# Заглушка для dataset.py — он импортирует MERGED_PAGES_DIR, но в инференсе не используется.
MERGED_PAGES_DIR = PROJECT_ROOT / "samples"
