import warnings
from dataclasses import dataclass

import numpy as np

from .config import HtmlDiffConfig
from .exceptions import InsufficientPagesWarning
from .features import build_feature_matrix
from .indexer import XPathIndex
from .orchestrator import run_star_diffs
from .parser import parse_html

__all__ = [
    'HtmlDiffConfig',
    'StabilityResult',
    'compute_stability_features',
    # sub-modules exposed for direct import
    'InsufficientPagesWarning',
]


@dataclass
class StabilityResult:
    feature_matrix: np.ndarray      # shape (N_nodes, 5)
    xpath_to_idx: dict[str, int]
    idx_to_xpath: dict[int, str]
    pages_compared: int


def compute_stability_features(
    pages: list[str],
    node_order: list[str] | None = None,
    config: HtmlDiffConfig | None = None,
    normalize: bool = False,
    domain: str = 'default',
) -> StabilityResult:
    """Compute per-node structural stability features for a set of HTML pages.

    Parameters
    ----------
    pages:
        List of raw HTML strings from the same domain.  The first page is used
        as the *base* tree; all others are diffed against it.
    node_order:
        Optional list of XPath strings defining the GNN node ordering.  When
        provided, the feature matrix rows correspond to this ordering.
    config:
        Optional :class:`HtmlDiffConfig`; defaults are used when ``None``.
    normalize:
        If ``True``, divide raw counts by the number of pages compared.
    domain:
        Domain label used for cache file organisation.

    Returns
    -------
    StabilityResult
    """
    if config is None:
        config = HtmlDiffConfig()

    if len(pages) < 2:
        warnings.warn(
            f"Only {len(pages)} page(s) provided for domain '{domain}'. "
            "Returning zero feature matrix.",
            InsufficientPagesWarning,
            stacklevel=2,
        )

    base_html = pages[0]
    base_tree = parse_html(base_html, config)

    if node_order is not None:
        index = XPathIndex.from_node_order(base_tree, node_order)
    else:
        index = XPathIndex(base_tree)

    if len(pages) < 2:
        matrix = np.zeros((len(index), 5), dtype=np.float32)
        return StabilityResult(
            feature_matrix=matrix,
            xpath_to_idx=index.xpath_to_idx,
            idx_to_xpath=index.idx_to_xpath,
            pages_compared=0,
        )

    _, events_per_diff = run_star_diffs(pages, config, domain=domain, base_tree=base_tree)
    matrix = build_feature_matrix(events_per_diff, index, config, normalize=normalize)

    # Expand compact matrix (n_unique_xpaths × 5) back to (len(node_order) × 5).
    # Duplicate XPaths in node_order (malformed HTML error-recovery artefacts) cause
    # from_node_order to assign contiguous indices to unique XPaths only, so the
    # compact matrix may have fewer rows than len(node_order). Each GNN node gets
    # the feature row of its first-occurrence XPath.
    if node_order is not None and len(matrix) != len(node_order):
        matrix_expanded = np.zeros((len(node_order), 5), dtype=np.float32)
        for i, xpath in enumerate(node_order):
            compact_idx = index.resolve(xpath)
            if compact_idx is not None:
                matrix_expanded[i] = matrix[compact_idx]
        matrix = matrix_expanded

    return StabilityResult(
        feature_matrix=matrix,
        xpath_to_idx=index.xpath_to_idx,
        idx_to_xpath=index.idx_to_xpath,
        pages_compared=len(pages) - 1,
    )
