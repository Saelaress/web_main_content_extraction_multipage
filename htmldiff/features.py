import numpy as np

from .differ import EditEvent, ChangeCategory
from .indexer import XPathIndex


def build_feature_matrix(
    events_per_diff: list[list[EditEvent]],
    index: XPathIndex,
    config,
    normalize: bool = False,
) -> np.ndarray:
    """Build a per-node feature matrix from a list of diff event lists.

    Returns an array of shape ``(len(index), 5)`` with columns:
        0 - n_structural
        1 - n_content
        2 - n_attr
        3 - n_total  (sum of cols 0-2)
        4 - n_pages_compared  (how many diffs contributed to each row)
    """
    n_nodes = len(index)
    matrix = np.zeros((n_nodes, 5), dtype=np.float32)

    for events in events_per_diff:
        for event in events:
            node_id = index.resolve_with_ancestor_fallback(event.xpath)
            if node_id is None:
                continue
            matrix[node_id, event.category.value] += 1

        # Every diff counts as one comparison for all nodes in the index
        matrix[:, 4] += 1

    # n_total = sum of structural + content + attr counts
    matrix[:, 3] = matrix[:, 0] + matrix[:, 1] + matrix[:, 2]

    if normalize:
        pages = matrix[:, 4]
        with np.errstate(divide='ignore', invalid='ignore'):
            for col in range(4):
                matrix[:, col] = np.where(pages > 0, matrix[:, col] / pages, 0.0)

    return matrix
