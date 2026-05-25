from lxml import etree


class XPathIndex:
    """Maps XPath strings to integer node IDs (DFS order) and vice-versa."""

    def __init__(self, root: etree._Element):
        tree = root.getroottree()
        self._xpath_to_idx: dict[str, int] = {}
        self._idx_to_xpath: dict[int, str] = {}

        idx = 0
        for el in root.iter():
            if callable(el.tag):
                # Skip comment / PI nodes — they have no stable XPath
                continue
            xpath = tree.getpath(el)
            self._xpath_to_idx[xpath] = idx
            self._idx_to_xpath[idx] = xpath
            idx += 1

    @classmethod
    def from_node_order(cls, root: etree._Element, node_order: list[str]) -> 'XPathIndex':
        """Build an index using a caller-supplied XPath ordering.

        ``node_order[i]`` is the XPath of the node with ID ``i``.
        Only XPaths present in ``node_order`` are indexed; any XPath from the
        actual tree that is absent from ``node_order`` will not be reachable
        via ``resolve``, but will still be reachable via ``resolve_with_ancestor_fallback``
        if an ancestor is in the index.
        """
        instance = cls.__new__(cls)
        instance._xpath_to_idx = {}
        instance._idx_to_xpath = {}

        idx = 0
        for xpath in node_order:
            if xpath not in instance._xpath_to_idx:
                instance._xpath_to_idx[xpath] = idx
                instance._idx_to_xpath[idx] = xpath
                idx += 1

        return instance

    def resolve(self, xpath: str) -> int | None:
        """Return node ID for an exact XPath match, or ``None``."""
        return self._xpath_to_idx.get(xpath)

    def resolve_with_ancestor_fallback(self, xpath: str) -> int | None:
        """Return node ID for the closest ancestor XPath that is in the index.

        Tries the exact XPath first, then walks up the tree by stripping the
        last path segment one step at a time until a match is found.
        """
        current = xpath
        while current:
            idx = self._xpath_to_idx.get(current)
            if idx is not None:
                return idx
            # Strip last segment
            parent = current.rsplit('/', 1)[0]
            if not parent or parent == current:
                break
            current = parent
        return None

    @property
    def xpath_to_idx(self) -> dict[str, int]:
        return self._xpath_to_idx

    @property
    def idx_to_xpath(self) -> dict[int, str]:
        return self._idx_to_xpath

    def __len__(self) -> int:
        return len(self._xpath_to_idx)
