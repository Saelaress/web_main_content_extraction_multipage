import hashlib
import json
from pathlib import Path

from .differ import EditEvent, ChangeCategory


# Increment when the diff pipeline changes in a way that invalidates old cache entries.
# v4: switched HTML parser from lxml.html to html5lib (consistent XPath with BS4)
# v5: switched back to lxml.html (6-7x faster, better BS4 XPath alignment in practice)
_CACHE_VERSION = b"v5"

# Fields that don't affect diff output — excluded from cache key
_KEY_EXCLUDE = frozenset({'n_workers', 'cache_dir', 'diff_timeout'})


def _cache_key(base_bytes: bytes, compare_html: str, config) -> str:
    """Compute a deterministic SHA-256 cache key for a diff pair."""
    h = hashlib.sha256()
    h.update(_CACHE_VERSION)
    h.update(base_bytes)
    h.update(compare_html.encode('utf-8', errors='replace'))
    relevant = {k: v for k, v in config.__dict__.items() if k not in _KEY_EXCLUDE}
    h.update(repr(sorted(relevant.items())).encode())
    return h.hexdigest()


def load_cached(cache_dir: str, domain: str, key: str) -> list[EditEvent] | None:
    """Load a cached diff result from disk, returning ``None`` if absent."""
    path = Path(cache_dir) / domain / f"{key}.json"
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return [EditEvent(xpath=e['xpath'], category=ChangeCategory(e['category'])) for e in data]


def save_cached(cache_dir: str, domain: str, key: str, events: list[EditEvent]) -> None:
    """Persist a diff result to disk under ``cache_dir/domain/key.json``."""
    path = Path(cache_dir) / domain
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"{key}.json", 'w', encoding='utf-8') as f:
        json.dump(
            [{'xpath': e.xpath, 'category': e.category.value} for e in events],
            f,
        )
