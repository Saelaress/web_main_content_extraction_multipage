from multiprocessing import Pool, cpu_count

from lxml import etree

from .config import HtmlDiffConfig
from .differ import diff_trees_safe, EditEvent
from .parser import parse_html
from .cache import _cache_key, load_cached, save_cached


def _worker(args: tuple) -> list[EditEvent]:
    """Multiprocessing worker: reconstruct config and run diff."""
    base_bytes, compare_html, config_dict = args
    config = HtmlDiffConfig(**config_dict)
    return diff_trees_safe(base_bytes, compare_html, config)


def _config_to_dict(config: HtmlDiffConfig) -> dict:
    """Serialise a HtmlDiffConfig to a plain dict safe for pickling."""
    d = {k: v for k, v in config.__dict__.items()}
    # Convert sets to lists for pickle / JSON safety
    d['ignored_attrs'] = list(config.ignored_attrs)
    d['strip_url_params_attrs'] = list(config.strip_url_params_attrs)
    return d


def run_star_diffs(
    pages: list[str],
    config: HtmlDiffConfig,
    domain: str = 'default',
    base_tree=None,
) -> tuple[bytes, list[list[EditEvent]]]:
    """Diff every page in *pages* against ``pages[0]`` (star topology).

    Returns ``(base_html_bytes, list_of_events_per_diff)`` where
    ``list_of_events_per_diff[i]`` corresponds to the diff of ``pages[0]``
    vs ``pages[i+1]``.

    *base_tree* may be supplied if the caller already parsed ``pages[0]``
    (e.g. to build an XPathIndex) — avoids a second html5lib parse.
    """
    base_html = pages[0]
    if base_tree is None:
        base_tree = parse_html(base_html, config)
    base_bytes: bytes = etree.tostring(
        base_tree, encoding='unicode', method='xml'
    ).encode('utf-8')

    compare_pages = pages[1:]
    if not compare_pages:
        return base_bytes, []

    config_dict = _config_to_dict(config)

    # Build the work list, resolving cache hits upfront
    args_list: list[tuple | None] = []
    cached_results: dict[int, list[EditEvent]] = {}

    for i, compare_html in enumerate(compare_pages):
        if config.cache_dir:
            key = _cache_key(base_bytes, compare_html, config)
            cached = load_cached(config.cache_dir, domain, key)
            if cached is not None:
                cached_results[i] = cached
                args_list.append(None)
                continue
        args_list.append((base_bytes, compare_html, config_dict))

    results_list: list[list[EditEvent] | None] = [None] * len(compare_pages)
    for i, r in cached_results.items():
        results_list[i] = r

    to_process = [(i, args) for i, args in enumerate(args_list) if args is not None]

    if not to_process:
        return base_bytes, results_list  # type: ignore[return-value]

    n_workers = config.n_workers if config.n_workers > 0 else max(1, cpu_count() - 1)
    n_workers = min(n_workers, len(to_process))

    if n_workers == 1 or len(to_process) == 1:
        for i, args in to_process:
            import time as _time
            _t0 = _time.perf_counter()
            base_b, cmp_html, _ = args
            # quick element count from serialised XML (no re-parse needed)
            n_base = base_b.count(b'<') - base_b.count(b'</')
            n_cmp  = cmp_html.count('<') - cmp_html.count('</')
            print(f"[diff] pair {i+1}/{len(compare_pages)} base~{n_base} cmp~{n_cmp} el", flush=True)
            events = _worker(args)
            print(f"[diff] pair {i+1} done in {_time.perf_counter()-_t0:.1f}s events={len(events)}", flush=True)
            results_list[i] = events
            if config.cache_dir:
                key = _cache_key(base_bytes, compare_pages[i], config)
                save_cached(config.cache_dir, domain, key, events)
    else:
        with Pool(processes=n_workers) as pool:
            worker_args = [args for _, args in to_process]
            worker_results = pool.map(_worker, worker_args)
        for (i, _), events in zip(to_process, worker_results):
            results_list[i] = events
            if config.cache_dir:
                key = _cache_key(base_bytes, compare_pages[i], config)
                save_cached(config.cache_dir, domain, key, events)

    return base_bytes, results_list  # type: ignore[return-value]
