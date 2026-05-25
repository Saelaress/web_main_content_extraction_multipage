import multiprocessing
import sys
import threading
import warnings
from dataclasses import dataclass
from enum import Enum

from xmldiff import actions as xa, main as xmain

from .config import HtmlDiffConfig
from .exceptions import DiffTimeoutWarning


class ChangeCategory(Enum):
    STRUCTURAL = 0   # InsertNode, DeleteNode, MoveNode, InsertComment
    CONTENT    = 1   # UpdateTextIn, UpdateTextAfter
    ATTRIBUTE  = 2   # UpdateAttrib, InsertAttrib, DeleteAttrib, RenameAttrib, RenameNode


@dataclass
class EditEvent:
    xpath: str
    category: ChangeCategory


def _classify_action(action) -> 'EditEvent | None':
    """Convert an xmldiff action into an EditEvent, or None if unrecognised."""
    if isinstance(action, xa.InsertNode):
        return EditEvent(xpath=action.target, category=ChangeCategory.STRUCTURAL)
    if isinstance(action, xa.DeleteNode):
        return EditEvent(xpath=action.node, category=ChangeCategory.STRUCTURAL)
    if isinstance(action, xa.MoveNode):
        return EditEvent(xpath=action.node, category=ChangeCategory.STRUCTURAL)
    if isinstance(action, xa.InsertComment):
        return EditEvent(xpath=action.target, category=ChangeCategory.STRUCTURAL)
    if isinstance(action, (xa.UpdateTextIn, xa.UpdateTextAfter)):
        return EditEvent(xpath=action.node, category=ChangeCategory.CONTENT)
    if isinstance(action, (xa.UpdateAttrib, xa.InsertAttrib, xa.DeleteAttrib, xa.RenameAttrib)):
        return EditEvent(xpath=action.node, category=ChangeCategory.ATTRIBUTE)
    if isinstance(action, xa.RenameNode):
        return EditEvent(xpath=action.node, category=ChangeCategory.ATTRIBUTE)
    return None


def _diff_inprocess(base_bytes: bytes, compare_html: str,
                    config: HtmlDiffConfig) -> list[EditEvent]:
    """Run diff entirely in-process and return a list of EditEvent."""
    from .parser import parse_html

    base_tree = parse_html(base_bytes.decode('utf-8', errors='replace'), config)
    compare_tree = parse_html(compare_html, config)

    actions = xmain.diff_trees(
        base_tree,
        compare_tree,
        diff_options={
            'fast_match': True,
            'ratio_mode': config.diff_ratio_mode,
            'uniqueattrs': config.uniqueattrs,
        },
    )

    events = []
    for action in actions:
        event = _classify_action(action)
        if event is not None:
            events.append(event)
    return events


# ---------------------------------------------------------------------------
# Subprocess worker (used only when timeout enforcement is required)
# ---------------------------------------------------------------------------

def _subprocess_worker(args: tuple) -> None:
    """Subprocess entry point: inject sys.path then run the diff."""
    sys_path, base_bytes, compare_html, config_dict, queue = args
    # Restore sys.path so the htmldiff package is importable
    for p in reversed(sys_path):
        if p not in sys.path:
            sys.path.insert(0, p)

    from htmldiff.config import HtmlDiffConfig as _Cfg
    from htmldiff.differ import _diff_inprocess

    try:
        config = _Cfg(**config_dict)
        events = _diff_inprocess(base_bytes, compare_html, config)
        queue.put(events)
    except Exception as exc:
        queue.put(exc)


def diff_trees_safe(base_bytes: bytes, compare_html: str,
                    config: HtmlDiffConfig) -> list[EditEvent]:
    """Diff two HTML pages and return a list of EditEvent objects.

    When ``config.diff_timeout`` is set to a finite value the diff is run
    inside a subprocess so the timeout can be enforced via
    ``Process.join(timeout)``.  A very large timeout (>= 1e9 s) causes the
    diff to run in-process to avoid the subprocess overhead during testing.

    Returns an empty list on timeout (with a :class:`DiffTimeoutWarning`) or
    on any internal error.
    """
    # No timeout — run in-process directly.
    if config.diff_timeout >= 1e9:
        try:
            return _diff_inprocess(base_bytes, compare_html, config)
        except Exception:
            return []

    # Threading timeout: no subprocess spawn overhead.
    # The thread cannot be forcibly killed, so it may continue running in the
    # background after we return [].  Use daemon=True so it does not block
    # program exit.  Acceptable trade-off: at most one stale thread per worker.
    result_holder: list[list[EditEvent]] = [[]]
    completed = threading.Event()

    def _run() -> None:
        try:
            result_holder[0] = _diff_inprocess(base_bytes, compare_html, config)
        except Exception:
            result_holder[0] = []
        finally:
            completed.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    _interval = 30.0
    _elapsed = 0.0
    while not completed.wait(timeout=_interval):
        _elapsed += _interval
        if _elapsed >= config.diff_timeout:
            break
        print(f"[diff] ожидание xmldiff {_elapsed:.0f}s / {config.diff_timeout:.0f}s ...", flush=True)

    if not completed.is_set():
        warnings.warn(
            f"xmldiff timed out after {config.diff_timeout}s; returning empty diff.",
            DiffTimeoutWarning,
            stacklevel=2,
        )
        return []

    return result_holder[0]
