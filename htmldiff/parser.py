import warnings
import lxml.html
from lxml import etree

from .config import HtmlDiffConfig
from .exceptions import EncodingWarning, PageTooLargeWarning
from .normalizer import normalize_attrs

_NOISE_TAGS = frozenset({
    'script', 'style', 'noscript', 'canvas', 'iframe', 'template',
    'i',        # icon elements (<i class="fa fa-*">) — no text, pure UI chrome
    'button',   # interactive widgets — volatile state, no structural signal
    'select', 'option', 'optgroup',  # form dropdowns — not content
})

# lxml.html does not namespace-prefix SVG/MathML, but keep the constants
# in case a future caller passes a tree built with a namespace-aware parser.
_SVG_NS = '{http://www.w3.org/2000/svg}'
_MATHML_NS = '{http://www.w3.org/1998/Math/MathML}'

# SVG / MathML container tags that lxml.html keeps in the tree (no namespace
# prefix in lxml.html output, so we filter by lower-cased tag name).
_FOREIGN_ROOT_TAGS = frozenset({'svg', 'math'})


def _decode_bytes(raw: bytes, config: HtmlDiffConfig) -> str:
    """Detect encoding and decode raw bytes to a Unicode string."""
    from charset_normalizer import from_bytes

    results = from_bytes(raw)
    best = results.best()

    if best is None or best.encoding is None:
        warnings.warn(
            "charset_normalizer could not detect encoding; decoding as UTF-8 with replacement.",
            EncodingWarning,
            stacklevel=3,
        )
        return raw.decode('utf-8', errors='replace')

    confidence = best.chaos  # chaos is the noise indicator; low = good
    # charset_normalizer exposes confidence via the result object
    # Use the first result from the matches list to get a proper confidence score
    matches = list(results)
    if matches:
        # `encoding_aliases` not guaranteed; use raw chaos score as proxy for confidence
        # chaos==0.0 is perfect, chaos==1.0 is garbage — treat chaos > 0.3 as low confidence
        if best.chaos > 0.3:
            warnings.warn(
                f"Low-confidence charset detection (chaos={best.chaos:.2f}); "
                "decoding with replacement characters.",
                EncodingWarning,
                stacklevel=3,
            )
            return raw.decode(best.encoding or 'utf-8', errors='replace')

    return str(best)


def parse_html(raw: str | bytes, config: HtmlDiffConfig) -> etree._Element:
    """Parse an HTML document and return the cleaned ``<html>`` root element.

    Uses ``lxml.html.fromstring`` — the libxml2 C parser. XPaths from
    ``tree.getpath()`` on this tree match BeautifulSoup(html5lib) XPaths on the
    same input to 99.95 % (see ``parser_swap_verification.md``).
    """

    # --- 1. Encoding detection / normalisation ---
    if isinstance(raw, bytes):
        text = _decode_bytes(raw, config)
    else:
        text = raw

    # --- 2. Parse with lxml.html (libxml2) ---
    # fromstring returns the document root; if the input has no <html> wrapper
    # we wrap it ourselves so XPaths always start with /html/...
    root = lxml.html.fromstring(text)
    # If the returned element isn't <html>, wrap so XPaths are stable.
    if not (isinstance(root.tag, str) and root.tag.lower() == 'html'):
        html_root = etree.Element('html')
        html_root.append(root)
        root = html_root

    # --- 3. Strip noise elements ---
    to_remove = []
    for el in root.iter():
        if callable(el.tag):
            to_remove.append(el)
            continue
        if not isinstance(el.tag, str):
            continue
        tag = el.tag
        # lxml.html does not namespace-prefix SVG/MathML, but if a caller ever
        # passes a namespace-aware tree we still strip those.
        if tag.startswith(_SVG_NS) or tag.startswith(_MATHML_NS):
            to_remove.append(el)
            continue
        if tag in _NOISE_TAGS:
            to_remove.append(el)
            continue
        # Drop SVG/MathML subtrees by walking down from their roots.
        if tag in _FOREIGN_ROOT_TAGS:
            to_remove.append(el)

    for el in to_remove:
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)

    # --- 4. Normalise whitespace-only text / tail nodes ---
    if config.normalize_whitespace_text:
        for el in root.iter():
            if callable(el.tag):
                continue
            if el.text and not el.text.strip():
                el.text = None
            if el.tail and not el.tail.strip():
                el.tail = None

    # --- 5. Normalise attributes ---
    normalize_attrs(root, config)

    # --- 6. Truncate if too many elements ---
    all_elements = list(root.iter())
    if len(all_elements) > config.max_elements:
        warnings.warn(
            f"Page has {len(all_elements)} elements, exceeding max_elements="
            f"{config.max_elements}. Truncating.",
            PageTooLargeWarning,
            stacklevel=2,
        )
        excess = all_elements[config.max_elements:]
        for el in excess:
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)

    return root
