import posixpath
import urllib.parse
from lxml import etree

from .config import HtmlDiffConfig

_BOOLEAN_ATTRS = frozenset({
    'checked', 'disabled', 'selected', 'readonly', 'multiple',
    'autofocus', 'required', 'defer', 'async',
})


def _strip_url_params(url: str) -> str:
    """Remove query string and fragment from a URL, normalise relative paths."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return url

    # If the URL has a scheme+netloc it is absolute — keep those, just drop query/fragment
    if parts.scheme or parts.netloc:
        cleaned = urllib.parse.urlunsplit((
            parts.scheme,
            parts.netloc,
            parts.path,
            '',
            '',
        ))
        return cleaned

    # Relative URL — normalise the path as well
    normalised_path = posixpath.normpath(parts.path) if parts.path else parts.path
    cleaned = urllib.parse.urlunsplit(('', '', normalised_path, '', ''))
    return cleaned


def _normalise_style(style: str) -> str:
    """Sort CSS declarations within a style attribute."""
    declarations = [d.strip() for d in style.split(';') if d.strip()]
    declarations.sort(key=lambda d: d.split(':', 1)[0].strip().lower())
    return ';'.join(declarations)


def normalize_attrs(tree: etree._Element, config: HtmlDiffConfig) -> None:
    """Normalise element attributes in-place across the whole tree."""
    for el in tree.iter():
        # Skip non-element nodes (comments, PIs, etc.)
        if callable(el.tag):
            continue

        attrib = el.attrib

        # 1. Remove explicitly ignored attributes
        for attr in config.ignored_attrs:
            if attr in attrib:
                del attrib[attr]

        # 2. Strip data-* attributes
        if config.strip_data_attrs:
            data_keys = [k for k in attrib if k.startswith('data-')]
            for k in data_keys:
                del attrib[k]

        # 3. Normalise class tokens (sort alphabetically)
        if config.normalize_class and 'class' in attrib:
            tokens = attrib['class'].split()
            attrib['class'] = ' '.join(sorted(tokens))

        # 4. Normalise style declarations (sort by property name)
        if config.normalize_style and 'style' in attrib:
            attrib['style'] = _normalise_style(attrib['style'])

        # 5. Strip query + fragment from URL-bearing attributes
        for attr in config.strip_url_params_attrs:
            if attr in attrib:
                attrib[attr] = _strip_url_params(attrib[attr])

        # 6. Normalise boolean attributes to empty string
        for attr in _BOOLEAN_ATTRS:
            if attr in attrib:
                attrib[attr] = ''

        # 7. Strip surrounding whitespace from aria-* attributes
        aria_keys = [k for k in attrib if k.startswith('aria-')]
        for k in aria_keys:
            attrib[k] = attrib[k].strip()
