from dataclasses import dataclass, field


@dataclass
class HtmlDiffConfig:
    ignored_attrs: set[str] = field(default_factory=set)
    strip_url_params_attrs: set[str] = field(default_factory=lambda: {'href', 'src', 'action'})
    strip_data_attrs: bool = True
    normalize_class: bool = True
    normalize_style: bool = True
    normalize_whitespace_text: bool = True
    max_elements: int = 10000
    attribute_inserts_to_ancestor: bool = True
    diff_ratio_mode: str = 'fast'   # 'fast' | 'accurate'
    uniqueattrs: list[str] = field(default_factory=lambda: ['id'])
    n_workers: int = -1             # -1 = cpu_count() - 1
    cache_dir: str | None = None
    diff_timeout: float = 30.0      # seconds before a single diff is abandoned

    def __post_init__(self):
        # Allow lists to be passed (e.g. when reconstructing from dict in worker)
        if isinstance(self.ignored_attrs, list):
            self.ignored_attrs = set(self.ignored_attrs)
        if isinstance(self.strip_url_params_attrs, list):
            self.strip_url_params_attrs = set(self.strip_url_params_attrs)
