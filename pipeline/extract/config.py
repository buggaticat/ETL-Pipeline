from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ExtractConfig:
    year: int = 2025
    polygon_base_url: str = "https://api.polygon.io"
    request_timeout: int = 30
    max_retries: int = 5
    request_interval_seconds: float = 12.5
    s3_prefix: str = "polygon/sp500/minute/2025/"
    s3_format: str = "parquet"
    checkpoint_key: str = "polygon/sp500/minute/2025/_checkpoint.json"

    @property
    def month_start(self) -> date:
        return date(self.year, 1, 1)


DEFAULT_CONFIG = ExtractConfig()
