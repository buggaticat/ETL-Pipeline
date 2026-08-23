from __future__ import annotations

from dataclasses import dataclass
import os


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return default if value is None else float(value)


@dataclass(frozen=True)
class TransformConfig:
    """Configuration values for the Spark transform job."""

    source_prefix: str = "s3a://nyc-tlc/polygon/sp500/minute/2025/"
    output_prefix: str = "s3a://nyc-tlc/polygon/sp500/minute_transformed/2025/"
    metadata_path: str | None = None
    year: int = 2025
    input_format: str = "parquet"
    output_format: str = "parquet"
    market_timezone: str = "America/New_York"
    regular_open_hour: int = 9
    regular_open_minute: int = 30
    regular_close_hour: int = 16
    premarket_start_hour: int = 4
    premarket_start_minute: int = 0
    afterhours_end_hour: int = 20
    afterhours_end_minute: int = 0
    spike_rolling_minutes: int = 21
    spike_threshold: float = 10.0
    rolling_close_window: int = 20


def load_config() -> TransformConfig:
    """Load transform configuration from environment variables."""
    year = _env_int("DATA_YEAR", 2025)
    return TransformConfig(
        source_prefix=_env(
            "TRANSFORM_SOURCE_PREFIX",
            f"s3a://nyc-tlc/polygon/sp500/minute/{year}/",
        ),
        output_prefix=_env(
            "TRANSFORM_OUTPUT_PREFIX",
            f"s3a://nyc-tlc/polygon/sp500/minute_transformed/{year}/",
        ),
        metadata_path=_env("TRANSFORM_METADATA_PATH"),
        year=year,
        input_format=_env("TRANSFORM_INPUT_FORMAT", "parquet") or "parquet",
        output_format=_env("TRANSFORM_OUTPUT_FORMAT", "parquet") or "parquet",
        market_timezone=_env("MARKET_TIMEZONE", "America/New_York") or "America/New_York",
        regular_open_hour=_env_int("REGULAR_OPEN_HOUR", 9),
        regular_open_minute=_env_int("REGULAR_OPEN_MINUTE", 30),
        regular_close_hour=_env_int("REGULAR_CLOSE_HOUR", 16),
        premarket_start_hour=_env_int("PREMARKET_START_HOUR", 4),
        premarket_start_minute=_env_int("PREMARKET_START_MINUTE", 0),
        afterhours_end_hour=_env_int("AFTERHOURS_END_HOUR", 20),
        afterhours_end_minute=_env_int("AFTERHOURS_END_MINUTE", 0),
        spike_rolling_minutes=_env_int("SPIKE_ROLLING_MINUTES", 21),
        spike_threshold=_env_float("SPIKE_THRESHOLD", 10.0),
        rolling_close_window=_env_int("ROLLING_CLOSE_WINDOW", 20),
    )
