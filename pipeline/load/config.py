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


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _required_env(name: str) -> str:
    value = _env(name)
    if value is None:
        raise ValueError(f"{name} is required.")
    return value


@dataclass(frozen=True)
class LoadConfig:
    """Configuration values for loading transformed S3 data into Snowflake."""

    s3_bucket: str = "<bucket>"
    s3_prefix: str = "polygon/sp500/minute_transformed/2025/"
    source_format: str = "parquet"
    snowflake_database: str = "MARKET_DATA"
    snowflake_schema: str = "PUBLIC"
    snowflake_table: str = "TRANSFORMED_MINUTE_BARS"
    snowflake_warehouse: str | None = None
    snowflake_role: str | None = None
    chunk_size: int = 100_000
    truncate_table: bool = True
    s3_region: str | None = "ap-southeast-1"


def load_config() -> LoadConfig:
    """Load Snowflake load configuration from environment variables."""
    year = _env_int("DATA_YEAR", 2025)
    return LoadConfig(
        s3_bucket=_required_env("S3_BUCKET"),
        s3_prefix=_env(
            "LOAD_SOURCE_PREFIX",
            f"polygon/sp500/minute_transformed/{year}/",
        )
        or f"polygon/sp500/minute_transformed/{year}/",
        source_format=_env("LOAD_SOURCE_FORMAT", "parquet") or "parquet",
        snowflake_database=_env("SNOWFLAKE_DATABASE", "MARKET_DATA") or "MARKET_DATA",
        snowflake_schema=_env("SNOWFLAKE_SCHEMA", "PUBLIC") or "PUBLIC",
        snowflake_table=_env("SNOWFLAKE_TABLE", "TRANSFORMED_MINUTE_BARS") or "TRANSFORMED_MINUTE_BARS",
        snowflake_warehouse=_env("SNOWFLAKE_WAREHOUSE"),
        snowflake_role=_env("SNOWFLAKE_ROLE"),
        chunk_size=_env_int("LOAD_CHUNK_SIZE", 100_000),
        truncate_table=_env_bool("LOAD_TRUNCATE_TABLE", True),
        s3_region=_env("AWS_REGION", "ap-southeast-1"),
    )
