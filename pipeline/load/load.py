from __future__ import annotations

import os
from pathlib import Path

import boto3
import snowflake.connector

from .config import LoadConfig, load_config


SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD")
SNOWFLAKE_PRIVATE_KEY_PATH = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
SNOWFLAKE_PRIVATE_KEY_PASSPHRASE = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
SNOWFLAKE_HOST = os.environ.get("SNOWFLAKE_HOST")


def _require_env(value: str | None, name: str) -> str:
    """Raise a clear error when a required environment variable is missing."""
    if not value:
        raise ValueError(f"{name} is required.")
    return value


def build_snowflake_connection(cfg: LoadConfig):
    """Create a Snowflake connection from environment-based secrets."""
    kwargs: dict[str, object] = {
        "account": _require_env(SNOWFLAKE_ACCOUNT, "SNOWFLAKE_ACCOUNT"),
        "user": _require_env(SNOWFLAKE_USER, "SNOWFLAKE_USER"),
    }

    if SNOWFLAKE_PASSWORD:
        kwargs["password"] = SNOWFLAKE_PASSWORD
    elif SNOWFLAKE_PRIVATE_KEY_PATH:
        key_path = Path(SNOWFLAKE_PRIVATE_KEY_PATH)
        if not key_path.exists():
            raise FileNotFoundError(f"SNOWFLAKE_PRIVATE_KEY_PATH does not exist: {key_path}")
        with key_path.open("rb") as handle:
            kwargs["private_key"] = handle.read()
        if SNOWFLAKE_PRIVATE_KEY_PASSPHRASE:
            kwargs["private_key_passphrase"] = SNOWFLAKE_PRIVATE_KEY_PASSPHRASE
    else:
        raise ValueError(
            "One of SNOWFLAKE_PASSWORD or SNOWFLAKE_PRIVATE_KEY_PATH is required."
        )

    if cfg.snowflake_warehouse:
        kwargs["warehouse"] = cfg.snowflake_warehouse
    if cfg.snowflake_role:
        kwargs["role"] = cfg.snowflake_role
    if SNOWFLAKE_HOST:
        kwargs["host"] = SNOWFLAKE_HOST

    return snowflake.connector.connect(**kwargs)


def _list_s3_keys(s3_client, bucket: str, prefix: str, source_format: str) -> list[str]:
    """Return all non-empty object keys under the configured S3 prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: list[str] = []
    suffix = f".{source_format}"
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            if not key.lower().endswith(suffix):
                continue
            keys.append(key)
    return keys


def _aws_credentials(s3_session: boto3.Session) -> tuple[str, str, str | None]:
    """Return frozen AWS credentials for Snowflake external stage access."""
    credentials = s3_session.get_credentials()
    if credentials is None:
        raise ValueError("Unable to resolve AWS credentials for S3 access.")
    frozen = credentials.get_frozen_credentials()
    if not frozen.access_key or not frozen.secret_key:
        raise ValueError("AWS access key and secret key are required for Snowflake COPY INTO.")
    return frozen.access_key, frozen.secret_key, frozen.token


def _file_format_sql(source_format: str) -> str:
    """Return the Snowflake file format clause for the configured source format."""
    if source_format == "parquet":
        return "TYPE = PARQUET"
    if source_format == "csv":
        return "TYPE = CSV SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '\"'"
    raise ValueError("LOAD_SOURCE_FORMAT must be either 'parquet' or 'csv'.")


def _copy_into_sql(
    stage_name: str,
    fully_qualified_table: str,
    source_format: str,
    pattern: str,
    truncate_table: bool,
) -> str:
    """Build the COPY INTO statement for loading staged S3 data."""
    file_format = _file_format_sql(source_format)
    copy_clause = [
        f"COPY INTO {fully_qualified_table}",
        f"FROM @{stage_name}",
        f"FILE_FORMAT = ({file_format})",
        f"PATTERN = '{pattern}'",
    ]
    if source_format == "parquet":
        copy_clause.append("MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE")
    if truncate_table:
        copy_clause.append("ON_ERROR = ABORT_STATEMENT")
    else:
        copy_clause.append("ON_ERROR = CONTINUE")
    return "\n".join(copy_clause)


def load_transformed_data(cfg: LoadConfig | None = None) -> int:
    """Load transformed data from S3 into Snowflake and return the row count."""
    active_cfg = cfg or load_config()
    source_format = active_cfg.source_format.strip().lower()
    s3_session = boto3.Session(region_name=active_cfg.s3_region)
    s3_client = s3_session.client("s3")
    object_keys = _list_s3_keys(s3_client, active_cfg.s3_bucket, active_cfg.s3_prefix, source_format)
    if not object_keys:
        raise FileNotFoundError(
            f"No objects found under s3://{active_cfg.s3_bucket}/{active_cfg.s3_prefix}"
        )

    connection = build_snowflake_connection(active_cfg)
    try:
        fully_qualified_table = (
            f"{active_cfg.snowflake_database}.{active_cfg.snowflake_schema}.{active_cfg.snowflake_table}"
        )
        access_key, secret_key, token = _aws_credentials(s3_session)
        stage_name = "LOAD_S3_STAGE"
        stage_url = f"s3://{active_cfg.s3_bucket}/{active_cfg.s3_prefix.rstrip('/')}/"
        pattern = rf".*\.({source_format})$"

        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {active_cfg.snowflake_database}")
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {active_cfg.snowflake_database}.{active_cfg.snowflake_schema}")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {fully_qualified_table} (
                    ticker STRING,
                    timestamp_utc TIMESTAMP_NTZ,
                    date DATE,
                    market_timestamp TIMESTAMP_NTZ,
                    market_date DATE,
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume NUMBER,
                    vwap FLOAT,
                    transactions NUMBER,
                    is_regular_hours BOOLEAN,
                    is_premarket BOOLEAN,
                    is_after_hours BOOLEAN,
                    session_tag STRING,
                    is_zero_volume BOOLEAN,
                    is_invalid_ohlc BOOLEAN,
                    is_duplicate BOOLEAN,
                    is_suspect_spike BOOLEAN,
                    is_gap BOOLEAN,
                    is_gap_boundary BOOLEAN,
                    gap_minutes_from_prev FLOAT,
                    gap_minutes_to_next FLOAT,
                    typical_price FLOAT,
                    price_range FLOAT,
                    rolling_close_avg FLOAT,
                    rolling_close_median FLOAT,
                    return_1m FLOAT,
                    log_return_1m FLOAT,
                    company_name STRING,
                    sector STRING,
                    industry STRING,
                    exchange STRING
                )
                """
            )
            stage_sql = [
                f"CREATE OR REPLACE TEMPORARY STAGE {stage_name}",
                f"URL = '{stage_url}'",
                "CREDENTIALS = (",
                f"  AWS_KEY_ID = '{access_key}'",
                f"  AWS_SECRET_KEY = '{secret_key}'",
            ]
            if token:
                stage_sql.append(f"  AWS_TOKEN = '{token}'")
            stage_sql.extend(
                [
                    ")",
                    f"FILE_FORMAT = ({_file_format_sql(source_format)})",
                ]
            )
            cursor.execute("\n".join(stage_sql))
            if active_cfg.truncate_table:
                cursor.execute(f"TRUNCATE TABLE {fully_qualified_table}")

        copy_sql = _copy_into_sql(
            stage_name=stage_name,
            fully_qualified_table=fully_qualified_table,
            source_format=source_format,
            pattern=pattern,
            truncate_table=active_cfg.truncate_table,
        )
        with connection.cursor() as cursor:
            cursor.execute(copy_sql)
            results = cursor.fetchall()
        rows_loaded = sum(int(row[3]) for row in results if len(row) > 3 and row[3] is not None)
        print(
            f"Loaded {rows_loaded:,} rows from {len(object_keys)} S3 objects into "
            f"{active_cfg.snowflake_database}.{active_cfg.snowflake_schema}.{active_cfg.snowflake_table}"
        )
        return rows_loaded
    finally:
        connection.close()


def run_load(cfg: LoadConfig | None = None) -> int:
    """Convenience entrypoint for orchestration layers such as Airflow."""
    return load_transformed_data(cfg)


def main() -> None:
    """Run the load job end to end."""
    run_load()


if __name__ == "__main__":
    main()
