from __future__ import annotations

import io
import os
import sys
import tempfile
from functools import reduce
from pathlib import Path
from pathlib import PurePosixPath
from shutil import which
from urllib.parse import urlparse

import pandas as pd
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from ..aws_auth import build_aws_session
from .config import TransformConfig, load_config


EXPECTED_COLUMNS = [
    "ticker",
    "timestamp_utc",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
]

METADATA_COLUMNS = ["ticker", "company_name", "sector", "industry", "exchange"]
RAW_COLUMNS = EXPECTED_COLUMNS + [c for c in METADATA_COLUMNS if c not in EXPECTED_COLUMNS]
FINAL_COLUMNS = EXPECTED_COLUMNS + [c for c in METADATA_COLUMNS if c != "ticker"]

S3A_SIMPLE_CREDENTIALS_PROVIDER = "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
S3A_TEMPORARY_CREDENTIALS_PROVIDER = "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider"
S3A_CLASSIC_INPUT_STREAM_FACTORY = (
    "org.apache.hadoop.fs.s3a.impl.streams.ClassicObjectInputStreamFactory"
)
PARQUET_SOURCE_SCHEMA = T.StructType(
    [
        T.StructField("ticker", T.StringType(), True),
        T.StructField("timestamp_utc", T.StringType(), True),
        T.StructField("date", T.StringType(), True),
        T.StructField("open", T.DoubleType(), True),
        T.StructField("high", T.DoubleType(), True),
        T.StructField("low", T.DoubleType(), True),
        T.StructField("close", T.DoubleType(), True),
        T.StructField("volume", T.DoubleType(), True),
        T.StructField("vwap", T.DoubleType(), True),
        T.StructField("transactions", T.DoubleType(), True),
        T.StructField("company_name", T.StringType(), True),
        T.StructField("sector", T.StringType(), True),
        T.StructField("industry", T.StringType(), True),
        T.StructField("exchange", T.StringType(), True),
    ]
)
TRANSFORM_SHUFFLE_PARTITIONS = int(os.environ.get("TRANSFORM_SHUFFLE_PARTITIONS", "128"))
TRANSFORM_DRIVER_MEMORY = os.environ.get("TRANSFORM_DRIVER_MEMORY", "8g")
TRANSFORM_EXECUTOR_MEMORY = os.environ.get("TRANSFORM_EXECUTOR_MEMORY", "8g")
TRANSFORM_ARROW_MAX_RECORDS_PER_BATCH = int(
    os.environ.get("TRANSFORM_ARROW_MAX_RECORDS_PER_BATCH", "1")
)


def _ensure_path(label: str, value: str) -> None:
    """Raise if a required local runtime path does not exist."""
    if not Path(value).exists():
        raise FileNotFoundError(f"{label} does not exist: {value}")


def build_spark() -> SparkSession:
    """Build a Spark session configured for local Windows S3A execution."""
    java_home = os.environ.get("JAVA_HOME")
    hadoop_home = os.environ.get("HADOOP_HOME")
    spark_home = os.environ.get("SPARK_HOME")
    python_bin = os.environ.get("PYSPARK_PYTHON", sys.executable)

    if java_home is None and os.name == "nt":
        java_home = r"C:\Program Files\Java\jdk-25.0.2"
    if java_home:
        os.environ["JAVA_HOME"] = java_home
    if hadoop_home:
        os.environ["HADOOP_HOME"] = hadoop_home
    if spark_home:
        os.environ["SPARK_HOME"] = spark_home
    os.environ["PYSPARK_PYTHON"] = python_bin
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_bin
    path_entries: list[str] = []
    if hadoop_home:
        path_entries.append(str(Path(hadoop_home) / "bin"))
    if spark_home:
        path_entries.append(str(Path(spark_home) / "bin"))
    if path_entries:
        os.environ["PATH"] = os.pathsep.join(path_entries + [os.environ["PATH"]])

    if java_home:
        _ensure_path("JAVA_HOME", java_home)
    if hadoop_home:
        _ensure_path("HADOOP_HOME", hadoop_home)
    if spark_home:
        _ensure_path("SPARK_HOME", spark_home)
    if which("java") is None and java_home is None:
        raise EnvironmentError(
            "Java is required for Spark but neither JAVA_HOME nor a java executable was found."
        )

    aws = build_aws_session(region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    local_tmp_dir = os.environ.get("SPARK_LOCAL_DIR") or tempfile.gettempdir()

    hadoop_aws_package = (
        "org.apache.hadoop:hadoop-aws:3.3.4,"
        "com.amazonaws:aws-java-sdk-bundle:1.12.262"
    )
    s3a_provider = (
        S3A_TEMPORARY_CREDENTIALS_PROVIDER if aws.token else S3A_SIMPLE_CREDENTIALS_PROVIDER
    )

    builder = (
        SparkSession.builder.appName("TLC-Transform")
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.jars.packages", hadoop_aws_package)
        .config("spark.driver.memory", TRANSFORM_DRIVER_MEMORY)
        .config("spark.executor.memory", TRANSFORM_EXECUTOR_MEMORY)
        .config("spark.local.dir", local_tmp_dir)
        .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={local_tmp_dir}")
        .config("spark.executor.extraJavaOptions", f"-Djava.io.tmpdir={local_tmp_dir}")
        .config(
            "spark.sql.execution.arrow.maxRecordsPerBatch",
            str(TRANSFORM_ARROW_MAX_RECORDS_PER_BATCH),
        )
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", s3a_provider)
        .config("spark.hadoop.fs.s3a.access.key", aws.access_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws.secret_key)
        .config("spark.hadoop.fs.s3a.endpoint", "s3.ap-southeast-1.amazonaws.com")
        .config("spark.hadoop.fs.s3a.endpoint.region", "ap-southeast-1")
        .config("spark.hadoop.fs.s3a.input.stream.factory.class", S3A_CLASSIC_INPUT_STREAM_FACTORY)
        .config("spark.hadoop.fs.s3a.connection.acquisition.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000")
        .config("spark.hadoop.fs.s3a.connection.idle.time", "60000")
        .config("spark.hadoop.fs.s3a.connection.request.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.timeout", "200000")
        .config("spark.hadoop.fs.s3a.connection.ttl", "86400000")
        .config("spark.hadoop.fs.s3a.buffer.dir", local_tmp_dir)
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400")
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")
        .config("spark.hadoop.fs.s3a.threads.max", "96")
        .config("spark.hadoop.fs.s3a.executor.capacity", "16")
        .config("spark.hadoop.fs.s3a.max.total.tasks", "32")
        .config("spark.hadoop.fs.s3a.connection.maximum", "200")
        .config("spark.sql.shuffle.partitions", str(TRANSFORM_SHUFFLE_PARTITIONS))
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
        .config("spark.sql.files.ignoreMissingFiles", "false")
    )

    spark = builder.getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", str(TRANSFORM_SHUFFLE_PARTITIONS))
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")
    spark.conf.set(
        "spark.sql.execution.arrow.maxRecordsPerBatch",
        str(TRANSFORM_ARROW_MAX_RECORDS_PER_BATCH),
    )
    hconf = spark.sparkContext._jsc.hadoopConfiguration()
    hconf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hconf.set("fs.s3a.aws.credentials.provider", s3a_provider)
    hconf.set("fs.s3a.access.key", aws.access_key)
    hconf.set("fs.s3a.secret.key", aws.secret_key)
    hconf.set("fs.s3a.endpoint", "s3.ap-southeast-1.amazonaws.com")
    hconf.set("fs.s3a.endpoint.region", "ap-southeast-1")
    hconf.set("fs.s3a.input.stream.factory.class", S3A_CLASSIC_INPUT_STREAM_FACTORY)
    hconf.set("fs.s3a.connection.acquisition.timeout", "60000")
    hconf.set("fs.s3a.connection.establish.timeout", "30000")
    hconf.set("fs.s3a.connection.idle.time", "60000")
    hconf.set("fs.s3a.connection.request.timeout", "60000")
    hconf.set("fs.s3a.connection.timeout", "200000")
    hconf.set("fs.s3a.connection.ttl", "86400000")
    hconf.set("hadoop.tmp.dir", local_tmp_dir)
    hconf.set("fs.s3a.buffer.dir", local_tmp_dir)
    hconf.set("fs.s3a.multipart.purge.age", "86400")
    hconf.set("fs.s3a.threads.keepalivetime", "60")
    hconf.set("fs.s3a.threads.max", "96")
    hconf.set("fs.s3a.executor.capacity", "16")
    hconf.set("fs.s3a.max.total.tasks", "32")
    hconf.set("fs.s3a.connection.maximum", "200")
    if aws.token:
        hconf.set("fs.s3a.session.token", aws.token)
    return spark


def _read_metadata(spark: SparkSession, cfg: TransformConfig) -> DataFrame | None:
    """Read optional ticker metadata from the configured path."""
    if not cfg.metadata_path:
        return None
    path = cfg.metadata_path.rstrip("/")
    if path.lower().endswith(".csv"):
        return spark.read.option("header", "true").option("inferSchema", "true").csv(path)
    return spark.read.parquet(path)


def _normalize_metadata(df: DataFrame) -> DataFrame:
    """Normalize metadata columns to a clean ticker-keyed dimension table."""
    if "ticker" not in df.columns:
        raise ValueError("Metadata must contain a ticker column.")
    selected = [c for c in METADATA_COLUMNS if c in df.columns]
    if "ticker" not in selected:
        selected = ["ticker"] + selected
    return (
        df.select(*selected)
        .withColumn("ticker", F.upper(F.trim(F.col("ticker"))))
        .dropDuplicates(["ticker"])
    )


def _standardize_schema(df: DataFrame) -> DataFrame:
    """Project raw source data into the canonical minute-bar schema."""
    for column in RAW_COLUMNS:
        if column not in df.columns:
            df = df.withColumn(column, F.lit(None))

    return (
        df.select(*RAW_COLUMNS)
        .withColumn("ticker", F.upper(F.trim(F.col("ticker"))))
        .withColumn("timestamp_utc", F.to_timestamp("timestamp_utc"))
        .withColumn("date", F.to_date("date"))
        .withColumn("open", F.col("open").cast(T.DoubleType()))
        .withColumn("high", F.col("high").cast(T.DoubleType()))
        .withColumn("low", F.col("low").cast(T.DoubleType()))
        .withColumn("close", F.col("close").cast(T.DoubleType()))
        .withColumn("volume", F.col("volume").cast(T.DoubleType()))
        .withColumn("vwap", F.col("vwap").cast(T.DoubleType()))
        .withColumn("transactions", F.col("transactions").cast(T.LongType()))
    )


def _add_quality_flags(df: DataFrame, cfg: TransformConfig) -> DataFrame:
    """Add market-session and basic validity flags."""
    market_ts = F.from_utc_timestamp(F.col("timestamp_utc"), cfg.market_timezone)
    minute_of_day = F.hour(market_ts) * F.lit(60) + F.minute(market_ts)
    regular_open = cfg.regular_open_hour * 60 + cfg.regular_open_minute
    regular_close = cfg.regular_close_hour * 60
    premarket_start = cfg.premarket_start_hour * 60 + cfg.premarket_start_minute
    afterhours_end = cfg.afterhours_end_hour * 60 + cfg.afterhours_end_minute

    invalid_ohlc = (
        F.col("open").isNull()
        | F.col("high").isNull()
        | F.col("low").isNull()
        | F.col("close").isNull()
        | F.col("volume").isNull()
        | (F.col("open") <= 0)
        | (F.col("high") <= 0)
        | (F.col("low") <= 0)
        | (F.col("close") <= 0)
        | (F.col("volume") < 0)
        | (F.col("high") < F.col("low"))
        | (F.col("high") < F.col("open"))
        | (F.col("high") < F.col("close"))
        | (F.col("low") > F.col("open"))
        | (F.col("low") > F.col("close"))
    )

    return (
        df.withColumn("market_timestamp", market_ts)
        .withColumn("market_date", F.to_date("market_timestamp"))
        .withColumn("minute_of_day", minute_of_day)
        .withColumn("is_regular_hours", (minute_of_day >= regular_open) & (minute_of_day < regular_close))
        .withColumn("is_premarket", (minute_of_day >= premarket_start) & (minute_of_day < regular_open))
        .withColumn("is_after_hours", (minute_of_day >= regular_close) & (minute_of_day < afterhours_end))
        .withColumn(
            "session_tag",
            F.when(F.col("is_regular_hours") & (minute_of_day < regular_open + 30), F.lit("open"))
            .when(F.col("is_regular_hours") & (minute_of_day >= regular_close - 30), F.lit("close"))
            .when(F.col("is_regular_hours"), F.lit("midday"))
            .when(F.col("is_premarket"), F.lit("premarket"))
            .when(F.col("is_after_hours"), F.lit("after_hours"))
            .otherwise(F.lit("off_hours")),
        )
        .withColumn("is_zero_volume", F.col("volume") == 0)
        .withColumn("is_invalid_ohlc", invalid_ohlc)
        .withColumn("has_null_key_fields", F.col("ticker").isNull() | F.col("timestamp_utc").isNull())
    )


def _dedupe_rows(df: DataFrame) -> DataFrame:
    """Keep one deterministic row per ticker and timestamp."""
    window = Window.partitionBy("ticker", "timestamp_utc").orderBy(
        F.col("has_null_key_fields").asc(),
        F.col("is_invalid_ohlc").asc(),
        F.col("volume").desc_nulls_last(),
        F.col("transactions").desc_nulls_last(),
        F.col("close").desc_nulls_last(),
    )
    group_window = Window.partitionBy("ticker", "timestamp_utc")
    return (
        df.withColumn("dedupe_rank", F.row_number().over(window))
        .withColumn("duplicate_group_size", F.count(F.lit(1)).over(group_window))
        .withColumn("is_duplicate", F.col("duplicate_group_size") > 1)
        .filter(F.col("dedupe_rank") == 1)
        .drop("dedupe_rank", "duplicate_group_size")
    )


def _add_gap_and_roll_features(df: DataFrame, cfg: TransformConfig) -> DataFrame:
    """Add gap markers and rolling price features for each ticker."""
    minute_window = Window.partitionBy("ticker").orderBy("timestamp_utc")
    prev_ts = F.lag("timestamp_utc").over(minute_window)
    next_ts = F.lead("timestamp_utc").over(minute_window)
    prev_close = F.lag("close").over(minute_window)
    gap_minutes = (
        (F.unix_timestamp("timestamp_utc") - F.unix_timestamp(prev_ts)) / F.lit(60)
    )
    next_gap_minutes = (
        (F.unix_timestamp(next_ts) - F.unix_timestamp("timestamp_utc")) / F.lit(60)
    )
    close_window = Window.partitionBy("ticker").orderBy("timestamp_utc").rowsBetween(-cfg.rolling_close_window + 1, 0)
    price_window = Window.partitionBy("ticker").orderBy("timestamp_utc").rowsBetween(-cfg.spike_rolling_minutes + 1, 0)

    return (
        df.withColumn("prev_timestamp_utc", prev_ts)
        .withColumn("next_timestamp_utc", next_ts)
        .withColumn("gap_minutes_from_prev", gap_minutes)
        .withColumn("gap_minutes_to_next", next_gap_minutes)
        .withColumn("is_gap", F.coalesce(F.col("gap_minutes_from_prev") > 1, F.lit(False)))
        .withColumn(
            "is_gap_boundary",
            F.coalesce(F.col("is_gap") | (F.col("gap_minutes_to_next") > 1), F.lit(False)),
        )
        .withColumn("typical_price", (F.col("high") + F.col("low") + F.col("close")) / F.lit(3.0))
        .withColumn("price_range", F.col("high") - F.col("low"))
        .withColumn("rolling_close_avg", F.avg("close").over(close_window))
        .withColumn("rolling_close_median", F.percentile_approx("close", 0.5, 100).over(price_window))
        .withColumn(
            "return_1m",
            F.when(prev_close.isNull() | (prev_close <= 0) | F.col("close").isNull() | (F.col("close") <= 0), F.lit(None))
            .otherwise((F.col("close") / prev_close) - F.lit(1.0)),
        )
        .withColumn(
            "log_return_1m",
            F.when(
                prev_close.isNull() | (prev_close <= 0) | F.col("close").isNull() | (F.col("close") <= 0),
                F.lit(None),
            ).otherwise(F.log(F.col("close") / prev_close)),
        )
        .withColumn(
            "is_suspect_spike",
            F.when(F.col("rolling_close_median").isNull(), F.lit(False))
            .otherwise(
                (F.col("close") > F.col("rolling_close_median") * F.lit(cfg.spike_threshold))
                | (F.col("close") < F.col("rolling_close_median") / F.lit(cfg.spike_threshold))
            ),
        )
    )


def _parse_s3a_uri(uri: str) -> tuple[str, str]:
    """Return the S3 bucket and key prefix from an s3a:// URI."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3a" or not parsed.netloc:
        raise ValueError(f"Expected an s3a URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _source_ticker_from_key(key: str, source_prefix: str) -> str:
    """Extract the ticker folder name from a raw S3 object key."""
    prefix_parts = PurePosixPath(source_prefix.rstrip("/")).parts
    key_parts = PurePosixPath(key).parts
    if len(key_parts) <= len(prefix_parts):
        raise ValueError(f"Cannot infer ticker from key: {key}")
    return key_parts[len(prefix_parts)]


def _normalize_raw_parquet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize one raw parquet file into a Spark-friendly row frame."""
    normalized = frame.copy()
    for column in RAW_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    normalized = normalized[RAW_COLUMNS]
    normalized["ticker"] = normalized["ticker"].astype("string").str.strip().str.upper()
    normalized["timestamp_utc"] = normalized["timestamp_utc"].astype("string")
    normalized["date"] = normalized["date"].astype("string")

    for column in ("open", "high", "low", "close", "volume", "vwap", "transactions"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")

    for column in ("company_name", "sector", "industry", "exchange"):
        normalized[column] = normalized[column].astype("string")

    return normalized


def _read_parquet_source_files(spark: SparkSession, source_files: list[str]) -> DataFrame:
    """Read parquet sources with PyArrow so mixed physical types do not crash Spark."""
    binary_files = spark.read.format("binaryFile").load(source_files).select("path", "content")

    def _decode_batches(iterator):
        for batch in iterator:
            for path, content in zip(batch["path"], batch["content"]):
                try:
                    table = pq.read_table(io.BytesIO(content))
                    frame = table.to_pandas()
                except Exception as exc:  # noqa: BLE001 - surface file-specific failures
                    raise RuntimeError(f"Unable to decode parquet file {path}") from exc
                yield _normalize_raw_parquet_frame(frame)

    return binary_files.mapInPandas(_decode_batches, schema=PARQUET_SOURCE_SCHEMA)


def _list_source_files(cfg: TransformConfig) -> list[str]:
    """Return concrete source file URIs under the configured prefix."""
    bucket, prefix = _parse_s3a_uri(cfg.source_prefix)
    suffix = ".parquet" if cfg.input_format == "parquet" else ".csv"
    aws = build_aws_session(region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    paginator = aws.session.client("s3").get_paginator("list_objects_v2")

    files: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/") or not key.lower().endswith(suffix):
                continue
            files.append(f"s3a://{bucket}/{key}")
    return files


def _group_source_files_by_ticker(cfg: TransformConfig, source_files: list[str]) -> dict[str, list[str]]:
    """Group source files by ticker folder so each ticker can be transformed independently."""
    grouped: dict[str, list[str]] = {}
    source_prefix = _parse_s3a_uri(cfg.source_prefix)[1]
    for uri in source_files:
        key = urlparse(uri).path.lstrip("/")
        ticker = _source_ticker_from_key(key, source_prefix)
        grouped.setdefault(ticker, []).append(uri)

    for ticker in grouped:
        grouped[ticker].sort()
    return grouped


def _read_source_file(spark: SparkSession, path: str, cfg: TransformConfig) -> DataFrame:
    """Read one raw source object without asking Spark to merge schemas."""
    if cfg.input_format == "parquet":
        return _read_parquet_source_files(spark, [path])
    if cfg.input_format == "csv":
        return spark.read.option("header", "true").option("inferSchema", "true").csv(path)
    raise ValueError("TRANSFORM_INPUT_FORMAT must be parquet or csv.")


def _read_source_data(spark: SparkSession, cfg: TransformConfig) -> DataFrame:
    """Read raw market data file-by-file so mixed file schemas do not collide."""
    source_files = _list_source_files(cfg)
    if not source_files:
        raise FileNotFoundError(f"No source files found under {cfg.source_prefix}")

    if cfg.input_format == "parquet":
        return _read_parquet_source_files(spark, source_files)

    frames = [_standardize_schema(_read_source_file(spark, path, cfg)) for path in source_files]
    return reduce(lambda left, right: left.unionByName(right), frames)


def _build_ticker_dataset(spark: SparkSession, cfg: TransformConfig, source_files: list[str]) -> DataFrame:
    """Build the transformed dataset for a single ticker folder."""
    if cfg.input_format == "parquet":
        source = _read_parquet_source_files(spark, source_files)
    else:
        frames = [_standardize_schema(_read_source_file(spark, path, cfg)) for path in source_files]
        source = reduce(lambda left, right: left.unionByName(right), frames)

    cleaned = (
        _standardize_schema(source)
        .filter(F.col("ticker").isNotNull() & F.col("timestamp_utc").isNotNull())
    )
    cleaned = _add_quality_flags(cleaned, cfg)
    cleaned = _dedupe_rows(cleaned)
    cleaned = _add_gap_and_roll_features(cleaned, cfg)
    return cleaned


def transform_market_data(spark: SparkSession, cfg: TransformConfig) -> DataFrame:
    """Read, clean, enrich, and return the full transformed market dataset."""
    print(f"Listing source files under {cfg.source_prefix}...", flush=True)
    source_files = _list_source_files(cfg)
    if not source_files:
        raise FileNotFoundError(f"No source files found under {cfg.source_prefix}")
    print(f"Found {len(source_files):,} source file(s).", flush=True)

    if cfg.input_format == "parquet":
        print("Transforming parquet sources...", flush=True)
        source = _read_source_data(spark, cfg)
        cleaned = (
            _standardize_schema(source)
            .filter(F.col("ticker").isNotNull() & F.col("timestamp_utc").isNotNull())
        )
        cleaned = _add_quality_flags(cleaned, cfg)
        cleaned = _dedupe_rows(cleaned)
        cleaned = _add_gap_and_roll_features(cleaned, cfg)
    else:
        print("Transforming CSV sources...", flush=True)
        source = _read_source_data(spark, cfg)
        cleaned = (
            _standardize_schema(source)
            .filter(F.col("ticker").isNotNull() & F.col("timestamp_utc").isNotNull())
        )
        cleaned = _add_quality_flags(cleaned, cfg)
        cleaned = _dedupe_rows(cleaned)
        cleaned = _add_gap_and_roll_features(cleaned, cfg)

    metadata = _read_metadata(spark, cfg)
    if metadata is not None:
        print("Joining metadata...", flush=True)
        metadata = _normalize_metadata(metadata)
        metadata_columns = [c for c in ["company_name", "sector", "industry", "exchange"] if c in metadata.columns]
        if metadata_columns:
            metadata = metadata.select(
                "ticker",
                *[F.col(c).alias(f"_metadata_{c}") for c in metadata_columns],
            )
            cleaned = cleaned.join(metadata, on="ticker", how="left")
            for column in metadata_columns:
                cleaned = cleaned.withColumn(
                    column,
                    F.coalesce(F.col(column), F.col(f"_metadata_{column}")),
                ).drop(f"_metadata_{column}")

    return cleaned.select(
        "ticker",
        "timestamp_utc",
        "date",
        "market_timestamp",
        "market_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
        "is_regular_hours",
        "is_premarket",
        "is_after_hours",
        "session_tag",
        "is_zero_volume",
        "is_invalid_ohlc",
        "is_duplicate",
        "is_suspect_spike",
        "is_gap",
        "is_gap_boundary",
        "gap_minutes_from_prev",
        "gap_minutes_to_next",
        "typical_price",
        "price_range",
        "rolling_close_avg",
        "rolling_close_median",
        "return_1m",
        "log_return_1m",
        "company_name",
        "sector",
        "industry",
        "exchange",
    )


def transform_all_symbols(spark: SparkSession, cfg: TransformConfig) -> DataFrame:
    """Compatibility wrapper that returns the unified transformed dataset."""
    return transform_market_data(spark, cfg)


def build_transformed_dataset(spark: SparkSession) -> DataFrame:
    """Build and return the fully transformed dataset for downstream loading.

    The caller is responsible for creating and stopping the Spark session.
    This ensures the session remains alive when the DataFrame is materialized.
    """
    cfg = load_config()
    return transform_all_symbols(spark, cfg)


def run_transform() -> DataFrame:
    """Convenience entrypoint for orchestration layers such as Airflow."""
    spark = build_spark()
    return build_transformed_dataset(spark)
    # NOTE: caller must call spark.stop() after materializing the DataFrame


def main() -> None:
    """Run the transform job and print a small sample for verification."""
    spark = build_spark()
    try:
        transformed = build_transformed_dataset(spark)
        print("Transformed schema:")
        transformed.printSchema()
        print("Sample rows:")
        transformed.show(10, truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
