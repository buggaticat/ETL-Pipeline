from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

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


def _ensure_path(label: str, value: str) -> None:
    """Raise if a required local runtime path does not exist."""
    if not Path(value).exists():
        raise FileNotFoundError(f"{label} does not exist: {value}")


def build_spark() -> SparkSession:
    """Build a Spark session configured for local Windows S3A execution."""
    java_home = os.environ.get("JAVA_HOME", r"C:\Program Files\Java\jdk-25.0.2")
    hadoop_home = os.environ.get("HADOOP_HOME", r"C:\hadoop")
    spark_home = os.environ.get("SPARK_HOME", r"C:\spark")
    python_bin = os.environ.get("PYSPARK_PYTHON", sys.executable)

    os.environ["JAVA_HOME"] = java_home
    os.environ["HADOOP_HOME"] = hadoop_home
    os.environ["SPARK_HOME"] = spark_home
    os.environ["PYSPARK_PYTHON"] = python_bin
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_bin
    os.environ["PATH"] = rf"{hadoop_home}\bin;{spark_home}\bin;" + os.environ["PATH"]

    _ensure_path("JAVA_HOME", java_home)
    _ensure_path("HADOOP_HOME", hadoop_home)
    _ensure_path("SPARK_HOME", spark_home)

    hadoop_aws_package = os.environ.get(
        "HADOOP_AWS_PACKAGE",
        "org.apache.hadoop:hadoop-aws:3.3.4",
    )

    builder = (
        SparkSession.builder.appName("TLC-Transform")
        .config("spark.jars.packages", hadoop_aws_package)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.auth.ProfileAWSCredentialsProvider",
        )
        .config("spark.hadoop.fs.s3a.endpoint.region", "us-east-1")
        .config("spark.hadoop.fs.s3a.connection.acquisition.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000")
        .config("spark.hadoop.fs.s3a.connection.idle.time", "60000")
        .config("spark.hadoop.fs.s3a.connection.request.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.timeout", "200000")
        .config("spark.hadoop.fs.s3a.connection.ttl", "86400000")
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")
        .config("spark.hadoop.fs.s3a.threads.max", "96")
        .config("spark.hadoop.fs.s3a.executor.capacity", "16")
        .config("spark.hadoop.fs.s3a.max.total.tasks", "32")
        .config("spark.hadoop.fs.s3a.connection.maximum", "200")
        .config("spark.sql.files.ignoreMissingFiles", "false")
    )

    spark = builder.getOrCreate()
    hconf = spark.sparkContext._jsc.hadoopConfiguration()
    hconf.set("fs.s3a.connection.acquisition.timeout", "60000")
    hconf.set("fs.s3a.connection.establish.timeout", "30000")
    hconf.set("fs.s3a.connection.idle.time", "60000")
    hconf.set("fs.s3a.connection.request.timeout", "60000")
    hconf.set("fs.s3a.connection.timeout", "200000")
    hconf.set("fs.s3a.connection.ttl", "86400000")
    hconf.set("fs.s3a.threads.keepalivetime", "60")
    hconf.set("fs.s3a.threads.max", "96")
    hconf.set("fs.s3a.executor.capacity", "16")
    hconf.set("fs.s3a.max.total.tasks", "32")
    hconf.set("fs.s3a.connection.maximum", "200")
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
    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df = df.withColumn(column, F.lit(None))

    return (
        df.select(*EXPECTED_COLUMNS)
        .withColumn("ticker", F.upper(F.trim(F.col("ticker"))))
        .withColumn("timestamp_utc", F.to_timestamp("timestamp_utc"))
        .withColumn("date", F.to_date("date"))
        .withColumn("open", F.col("open").cast(T.DoubleType()))
        .withColumn("high", F.col("high").cast(T.DoubleType()))
        .withColumn("low", F.col("low").cast(T.DoubleType()))
        .withColumn("close", F.col("close").cast(T.DoubleType()))
        .withColumn("volume", F.col("volume").cast(T.LongType()))
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


def transform_market_data(spark: SparkSession, cfg: TransformConfig) -> DataFrame:
    """Read, clean, enrich, and return the full transformed market dataset."""
    if cfg.input_format == "parquet":
        source = spark.read.parquet(cfg.source_prefix)
    elif cfg.input_format == "csv":
        source = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .option("recursiveFileLookup", "true")
            .csv(cfg.source_prefix)
        )
    else:
        raise ValueError("TRANSFORM_INPUT_FORMAT must be parquet or csv.")

    cleaned = (
        _standardize_schema(source)
        .filter(F.col("ticker").isNotNull() & F.col("timestamp_utc").isNotNull())
    )
    cleaned = _add_quality_flags(cleaned, cfg)
    cleaned = _dedupe_rows(cleaned)
    cleaned = _add_gap_and_roll_features(cleaned, cfg)
    metadata = _read_metadata(spark, cfg)
    if metadata is not None:
        metadata = _normalize_metadata(metadata)
        cleaned = cleaned.join(metadata, on="ticker", how="left")

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
        *([c for c in ["company_name", "sector", "industry", "exchange"] if c in cleaned.columns]),
    )


def transform_all_symbols(spark: SparkSession, cfg: TransformConfig) -> DataFrame:
    """Compatibility wrapper that returns the unified transformed dataset."""
    return transform_market_data(spark, cfg)


def build_transformed_dataset(spark: SparkSession | None = None) -> DataFrame:
    """Build and return the fully transformed dataset for downstream loading.

    If a Spark session is not provided, this function creates one, runs the
    transform, and closes the session before returning the final DataFrame.
    """
    cfg = load_config()
    owns_spark = spark is None
    active_spark = spark or build_spark()

    try:
        return transform_all_symbols(active_spark, cfg)
    finally:
        if owns_spark:
            active_spark.stop()


def run_transform() -> DataFrame:
    """Convenience entrypoint for orchestration layers such as Airflow."""
    return build_transformed_dataset()


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
