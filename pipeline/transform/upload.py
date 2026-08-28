from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from .config import TransformConfig, load_config
from .transform import (
    _build_ticker_dataset,
    _group_source_files_by_ticker,
    _list_source_files,
    build_spark,
    build_transformed_dataset,
)


def _output_mode(cfg: TransformConfig) -> str:
    """Map the configured output format to a Spark writer mode."""
    fmt = cfg.output_format.strip().lower()
    if fmt not in {"parquet", "csv"}:
        raise ValueError("TRANSFORM_OUTPUT_FORMAT must be either 'parquet' or 'csv'.")
    return fmt


def upload_transformed_data(
    spark: SparkSession,
    cfg: TransformConfig,
    transformed: DataFrame | None = None,
    output_path: str | None = None,
) -> str:
    """Write one transformed dataset back to S3 and return the destination path."""
    output_format = _output_mode(cfg)
    output_path = (output_path or cfg.output_prefix).rstrip("/")
    frame = transformed if transformed is not None else build_transformed_dataset(spark)

    print(f"Writing transformed data to {output_path} as {output_format}...", flush=True)
    writer = frame.write.mode("overwrite")
    if output_format == "parquet":
        writer.parquet(output_path)
    else:
        writer.option("header", "true").csv(output_path)
    return output_path


def _write_ticker_outputs(spark: SparkSession, cfg: TransformConfig) -> list[str]:
    """Transform and upload each ticker independently."""
    source_files = _list_source_files(cfg)
    if not source_files:
        raise FileNotFoundError(f"No source files found under {cfg.source_prefix}")

    grouped = _group_source_files_by_ticker(cfg, source_files)
    if not grouped:
        raise FileNotFoundError(f"No ticker groups found under {cfg.source_prefix}")

    output_prefix = cfg.output_prefix.rstrip("/")
    destinations: list[str] = []
    total = len(grouped)

    for index, ticker in enumerate(sorted(grouped), start=1):
        print(f"Transforming ticker {ticker} ({index}/{total})...", flush=True)
        transformed = _build_ticker_dataset(spark, cfg, grouped[ticker])
        destination = f"{output_prefix}/{ticker}"
        upload_transformed_data(spark, cfg, transformed, destination)
        destinations.append(destination)
        print(f"Uploaded ticker {ticker} to {destination}", flush=True)

    return destinations


def run_upload(spark: SparkSession | None = None) -> str:
    """Convenience entrypoint for orchestrating the upload stage."""
    cfg = load_config()
    print(f"Starting transform upload for source {cfg.source_prefix}", flush=True)
    owns_spark = spark is None
    active_spark = spark or build_spark()

    try:
        destinations = _write_ticker_outputs(active_spark, cfg)
        destination_root = cfg.output_prefix.rstrip("/")
        print(
            f"Uploaded {len(destinations):,} ticker dataset(s) under {destination_root}",
            flush=True,
        )
        return destination_root
    finally:
        if owns_spark:
            active_spark.stop()


def main() -> None:
    """Run the upload job end to end."""
    run_upload()


if __name__ == "__main__":
    main()
