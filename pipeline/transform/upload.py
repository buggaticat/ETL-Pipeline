from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from .config import TransformConfig, load_config
from .transform import build_spark, build_transformed_dataset


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
) -> str:
    """Write the transformed dataset back to S3 and return the destination path."""
    output_format = _output_mode(cfg)
    output_path = cfg.output_prefix.rstrip("/")
    frame = transformed if transformed is not None else build_transformed_dataset(spark)

    writer = frame.write.mode("overwrite")
    if output_format == "parquet":
        writer.parquet(output_path)
    else:
        writer.option("header", "true").csv(output_path)
    return output_path


def run_upload(spark: SparkSession | None = None) -> str:
    """Convenience entrypoint for orchestrating the upload stage."""
    cfg = load_config()
    owns_spark = spark is None
    active_spark = spark or build_spark()

    try:
        transformed = build_transformed_dataset(active_spark)
        destination = upload_transformed_data(active_spark, cfg, transformed)
        print(f"Uploaded transformed dataset to {destination}")
        return destination
    finally:
        if owns_spark:
            active_spark.stop()


def main() -> None:
    """Run the upload job end to end."""
    run_upload()


if __name__ == "__main__":
    main()
