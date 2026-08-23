from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def _configure_runtime_env(target_date: str) -> None:
    """Align the existing layer defaults before each task imports its module."""
    year = os.environ.get("DATA_YEAR", "2025")
    bucket = os.environ.get("S3_BUCKET", "nyc-tlc")
    extract_prefix = f"polygon/sp500/minute/{year}/"
    transformed_prefix = f"polygon/sp500/minute_transformed/{year}/"

    os.environ["S3_BUCKET"] = bucket
    os.environ["S3_PREFIX"] = extract_prefix
    os.environ["CHECKPOINT_KEY"] = f"polygon/sp500/minute/{year}/_checkpoint.json"
    os.environ["DATA_END_DATE"] = target_date
    os.environ["DATA_LOOKBACK_DAYS"] = "2"
    os.environ.pop("DATA_START_DATE", None)
    os.environ["TRANSFORM_SOURCE_PREFIX"] = f"s3a://{bucket}/{extract_prefix}"
    os.environ["TRANSFORM_OUTPUT_PREFIX"] = f"s3a://{bucket}/{transformed_prefix}"
    os.environ["LOAD_SOURCE_PREFIX"] = transformed_prefix


def run_extract(**context) -> None:
    """Run the existing Polygon extract job."""
    target_date = (context["data_interval_end"].date() - timedelta(days=1)).isoformat()
    _configure_runtime_env(target_date)
    from pipeline.extract.read import run_extract as extract

    extract()


def run_transform(**context) -> str:
    """Run the existing Spark transform and upload job."""
    target_date = (context["data_interval_end"].date() - timedelta(days=1)).isoformat()
    _configure_runtime_env(target_date)
    from pipeline.transform.upload import run_upload

    return run_upload()


def run_load(**context) -> int:
    """Run the existing Snowflake load job."""
    target_date = (context["data_interval_end"].date() - timedelta(days=1)).isoformat()
    _configure_runtime_env(target_date)
    from pipeline.load.load import run_load as load

    return load()


with DAG(
    dag_id="polygon_sp500_daily_etl",
    default_args=DEFAULT_ARGS,
    description="Daily Polygon -> Spark -> Snowflake pipeline for S&P 500 minute data",
    schedule_interval="0 2 * * *",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["polygon", "spark", "snowflake", "etl"],
) as dag:
    extract_task = PythonOperator(
        task_id="extract_polygon_data",
        python_callable=run_extract,
    )

    transform_task = PythonOperator(
        task_id="transform_with_spark",
        python_callable=run_transform,
    )

    load_task = PythonOperator(
        task_id="load_into_snowflake",
        python_callable=run_load,
    )

    extract_task >> transform_task >> load_task
