import io
import json
import os
import time
from datetime import date, timedelta
from typing import Iterable

import boto3
import pandas as pd
import requests
from tqdm import tqdm

from .config import DEFAULT_CONFIG


POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", DEFAULT_CONFIG.s3_prefix)
CHECKPOINT_KEY = os.environ.get("CHECKPOINT_KEY", DEFAULT_CONFIG.checkpoint_key)
POLYGON_BASE_URL = os.environ.get("POLYGON_BASE_URL", DEFAULT_CONFIG.polygon_base_url)
REQUEST_TIMEOUT = int(os.environ.get("POLYGON_REQUEST_TIMEOUT", str(DEFAULT_CONFIG.request_timeout)))
MAX_RETRIES = int(os.environ.get("POLYGON_MAX_RETRIES", str(DEFAULT_CONFIG.max_retries)))
YEAR = int(os.environ.get("DATA_YEAR", str(DEFAULT_CONFIG.year)))
REQUEST_INTERVAL_SECONDS = float(
    os.environ.get("POLYGON_REQUEST_INTERVAL_SECONDS", str(DEFAULT_CONFIG.request_interval_seconds))
)
S3_FORMAT = os.environ.get("S3_FORMAT", DEFAULT_CONFIG.s3_format).strip().lower()
WIKI_URL = os.environ.get(
    "SP500_WIKI_URL",
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
)

DATA_COLUMNS = [
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


def _require_env(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required.")
    return value


def get_sp500_symbols() -> list[str]:
    response = requests.get(
        WIKI_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            )
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    tables = pd.read_html(response.text)
    constituents = tables[0]
    symbols = constituents["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return sorted(symbols.unique().tolist())


def month_ranges(year: int) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current = date(year, 1, 1)
    while current.year == year:
        if current.month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        end = next_month - timedelta(days=1)
        ranges.append((current, end))
        current = next_month
    return ranges


def fetch_polygon_minute_bars(
    session: requests.Session,
    ticker: str,
    start_date: date,
    end_date: date,
) -> list[dict]:
    url = (
        f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/"
        f"{start_date.isoformat()}/{end_date.isoformat()}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 429:
            time.sleep(REQUEST_INTERVAL_SECONDS)
            continue
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "OK":
            raise RuntimeError(
                f"Polygon request failed for {ticker} {start_date} to {end_date}: {payload}"
            )
        return payload.get("results", [])

    raise RuntimeError(
        f"Polygon rate limit retries exhausted for {ticker} {start_date} to {end_date}."
    )


def results_to_rows(ticker: str, results: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in results:
        timestamp = pd.to_datetime(row["t"], unit="ms", utc=True)
        rows.append(
            {
                "ticker": ticker,
                "timestamp_utc": timestamp.isoformat(),
                "date": timestamp.date().isoformat(),
                "open": row.get("o"),
                "high": row.get("h"),
                "low": row.get("l"),
                "close": row.get("c"),
                "volume": row.get("v"),
                "vwap": row.get("vw"),
                "transactions": row.get("n"),
            }
        )
    return rows


def window_key(prefix: str, ticker: str, start_date: date, end_date: date) -> str:
    suffix = "parquet" if S3_FORMAT == "parquet" else "csv"
    return (
        f"{prefix.rstrip('/')}/{ticker}/"
        f"{start_date.isoformat()}_{end_date.isoformat()}.{suffix}"
    )


def upload_window_to_s3(
    s3,
    bucket: str,
    prefix: str,
    ticker: str,
    start_date: date,
    end_date: date,
    rows: list[dict],
) -> str:
    object_key = window_key(prefix, ticker, start_date, end_date)
    frame = pd.DataFrame(rows, columns=DATA_COLUMNS)

    if S3_FORMAT == "parquet":
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        body = buffer.getvalue()
        content_type = "application/octet-stream"
    elif S3_FORMAT == "csv":
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        body = buffer.getvalue().encode("utf-8")
        content_type = "text/csv"
    else:
        raise ValueError("S3_FORMAT must be either 'parquet' or 'csv'.")

    s3.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=body,
        ContentType=content_type,
    )
    return object_key


def load_checkpoint(s3, bucket: str, checkpoint_key: str) -> dict:
    try:
        response = s3.get_object(Bucket=bucket, Key=checkpoint_key)
    except s3.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return {"symbol_index": 0, "window_index": 0}
        raise

    payload = json.loads(response["Body"].read().decode("utf-8"))
    return {
        "year": int(payload.get("year", YEAR)),
        "s3_prefix": payload.get("s3_prefix", S3_PREFIX),
        "s3_format": payload.get("s3_format", S3_FORMAT),
        "processed_symbols": list(payload.get("processed_symbols", [])),
        "current_symbol": payload.get("current_symbol"),
        "window_index": int(payload.get("window_index", 0)),
    }


def save_checkpoint(
    s3,
    bucket: str,
    checkpoint_key: str,
    processed_symbols: list[str],
    current_symbol: str | None,
    window_index: int,
) -> None:
    payload = {
        "year": YEAR,
        "s3_prefix": S3_PREFIX,
        "s3_format": S3_FORMAT,
        "processed_symbols": processed_symbols,
        "current_symbol": current_symbol,
        "window_index": window_index,
        "updated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    s3.put_object(
        Bucket=bucket,
        Key=checkpoint_key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def main() -> None:
    _require_env(POLYGON_API_KEY, "POLYGON_API_KEY")
    bucket = _require_env(S3_BUCKET, "S3_BUCKET")

    symbols = get_sp500_symbols()
    windows = month_ranges(YEAR)
    s3 = boto3.client("s3")
    session = requests.Session()
    checkpoint = load_checkpoint(s3, bucket, CHECKPOINT_KEY)
    if checkpoint["year"] != YEAR or checkpoint["s3_prefix"] != S3_PREFIX or checkpoint["s3_format"] != S3_FORMAT:
        raise ValueError(
            "Checkpoint configuration does not match the current run. "
            f"checkpoint(year={checkpoint['year']}, prefix={checkpoint['s3_prefix']}, format={checkpoint['s3_format']}) "
            f"!= run(year={YEAR}, prefix={S3_PREFIX}, format={S3_FORMAT})."
        )

    processed_symbols = set(str(symbol) for symbol in checkpoint["processed_symbols"])
    current_symbol = checkpoint["current_symbol"]
    if current_symbol and current_symbol in symbols:
        start_symbol_index = symbols.index(current_symbol)
    else:
        start_symbol_index = next((i for i, symbol in enumerate(symbols) if symbol not in processed_symbols), len(symbols))
        current_symbol = symbols[start_symbol_index] if start_symbol_index < len(symbols) else None
    start_window_index = int(checkpoint["window_index"]) if current_symbol and current_symbol in symbols else 0
    completed_steps = sum(1 for symbol in symbols if symbol in processed_symbols) * len(windows) + start_window_index

    total_steps = len(symbols) * len(windows)
    print(
        f"Starting run for {len(symbols)} S&P 500 symbols in {YEAR}: "
        f"{len(windows)} monthly windows each, format {S3_FORMAT}, "
        f"request interval {REQUEST_INTERVAL_SECONDS:.1f}s, "
        f"resume symbol {start_symbol_index}, window {start_window_index}"
    )

    with tqdm(total=total_steps, desc="Fetching and uploading", unit="window") as progress:
        if completed_steps:
            progress.update(completed_steps)

        for symbol_index, symbol in enumerate(symbols[start_symbol_index:], start=start_symbol_index):
            window_start = start_window_index if symbol_index == start_symbol_index else 0
            for window_index, (start_date, end_date) in enumerate(windows[window_start:], start=window_start):
                rows = results_to_rows(
                    symbol,
                    fetch_polygon_minute_bars(session, symbol, start_date, end_date),
                )
                object_key = upload_window_to_s3(
                    s3,
                    bucket,
                    S3_PREFIX,
                    symbol,
                    start_date,
                    end_date,
                    rows,
                )
                print(
                    f"Uploaded {symbol} {start_date.isoformat()} to {end_date.isoformat()} "
                    f"({len(rows):,} rows) -> s3://{bucket}/{object_key}"
                )
                rows = []
                save_checkpoint(
                    s3,
                    bucket,
                    CHECKPOINT_KEY,
                    sorted(processed_symbols),
                    symbol,
                    window_index + 1,
                )
                time.sleep(REQUEST_INTERVAL_SECONDS)
                progress.update(1)
            start_window_index = 0
            processed_symbols.add(symbol)
            next_symbol = symbols[symbol_index + 1] if symbol_index + 1 < len(symbols) else None
            save_checkpoint(
                s3,
                bucket,
                CHECKPOINT_KEY,
                sorted(processed_symbols),
                next_symbol,
                0,
            )


if __name__ == "__main__":
    main()
