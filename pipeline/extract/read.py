import io
import json
import os
import time
from datetime import date, timedelta
from typing import Iterable

import boto3
from bs4 import BeautifulSoup
import pandas as pd
import requests
from requests import exceptions as requests_exceptions
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
DATA_START_DATE = os.environ.get("DATA_START_DATE")
DATA_END_DATE = os.environ.get("DATA_END_DATE")
LOOKBACK_DAYS = int(os.environ.get("DATA_LOOKBACK_DAYS", "2"))
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
    """Raise a clear error when a required environment variable is missing."""
    if not value:
        raise ValueError(f"{name} is required.")
    return value


def get_sp500_symbols() -> list[str]:
    """Scrape the current S&P 500 constituent tickers from Wikipedia."""
    response = requests.get(
        WIKI_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise RuntimeError("Could not find the S&P 500 constituents table on Wikipedia.")

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    try:
        symbol_index = headers.index("Symbol")
    except ValueError as exc:
        raise RuntimeError("S&P 500 constituents table does not contain a Symbol column.") from exc

    symbols: list[str] = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) <= symbol_index:
            continue
        symbol = cells[symbol_index].get_text(strip=True).replace(".", "-")
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        raise RuntimeError("No S&P 500 symbols could be parsed from Wikipedia.")
    return symbols


def month_ranges(year: int) -> list[tuple[date, date]]:
    """Return inclusive monthly date ranges for the requested year."""
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


def selected_ranges(year: int) -> list[tuple[date, date]]:
    """Return the configured extract ranges, preferring explicit daily bounds."""
    if DATA_END_DATE:
        end = date.fromisoformat(DATA_END_DATE)
        if DATA_START_DATE:
            start = date.fromisoformat(DATA_START_DATE)
            if end < start:
                raise ValueError("DATA_END_DATE must be on or after DATA_START_DATE.")
            return [(start, end)]

        windows: list[tuple[date, date]] = []
        for offset in range(max(1, LOOKBACK_DAYS)):
            day = end - timedelta(days=offset)
            windows.append((day, day))
        return windows
    return month_ranges(year)


def fetch_polygon_minute_bars(
    session: requests.Session,
    ticker: str,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """Fetch Polygon minute bars for one ticker and one date range."""
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
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429:
                print(
                    f"Polygon rate limited for {ticker} {start_date.isoformat()} to {end_date.isoformat()} "
                    f"on attempt {attempt}/{MAX_RETRIES}."
                )
                continue
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "OK":
                raise RuntimeError(
                    f"Polygon request failed for {ticker} {start_date} to {end_date}: {payload}"
                )
            return payload.get("results", [])
        except (requests_exceptions.ReadTimeout, requests_exceptions.Timeout) as exc:
            print(
                f"Polygon timed out for {ticker} {start_date.isoformat()} to {end_date.isoformat()} "
                f"on attempt {attempt}/{MAX_RETRIES}: {exc}"
            )
        except requests_exceptions.RequestException as exc:
            if getattr(exc, "response", None) is not None and exc.response.status_code == 429:
                print(
                    f"Polygon rate limited for {ticker} {start_date.isoformat()} to {end_date.isoformat()} "
                    f"on attempt {attempt}/{MAX_RETRIES}."
                )
            else:
                raise

        if attempt < MAX_RETRIES:
            time.sleep(REQUEST_INTERVAL_SECONDS * attempt)

    raise RuntimeError(
        f"Polygon rate limit retries exhausted for {ticker} {start_date} to {end_date}."
    )


def fetch_polygon_minute_bars_or_empty(
    session: requests.Session,
    ticker: str,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """Fetch minute bars, returning an empty list when Polygon has no data yet."""
    try:
        results = fetch_polygon_minute_bars(session, ticker, start_date, end_date)
    except requests.HTTPError as exc:
        response = exc.response
        if response is not None and response.status_code in {404, 429}:
            print(
                f"Skipping {ticker} {start_date.isoformat()} to {end_date.isoformat()} "
                f"because Polygon returned {response.status_code}."
            )
            return []
        raise
    finally:
        time.sleep(REQUEST_INTERVAL_SECONDS)

    if not results:
        print(
            f"Skipping {ticker} {start_date.isoformat()} to {end_date.isoformat()} "
            "because Polygon returned no results."
        )
    return results


def results_to_rows(ticker: str, results: Iterable[dict]) -> list[dict]:
    """Convert Polygon aggregate results into the canonical row structure."""
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
    """Build the S3 object key for one ticker window."""
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
    """Serialize one window of rows and upload it to S3."""
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
    """Load the extract checkpoint or return a fresh default state."""
    try:
        response = s3.get_object(Bucket=bucket, Key=checkpoint_key)
    except s3.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return {
                "year": YEAR,
                "s3_prefix": S3_PREFIX,
                "s3_format": S3_FORMAT,
                "extract_mode": "daily" if DATA_START_DATE and DATA_END_DATE else "monthly",
                "data_start_date": DATA_START_DATE,
                "data_end_date": DATA_END_DATE,
                "processed_symbols": [],
                "current_symbol": None,
                "window_index": 0,
            }
        raise

    payload = json.loads(response["Body"].read().decode("utf-8"))
    return {
        "year": int(payload.get("year", YEAR)),
        "s3_prefix": payload.get("s3_prefix", S3_PREFIX),
        "s3_format": payload.get("s3_format", S3_FORMAT),
        "extract_mode": payload.get(
            "extract_mode",
            "daily" if DATA_START_DATE and DATA_END_DATE else "monthly",
        ),
        "data_start_date": payload.get("data_start_date", DATA_START_DATE),
        "data_end_date": payload.get("data_end_date", DATA_END_DATE),
        "processed_symbols": list(payload.get("processed_symbols", [])),
        "current_symbol": payload.get("current_symbol"),
        "window_index": int(payload.get("window_index", 0)),
    }


def save_checkpoint(
    s3,
    bucket: str,
    checkpoint_key: str,
    extract_mode: str,
    data_start_date: str | None,
    data_end_date: str | None,
    processed_symbols: list[str],
    current_symbol: str | None,
    window_index: int,
) -> None:
    """Persist the extract checkpoint so the job can resume safely."""
    payload = {
        "year": YEAR,
        "s3_prefix": S3_PREFIX,
        "s3_format": S3_FORMAT,
        "extract_mode": extract_mode,
        "data_start_date": data_start_date,
        "data_end_date": data_end_date,
        "processed_symbols": processed_symbols,
        "current_symbol": current_symbol,
        "window_index": window_index,
        "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    s3.put_object(
        Bucket=bucket,
        Key=checkpoint_key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def run_extract() -> None:
    """Run the extract job end to end and upload rows to S3."""
    _require_env(POLYGON_API_KEY, "POLYGON_API_KEY")
    bucket = _require_env(S3_BUCKET, "S3_BUCKET")

    symbols = get_sp500_symbols()
    windows = selected_ranges(YEAR)
    s3 = boto3.client("s3")
    session = requests.Session()
    checkpoint = load_checkpoint(s3, bucket, CHECKPOINT_KEY)
    if checkpoint["year"] != YEAR or checkpoint["s3_prefix"] != S3_PREFIX or checkpoint["s3_format"] != S3_FORMAT:
        raise ValueError(
            "Checkpoint configuration does not match the current run. "
            f"checkpoint(year={checkpoint['year']}, prefix={checkpoint['s3_prefix']}, format={checkpoint['s3_format']}) "
            f"!= run(year={YEAR}, prefix={S3_PREFIX}, format={S3_FORMAT})."
        )
    expected_mode = "daily" if DATA_START_DATE and DATA_END_DATE else "monthly"
    if checkpoint["extract_mode"] != expected_mode:
        raise ValueError(
            f"Checkpoint extract mode {checkpoint['extract_mode']} does not match current run mode {expected_mode}."
        )
    if checkpoint["data_start_date"] != DATA_START_DATE or checkpoint["data_end_date"] != DATA_END_DATE:
        raise ValueError(
            "Checkpoint date window does not match the current run window."
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
    extracted_object_keys: list[str] = []
    print(
        f"Starting run for {len(symbols)} S&P 500 symbols in {YEAR}: "
        f"{len(windows)} date window(s), format {S3_FORMAT}, "
        f"request interval {REQUEST_INTERVAL_SECONDS:.1f}s, "
        f"resume symbol {start_symbol_index}, window {start_window_index}"
    )

    with tqdm(total=total_steps, desc="Fetching and uploading", unit="window") as progress:
        if completed_steps:
            progress.update(completed_steps)

        for symbol_index, symbol in enumerate(symbols[start_symbol_index:], start=start_symbol_index):
            window_start = start_window_index if symbol_index == start_symbol_index else 0
            for window_index, (start_date, end_date) in enumerate(windows[window_start:], start=window_start):
                raw_results = fetch_polygon_minute_bars_or_empty(session, symbol, start_date, end_date)
                if not raw_results:
                    progress.update(1)
                    continue
                rows = results_to_rows(symbol, raw_results)
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
                extracted_object_keys.append(object_key)
                save_checkpoint(
                    s3,
                    bucket,
                    CHECKPOINT_KEY,
                    expected_mode,
                    DATA_START_DATE,
                    DATA_END_DATE,
                    sorted(processed_symbols),
                    symbol,
                    window_index + 1,
                )
                progress.update(1)
            start_window_index = 0
            processed_symbols.add(symbol)
            next_symbol = symbols[symbol_index + 1] if symbol_index + 1 < len(symbols) else None
            save_checkpoint(
                s3,
                bucket,
                CHECKPOINT_KEY,
                expected_mode,
                DATA_START_DATE,
                DATA_END_DATE,
                sorted(processed_symbols),
                next_symbol,
                0,
            )

    print(
        "Extract complete: "
        f"{len(symbols)} symbols, {len(windows)} windows, "
        f"{len(extracted_object_keys)} uploaded objects."
    )


def main() -> None:
    """Run the extract job end to end."""
    run_extract()


if __name__ == "__main__":
    main()
