import io
import json
import os
import time
from datetime import date, timedelta
from typing import Iterable

from bs4 import BeautifulSoup
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests import exceptions as requests_exceptions
from tqdm import tqdm

from ..aws_auth import build_aws_session
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
POLYGON_REFERENCE_LOOKUPS_ENABLED = os.environ.get("POLYGON_REFERENCE_LOOKUPS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
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
    "company_name",
    "sector",
    "industry",
    "exchange",
]

INTEGER_COLUMNS = ["volume", "transactions"]
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string()),
        pa.field("timestamp_utc", pa.string()),
        pa.field("date", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("vwap", pa.float64()),
        pa.field("transactions", pa.int64()),
        pa.field("company_name", pa.string()),
        pa.field("sector", pa.string()),
        pa.field("industry", pa.string()),
        pa.field("exchange", pa.string()),
    ]
)

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

    header_row = table.find("tr")
    if header_row is None:
        raise RuntimeError("S&P 500 constituents table is missing a header row.")
    headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
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


def _load_wikipedia_constituent_metadata() -> dict[str, dict[str, str | None]]:
    """Scrape company metadata from Wikipedia for fallback use."""
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

    header_row = table.find("tr")
    if header_row is None:
        raise RuntimeError("S&P 500 constituents table is missing a header row.")
    headers = [th.get_text(strip=True) for th in header_row.find_all("th")]

    def _find_header(*candidates: str) -> int | None:
        for candidate in candidates:
            if candidate in headers:
                return headers.index(candidate)
        return None

    symbol_index = _find_header("Symbol")
    company_index = _find_header("Security", "Company", "Name")
    sector_index = _find_header("GICS Sector", "Sector")
    industry_index = _find_header("GICS Sub-Industry", "Sub-Industry", "Industry")

    if symbol_index is None:
        raise RuntimeError("S&P 500 constituents table is missing a Symbol column.")

    metadata: dict[str, dict[str, str | None]] = {}
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) <= symbol_index:
            continue
        symbol = cells[symbol_index].get_text(strip=True).replace(".", "-")
        if symbol and symbol not in metadata:
            metadata[symbol] = {
                "company_name": (
                    cells[company_index].get_text(strip=True)
                    if company_index is not None and len(cells) > company_index
                    else None
                ),
                "sector": (
                    cells[sector_index].get_text(strip=True)
                    if sector_index is not None and len(cells) > sector_index
                    else None
                ),
                "industry": (
                    cells[industry_index].get_text(strip=True)
                    if industry_index is not None and len(cells) > industry_index
                    else None
                ),
            }
    return metadata


def _normalize_polygon_ticker_details(payload: dict) -> dict[str, str | None]:
    """Normalize Polygon ticker details into the metadata columns we load downstream."""
    result = payload.get("results")
    if isinstance(result, list):
        result = result[0] if result else {}
    elif isinstance(result, dict):
        result = result
    elif "ticker" in payload:
        result = payload
    else:
        result = {}

    company_name = result.get("name")
    return {
        "company_name": company_name,
        "exchange": result.get("primary_exchange"),
    }


def fetch_polygon_ticker_metadata(
    session: requests.Session,
    ticker: str,
) -> dict[str, str | None]:
    """Fetch ticker metadata from Polygon's reference data endpoints."""
    candidates = [
        (
            f"{POLYGON_BASE_URL}/v3/reference/tickers",
            {
                "ticker": ticker,
                "limit": 1,
                "active": "true",
                "apiKey": POLYGON_API_KEY,
            },
        ),
        (
            f"{POLYGON_BASE_URL}/v3/reference/tickers/{ticker}",
            {"apiKey": POLYGON_API_KEY},
        ),
    ]

    best_metadata: dict[str, str | None] | None = None
    best_score = -1

    for url, params in candidates:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if getattr(response, "status_code", 200) == 429:
                    print(
                        f"Polygon reference lookup rate limited for {ticker} "
                        f"on attempt {attempt}/{MAX_RETRIES}."
                    )
                    continue
                response.raise_for_status()
                payload = response.json()
                metadata = _normalize_polygon_ticker_details(payload)
                score = sum(
                    1
                    for value in metadata.values()
                    if value is not None and value != "Unknown"
                )
                if score > best_score:
                    best_metadata = metadata
                    best_score = score
                break
            except (requests_exceptions.ReadTimeout, requests_exceptions.Timeout) as exc:
                print(
                    f"Polygon reference lookup timed out for {ticker} "
                    f"on attempt {attempt}/{MAX_RETRIES}: {exc}"
                )
            except requests_exceptions.RequestException as exc:
                if getattr(exc, "response", None) is not None and exc.response.status_code == 429:
                    print(
                        f"Polygon reference lookup rate limited for {ticker} "
                        f"on attempt {attempt}/{MAX_RETRIES}."
                    )
                else:
                    if url.endswith(f"/{ticker}") and getattr(exc, "response", None) is not None:
                        break
                    raise

            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_INTERVAL_SECONDS * attempt)

    return best_metadata or {"company_name": None, "exchange": None}


def _build_metadata_by_symbol(
    session: requests.Session,
    symbols: list[str],
    wikipedia_metadata: dict[str, dict[str, str | None]],
    include_polygon_reference: bool,
) -> dict[str, dict[str, str | None]]:
    """Build the per-symbol metadata map used during extraction."""
    metadata_by_symbol: dict[str, dict[str, str | None]] = {}
    for symbol in symbols:
        fallback_metadata = wikipedia_metadata.get(symbol, {})
        polygon_metadata = {"company_name": None, "exchange": None}
        if include_polygon_reference:
            polygon_metadata = fetch_polygon_ticker_metadata(session, symbol)
        metadata_by_symbol[symbol] = {
            "company_name": polygon_metadata.get("company_name") or fallback_metadata.get("company_name"),
            "sector": fallback_metadata.get("sector"),
            "industry": fallback_metadata.get("industry"),
            "exchange": polygon_metadata.get("exchange") if include_polygon_reference else None,
        }
    return metadata_by_symbol


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


def results_to_rows(
    ticker: str,
    results: Iterable[dict],
    metadata: dict[str, str | None] | None = None,
) -> list[dict]:
    """Convert Polygon aggregate results into the canonical row structure."""
    metadata = metadata or {}
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
                "company_name": metadata.get("company_name"),
                "sector": metadata.get("sector"),
                "industry": metadata.get("industry"),
                "exchange": metadata.get("exchange"),
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


def _normalize_output_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Force the raw extract schema to stay stable before serialization."""
    normalized = frame.copy()
    for column in INTEGER_COLUMNS:
        if column in normalized.columns:
            numeric = pd.to_numeric(normalized[column], errors="coerce")
            normalized[column] = numeric.round().astype("Int64")
    return normalized


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
    frame = _normalize_output_frame(pd.DataFrame(rows, columns=DATA_COLUMNS))

    if S3_FORMAT == "parquet":
        buffer = io.BytesIO()
        table = pa.Table.from_pandas(frame, schema=PARQUET_SCHEMA, preserve_index=False)
        pq.write_table(table, buffer)
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


def _fresh_checkpoint_state() -> dict[str, object]:
    """Return the default checkpoint state for the current run."""
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


def run_extract() -> None:
    """Run the extract job end to end and upload rows to S3."""
    _require_env(POLYGON_API_KEY, "POLYGON_API_KEY")
    bucket = _require_env(S3_BUCKET, "S3_BUCKET")

    symbols = get_sp500_symbols()
    wikipedia_metadata = _load_wikipedia_constituent_metadata()
    windows = selected_ranges(YEAR)
    aws = build_aws_session()
    s3 = aws.session.client("s3")
    session = requests.Session()
    checkpoint = load_checkpoint(s3, bucket, CHECKPOINT_KEY)
    expected_mode = "daily" if DATA_START_DATE and DATA_END_DATE else "monthly"
    checkpoint_mismatch = (
        checkpoint["year"] != YEAR
        or checkpoint["s3_prefix"] != S3_PREFIX
        or checkpoint["s3_format"] != S3_FORMAT
        or checkpoint["extract_mode"] != expected_mode
        or checkpoint["data_start_date"] != DATA_START_DATE
        or checkpoint["data_end_date"] != DATA_END_DATE
    )
    if checkpoint_mismatch:
        print(
            "Ignoring stale checkpoint and starting fresh for the current run window.",
            flush=True,
        )
        checkpoint = _fresh_checkpoint_state()

    processed_symbols = set(str(symbol) for symbol in checkpoint["processed_symbols"])
    current_symbol = checkpoint["current_symbol"]
    if current_symbol and current_symbol in symbols:
        start_symbol_index = symbols.index(current_symbol)
    else:
        start_symbol_index = next((i for i, symbol in enumerate(symbols) if symbol not in processed_symbols), len(symbols))
        current_symbol = symbols[start_symbol_index] if start_symbol_index < len(symbols) else None
    start_window_index = int(checkpoint["window_index"]) if current_symbol and current_symbol in symbols else 0
    completed_steps = sum(1 for symbol in symbols if symbol in processed_symbols) * len(windows) + start_window_index

    metadata_by_symbol = _build_metadata_by_symbol(
        session,
        symbols,
        wikipedia_metadata,
        include_polygon_reference=POLYGON_REFERENCE_LOOKUPS_ENABLED,
    )

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
                rows = results_to_rows(symbol, raw_results, metadata_by_symbol.get(symbol))
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
