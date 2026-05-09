"""
Download 25 years of NIFTY 50 Total Return Index (TRI) historical data from niftyindices.com.
- Chunks the range into 90-day windows (the API is slow and prohibitive on full-year calls).
- Saves each chunk to a per-chunk CSV under ./_chunks_tri so reruns skip done work.
- Retries with backoff on timeout / 5xx.
- Final step concatenates everything into nifty50tri_<start>_<end>.csv.
"""
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUT_DIR = Path(__file__).parent
CHUNK_DIR = OUT_DIR / "_chunks_tri"
CHUNK_DIR.mkdir(exist_ok=True)

INDEX = "NIFTY 50"
END = date.today()
START = date(2000, 1, 1)
CHUNK_DAYS = 90
TIMEOUT = (15, 120)  # (connect, read)

URL = "https://niftyindices.com/Backpage.aspx/getTotalReturnIndexString"
HEADERS = {
    "Content-Type": "application/json; charset=UTF-8",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.niftyindices.com/reports/historical-data",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
}

session = requests.Session()
session.headers.update(HEADERS)
retry = Retry(
    total=5,
    backoff_factor=2,                 # 2,4,8,16,32s
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST"]),
)
session.mount("https://", HTTPAdapter(max_retries=retry))
session.get("https://www.niftyindices.com/reports/historical-data", timeout=TIMEOUT)


def fmt(d: date) -> str:
    return d.strftime("%d-%b-%Y")


def fetch(start: date, end: date) -> pd.DataFrame:
    cinfo = json.dumps(
        {
            "name": INDEX,
            "startDate": fmt(start),
            "endDate": fmt(end),
            "indexName": INDEX,
        }
    )
    last_err = None
    for attempt in range(4):
        try:
            r = session.post(URL, data=json.dumps({"cinfo": cinfo}), timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()["d"]
            rows = json.loads(payload) if isinstance(payload, str) else payload
            return pd.DataFrame(rows)
        except (requests.Timeout, requests.ConnectionError, ValueError) as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"    retry in {wait}s ({type(e).__name__})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed {start}..{end}: {last_err}")


def chunks(start: date, end: date, days: int):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


for s, e in chunks(START, END, CHUNK_DAYS):
    fp = CHUNK_DIR / f"{s.isoformat()}_{e.isoformat()}.csv"
    if fp.exists() and fp.stat().st_size > 0:
        print(f"  skip {s}..{e} (have {fp.name})", flush=True)
        continue
    t0 = time.time()
    print(f"  fetching {s}..{e} ...", end=" ", flush=True)
    df = fetch(s, e)
    df.to_csv(fp, index=False)
    print(f"{len(df)} rows in {time.time()-t0:.1f}s", flush=True)
    time.sleep(0.5)

# Combine — skip empty / unparseable chunk files
def _safe_read(p):
    try:
        df = pd.read_csv(p)
        return df if not df.empty else None
    except pd.errors.EmptyDataError:
        print(f"  (skipping empty chunk: {p.name})")
        return None

frames = [d for p in sorted(CHUNK_DIR.glob("*.csv")) for d in [_safe_read(p)] if d is not None]
full = pd.concat(frames, ignore_index=True)

date_col = next(c for c in full.columns if "date" in c.lower() or c == "HistoricalDate")
full[date_col] = pd.to_datetime(full[date_col], errors="coerce")
full = (
    full.dropna(subset=[date_col])
    .drop_duplicates(subset=[date_col])
    .sort_values(date_col)
    .reset_index(drop=True)
)

out = OUT_DIR / f"nifty50tri_{START.year}_{END.year}.csv"
full.to_csv(out, index=False)
print(f"\nSaved {len(full):,} rows -> {out}")
