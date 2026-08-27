import sqlite3
import time
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

BASE_URL       = "https://www.ebi.ac.uk/chembl/api/data/assay"
ASSAY_TYPES    = ["F", "B"]
CONFIDENCE     = 9
PAGE_SIZE      = 1000
MAX_RETRIES    = 6
BACKOFF_FACTOR = 2.0
TIMEOUT        = 60

COLUMNS = [
    "assay_chembl_id",
    "assay_type",
    "assay_organism",
    "confidence_score",
    "description",
    "bao_format",
    "bao_label",
    "document_chembl_id",
    "target_chembl_id",
    "relationship_type",
    "assay_cell_type",
    "assay_tissue",
    "assay_subcellular_fraction",
    "assay_parameters",
    "src_id",
]

DDL = """
CREATE TABLE IF NOT EXISTS assays (
    assay_chembl_id              TEXT PRIMARY KEY,
    assay_type                   TEXT,
    assay_organism               TEXT,
    confidence_score             INTEGER,
    description                  TEXT,
    bao_format                   TEXT,
    bao_label                    TEXT,
    document_chembl_id           TEXT,
    target_chembl_id             TEXT,
    relationship_type            TEXT,
    assay_cell_type              TEXT,
    assay_tissue                 TEXT,
    assay_subcellular_fraction   TEXT,
    assay_parameters             TEXT,
    src_id                       TEXT,
    fetched_at                   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("01_download_assays.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

def build_session() -> requests.Session:
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.headers.update({"Accept": "application/json"})
    return session

def fetch_page(session: requests.Session, assay_type: str, offset: int) -> dict:
    params = {
        "assay_type":      assay_type,
        "confidence_score": CONFIDENCE,
        "limit":           PAGE_SIZE,
        "offset":          offset,
        "format":          "json",
    }
    attempt = 0
    delay   = 2.0
    while True:
        attempt += 1
        try:
            r = session.get(BASE_URL, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            log.warning("HTTP %s on attempt %d (offset=%d, type=%s)",
                        r.status_code, attempt, offset, assay_type)
        except requests.exceptions.RequestException as exc:
            log.warning("Request error on attempt %d: %s", attempt, exc)

        if attempt >= MAX_RETRIES:
            raise RuntimeError(
                f"Failed to fetch offset={offset} type={assay_type} "
                f"after {MAX_RETRIES} attempts."
            )
        log.info("Retrying in %.0fs …", delay)
        time.sleep(delay)
        delay = min(delay * BACKOFF_FACTOR, 120)

def iter_assays(session: requests.Session,
                assay_type: str,
                start_offset: int = 0) -> Iterator[Dict]:
    offset = start_offset
    total  = None

    with tqdm(
        desc=f"Downloading type={assay_type}",
        unit=" assays",
        total=None,
        dynamic_ncols=True,
    ) as bar:
        if start_offset > 0:
            bar.update(start_offset)

        while True:
            page   = fetch_page(session, assay_type, offset)
            page_meta  = page.get("page_meta", {})
            records    = page.get("assays", [])

            if total is None:
                total = page_meta.get("total_count", 0)
                bar.total = total
                bar.refresh()

            if not records:
                break

            for rec in records:
                yield rec

            bar.update(len(records))
            offset += len(records)

            if offset >= total:
                break

            time.sleep(0.3)

def open_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.executescript(DDL)
    con.commit()
    return con

def get_meta(con: sqlite3.Connection, key: str, default=None):
    row = con.execute("SELECT value FROM _meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default

def set_meta(con: sqlite3.Connection, key: str, value: str):
    con.execute("INSERT OR REPLACE INTO _meta(key, value) VALUES (?,?)", (key, value))

def record_to_row(rec: dict) -> Dict:
    import json
    row = {}
    for col in COLUMNS:
        val = rec.get(col)
        if isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False)
        row[col] = val
    return row

def upsert_batch(con: sqlite3.Connection, rows: List[Dict]):
    if not rows:
        return
    placeholders = ", ".join(f":{c}" for c in COLUMNS)
    col_names    = ", ".join(COLUMNS)
    sql = (
        f"INSERT OR IGNORE INTO assays ({col_names}) "
        f"VALUES ({placeholders})"
    )
    con.executemany(sql, rows)

def export_tsv(con: sqlite3.Connection, path: Path):
    log.info("Exporting TSV → %s", path)
    rows = con.execute("SELECT * FROM assays ORDER BY assay_chembl_id").fetchall()
    if not rows:
        log.warning("No rows to export.")
        return
    with open(path, "w", encoding="utf-8") as fh:
        headers = [desc[0] for desc in con.execute("SELECT * FROM assays LIMIT 0").description]
        fh.write("\t".join(headers) + "\n")
        for row in tqdm(rows, desc="Writing TSV", unit=" rows", dynamic_ncols=True):
            fh.write("\t".join("" if v is None else str(v) for v in row) + "\n")
    log.info("TSV written: %d rows", len(rows))

def main():
    parser = argparse.ArgumentParser(
        description="Download ChEMBL assays (type F/B, confidence 9) to SQLite."
    )
    parser.add_argument(
        "--db", default="bronze_assays.db",
        help="Path to output SQLite database (default: bronze_assays.db)"
    )
    parser.add_argument(
        "--tsv", default="bronze_assays.tsv",
        help="Path to optional TSV export (default: bronze_assays.tsv; set to '' to skip)"
    )
    parser.add_argument(
        "--types", nargs="+", default=ASSAY_TYPES,
        help="Assay types to download (default: F B)"
    )
    parser.add_argument(
        "--batch", type=int, default=500,
        help="Records per DB commit (default: 500)"
    )
    args = parser.parse_args()

    db_path  = Path(args.db)
    tsv_path = Path(args.tsv) if args.tsv else None

    log.info("=" * 60)
    log.info("ChEMBL Assay Downloader")
    log.info("  DB       : %s", db_path)
    log.info("  types    : %s", args.types)
    log.info("  confidence: %d", CONFIDENCE)
    log.info("=" * 60)

    con     = open_db(db_path)
    session = build_session()

    total_written = 0

    for atype in args.types:
        resume_key    = f"offset_{atype}"
        start_offset  = int(get_meta(con, resume_key, 0))

        if start_offset > 0:
            log.info("Resuming type=%s from offset %d", atype, start_offset)

        batch    = []
        offset   = start_offset

        for rec in iter_assays(session, atype, start_offset=start_offset):
            batch.append(record_to_row(rec))
            offset += 1

            if len(batch) >= args.batch:
                upsert_batch(con, batch)
                set_meta(con, resume_key, str(offset))
                con.commit()
                total_written += len(batch)
                batch = []

        if batch:
            upsert_batch(con, batch)
            set_meta(con, resume_key, str(offset))
            con.commit()
            total_written += len(batch)

        log.info("type=%s done. Offset reached: %d", atype, offset)

    count = con.execute("SELECT COUNT(*) FROM assays").fetchone()[0]
    log.info("─" * 60)
    log.info("Total rows in DB : %d", count)
    log.info("Rows written now : %d", total_written)

    if tsv_path:
        export_tsv(con, tsv_path)

    con.close()
    log.info("Done. ✓")

if __name__ == "__main__":
    main()
