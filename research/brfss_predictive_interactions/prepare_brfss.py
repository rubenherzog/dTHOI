from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

XCOLS = [
    "EXERANY2", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "ASTHMA3", "CHCSCNC1",
    "CHCOCNC1", "CHCCOPD3", "ADDEPEV3", "CHCKDNY2", "HAVARTH4", "VETERAN3",
]
KEEP = XCOLS + ["_BMI5", "_LLCPWT"]


def _selected_chunks(path: Path, *, chunksize: int = 25_000):
    """Yield only analysis columns from the CDC XPORT file in bounded chunks."""
    reader = pd.read_sas(path, format="xport", encoding="latin1", chunksize=chunksize)
    for chunk in reader:
        missing = [c for c in KEEP if c not in chunk.columns]
        if missing:
            raise KeyError(f"Missing BRFSS columns: {missing}")
        yield chunk[KEEP].copy()


def read_xpt(path: Path) -> pd.DataFrame:
    """Read a CDC XPT or one-file ZIP while retaining only analysis columns."""
    if path.suffix.lower() != ".zip":
        return pd.concat(_selected_chunks(path), ignore_index=True)

    with zipfile.ZipFile(path) as archive:
        # The official 2023 archive contains a trailing space in `LLCP2023.XPT `.
        members = [n for n in archive.namelist() if n.strip().upper().endswith(".XPT")]
        if len(members) != 1:
            raise ValueError("Expected exactly one .XPT member in the ZIP.")
        with tempfile.TemporaryDirectory() as tmp:
            extracted = Path(tmp) / "LLCP2023.XPT"
            with archive.open(members[0]) as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            return pd.concat(_selected_chunks(extracted), ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="LLCP2023.XPT or LLCP2023XPT.zip")
    parser.add_argument("output", type=Path, help="Output selected.pkl")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = read_xpt(args.input)
    frame.to_pickle(args.output)
    print(f"wrote {len(frame):,} rows to {args.output}")


if __name__ == "__main__":
    main()
