"""Download SPDR sector ETF prices for the real-data benchmark."""
from __future__ import annotations

import os
import sys

import pandas as pd
import yfinance as yf

TICKERS = ["XLE", "XLB", "XLI", "XLY", "XLP", "XLF", "XLK", "XLC", "XLU", "XLRE"]
START = "2018-06-19"  # XLC inception
END = "2025-12-31"
OUT_PATH = "data/real/sector_etfs.csv"


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = yf.download(
        TICKERS, start=START, end=END, interval="1d",
        auto_adjust=True, progress=False, group_by="ticker",
    )
    close_cols = {}
    for t in TICKERS:
        if t in df.columns.get_level_values(0):
            close_cols[t] = df[t]["Close"]
        else:
            print(f"WARN: missing ticker {t}", file=sys.stderr)
    wide = pd.DataFrame(close_cols).dropna()
    wide.index.name = "Date"
    wide.to_csv(OUT_PATH)
    print(f"Saved {wide.shape[0]} rows × {wide.shape[1]} tickers → {OUT_PATH}")
    print(f"Range: {wide.index.min().date()} → {wide.index.max().date()}")


if __name__ == "__main__":
    main()
