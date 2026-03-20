"""
fetch_data.py — Weekly NIFTY 1-minute data fetcher
Run every Friday after market close to accumulate training data.

Usage:
    python scripts/fetch_data.py
"""

import os
import yfinance as yf
import pandas as pd


def fetch_and_append():
    csv_path = "data/NIFTY_1min.csv"

    # Download latest 7 days of 1-minute data
    print("Downloading latest NIFTY 1-minute data...")
    new_df = yf.download("^NSEI", period="7d", interval="1m", progress=False)

    if len(new_df) == 0:
        print("No data returned from Yahoo Finance. Try again later.")
        return

    # Flatten MultiIndex columns and normalise names
    new_df = new_df.reset_index()
    new_df.columns = [
        c[0].lower() if isinstance(c, tuple) else c.lower()
        for c in new_df.columns
    ]

    # Ensure date column exists
    if "date" not in new_df.columns:
        new_df = new_df.rename(columns={new_df.columns[0]: "date"})

    # Convert date to plain string for consistent CSV storage
    new_df["date"] = new_df["date"].astype(str)

    # Keep only the columns we need
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in new_df.columns]
    new_df = new_df[keep]

    # Append to existing CSV or create new one
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path, dtype={"date": str})
        before = len(existing)

        combined = pd.concat([existing, new_df], ignore_index=True)
        combined["date"] = combined["date"].astype(str)
        combined = (
            combined
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        combined.to_csv(csv_path, index=False)
        added = len(combined) - before

        print(f"Updated : {len(combined):,} total bars")
        print(f"Was     : {before:,} bars")
        print(f"Added   : {added:,} new bars")
        print(f"From    : {combined['date'].iloc[0]}")
        print(f"To      : {combined['date'].iloc[-1]}")
    else:
        os.makedirs("data", exist_ok=True)
        new_df.to_csv(csv_path, index=False)
        print(f"Created : {len(new_df):,} bars")
        print(f"From    : {new_df['date'].iloc[0]}")
        print(f"To      : {new_df['date'].iloc[-1]}")

    print("\nDone. Run training when you have enough data:")
    print("  python scripts/train_model.py --data data/NIFTY_1min.csv --tp 0.1 --sl 0.08 --lookahead 15")


if __name__ == "__main__":
    fetch_and_append()