"""Download and load the nycflights13 flights and weather tables."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FILES = ("flights.csv", "weather.csv")


def download(raw_dir: Path = RAW_DIR) -> None:
    """Fetch the CSVs if they are not already on disk."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = raw_dir / name
        if target.exists():
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", target)


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (flights, weather), downloading first if needed."""
    download(raw_dir)
    flights = pd.read_csv(raw_dir / "flights.csv", index_col=0)
    weather = pd.read_csv(raw_dir / "weather.csv", index_col=0)
    return flights, weather


if __name__ == "__main__":
    download()
