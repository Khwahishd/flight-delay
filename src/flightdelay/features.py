"""Feature engineering with an explicit notion of *what is known when*.

Two prediction scenarios are supported:

``scheduled``
    Predict at schedule time. Only the timetable, the route, and the weather
    observed for the scheduled departure hour are used.
``departure``
    Predict at pushback. Adds ``dep_delay``, which is unknown until the
    aircraft actually leaves the gate.

Using ``dep_delay`` in the ``scheduled`` scenario would be target leakage, so
the scenario is enforced here rather than left to the caller.
"""
from __future__ import annotations

import pandas as pd

DELAY_THRESHOLD_MIN = 15  # standard "late" definition (FAA: 15+ minutes)

CATEGORICAL = ["carrier", "origin", "dest"]
WEATHER = [
    "temp", "dewp", "humid", "wind_dir", "wind_speed",
    "wind_gust", "precip", "pressure", "visib",
]
SCHEDULE_NUMERIC = [
    "distance", "sched_dep_min", "sched_arr_min", "month",
    "day_of_week", "origin_hourly_departures",
]
SCENARIOS = ("scheduled", "departure")


def hhmm_to_minutes(hhmm: pd.Series) -> pd.Series:
    """Convert clock times like 1530 into minutes since midnight (930)."""
    return (hhmm // 100) * 60 + (hhmm % 100)


def build_dataset(
    flights: pd.DataFrame, weather: pd.DataFrame, scenario: str = "scheduled"
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where y = 1 if the flight arrived >= 15 minutes late.

    Cancelled and diverted flights (no ``arr_delay``) are dropped. ``X`` keeps
    the original flight order, so chronological splits stay valid.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")

    weather = weather.drop_duplicates(subset=["origin", "time_hour"])
    df = flights.merge(
        weather[["origin", "time_hour", *WEATHER]],
        on=["origin", "time_hour"],
        how="left",
        validate="many_to_one",
    )

    # Congestion proxy: scheduled departures from the same airport in the same
    # hour. Computed from the timetable, so it is known in advance.
    df["origin_hourly_departures"] = df.groupby(["origin", "time_hour"])[
        "flight"
    ].transform("size")
    df["sched_dep_min"] = hhmm_to_minutes(df["sched_dep_time"])
    df["sched_arr_min"] = hhmm_to_minutes(df["sched_arr_time"])
    df["day_of_week"] = pd.to_datetime(df[["year", "month", "day"]]).dt.day_of_week

    df = df.dropna(subset=["arr_delay"]).reset_index(drop=True)

    numeric = SCHEDULE_NUMERIC + WEATHER
    if scenario == "departure":
        numeric = numeric + ["dep_delay"]
    X = df[CATEGORICAL + numeric].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype("category")
    y = (df["arr_delay"] >= DELAY_THRESHOLD_MIN).astype(int)
    return X, y


def time_split(
    X: pd.DataFrame, y: pd.Series, test_months: tuple[int, ...] = (10, 11, 12)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological split: train on earlier months, test on later ones.

    A random split would let the model see the same weather systems and
    schedule patterns on both sides, which overstates real-world accuracy.
    """
    is_test = X["month"].isin(test_months)
    if not is_test.any() or is_test.all():
        raise ValueError("test_months leaves train or test set empty")
    if X.loc[is_test, "month"].min() <= X.loc[~is_test, "month"].max():
        raise ValueError("test_months must all be later than every training month")
    return X[~is_test], X[is_test], y[~is_test], y[is_test]
