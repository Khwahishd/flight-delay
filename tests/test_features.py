import pandas as pd
import pytest

from flightdelay.features import (
    build_dataset,
    hhmm_to_minutes,
    time_split,
)


def make_frames():
    flights = pd.DataFrame(
        {
            "year": 2013,
            "month": [1, 1, 1, 10, 11],
            "day": [1, 1, 2, 5, 6],
            "sched_dep_time": [515, 515, 1530, 800, 900],
            "sched_arr_time": [819, 830, 1800, 1000, 1100],
            "dep_delay": [2, 30, -3, 10, 0],
            "arr_delay": [11, 40, None, 15, -5],  # row 2 is cancelled
            "carrier": ["UA", "UA", "AA", "DL", "DL"],
            "flight": [1, 2, 3, 4, 5],
            "origin": ["EWR", "EWR", "JFK", "LGA", "LGA"],
            "dest": ["IAH", "ORD", "LAX", "ATL", "ATL"],
            "distance": [1400, 719, 2475, 762, 762],
            "time_hour": [
                "2013-01-01T10:00:00Z", "2013-01-01T10:00:00Z",
                "2013-01-02T20:00:00Z", "2013-10-05T12:00:00Z",
                "2013-11-06T13:00:00Z",
            ],
        }
    )
    weather = pd.DataFrame(
        {
            "origin": ["EWR", "EWR", "JFK", "LGA", "LGA"],
            "time_hour": [
                "2013-01-01T10:00:00Z", "2013-01-01T10:00:00Z",  # duplicate key
                "2013-01-02T20:00:00Z", "2013-10-05T12:00:00Z",
                "2013-11-06T13:00:00Z",
            ],
            "temp": 40.0, "dewp": 30.0, "humid": 60.0, "wind_dir": 270,
            "wind_speed": 10.0, "wind_gust": None, "precip": 0.0,
            "pressure": 1012.0, "visib": 10.0,
        }
    )
    return flights, weather


def test_hhmm_to_minutes():
    assert hhmm_to_minutes(pd.Series([0, 515, 1530, 2359])).tolist() == [0, 315, 930, 1439]


def test_target_and_dropped_rows():
    X, y = build_dataset(*make_frames())
    assert len(X) == 4  # cancelled flight removed
    assert y.tolist() == [0, 1, 1, 0]  # arr_delay >= 15 is "late"


def test_weather_join_does_not_duplicate_flights():
    X, _ = build_dataset(*make_frames())
    assert len(X) == 4
    assert X["temp"].notna().all()


def test_scheduled_scenario_excludes_dep_delay():
    X, _ = build_dataset(*make_frames(), scenario="scheduled")
    assert "dep_delay" not in X.columns


def test_departure_scenario_includes_dep_delay():
    X, _ = build_dataset(*make_frames(), scenario="departure")
    assert "dep_delay" in X.columns


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        build_dataset(*make_frames(), scenario="oracle")


def test_congestion_counts_same_origin_hour():
    X, _ = build_dataset(*make_frames())
    # first two flights share EWR @ 10:00Z
    assert X["origin_hourly_departures"].tolist()[:2] == [2, 2]


def test_time_split_is_chronological():
    X, y = build_dataset(*make_frames())
    X_tr, X_te, y_tr, y_te = time_split(X, y, test_months=(10, 11, 12))
    assert X_tr["month"].max() < X_te["month"].min()
    assert len(X_tr) + len(X_te) == len(X)
    assert len(y_tr) == len(X_tr) and len(y_te) == len(X_te)


def test_time_split_rejects_non_future_test_months():
    X, y = build_dataset(*make_frames())
    with pytest.raises(ValueError):
        time_split(X, y, test_months=(1,))
