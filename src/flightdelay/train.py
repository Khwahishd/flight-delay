"""Train and evaluate all models for each scenario.

Usage:
    python -m flightdelay.train                  # both scenarios
    python -m flightdelay.train --scenario scheduled
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_raw
from .evaluate import plot_curves, plot_importance, score
from .features import SCENARIOS, build_dataset, time_split
from .models import make_hgb, make_logistic, make_prior_baseline

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def run(scenario: str) -> dict:
    flights, weather = load_raw()
    X, y = build_dataset(flights, weather, scenario)
    X_train, X_test, y_train, y_test = time_split(X, y)
    print(f"[{scenario}] train={len(X_train):,} (late {y_train.mean():.1%})  "
          f"test={len(X_test):,} (late {y_test.mean():.1%})")

    numeric = [c for c in X.columns if c not in ("carrier", "origin", "dest")]
    models = {
        "Prior baseline": make_prior_baseline(),
        "Logistic regression": make_logistic(numeric),
        "HistGradientBoosting": make_hgb(),
    }
    probas, results = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probas[name] = model.predict_proba(X_test)[:, 1]
        results[name] = score(y_test, probas[name])
        print(f"  {name:22s} " + "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    fig_dir = REPORTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(y_test, probas, fig_dir / f"curves_{scenario}.png")
    importance = plot_importance(models["HistGradientBoosting"], X_test, y_test,
                                 fig_dir / f"importance_{scenario}.png")

    summary = {
        "scenario": scenario,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "late_rate_train": float(y_train.mean()),
        "late_rate_test": float(y_test.mean()),
        "metrics": results,
        "permutation_importance": importance,
    }
    (REPORTS / f"metrics_{scenario}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    args = parser.parse_args()
    for scenario in SCENARIOS if args.scenario == "all" else [args.scenario]:
        run(scenario)


if __name__ == "__main__":
    main()
