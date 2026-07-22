"""Continuous, shrunken station-specific blend of station-stack and recency.

The prior hard-routing supermodel found a small set of stations with stable
recency signal. This follow-up replaces hard routes with regularised continuous
weights. Each weight is fit only on meta-training OOF rows, clipped, then
shrunk toward station-stack before it is applied to a held-out fold or test.
API_TEST is not read by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_lightgbm_residual_experiment import rmse
from run_station_aware_experiment import write_submission


STATION_DIR = Path("output_station_aware")
RECENCY_DIR = Path("output_catboost_recency")
FEATURE_DIR = Path("output_catboost_experiments")
OUTPUT_DIR = Path("output_station_shrunk_recency")
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv")

CONTROL_OOF_PATH = STATION_DIR / "oof_predictions.parquet"
RECENCY_OOF_PATH = RECENCY_DIR / "oof_predictions.parquet"
CONTROL_SUBMISSION_PATH = STATION_DIR / "submission_station_stack.csv"
RECENCY_SUBMISSION_PATH = RECENCY_DIR / "submission_recency_blend.csv"
TEST_FEATURES_PATH = FEATURE_DIR / "test_features.parquet"

# Fixed before this validation run: allow a challenger to contribute at most
# 50%, then apply 50% shrinkage toward the station-stack default.
MAX_RAW_RECENCY_WEIGHT = 0.50
SHRINKAGE = 0.50
MAX_WORST_FOLD_DELTA = 0.001

KEY_COLUMNS = ["fold", "datetime", "nama_pos", "tma_mdpl"]


def load_oof() -> pd.DataFrame:
    control = pd.read_parquet(CONTROL_OOF_PATH)[
        KEY_COLUMNS + ["prediction_station_stack"]
    ]
    recency = pd.read_parquet(RECENCY_OOF_PATH)[
        ["fold", "datetime", "nama_pos", "prediction_selected"]
    ].rename(columns={"prediction_selected": "prediction_recency"})
    frame = control.merge(recency, on=KEY_COLUMNS[:-1], validate="one_to_one")
    if len(frame) != len(control) or frame["prediction_recency"].isna().any():
        raise ValueError("Recency OOF does not align with station-stack OOF")
    return frame


def fit_station_weights(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for station, group in frame.groupby("nama_pos", observed=True, sort=True):
        control = group["prediction_station_stack"].to_numpy(dtype=float)
        recency = group["prediction_recency"].to_numpy(dtype=float)
        truth = group["tma_mdpl"].to_numpy(dtype=float)
        delta = recency - control
        denominator = float(np.dot(delta, delta))
        raw_weight = (
            0.0
            if denominator <= np.finfo(float).eps
            else float(np.dot(truth - control, delta) / denominator)
        )
        raw_weight = float(np.clip(raw_weight, 0.0, MAX_RAW_RECENCY_WEIGHT))
        weight = SHRINKAGE * raw_weight
        rows.append(
            {
                "nama_pos": station,
                "raw_recency_weight": raw_weight,
                "recency_weight": weight,
                "station_stack_weight": 1.0 - weight,
                "fit_rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def apply_weights(frame: pd.DataFrame, weights: pd.DataFrame) -> np.ndarray:
    lookup = weights.set_index("nama_pos")["recency_weight"]
    weight = frame["nama_pos"].map(lookup)
    if weight.isna().any():
        raise ValueError("Missing station blend weight")
    return (
        (1.0 - weight.to_numpy(dtype=float))
        * frame["prediction_station_stack"].to_numpy(dtype=float)
        + weight.to_numpy(dtype=float) * frame["prediction_recency"].to_numpy(dtype=float)
    )


def scores(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    work = frame[["fold", "tma_mdpl"]].copy()
    work["prediction"] = prediction
    return {
        fold: rmse(group["tma_mdpl"], group["prediction"])
        for fold, group in work.groupby("fold", sort=False)
    }


def metric_rows(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    by_fold = scores(frame, frame[column].to_numpy(dtype=float))
    return [
        *(
            {"model": column.removeprefix("prediction_"), "fold": fold, "rmse": score}
            for fold, score in by_fold.items()
        ),
        {
            "model": column.removeprefix("prediction_"),
            "fold": "mean_fold_rmse",
            "rmse": float(np.mean(list(by_fold.values()))),
        },
        {
            "model": column.removeprefix("prediction_"),
            "fold": "pooled_rmse",
            "rmse": rmse(frame["tma_mdpl"], frame[column]),
        },
    ]


def leave_one_fold_out(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    weight_parts: list[pd.DataFrame] = []
    for held_out in frame["fold"].drop_duplicates():
        training = frame[frame["fold"] != held_out]
        held = frame[frame["fold"] == held_out]
        weights = fit_station_weights(training)
        prediction = apply_weights(held, weights)
        control = held["prediction_station_stack"].to_numpy(dtype=float)
        audit_rows.append(
            {
                "held_out_fold": held_out,
                "held_out_rmse": rmse(held["tma_mdpl"], prediction),
                "control_rmse": rmse(held["tma_mdpl"], control),
                "mean_recency_weight": float(weights["recency_weight"].mean()),
                "nonzero_station_count": int((weights["recency_weight"] > 0).sum()),
            }
        )
        held_part = held[["fold", "datetime", "nama_pos"]].copy()
        held_part["prediction"] = prediction
        parts.append(held_part)
        weights = weights.copy()
        weights.insert(0, "held_out_fold", held_out)
        weight_parts.append(weights)
    audit = pd.DataFrame(audit_rows)
    audit["delta_vs_control"] = audit["held_out_rmse"] - audit["control_rmse"]
    prediction_frame = pd.concat(parts, ignore_index=True)
    ordered = frame[["fold", "datetime", "nama_pos"]].merge(
        prediction_frame, on=["fold", "datetime", "nama_pos"], validate="one_to_one"
    )
    return audit, ordered["prediction"].to_numpy(dtype=float), pd.concat(weight_parts)


def load_test_predictions(sample: pd.DataFrame) -> pd.DataFrame:
    test = pd.read_parquet(TEST_FEATURES_PATH)[["id", "nama_pos"]]
    test = sample[["id"]].merge(test, on="id", validate="one_to_one")
    for name, path in (
        ("station_stack", CONTROL_SUBMISSION_PATH),
        ("recency", RECENCY_SUBMISSION_PATH),
    ):
        submission = pd.read_csv(path)
        test = test.merge(
            submission[["id", "tma_mdpl"]].rename(
                columns={"tma_mdpl": f"prediction_{name}"}
            ),
            on="id",
            validate="one_to_one",
        )
    columns = ["prediction_station_stack", "prediction_recency"]
    if test[columns].isna().any().any() or not np.isfinite(
        test[columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid test component predictions")
    return test


def main() -> None:
    required = [
        CONTROL_OOF_PATH,
        RECENCY_OOF_PATH,
        CONTROL_SUBMISSION_PATH,
        RECENCY_SUBMISSION_PATH,
        TEST_FEATURES_PATH,
        SAMPLE_SUBMISSION_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_oof()

    final_weights = fit_station_weights(frame)
    frame["prediction_shrunk_recency_fitted"] = apply_weights(frame, final_weights)
    audit, lofo_prediction, lofo_weights = leave_one_fold_out(frame)
    frame["prediction_shrunk_recency_lofo"] = lofo_prediction
    metrics = pd.DataFrame(
        metric_rows(frame, "prediction_station_stack")
        + metric_rows(frame, "prediction_recency")
        + metric_rows(frame, "prediction_shrunk_recency_fitted")
        + metric_rows(frame, "prediction_shrunk_recency_lofo")
    )
    pivot = metrics.pivot(index="model", columns="fold", values="rmse")
    audit_pass = bool(
        pivot.loc["shrunk_recency_lofo", "mean_fold_rmse"]
        < pivot.loc["station_stack", "mean_fold_rmse"]
        and pivot.loc["shrunk_recency_lofo", "pooled_rmse"]
        < pivot.loc["station_stack", "pooled_rmse"]
        and float(audit["delta_vs_control"].max()) <= MAX_WORST_FOLD_DELTA
    )

    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    test = load_test_predictions(sample)
    test_prediction = apply_weights(test, final_weights)
    write_submission(
        OUTPUT_DIR / "submission_station_shrunk_recency.csv",
        test["id"],
        test_prediction,
    )

    frame.to_parquet(OUTPUT_DIR / "oof_predictions.parquet", index=False)
    final_weights.to_csv(OUTPUT_DIR / "final_station_weights.csv", index=False)
    lofo_weights.to_csv(OUTPUT_DIR / "leave_one_fold_out_weights.csv", index=False)
    audit.to_csv(OUTPUT_DIR / "leave_one_fold_out_audit.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    summary = {
        "selection_protocol": {
            "control": "station_stack",
            "challenger": "recency_blend",
            "max_raw_recency_weight": MAX_RAW_RECENCY_WEIGHT,
            "shrinkage": SHRINKAGE,
            "max_final_recency_weight": MAX_RAW_RECENCY_WEIGHT * SHRINKAGE,
            "max_worst_fold_delta": MAX_WORST_FOLD_DELTA,
        },
        "pre_api_decision": {
            "api_test_evaluated": False,
            "audit_pass": audit_pass,
            "recommendation": (
                "frozen_candidate_eligible_for_one_way_api_verification"
                if audit_pass
                else "retain_station_stack_control"
            ),
        },
        "final_weight_summary": {
            "mean": float(final_weights["recency_weight"].mean()),
            "nonzero_station_count": int((final_weights["recency_weight"] > 0).sum()),
            "max": float(final_weights["recency_weight"].max()),
        },
        "leave_one_fold_out": audit.to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
        "submission": str(OUTPUT_DIR / "submission_station_shrunk_recency.csv"),
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
