"""Station-aware anchor and local linear-trend residual experiment.

All anchor parameters, Ridge regularization, and ensemble weights are selected
from the existing four rolling-origin OOF folds. API_TEST is deliberately not
read by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from run_lightgbm_residual_experiment import (
    apply_seasonal_lookup,
    isolated_spike_weights,
    rmse,
    seasonal_lookup,
)


CAT_DIR = Path("output_catboost_experiments")
LGB_DIR = Path("output_lightgbm_residual")
OUTPUT_DIR = Path("output_station_aware")

TRAIN_PATH = CAT_DIR / "train_features.parquet"
TEST_PATH = CAT_DIR / "test_features.parquet"
CAT_OOF_PATH = CAT_DIR / "oof_predictions.parquet"
CAT_RAW_SUBMISSION_PATH = CAT_DIR / "submission_rmse_selected.csv"
CAT_ANCHOR_SUBMISSION_PATH = CAT_DIR / "submission_rmse_anchor.csv"
LGB_OOF_PATH = LGB_DIR / "oof_predictions.parquet"
LGB_SUBMISSION_PATH = LGB_DIR / "submission_lightgbm_residual.csv"
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv")

FOLDS = (
    ("sep_2023", "2023-09-18 18:00:00", "2023-09-19 06:00:00", "2024-05-18 18:00:00"),
    ("may_2024", "2024-05-18 18:00:00", "2024-05-19 06:00:00", "2025-01-18 18:00:00"),
    ("sep_2024", "2024-09-18 18:00:00", "2024-09-19 06:00:00", "2025-05-18 18:00:00"),
    ("jan_2025", "2025-01-18 18:00:00", "2025-01-19 06:00:00", "2025-09-18 18:00:00"),
)

GLOBAL_ALPHA = 0.75
TAU_DAYS = 365.0
LOCAL_ALPHA_SHRINKAGE = 0.50
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1_000.0)
MIN_STATION_ROWS = 90

LOCAL_FEATURES = [
    "rainfall_mm",
    "rainfall_mm_sum_24h",
    "rainfall_mm_sum_72h",
    "rainfall_mm_sum_168h",
    "rainfall_mm_api_halflife_48h",
    "rainfall_mm_api_halflife_168h",
    "upstream_rainfall_mm_sum_24h",
    "upstream_rainfall_mm_sum_72h",
    "upstream_rainfall_mm_sum_168h",
    "upstream_rainfall_mm_api_halflife_48h",
    "upstream_rainfall_mm_api_halflife_168h",
    "soil_moisture_0_7cm",
    "soil_moisture_28_100cm",
    "temperature_c_mean_24h",
    "humidity_pct_mean_24h",
    "surface_pressure_hpa_mean_24h",
    "nino_34",
    "rmm1",
    "rmm2",
    "dayofyear_sin",
    "dayofyear_cos",
    "hour_sin",
    "hour_cos",
]


def prediction_metrics(
    frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        for fold, group in frame.groupby("fold", sort=False):
            rows.append(
                {
                    "model": column.removeprefix("prediction_"),
                    "fold": fold,
                    "rows": len(group),
                    "rmse": rmse(group["tma_mdpl"], group[column]),
                }
            )
        rows.append(
            {
                "model": column.removeprefix("prediction_"),
                "fold": "pooled",
                "rows": len(frame),
                "rmse": rmse(frame["tma_mdpl"], frame[column]),
            }
        )
    return pd.DataFrame(rows)


def tune_station_anchor(cat_oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = cat_oof.copy()
    frame["anchor_basis"] = (
        np.exp(-frame["horizon_days"].to_numpy(dtype=float) / TAU_DAYS)
        * frame["state_anomaly"].to_numpy(dtype=float)
    )
    frame["prediction_global_anchor"] = (
        frame["prediction_raw_rmse_selected"] + GLOBAL_ALPHA * frame["anchor_basis"]
    )
    records: list[dict[str, object]] = []
    alphas = np.linspace(-0.25, 1.50, 36)
    for station, station_frame in frame.groupby("nama_pos", sort=True):
        candidates: list[dict[str, float]] = []
        for alpha in alphas:
            fold_scores = []
            for _, fold_frame in station_frame.groupby("fold", sort=False):
                prediction = (
                    fold_frame["prediction_raw_rmse_selected"].to_numpy()
                    + alpha * fold_frame["anchor_basis"].to_numpy()
                )
                fold_scores.append(rmse(fold_frame["tma_mdpl"], prediction))
            candidates.append(
                {"local_alpha": float(alpha), "mean_fold_rmse": float(np.mean(fold_scores))}
            )
        best = min(candidates, key=lambda row: row["mean_fold_rmse"])
        shrunk_alpha = (
            GLOBAL_ALPHA
            + LOCAL_ALPHA_SHRINKAGE * (best["local_alpha"] - GLOBAL_ALPHA)
        )
        records.append(
            {
                "nama_pos": station,
                "local_alpha": best["local_alpha"],
                "shrunk_alpha": float(shrunk_alpha),
                "local_mean_fold_rmse": best["mean_fold_rmse"],
            }
        )
    parameters = pd.DataFrame(records)
    alpha_lookup = parameters.set_index("nama_pos")["shrunk_alpha"]
    frame["station_alpha"] = frame["nama_pos"].map(alpha_lookup)
    frame["prediction_station_anchor"] = (
        frame["prediction_raw_rmse_selected"]
        + frame["station_alpha"] * frame["anchor_basis"]
    )
    return frame, parameters


def numeric_design(
    fit: pd.DataFrame, target: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    origin = pd.Timestamp("2023-01-01 00:00:00")
    fit_x = fit[LOCAL_FEATURES].astype(float).copy()
    target_x = target[LOCAL_FEATURES].astype(float).copy()
    fit_x["trend_days"] = (
        fit["datetime"] - origin
    ).dt.total_seconds().to_numpy() / 86_400.0
    target_x["trend_days"] = (
        target["datetime"] - origin
    ).dt.total_seconds().to_numpy() / 86_400.0
    fit_values = fit_x.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    target_values = target_x.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    medians = np.nanmedian(fit_values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    fit_values = np.where(np.isfinite(fit_values), fit_values, medians)
    target_values = np.where(np.isfinite(target_values), target_values, medians)
    centers = fit_values.mean(axis=0)
    scales = fit_values.std(axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    return (fit_values - centers) / scales, (target_values - centers) / scales


def fit_local_ridge(
    fit: pd.DataFrame,
    target: pd.DataFrame,
    fit_baseline: np.ndarray,
    target_baseline: np.ndarray,
    alpha: float,
) -> np.ndarray:
    output = np.full(len(target), np.nan, dtype=float)
    fit_work = fit.copy()
    target_work = target.copy()
    fit_work["_baseline"] = fit_baseline
    target_work["_baseline"] = target_baseline
    weights, _ = isolated_spike_weights(fit_work)
    fit_work["_weight"] = weights
    for station, target_station in target_work.groupby("nama_pos", sort=False):
        fit_station = fit_work[fit_work["nama_pos"] == station]
        if len(fit_station) < MIN_STATION_ROWS:
            output[target_station.index.to_numpy()] = target_station["_baseline"].to_numpy()
            continue
        x_fit, x_target = numeric_design(fit_station, target_station)
        y_fit = (
            fit_station["tma_mdpl"].to_numpy(dtype=float)
            - fit_station["_baseline"].to_numpy(dtype=float)
        )
        model = Ridge(alpha=alpha)
        model.fit(x_fit, y_fit, sample_weight=fit_station["_weight"].to_numpy())
        output[target_station.index.to_numpy()] = (
            target_station["_baseline"].to_numpy(dtype=float) + model.predict(x_target)
        )
    if not np.isfinite(output).all():
        raise ValueError("Local Ridge produced missing or non-finite predictions")
    return output


def validate_local_ridge(train: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold, train_end, valid_start, valid_end in FOLDS:
        fit = train[train["datetime"] <= pd.Timestamp(train_end)].copy()
        valid = train[
            train["datetime"].between(
                pd.Timestamp(valid_start), pd.Timestamp(valid_end), inclusive="both"
            )
        ].copy()
        eligible = fit.groupby("nama_pos", observed=True).size().loc[
            lambda count: count >= MIN_STATION_ROWS
        ].index
        fit = fit[fit["nama_pos"].isin(eligible)].copy()
        valid = valid[valid["nama_pos"].isin(eligible)].copy()
        # Dense positional indices make assignment inside fit_local_ridge safe.
        valid = valid.reset_index(drop=True)
        lookup = seasonal_lookup(fit)
        fit_baseline = apply_seasonal_lookup(fit, lookup)
        valid_baseline = apply_seasonal_lookup(valid, lookup)
        result = valid[["datetime", "nama_pos", "tma_mdpl"]].copy()
        result.insert(0, "fold", fold)
        print(f"[{fold}] local Ridge: fit={len(fit):,} valid={len(valid):,}", flush=True)
        for alpha in RIDGE_ALPHAS:
            prediction = fit_local_ridge(
                fit, valid, fit_baseline, valid_baseline, alpha
            )
            result[f"prediction_ridge_{alpha:g}"] = prediction
            print(
                f"  alpha={alpha:g} RMSE={rmse(valid['tma_mdpl'], prediction):.6f}",
                flush=True,
            )
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def select_ridge_alpha(oof: pd.DataFrame) -> float:
    scores = {}
    for alpha in RIDGE_ALPHAS:
        column = f"prediction_ridge_{alpha:g}"
        fold_scores = [
            rmse(group["tma_mdpl"], group[column])
            for _, group in oof.groupby("fold", sort=False)
        ]
        scores[alpha] = float(np.mean(fold_scores))
    return float(min(scores, key=scores.get))


def tune_simplex_blend(
    oof: pd.DataFrame, ridge_column: str
) -> tuple[pd.DataFrame, dict[str, float]]:
    cat = oof["prediction_station_anchor"].to_numpy(dtype=float)
    lgb = oof["prediction_raw_residual"].to_numpy(dtype=float)
    ridge = oof[ridge_column].to_numpy(dtype=float)
    cat_fold = {
        fold: rmse(group["tma_mdpl"], group["prediction_station_anchor"])
        for fold, group in oof.groupby("fold", sort=False)
    }
    rows: list[dict[str, float]] = []
    for lgb_weight in np.arange(0.0, 0.401, 0.01):
        for ridge_weight in np.arange(0.0, 0.401, 0.01):
            if lgb_weight + ridge_weight > 0.60:
                continue
            cat_weight = 1.0 - lgb_weight - ridge_weight
            prediction = cat_weight * cat + lgb_weight * lgb + ridge_weight * ridge
            candidate = oof[["fold", "tma_mdpl"]].copy()
            candidate["prediction"] = prediction
            scores = {
                fold: rmse(group["tma_mdpl"], group["prediction"])
                for fold, group in candidate.groupby("fold", sort=False)
            }
            rows.append(
                {
                    "catboost_weight": float(cat_weight),
                    "lightgbm_weight": float(lgb_weight),
                    "ridge_weight": float(ridge_weight),
                    "mean_fold_rmse": float(np.mean(list(scores.values()))),
                    "pooled_rmse": rmse(candidate["tma_mdpl"], candidate["prediction"]),
                    "worst_fold_delta": float(max(scores[f] - cat_fold[f] for f in scores)),
                    **{f"rmse_{fold}": value for fold, value in scores.items()},
                }
            )
    grid = pd.DataFrame(rows)
    eligible = grid[grid["worst_fold_delta"] <= 0.01]
    selected = (eligible if not eligible.empty else grid).sort_values(
        ["mean_fold_rmse", "pooled_rmse"]
    ).iloc[0].to_dict()
    return grid, selected


def tune_station_lightgbm_weights(
    oof: pd.DataFrame,
    global_lightgbm_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    records: list[dict[str, float | str]] = []
    for station, group in oof.groupby("nama_pos", sort=True):
        cat = group["prediction_station_anchor"].to_numpy(dtype=float)
        lgb = group["prediction_raw_residual"].to_numpy(dtype=float)
        truth = group["tma_mdpl"].to_numpy(dtype=float)
        delta = lgb - cat
        denominator = float(np.dot(delta, delta))
        local_weight = (
            global_lightgbm_weight
            if denominator <= np.finfo(float).eps
            else float(np.clip(np.dot(truth - cat, delta) / denominator, 0.0, 1.0))
        )
        records.append(
            {"nama_pos": station, "local_lightgbm_weight": local_weight}
        )
    parameters = pd.DataFrame(records)
    local_lookup = parameters.set_index("nama_pos")["local_lightgbm_weight"]
    rows: list[dict[str, float]] = []
    station_local = oof["nama_pos"].map(local_lookup).to_numpy(dtype=float)
    cat = oof["prediction_station_anchor"].to_numpy(dtype=float)
    lgb = oof["prediction_raw_residual"].to_numpy(dtype=float)
    baseline_by_fold = {
        fold: rmse(group["tma_mdpl"], group["prediction_station_anchor"])
        for fold, group in oof.groupby("fold", sort=False)
    }
    for shrinkage in np.linspace(0.0, 1.0, 21):
        weights = global_lightgbm_weight + shrinkage * (
            station_local - global_lightgbm_weight
        )
        prediction = (1.0 - weights) * cat + weights * lgb
        candidate = oof[["fold", "tma_mdpl"]].copy()
        candidate["prediction"] = prediction
        scores = {
            fold: rmse(group["tma_mdpl"], group["prediction"])
            for fold, group in candidate.groupby("fold", sort=False)
        }
        rows.append(
            {
                "station_weight_shrinkage": float(shrinkage),
                "mean_fold_rmse": float(np.mean(list(scores.values()))),
                "pooled_rmse": rmse(candidate["tma_mdpl"], candidate["prediction"]),
                "worst_fold_delta": float(
                    max(scores[fold] - baseline_by_fold[fold] for fold in scores)
                ),
                **{f"rmse_{fold}": value for fold, value in scores.items()},
            }
        )
    grid = pd.DataFrame(rows)
    eligible = grid[grid["worst_fold_delta"] <= 0.01]
    selected = (eligible if not eligible.empty else grid).sort_values(
        ["mean_fold_rmse", "pooled_rmse"]
    ).iloc[0].to_dict()
    shrinkage = float(selected["station_weight_shrinkage"])
    parameters["shrunk_lightgbm_weight"] = (
        global_lightgbm_weight
        + shrinkage
        * (parameters["local_lightgbm_weight"] - global_lightgbm_weight)
    )
    return parameters, grid, selected


def write_submission(path: Path, ids: pd.Series, prediction: np.ndarray) -> None:
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    predicted = pd.DataFrame({"id": ids, "tma_mdpl": prediction})
    submission = sample[["id"]].merge(predicted, on="id", how="left", validate="one_to_one")
    if submission["tma_mdpl"].isna().any() or not np.isfinite(submission["tma_mdpl"]).all():
        raise ValueError(f"Invalid submission: {path}")
    submission.to_csv(path, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        TRAIN_PATH, TEST_PATH, CAT_OOF_PATH, CAT_RAW_SUBMISSION_PATH,
        CAT_ANCHOR_SUBMISSION_PATH, LGB_OOF_PATH, LGB_SUBMISSION_PATH,
        SAMPLE_SUBMISSION_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    cat_oof = pd.read_parquet(CAT_OOF_PATH)
    anchored_oof, station_parameters = tune_station_anchor(cat_oof)
    ridge_oof = validate_local_ridge(train)
    lgb_oof = pd.read_parquet(LGB_OOF_PATH)[
        ["fold", "datetime", "nama_pos", "prediction_raw_residual"]
    ]
    oof = anchored_oof.merge(
        ridge_oof,
        on=["fold", "datetime", "nama_pos", "tma_mdpl"],
        how="inner",
        validate="one_to_one",
    ).merge(
        lgb_oof,
        on=["fold", "datetime", "nama_pos"],
        how="inner",
        validate="one_to_one",
    )
    if len(oof) != len(cat_oof):
        raise ValueError("OOF components do not align exactly")

    ridge_alpha = select_ridge_alpha(oof)
    ridge_column = f"prediction_ridge_{ridge_alpha:g}"
    blend_grid, selected_blend = tune_simplex_blend(oof, ridge_column)
    oof["prediction_station_stack"] = (
        selected_blend["catboost_weight"] * oof["prediction_station_anchor"]
        + selected_blend["lightgbm_weight"] * oof["prediction_raw_residual"]
        + selected_blend["ridge_weight"] * oof[ridge_column]
    )
    station_blend_parameters, station_blend_grid, selected_station_blend = (
        tune_station_lightgbm_weights(
            oof, float(selected_blend["lightgbm_weight"])
        )
    )
    station_lgb_lookup = station_blend_parameters.set_index("nama_pos")[
        "shrunk_lightgbm_weight"
    ]
    oof_station_lgb_weight = oof["nama_pos"].map(station_lgb_lookup).to_numpy()
    oof["prediction_station_weighted_stack"] = (
        (1.0 - oof_station_lgb_weight) * oof["prediction_station_anchor"]
        + oof_station_lgb_weight * oof["prediction_raw_residual"]
    )
    metric_columns = [
        "prediction_raw_rmse_selected",
        "prediction_global_anchor",
        "prediction_station_anchor",
        ridge_column,
        "prediction_raw_residual",
        "prediction_station_stack",
        "prediction_station_weighted_stack",
    ]
    metrics = prediction_metrics(oof, metric_columns)
    metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    station_parameters.to_csv(OUTPUT_DIR / "station_anchor_parameters.csv", index=False)
    station_blend_parameters.to_csv(
        OUTPUT_DIR / "station_blend_parameters.csv", index=False
    )
    blend_grid.to_csv(OUTPUT_DIR / "blend_grid.csv", index=False)
    station_blend_grid.to_csv(
        OUTPUT_DIR / "station_blend_grid.csv", index=False
    )
    oof.to_parquet(OUTPUT_DIR / "oof_predictions.parquet", index=False)

    raw_submission = pd.read_csv(CAT_RAW_SUBMISSION_PATH)
    anchor_submission = pd.read_csv(CAT_ANCHOR_SUBMISSION_PATH)
    lgb_submission = pd.read_csv(LGB_SUBMISSION_PATH)
    ids = test["id"]
    raw = ids.to_frame().merge(raw_submission, on="id", validate="one_to_one")["tma_mdpl"].to_numpy()
    global_anchor = ids.to_frame().merge(anchor_submission, on="id", validate="one_to_one")["tma_mdpl"].to_numpy()
    lgb = ids.to_frame().merge(lgb_submission, on="id", validate="one_to_one")["tma_mdpl"].to_numpy()
    correction_basis = (global_anchor - raw) / GLOBAL_ALPHA
    alpha_lookup = station_parameters.set_index("nama_pos")["shrunk_alpha"]
    station_alpha = test["nama_pos"].map(alpha_lookup).to_numpy(dtype=float)
    station_anchor = raw + station_alpha * correction_basis

    final_lookup = seasonal_lookup(train)
    train_baseline = apply_seasonal_lookup(train, final_lookup)
    test_baseline = apply_seasonal_lookup(test, final_lookup)
    ridge_prediction = fit_local_ridge(
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        train_baseline,
        test_baseline,
        ridge_alpha,
    )
    stack_prediction = (
        selected_blend["catboost_weight"] * station_anchor
        + selected_blend["lightgbm_weight"] * lgb
        + selected_blend["ridge_weight"] * ridge_prediction
    )
    final_station_lgb_weight = test["nama_pos"].map(station_lgb_lookup).to_numpy()
    station_weighted_stack = (
        (1.0 - final_station_lgb_weight) * station_anchor
        + final_station_lgb_weight * lgb
    )
    write_submission(OUTPUT_DIR / "submission_station_anchor.csv", ids, station_anchor)
    write_submission(OUTPUT_DIR / "submission_local_ridge.csv", ids, ridge_prediction)
    write_submission(OUTPUT_DIR / "submission_station_stack.csv", ids, stack_prediction)
    write_submission(
        OUTPUT_DIR / "submission_station_weighted_stack.csv",
        ids,
        station_weighted_stack,
    )

    summary = {
        "ridge_alpha": ridge_alpha,
        "anchor_tau_days": TAU_DAYS,
        "anchor_local_shrinkage": LOCAL_ALPHA_SHRINKAGE,
        "selected_blend": selected_blend,
        "selected_station_blend": selected_station_blend,
        "validation_metrics": metrics.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
