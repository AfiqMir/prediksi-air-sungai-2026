"""Validate a leakage-safe direct LightGBM residual model and CatBoost blend.

The experiment reuses the exact non-recursive feature matrices and rolling-origin
folds from the CatBoost RMSE-anchor pipeline.  Seasonal baselines and residual
scales are fitted independently inside every fold.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


FEATURE_DIR = Path("output_catboost_experiments")
OUTPUT_DIR = Path("output_lightgbm_residual")
TRAIN_PATH = FEATURE_DIR / "train_features.parquet"
TEST_PATH = FEATURE_DIR / "test_features.parquet"
FEATURE_METADATA_PATH = FEATURE_DIR / "feature_metadata.json"
CATBOOST_OOF_PATH = FEATURE_DIR / "oof_predictions.parquet"
CATBOOST_SUBMISSION_PATH = FEATURE_DIR / "submission_rmse_anchor.csv"
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv")

FOLDS = (
    ("sep_2023", "2023-09-18 18:00:00", "2023-09-19 06:00:00", "2024-05-18 18:00:00"),
    ("may_2024", "2024-05-18 18:00:00", "2024-05-19 06:00:00", "2025-01-18 18:00:00"),
    ("sep_2024", "2024-09-18 18:00:00", "2024-09-19 06:00:00", "2025-05-18 18:00:00"),
    ("jan_2025", "2025-01-18 18:00:00", "2025-01-19 06:00:00", "2025-09-18 18:00:00"),
)

VARIANTS = ("raw_residual", "scaled_residual")
MIN_LABELS_BEFORE_CUTOFF = 90
MAX_ESTIMATORS = 3_000
EARLY_STOPPING_ROUNDS = 150
RANDOM_STATE = 17
CATBOOST_ANCHOR_ALPHA = 0.75
CATBOOST_ANCHOR_TAU_DAYS = 365.0


def rmse(actual: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(actual) - np.asarray(prediction)))))


def seasonal_lookup(fit: pd.DataFrame) -> dict[str, object]:
    reference = fit.assign(
        month=fit["datetime"].dt.month,
        hour=fit["datetime"].dt.hour,
    )
    return {
        "seasonal": reference.groupby(
            ["nama_pos", "month", "hour"], observed=True
        )["tma_mdpl"].median(),
        "station": reference.groupby("nama_pos", observed=True)["tma_mdpl"].median(),
        "global": float(reference["tma_mdpl"].median()),
    }


def apply_seasonal_lookup(frame: pd.DataFrame, lookup: dict[str, object]) -> np.ndarray:
    key = pd.MultiIndex.from_arrays(
        [frame["nama_pos"], frame["datetime"].dt.month, frame["datetime"].dt.hour]
    )
    prediction = lookup["seasonal"].reindex(key).to_numpy(dtype=float, copy=True)
    missing = ~np.isfinite(prediction)
    if missing.any():
        station_fallback = frame.loc[missing, "nama_pos"].map(lookup["station"])
        prediction[missing] = station_fallback.fillna(lookup["global"]).to_numpy(dtype=float)
    return prediction


def residual_scales(stations: pd.Series, residual: np.ndarray) -> pd.Series:
    work = pd.DataFrame({"nama_pos": stations.to_numpy(), "residual": residual})
    grouped = work.groupby("nama_pos", observed=True)["residual"]
    scale = ((grouped.quantile(0.75) - grouped.quantile(0.25)) / 1.349).clip(lower=0.05)
    return scale


def isolated_spike_weights(train: pd.DataFrame) -> tuple[np.ndarray, int]:
    ordered = train.sort_values(["datetime", "nama_pos"]).copy()
    group = ordered.groupby("nama_pos", sort=False, observed=True)
    ordered["prev_y"] = group["tma_mdpl"].shift(1)
    ordered["next_y"] = group["tma_mdpl"].shift(-1)
    ordered["prev_t"] = group["datetime"].shift(1)
    ordered["next_t"] = group["datetime"].shift(-1)
    diff_scale = group["tma_mdpl"].transform(
        lambda values: values.diff().abs().median() * 1.4826
    ).clip(lower=0.05)
    grouped_y = ordered.groupby("nama_pos", observed=True)["tma_mdpl"]
    robust_scale = (
        (grouped_y.transform("quantile", 0.75) - grouped_y.transform("quantile", 0.25))
        / 1.349
    ).clip(lower=0.05)
    midpoint = (ordered["prev_y"] + ordered["next_y"]) / 2.0
    threshold = np.maximum(12.0 * diff_scale, 8.0 * robust_scale)
    neighbors_agree = (ordered["prev_y"] - ordered["next_y"]).abs() <= np.maximum(
        4.0 * diff_scale, 2.0 * robust_scale
    )
    adjacent = (
        ((ordered["datetime"] - ordered["prev_t"]) <= pd.Timedelta(hours=12))
        & ((ordered["next_t"] - ordered["datetime"]) <= pd.Timedelta(hours=12))
    )
    spike = adjacent & neighbors_agree & ((ordered["tma_mdpl"] - midpoint).abs() > threshold)
    ordered["weight"] = np.where(spike, 0.05, 1.0)
    return ordered["weight"].reindex(train.index).to_numpy(dtype=float), int(spike.sum())


def prepare_categories(
    train: pd.DataFrame, test: pd.DataFrame, categorical: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    for column in categorical:
        values = pd.concat([train[column], test[column]], ignore_index=True).fillna("__MISSING__")
        categories = sorted(values.astype(str).unique().tolist())
        dtype = pd.CategoricalDtype(categories=categories)
        train[column] = train[column].fillna("__MISSING__").astype(str).astype(dtype)
        test[column] = test[column].fillna("__MISSING__").astype(str).astype(dtype)
    return train, test


def make_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=MAX_ESTIMATORS,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def target_for_variant(
    variant: str,
    frame: pd.DataFrame,
    baseline: np.ndarray,
    scale: pd.Series,
) -> np.ndarray:
    residual = frame["tma_mdpl"].to_numpy(dtype=float) - baseline
    if variant == "raw_residual":
        return residual
    return residual / frame["nama_pos"].map(scale).to_numpy(dtype=float)


def prediction_for_variant(
    variant: str,
    prediction: np.ndarray,
    frame: pd.DataFrame,
    baseline: np.ndarray,
    scale: pd.Series,
) -> np.ndarray:
    if variant == "scaled_residual":
        prediction = prediction * frame["nama_pos"].map(scale).to_numpy(dtype=float)
    return baseline + prediction


def catboost_anchor_oof() -> pd.DataFrame:
    cat = pd.read_parquet(CATBOOST_OOF_PATH)
    correction = (
        CATBOOST_ANCHOR_ALPHA
        * np.exp(-cat["horizon_days"].to_numpy(dtype=float) / CATBOOST_ANCHOR_TAU_DAYS)
        * cat["state_anomaly"].to_numpy(dtype=float)
    )
    result = cat[["fold", "datetime", "nama_pos", "tma_mdpl"]].copy()
    result["prediction_catboost_anchor"] = (
        cat["prediction_raw_rmse_selected"].to_numpy(dtype=float) + correction
    )
    return result


def metric_rows(oof: pd.DataFrame, prediction_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for prediction_column in prediction_columns:
        for fold, group in oof.groupby("fold", sort=False):
            rows.append(
                {
                    "model": prediction_column.removeprefix("prediction_"),
                    "fold": fold,
                    "rows": len(group),
                    "rmse": rmse(group["tma_mdpl"], group[prediction_column]),
                }
            )
        rows.append(
            {
                "model": prediction_column.removeprefix("prediction_"),
                "fold": "pooled",
                "rows": len(oof),
                "rmse": rmse(oof["tma_mdpl"], oof[prediction_column]),
            }
        )
    return pd.DataFrame(rows)


def tune_blend(oof: pd.DataFrame, lgb_column: str) -> tuple[pd.DataFrame, dict[str, float]]:
    cat = oof["prediction_catboost_anchor"].to_numpy(dtype=float)
    lgb_prediction = oof[lgb_column].to_numpy(dtype=float)
    truth = oof["tma_mdpl"].to_numpy(dtype=float)
    delta = lgb_prediction - cat
    denominator = float(np.dot(delta, delta))
    pooled_weight = 0.0 if denominator == 0 else float(np.clip(np.dot(truth - cat, delta) / denominator, 0, 1))

    rows: list[dict[str, float]] = []
    cat_fold_rmse = {
        fold: rmse(group["tma_mdpl"], group["prediction_catboost_anchor"])
        for fold, group in oof.groupby("fold", sort=False)
    }
    for lgb_weight in np.linspace(0.0, 1.0, 1001):
        prediction = cat + lgb_weight * delta
        candidate = oof[["fold", "tma_mdpl"]].copy()
        candidate["prediction"] = prediction
        scores = {
            fold: rmse(group["tma_mdpl"], group["prediction"])
            for fold, group in candidate.groupby("fold", sort=False)
        }
        rows.append(
            {
                "lightgbm_weight": float(lgb_weight),
                "catboost_weight": float(1.0 - lgb_weight),
                "mean_fold_rmse": float(np.mean(list(scores.values()))),
                "pooled_rmse": rmse(truth, prediction),
                "worst_fold_delta": float(max(scores[f] - cat_fold_rmse[f] for f in scores)),
                **{f"rmse_{fold}": value for fold, value in scores.items()},
            }
        )
    grid = pd.DataFrame(rows)
    conservative = grid[grid["worst_fold_delta"] <= 1e-12]
    selected = (conservative if not conservative.empty else grid).sort_values(
        ["mean_fold_rmse", "pooled_rmse"]
    ).iloc[0].to_dict()
    selected["pooled_optimal_lightgbm_weight"] = pooled_weight
    return grid, selected


def validate(
    train: pd.DataFrame,
    feature_columns: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    parts: list[pd.DataFrame] = []
    best_iterations: dict[str, list[int]] = {variant: [] for variant in VARIANTS}
    for fold, train_end, valid_start, valid_end in FOLDS:
        fit = train[train["datetime"] <= pd.Timestamp(train_end)].copy()
        valid = train[
            train["datetime"].between(pd.Timestamp(valid_start), pd.Timestamp(valid_end), inclusive="both")
        ].copy()
        eligible = fit.groupby("nama_pos", observed=True).size().loc[lambda count: count >= MIN_LABELS_BEFORE_CUTOFF].index
        fit = fit[fit["nama_pos"].isin(eligible)]
        valid = valid[valid["nama_pos"].isin(eligible)]
        lookup = seasonal_lookup(fit)
        fit_baseline = apply_seasonal_lookup(fit, lookup)
        valid_baseline = apply_seasonal_lookup(valid, lookup)
        scale = residual_scales(fit["nama_pos"], fit["tma_mdpl"].to_numpy() - fit_baseline)
        weights, spike_count = isolated_spike_weights(fit)
        fold_oof = valid[["datetime", "nama_pos", "tma_mdpl"]].copy()
        fold_oof.insert(0, "fold", fold)
        fold_oof["prediction_seasonal"] = valid_baseline
        print(f"[{fold}] fit={len(fit):,} valid={len(valid):,} spikes={spike_count}", flush=True)
        for variant in VARIANTS:
            model = make_model()
            y_fit = target_for_variant(variant, fit, fit_baseline, scale)
            y_valid = target_for_variant(variant, valid, valid_baseline, scale)
            model.fit(
                fit[feature_columns],
                y_fit,
                sample_weight=weights,
                categorical_feature=categorical,
                eval_set=[(valid[feature_columns], y_valid)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            prediction = prediction_for_variant(
                variant, model.predict(valid[feature_columns]), valid, valid_baseline, scale
            )
            fold_oof[f"prediction_{variant}"] = prediction
            iteration = int(model.best_iteration_)
            best_iterations[variant].append(iteration)
            print(f"  {variant}: RMSE={rmse(valid['tma_mdpl'], prediction):.6f} iter={iteration}", flush=True)
        parts.append(fold_oof)
        pd.concat(parts, ignore_index=True).to_parquet(OUTPUT_DIR / "oof_lightgbm.parquet", index=False)
    final_iterations = {
        variant: int(np.median(iterations)) for variant, iterations in best_iterations.items()
    }
    oof = pd.concat(parts, ignore_index=True)
    metrics = metric_rows(oof, ["prediction_seasonal", *[f"prediction_{v}" for v in VARIANTS]])
    return oof, metrics, final_iterations


def train_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    categorical: list[str],
    variant: str,
    iterations: int,
) -> np.ndarray:
    lookup = seasonal_lookup(train)
    train_baseline = apply_seasonal_lookup(train, lookup)
    test_baseline = apply_seasonal_lookup(test, lookup)
    scale = residual_scales(train["nama_pos"], train["tma_mdpl"].to_numpy() - train_baseline)
    weights, spike_count = isolated_spike_weights(train)
    target = target_for_variant(variant, train, train_baseline, scale)
    model = make_model().set_params(n_estimators=iterations)
    print(f"[final] variant={variant} iterations={iterations} spikes={spike_count}", flush=True)
    model.fit(
        train[feature_columns], target, sample_weight=weights, categorical_feature=categorical
    )
    model.booster_.save_model(str(OUTPUT_DIR / f"lightgbm_{variant}.txt"))
    return prediction_for_variant(
        variant, model.predict(test[feature_columns]), test, test_baseline, scale
    )


def write_submission(path: Path, ids: pd.Series, prediction: np.ndarray) -> None:
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    predicted = pd.DataFrame({"id": ids, "tma_mdpl": prediction})
    submission = sample[["id"]].merge(predicted, on="id", how="left", validate="one_to_one")
    if submission["tma_mdpl"].isna().any() or not np.isfinite(submission["tma_mdpl"]).all():
        raise ValueError(f"Invalid prediction in {path}")
    submission.to_csv(path, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    required = [TRAIN_PATH, TEST_PATH, FEATURE_METADATA_PATH, CATBOOST_OOF_PATH, CATBOOST_SUBMISSION_PATH, SAMPLE_SUBMISSION_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    metadata = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    feature_columns = metadata["features"]
    categorical = metadata["categorical"]
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    train, test = prepare_categories(train, test, categorical)

    lgb_oof, metrics, final_iterations = validate(train, feature_columns, categorical)
    cat_oof = catboost_anchor_oof()
    oof = cat_oof.merge(
        lgb_oof,
        on=["fold", "datetime", "nama_pos", "tma_mdpl"],
        how="inner",
        validate="one_to_one",
    )
    if len(oof) != len(cat_oof) or len(oof) != len(lgb_oof):
        raise ValueError("CatBoost and LightGBM OOF rows do not align exactly")
    all_metrics = metric_rows(
        oof,
        ["prediction_catboost_anchor", "prediction_seasonal", *[f"prediction_{v}" for v in VARIANTS]],
    )
    variant_mean = (
        all_metrics[all_metrics["model"].isin(VARIANTS) & (all_metrics["fold"] != "pooled")]
        .groupby("model")["rmse"].mean()
        .sort_values()
    )
    selected_variant = str(variant_mean.index[0])
    grid, blend = tune_blend(oof, f"prediction_{selected_variant}")
    lgb_weight = float(blend["lightgbm_weight"])
    oof["prediction_selected_blend"] = (
        (1.0 - lgb_weight) * oof["prediction_catboost_anchor"]
        + lgb_weight * oof[f"prediction_{selected_variant}"]
    )
    all_metrics = metric_rows(
        oof,
        ["prediction_catboost_anchor", "prediction_seasonal", *[f"prediction_{v}" for v in VARIANTS], "prediction_selected_blend"],
    )
    all_metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    oof.to_parquet(OUTPUT_DIR / "oof_predictions.parquet", index=False)
    grid.to_csv(OUTPUT_DIR / "blend_grid.csv", index=False)

    lgb_test = train_final(
        train, test, feature_columns, categorical, selected_variant, final_iterations[selected_variant]
    )
    write_submission(OUTPUT_DIR / "submission_lightgbm_residual.csv", test["id"], lgb_test)
    cat_submission = pd.read_csv(CATBOOST_SUBMISSION_PATH)
    cat_by_id = test[["id"]].merge(cat_submission, on="id", how="left", validate="one_to_one")["tma_mdpl"].to_numpy()
    blend_test = (1.0 - lgb_weight) * cat_by_id + lgb_weight * lgb_test
    write_submission(OUTPUT_DIR / "submission_catboost_lightgbm_blend.csv", test["id"], blend_test)
    pooled_lgb_weight = float(blend["pooled_optimal_lightgbm_weight"])
    pooled_blend_test = (
        (1.0 - pooled_lgb_weight) * cat_by_id + pooled_lgb_weight * lgb_test
    )
    write_submission(
        OUTPUT_DIR / "submission_catboost_lightgbm_pooled_blend.csv",
        test["id"],
        pooled_blend_test,
    )

    summary = {
        "selected_variant": selected_variant,
        "final_iterations": final_iterations,
        "selected_blend": blend,
        "validation_metrics": all_metrics.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(all_metrics.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
