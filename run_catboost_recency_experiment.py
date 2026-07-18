"""Recency-weighted CatBoost ensemble experiment.

The experiment keeps the existing feature table, target transform, CatBoost
configuration, folds, and station-aware state correction.  Only the amount of
influence assigned to old observations changes.  Model and blend selection use
rolling-origin OOF predictions; API_TEST is intentionally not read here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from run_catboost_rmse_anchor_experiment import (
    MAX_ITERATIONS,
    best_selection_iterations,
    build_or_load_features,
    load_pipeline_namespace,
    original_rmse,
)


OUTPUT_DIR = Path("output_catboost_recency")
CAT_DIR = Path("output_catboost_experiments")
STATION_DIR = Path("output_station_aware")
CAT_OOF_PATH = CAT_DIR / "oof_predictions.parquet"
STATION_OOF_PATH = STATION_DIR / "oof_predictions.parquet"
RAW_SUBMISSION_PATH = CAT_DIR / "submission_rmse_selected.csv"
CONTROL_SUBMISSION_PATH = STATION_DIR / "submission_station_anchor.csv"
STATION_PARAMETERS_PATH = STATION_DIR / "station_anchor_parameters.csv"
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv")

VARIANTS: dict[str, dict[str, float | str]] = {
    "exp_180d": {"kind": "exponential", "days": 180.0},
    "exp_365d": {"kind": "exponential", "days": 365.0},
    "exp_730d": {"kind": "exponential", "days": 730.0},
    "recent_540d": {"kind": "window", "days": 540.0},
}
FINAL_SEEDS = (17, 41, 83)


def make_model(config, seed: int, iterations: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        loss_function=f"Huber:delta={config.huber_delta}",
        eval_metric="RMSE",
        l2_leaf_reg=config.l2_leaf_reg,
        random_seed=seed,
        random_strength=0.8,
        bootstrap_type="Bayesian",
        bagging_temperature=0.7,
        allow_writing_files=False,
        thread_count=-1,
        verbose=200,
    )


def apply_recency(
    fit: pd.DataFrame,
    specification: dict[str, float | str],
    base_weights: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    cutoff = fit["datetime"].max()
    age_days = (
        (cutoff - fit["datetime"]).dt.total_seconds() / 86_400.0
    ).to_numpy(dtype=float)
    days = float(specification["days"])
    if specification["kind"] == "window":
        keep = age_days <= days
        return fit.loc[keep].copy(), base_weights[keep]
    recency = np.exp(-np.log(2.0) * age_days / days)
    weights = base_weights * recency
    weights /= weights.mean()
    return fit.copy(), weights


def add_fixed_station_anchor(frame: pd.DataFrame) -> pd.DataFrame:
    parameters = pd.read_csv(STATION_PARAMETERS_PATH)
    alpha = parameters.set_index("nama_pos")["shrunk_alpha"]
    result = frame.copy()
    result["station_alpha"] = result["nama_pos"].map(alpha)
    if result["station_alpha"].isna().any():
        raise ValueError("Missing station anchor parameter")
    for variant in VARIANTS:
        result[f"prediction_{variant}_anchor"] = (
            result[f"prediction_{variant}"]
            + result["station_alpha"] * result["anchor_basis"]
        )
    return result


def validate(
    train: pd.DataFrame,
    columns: list[str],
    namespace: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = namespace["ModelConfig"]()
    categorical = [
        column
        for column in namespace["CATEGORICAL_COLUMNS"]
        if column in columns
    ]
    reference = pd.read_parquet(STATION_OOF_PATH)[
        [
            "fold",
            "datetime",
            "nama_pos",
            "tma_mdpl",
            "anchor_basis",
            "prediction_station_anchor",
        ]
    ].copy()
    metric_rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    model_dir = OUTPUT_DIR / "validation_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for fold_name, train_end, valid_start, valid_end in namespace["FOLDS"]:
        full_fit = train[train["datetime"] <= pd.Timestamp(train_end)].copy()
        valid = train[
            train["datetime"].between(
                pd.Timestamp(valid_start), pd.Timestamp(valid_end), inclusive="both"
            )
        ].copy()
        eligible = (
            full_fit.groupby("nama_pos", observed=True)
            .size()
            .loc[lambda count: count >= 90]
            .index
        )
        full_fit = full_fit[full_fit["nama_pos"].isin(eligible)]
        valid = valid[valid["nama_pos"].isin(eligible)]
        base_weights, spike_count = namespace["isolated_spike_weights"](full_fit)
        fold_prediction = valid[["datetime", "nama_pos"]].copy()
        fold_prediction.insert(0, "fold", fold_name)

        for variant, specification in VARIANTS.items():
            fit, weights = apply_recency(full_fit, specification, base_weights)
            stats = namespace["fit_target_stats"](fit)
            y_fit = namespace["normalize_target"](fit, stats)
            y_valid = namespace["normalize_target"](valid, stats)
            fit_pool = Pool(
                namespace["_xy"](fit, columns),
                y_fit,
                cat_features=categorical,
                weight=weights,
            )
            valid_pool = Pool(
                namespace["_xy"](valid, columns),
                y_valid,
                cat_features=categorical,
            )
            path = model_dir / f"{fold_name}_{variant}.cbm"
            model = make_model(config, config.seeds[0], MAX_ITERATIONS)
            print(
                f"\n[{fold_name}/{variant}] fit={len(fit):,}, "
                f"validation={len(valid):,}, effective_weight={weights.sum():,.1f}",
                flush=True,
            )
            if path.exists():
                model.load_model(str(path))
                print("Reusing cached model", flush=True)
            else:
                model.fit(fit_pool, eval_set=valid_pool, use_best_model=False)
                model.save_model(str(path))

            selection = best_selection_iterations(
                model,
                valid_pool,
                valid["tma_mdpl"].to_numpy(dtype=float),
                y_valid,
                valid["nama_pos"],
                stats,
                namespace,
            )
            iteration = selection["raw_rmse_selected"]
            prediction_z = model.predict(valid_pool, ntree_end=iteration)
            prediction = namespace["denormalize_prediction"](
                prediction_z, valid["nama_pos"], stats
            )
            fold_prediction[f"prediction_{variant}"] = prediction
            metric_rows.append(
                {
                    "variant": variant,
                    "fold": fold_name,
                    "fit_rows": len(fit),
                    "validation_rows": len(valid),
                    "spikes_downweighted": spike_count,
                    "iteration": iteration,
                    "raw_rmse": original_rmse(valid["tma_mdpl"], prediction),
                }
            )
            print(
                f"[{fold_name}/{variant}] raw RMSE="
                f"{metric_rows[-1]['raw_rmse']:.6f}, iteration={iteration}",
                flush=True,
            )

        predictions.append(fold_prediction)
        pd.concat(predictions, ignore_index=True).to_parquet(
            OUTPUT_DIR / "oof_recency_raw.parquet", index=False
        )
        pd.DataFrame(metric_rows).to_csv(
            OUTPUT_DIR / "tree_selection_metrics.csv", index=False
        )

    recency = pd.concat(predictions, ignore_index=True)
    merged = reference.merge(
        recency,
        on=["fold", "datetime", "nama_pos"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(reference):
        raise ValueError("Recency OOF does not align with control OOF")
    return add_fixed_station_anchor(merged), pd.DataFrame(metric_rows)


def fold_scores(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    work = frame[["fold", "tma_mdpl"]].copy()
    work["prediction"] = prediction
    return {
        fold: original_rmse(group["tma_mdpl"], group["prediction"])
        for fold, group in work.groupby("fold", sort=False)
    }


def select_candidate(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    truth = oof["tma_mdpl"].to_numpy(dtype=float)
    control = oof["prediction_station_anchor"].to_numpy(dtype=float)
    control_folds = fold_scores(oof, control)
    control_mean = float(np.mean(list(control_folds.values())))
    control_pooled = original_rmse(truth, control)
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        recency = oof[f"prediction_{variant}_anchor"].to_numpy(dtype=float)
        for weight in np.linspace(0.0, 1.0, 101):
            prediction = (1.0 - weight) * control + weight * recency
            scores = fold_scores(oof, prediction)
            deltas = {fold: scores[fold] - control_folds[fold] for fold in scores}
            rows.append(
                {
                    "variant": variant,
                    "recency_weight": float(weight),
                    "control_weight": float(1.0 - weight),
                    "mean_fold_rmse": float(np.mean(list(scores.values()))),
                    "pooled_rmse": original_rmse(truth, prediction),
                    "mean_fold_delta": float(np.mean(list(deltas.values()))),
                    "pooled_delta": original_rmse(truth, prediction) - control_pooled,
                    "worst_fold_delta": float(max(deltas.values())),
                    **{f"rmse_{fold}": score for fold, score in scores.items()},
                    **{f"delta_{fold}": delta for fold, delta in deltas.items()},
                }
            )
    grid = pd.DataFrame(rows)
    robust = grid[
        (grid["mean_fold_delta"] < 0.0)
        & (grid["pooled_delta"] < 0.0)
        & (grid["worst_fold_delta"] <= 0.01)
        & (grid["delta_sep_2023"] <= 0.01)
        & (grid["delta_sep_2024"] <= 0.01)
    ].sort_values(["mean_fold_rmse", "pooled_rmse", "worst_fold_delta"])
    if robust.empty:
        selected = grid[grid["recency_weight"] == 0.0].iloc[0]
        note = "No robust recency candidate; retained station-anchor control."
    else:
        selected = robust.iloc[0]
        note = "Selected from OOF under pooled, mean-fold, stress-fold constraints."
    result = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in selected.to_dict().items()
    }
    result["selection_note"] = note
    result["control_mean_fold_rmse"] = control_mean
    result["control_pooled_rmse"] = control_pooled
    return grid, result


def train_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    namespace: dict,
    metrics: pd.DataFrame,
    selection: dict[str, object],
) -> Path:
    variant = str(selection["variant"])
    recency_weight = float(selection["recency_weight"])
    output_path = OUTPUT_DIR / "submission_recency_blend.csv"
    if recency_weight == 0.0:
        control = pd.read_csv(CONTROL_SUBMISSION_PATH)
        control.to_csv(output_path, index=False)
        return output_path

    config = namespace["ModelConfig"]()
    categorical = [
        column
        for column in namespace["CATEGORICAL_COLUMNS"]
        if column in columns
    ]
    base_weights, _ = namespace["isolated_spike_weights"](train)
    fit, weights = apply_recency(train, VARIANTS[variant], base_weights)
    stats = namespace["fit_target_stats"](fit)
    y_fit = namespace["normalize_target"](fit, stats)
    train_pool = Pool(
        namespace["_xy"](fit, columns),
        y_fit,
        cat_features=categorical,
        weight=weights,
    )
    test_pool = Pool(namespace["_xy"](test, columns), cat_features=categorical)
    iterations = int(
        np.median(metrics.loc[metrics["variant"] == variant, "iteration"])
    )
    predictions = []
    model_dir = OUTPUT_DIR / "final_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for seed in FINAL_SEEDS:
        print(
            f"Training final {variant}, seed={seed}, iterations={iterations}",
            flush=True,
        )
        model = make_model(config, seed, iterations)
        model.fit(train_pool)
        model.save_model(str(model_dir / f"{variant}_seed_{seed}.cbm"))
        predictions.append(model.predict(test_pool))
    prediction_z = np.mean(predictions, axis=0)
    prediction = namespace["denormalize_prediction"](
        prediction_z, test["nama_pos"], stats
    )

    raw = pd.read_csv(RAW_SUBMISSION_PATH)
    control = pd.read_csv(CONTROL_SUBMISSION_PATH)
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    if not raw["id"].equals(sample["id"]) or not control["id"].equals(sample["id"]):
        raise ValueError("Cached submissions do not align with sample submission")
    station_correction = control["tma_mdpl"].to_numpy() - raw["tma_mdpl"].to_numpy()
    recency_anchor = prediction + station_correction
    blended = (
        (1.0 - recency_weight) * control["tma_mdpl"].to_numpy()
        + recency_weight * recency_anchor
    )
    if not np.isfinite(blended).all():
        raise ValueError("Non-finite final predictions")
    submission = pd.DataFrame({"id": sample["id"], "tma_mdpl": blended})
    submission.to_csv(output_path, index=False)
    pd.DataFrame({"id": sample["id"], "tma_mdpl": recency_anchor}).to_csv(
        OUTPUT_DIR / f"submission_{variant}_anchor.csv", index=False
    )
    selection["final_iterations"] = iterations
    selection["final_seeds"] = list(FINAL_SEEDS)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    namespace = load_pipeline_namespace()
    train, test, columns = build_or_load_features(namespace)
    oof, metrics = validate(train, columns, namespace)
    grid, selection = select_candidate(oof)
    weight = float(selection["recency_weight"])
    variant = str(selection["variant"])
    oof["prediction_selected"] = (
        (1.0 - weight) * oof["prediction_station_anchor"]
        + weight * oof[f"prediction_{variant}_anchor"]
    )
    oof.to_parquet(OUTPUT_DIR / "oof_predictions.parquet", index=False)
    grid.to_csv(OUTPUT_DIR / "blend_grid.csv", index=False)
    submission_path = train_final(
        train, test, columns, namespace, metrics, selection
    )
    selection["submission"] = str(submission_path)
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print("\nSelected candidate:", flush=True)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
