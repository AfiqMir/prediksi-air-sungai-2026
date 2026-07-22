"""Leakage-safe ExtraTrees residual expert and station-stack blend.

ExtraTrees is trained directly on residuals from a train-only station/month/hour
seasonal baseline. Two predeclared regularisation settings are compared on the
four rolling-origin folds. Model choice and blend weight use OOF only, while an
outer leave-one-fold-out audit tests the whole choice procedure. API_TEST is
never read by this script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from run_lightgbm_residual_experiment import (
    apply_seasonal_lookup,
    isolated_spike_weights,
    rmse,
    seasonal_lookup,
)
from run_station_aware_experiment import FOLDS, write_submission


FEATURE_DIR = Path("output_catboost_experiments")
STATION_DIR = Path("output_station_aware")
OUTPUT_DIR = Path("output_extratrees_residual")
TRAIN_PATH = FEATURE_DIR / "train_features.parquet"
TEST_PATH = FEATURE_DIR / "test_features.parquet"
METADATA_PATH = FEATURE_DIR / "feature_metadata.json"
STATION_OOF_PATH = STATION_DIR / "oof_predictions.parquet"
STATION_SUBMISSION_PATH = STATION_DIR / "submission_station_stack.csv"
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv")

MIN_LABELS_BEFORE_CUTOFF = 90
MAX_EXPERT_WEIGHT = 0.30
WEIGHT_STEP = 0.01
MAX_WORST_FOLD_DELTA = 0.005
RANDOM_STATE = 2026


@dataclass(frozen=True)
class Spec:
    name: str
    min_samples_leaf: int
    max_features: float


# Fixed before validation: the first favours a diverse high-variance expert;
# the second is deliberately smoother to resist station-specific outliers.
SPECS = (
    Spec("diverse_leaf_3", min_samples_leaf=3, max_features=0.70),
    Spec("smooth_leaf_10", min_samples_leaf=10, max_features=1.00),
)


def make_preprocessor(
    numeric: list[str], categorical: list[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def clean_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame[columns].copy()
    for column in result:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)
        else:
            result[column] = result[column].astype("string").fillna("__MISSING__")
    return result


def make_model(spec: Spec) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        # 80 trees keeps the four-fold, two-spec audit practical on this CPU
        # while retaining ExtraTrees' deliberately diverse random partitions.
        n_estimators=80,
        max_features=spec.max_features,
        min_samples_leaf=spec.min_samples_leaf,
        max_depth=None,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def metric_by_fold(
    frame: pd.DataFrame, prediction: np.ndarray
) -> dict[str, float]:
    work = frame[["fold", "tma_mdpl"]].copy()
    work["prediction"] = prediction
    return {
        fold: rmse(group["tma_mdpl"], group["prediction"])
        for fold, group in work.groupby("fold", sort=False)
    }


def prediction_metrics(
    frame: pd.DataFrame, column: str
) -> list[dict[str, object]]:
    scores = metric_by_fold(frame, frame[column].to_numpy(dtype=float))
    return [
        *(
            {"model": column.removeprefix("prediction_"), "fold": fold, "rmse": score}
            for fold, score in scores.items()
        ),
        {
            "model": column.removeprefix("prediction_"),
            "fold": "mean_fold_rmse",
            "rmse": float(np.mean(list(scores.values()))),
        },
        {
            "model": column.removeprefix("prediction_"),
            "fold": "pooled_rmse",
            "rmse": rmse(frame["tma_mdpl"], frame[column]),
        },
    ]


def validate(
    train: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    metrics_path = OUTPUT_DIR / "model_metrics.csv"
    rows: list[dict[str, object]] = (
        pd.read_csv(metrics_path).to_dict(orient="records")
        if metrics_path.exists()
        else []
    )
    cache_path = OUTPUT_DIR / "oof_extratrees.parquet"
    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    for fold, train_end, valid_start, valid_end in FOLDS:
        fit = train[train["datetime"] <= pd.Timestamp(train_end)].copy()
        valid = train[
            train["datetime"].between(
                pd.Timestamp(valid_start), pd.Timestamp(valid_end), inclusive="both"
            )
        ].copy()
        eligible = (
            fit.groupby("nama_pos", observed=True)
            .size()
            .loc[lambda count: count >= MIN_LABELS_BEFORE_CUTOFF]
            .index
        )
        fit = fit[fit["nama_pos"].isin(eligible)]
        valid = valid[valid["nama_pos"].isin(eligible)]
        expected_columns = {
            "fold",
            "datetime",
            "nama_pos",
            "tma_mdpl",
            "prediction_seasonal",
            *(f"prediction_extratrees_{spec.name}" for spec in SPECS),
        }
        cached_fold = cached[cached["fold"] == fold] if "fold" in cached else pd.DataFrame()
        if (
            not cached_fold.empty
            and expected_columns.issubset(cached_fold.columns)
            and len(cached_fold) == len(valid)
        ):
            parts.append(cached_fold[list(expected_columns)].copy())
            print(f"[{fold}] reusing cached ExtraTrees OOF", flush=True)
            continue
        lookup = seasonal_lookup(fit)
        fit_baseline = apply_seasonal_lookup(fit, lookup)
        valid_baseline = apply_seasonal_lookup(valid, lookup)
        target = fit["tma_mdpl"].to_numpy(dtype=float) - fit_baseline
        weights, spike_count = isolated_spike_weights(fit)
        preprocessor = make_preprocessor(numeric, categorical)
        x_fit = preprocessor.fit_transform(clean_features(fit, features))
        x_valid = preprocessor.transform(clean_features(valid, features))
        fold_oof = valid[["datetime", "nama_pos", "tma_mdpl"]].copy()
        fold_oof.insert(0, "fold", fold)
        fold_oof["prediction_seasonal"] = valid_baseline
        print(
            f"[{fold}] fit={len(fit):,} valid={len(valid):,} "
            f"encoded_features={x_fit.shape[1]:,} spikes={spike_count}",
            flush=True,
        )
        for spec in SPECS:
            model = make_model(spec)
            model.fit(x_fit, target, sample_weight=weights)
            prediction = valid_baseline + model.predict(x_valid)
            column = f"prediction_extratrees_{spec.name}"
            fold_oof[column] = prediction
            score = rmse(valid["tma_mdpl"], prediction)
            rows.append(
                {
                    "fold": fold,
                    "spec": spec.name,
                    "fit_rows": len(fit),
                    "validation_rows": len(valid),
                    "encoded_features": int(x_fit.shape[1]),
                    "spikes_downweighted": spike_count,
                    "rmse": score,
                }
            )
            print(f"  {spec.name}: RMSE={score:.6f}", flush=True)
        parts.append(fold_oof)
        pd.concat(parts, ignore_index=True).to_parquet(
            OUTPUT_DIR / "oof_extratrees.parquet", index=False
        )
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(rows)


def select_spec(frame: pd.DataFrame, folds: set[str] | None = None) -> str:
    work = frame if folds is None else frame[frame["fold"].isin(folds)]
    scores: list[tuple[float, float, str]] = []
    for spec in SPECS:
        column = f"prediction_extratrees_{spec.name}"
        by_fold = metric_by_fold(work, work[column].to_numpy(dtype=float))
        scores.append(
            (
                float(np.mean(list(by_fold.values()))),
                rmse(work["tma_mdpl"], work[column]),
                spec.name,
            )
        )
    return min(scores)[2]


def select_blend(
    frame: pd.DataFrame,
    spec_name: str,
    folds: set[str] | None = None,
) -> dict[str, float]:
    work = frame if folds is None else frame[frame["fold"].isin(folds)]
    control = work["prediction_station_stack"].to_numpy(dtype=float)
    expert = work[f"prediction_extratrees_{spec_name}"].to_numpy(dtype=float)
    truth = work["tma_mdpl"].to_numpy(dtype=float)
    control_scores = metric_by_fold(work, control)
    candidates: list[dict[str, float]] = []
    for weight in np.arange(0.0, MAX_EXPERT_WEIGHT + WEIGHT_STEP / 2, WEIGHT_STEP):
        prediction = (1.0 - weight) * control + weight * expert
        scores = metric_by_fold(work, prediction)
        candidates.append(
            {
                "expert_weight": float(weight),
                "station_stack_weight": float(1.0 - weight),
                "mean_fold_rmse": float(np.mean(list(scores.values()))),
                "pooled_rmse": rmse(truth, prediction),
                "worst_fold_delta": float(
                    max(scores[fold] - control_scores[fold] for fold in scores)
                ),
                **{f"rmse_{fold}": score for fold, score in scores.items()},
            }
        )
    grid = pd.DataFrame(candidates)
    eligible = grid[grid["worst_fold_delta"] <= MAX_WORST_FOLD_DELTA]
    chosen = (eligible if not eligible.empty else grid).sort_values(
        ["mean_fold_rmse", "pooled_rmse"]
    ).iloc[0]
    return chosen.to_dict() | {"grid": grid}


def leave_one_fold_out_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_folds = list(frame["fold"].drop_duplicates())
    for held_out in all_folds:
        training_folds = set(all_folds) - {held_out}
        spec_name = select_spec(frame, training_folds)
        selected = select_blend(frame, spec_name, training_folds)
        held = frame[frame["fold"] == held_out]
        prediction = (
            float(selected["station_stack_weight"])
            * held["prediction_station_stack"].to_numpy(dtype=float)
            + float(selected["expert_weight"])
            * held[f"prediction_extratrees_{spec_name}"].to_numpy(dtype=float)
        )
        control = held["prediction_station_stack"].to_numpy(dtype=float)
        rows.append(
            {
                "held_out_fold": held_out,
                "selected_spec": spec_name,
                "expert_weight": float(selected["expert_weight"]),
                "held_out_rmse": rmse(held["tma_mdpl"], prediction),
                "control_rmse": rmse(held["tma_mdpl"], control),
            }
        )
    audit = pd.DataFrame(rows)
    audit["delta_vs_control"] = audit["held_out_rmse"] - audit["control_rmse"]
    return audit


def train_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    spec_name: str,
) -> np.ndarray:
    spec = next(spec for spec in SPECS if spec.name == spec_name)
    lookup = seasonal_lookup(train)
    baseline = apply_seasonal_lookup(train, lookup)
    test_baseline = apply_seasonal_lookup(test, lookup)
    target = train["tma_mdpl"].to_numpy(dtype=float) - baseline
    weights, spike_count = isolated_spike_weights(train)
    preprocessor = make_preprocessor(numeric, categorical)
    x_train = preprocessor.fit_transform(clean_features(train, features))
    x_test = preprocessor.transform(clean_features(test, features))
    model = make_model(spec)
    print(
        f"[final] {spec.name}: train={len(train):,} encoded_features={x_train.shape[1]:,} "
        f"spikes={spike_count}",
        flush=True,
    )
    model.fit(x_train, target, sample_weight=weights)
    return test_baseline + model.predict(x_test)


def main() -> None:
    required = [
        TRAIN_PATH,
        TEST_PATH,
        METADATA_PATH,
        STATION_OOF_PATH,
        STATION_SUBMISSION_PATH,
        SAMPLE_SUBMISSION_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    features = list(metadata["features"])
    categorical = [column for column in metadata["categorical"] if column in features]
    numeric = [column for column in features if column not in categorical]
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    for frame in (train, test):
        frame["datetime"] = pd.to_datetime(frame["datetime"])

    extratrees_oof, model_metrics = validate(train, features, numeric, categorical)
    control = pd.read_parquet(STATION_OOF_PATH)[
        ["fold", "datetime", "nama_pos", "tma_mdpl", "prediction_station_stack"]
    ]
    oof = control.merge(
        extratrees_oof,
        on=["fold", "datetime", "nama_pos", "tma_mdpl"],
        validate="one_to_one",
    )
    if len(oof) != len(control):
        raise ValueError("ExtraTrees OOF does not align with station-stack OOF")

    selected_spec = select_spec(oof)
    selected_blend = select_blend(oof, selected_spec)
    blend_grid = selected_blend.pop("grid")
    audit = leave_one_fold_out_audit(oof)
    oof["prediction_extratrees_selected"] = oof[
        f"prediction_extratrees_{selected_spec}"
    ]
    oof["prediction_extratrees_blend"] = (
        float(selected_blend["station_stack_weight"])
        * oof["prediction_station_stack"]
        + float(selected_blend["expert_weight"])
        * oof["prediction_extratrees_selected"]
    )
    metrics = pd.DataFrame(
        prediction_metrics(oof, "prediction_station_stack")
        + prediction_metrics(oof, "prediction_extratrees_selected")
        + prediction_metrics(oof, "prediction_extratrees_blend")
    )
    score_pivot = metrics.pivot(index="model", columns="fold", values="rmse")
    audit_pass = bool(
        score_pivot.loc["extratrees_blend", "mean_fold_rmse"]
        < score_pivot.loc["station_stack", "mean_fold_rmse"]
        and score_pivot.loc["extratrees_blend", "pooled_rmse"]
        < score_pivot.loc["station_stack", "pooled_rmse"]
        and float(audit["delta_vs_control"].max()) <= MAX_WORST_FOLD_DELTA
        and float(selected_blend["expert_weight"]) > 0.0
    )

    test_expert = train_final(train, test, features, numeric, categorical, selected_spec)
    stack_submission = pd.read_csv(STATION_SUBMISSION_PATH)
    stack_prediction = test[["id"]].merge(
        stack_submission, on="id", validate="one_to_one"
    )["tma_mdpl"].to_numpy(dtype=float)
    test_blend = (
        float(selected_blend["station_stack_weight"]) * stack_prediction
        + float(selected_blend["expert_weight"]) * test_expert
    )
    write_submission(OUTPUT_DIR / "submission_extratrees_residual.csv", test["id"], test_expert)
    write_submission(
        OUTPUT_DIR / "submission_extratrees_station_stack_blend.csv",
        test["id"],
        test_blend,
    )

    oof.to_parquet(OUTPUT_DIR / "oof_predictions.parquet", index=False)
    model_metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    blend_grid.to_csv(OUTPUT_DIR / "blend_grid.csv", index=False)
    audit.to_csv(OUTPUT_DIR / "leave_one_fold_out_audit.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    summary = {
        "selection_protocol": {
            "specs": [spec.__dict__ for spec in SPECS],
            "max_expert_weight": MAX_EXPERT_WEIGHT,
            "max_worst_fold_delta": MAX_WORST_FOLD_DELTA,
        },
        "selected_spec": selected_spec,
        "selected_blend": selected_blend,
        "leave_one_fold_out": audit.to_dict(orient="records"),
        "pre_api_decision": {
            "api_test_evaluated": False,
            "audit_pass": audit_pass,
            "recommendation": (
                "frozen_candidate_eligible_for_one_way_api_verification"
                if audit_pass
                else "retain_station_stack_control"
            ),
        },
        "metrics": metrics.to_dict(orient="records"),
        "submission": str(OUTPUT_DIR / "submission_extratrees_station_stack_blend.csv"),
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
