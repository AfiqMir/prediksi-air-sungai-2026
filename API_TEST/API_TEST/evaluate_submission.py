#!/usr/bin/env python3
"""
evaluate_submission.py

Membandingkan submission dengan nilai manual API, lalu memperkirakan
RMSE kompetisi menggunakan regresi yang dikalibrasi dari 7 submission.

Pemakaian paling singkat:
    python evaluate_submission.py submission_gnn_ibrahim.csv

Script otomatis:
- mencari api_manual_reference_clean.csv;
- membuat folder output berdasarkan nama submission;
- menghitung RMSE, MAE, bias, median error, dan max error;
- menghitung metrik per jam, per pos, dan per tanggal;
- memperkirakan RMSE kompetisi.

Pemakaian opsional:
    python evaluate_submission.py submission.csv answer_key.csv

    python evaluate_submission.py submission.csv \
        --answer-key answer_key.csv \
        --output-dir hasil_evaluasi

Format submission:
    id,tma_mdpl

Format kunci yang didukung:
1. id + api_tma_mdpl
2. id + tma_mdpl
3. datetime + station_key + api_tma_mdpl
4. datetime + nama_pos + api_tma_mdpl
5. datetime + api_station_name + api_tma_mdpl
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# REGRESI RMSE API -> RMSE KOMPETISI
#
# Berdasarkan 7 submission:
# y^2 = intercept + slope * x^2
#
# x = RMSE API
# y = estimasi RMSE kompetisi
# ============================================================

REGRESSION_INTERCEPT = 1.8721837737
REGRESSION_SLOPE = 0.9327343204
REGRESSION_SAMPLE_COUNT = 7
REGRESSION_RESIDUAL_RMSE = 0.01482


DEFAULT_ANSWER_KEY_NAMES = (
    "api_manual_reference_clean.csv",
    "api_manual_reference.csv",
)


# ============================================================
# UTILITAS
# ============================================================

def normalize_station_name(value: object) -> str:
    """Menormalkan nama pos untuk pencocokan yang lebih tahan format."""
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value).strip())
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = text.casefold()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_submission_id(series: pd.Series) -> pd.DataFrame:
    """
    Memecah ID berbentuk:
        YYYY-MM-DD HH:MM:SS - Nama Pos
    """
    ids = series.astype(str)

    datetime_values = pd.to_datetime(
        ids.str[:19],
        errors="coerce",
    )

    station_names = (
        ids.str[19:]
        .str.replace(r"^[\s_\-|:;]+", "", regex=True)
        .str.strip()
    )

    return pd.DataFrame(
        {
            "datetime": datetime_values,
            "nama_pos": station_names,
            "station_key": station_names.map(normalize_station_name),
        }
    )


def estimate_competition_rmse(api_rmse: float) -> float:
    """
    Mengestimasi RMSE kompetisi dari RMSE API.

    Formula:
        sqrt(1.8721837737 + 0.9327343204 * api_rmse^2)
    """
    if not math.isfinite(api_rmse) or api_rmse < 0:
        return math.nan

    value_inside_sqrt = (
        REGRESSION_INTERCEPT
        + REGRESSION_SLOPE * api_rmse**2
    )

    if value_inside_sqrt < 0:
        return math.nan

    return math.sqrt(value_inside_sqrt)


def make_output_dir_name(submission_path: Path) -> str:
    """
    Contoh:
        submission_gnn_ibrahim.csv -> evaluation_gnn_ibrahim
        submission_cb.csv          -> evaluation_cb
    """
    stem = submission_path.stem.strip()
    lower_stem = stem.casefold()

    if lower_stem.startswith("submission_"):
        stem = stem[len("submission_"):]
    elif lower_stem.startswith("submission"):
        remainder = stem[len("submission"):].lstrip("_- ")
        if remainder:
            stem = remainder

    stem = re.sub(r"[^\w.-]+", "_", stem).strip("_.-")
    if not stem:
        stem = "submission"

    return f"evaluation_{stem}"


def find_default_answer_key(
    submission_path: Path,
    script_path: Path,
) -> Path:
    """
    Mencari kunci secara berurutan di:
    1. folder submission;
    2. folder script;
    3. current working directory;
    4. subfolder api_review pada lokasi-lokasi tersebut.
    """
    base_directories = [
        submission_path.resolve().parent,
        script_path.resolve().parent,
        Path.cwd().resolve(),
    ]

    candidates: list[Path] = []
    seen: set[Path] = set()

    for base_dir in base_directories:
        for filename in DEFAULT_ANSWER_KEY_NAMES:
            for candidate in (
                base_dir / filename,
                base_dir / "api_review" / filename,
            ):
                resolved = candidate.resolve()
                if resolved not in seen:
                    candidates.append(resolved)
                    seen.add(resolved)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Kunci jawaban tidak ditemukan otomatis.\n"
        "Letakkan api_manual_reference_clean.csv di folder yang sama "
        "dengan script/submission, atau berikan path secara manual.\n"
        f"Lokasi yang sudah diperiksa:\n{checked}"
    )


# ============================================================
# MEMBACA DATA
# ============================================================

def read_submission(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"File submission tidak ditemukan: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {"id", "tma_mdpl"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Submission harus memiliki kolom {sorted(required_columns)}. "
            f"Kolom yang hilang: {sorted(missing_columns)}"
        )

    parsed = parse_submission_id(df["id"])

    if parsed["datetime"].isna().any():
        examples = (
            df.loc[parsed["datetime"].isna(), "id"]
            .head(5)
            .tolist()
        )
        raise ValueError(
            "Ada ID submission dengan timestamp tidak valid. "
            f"Contoh: {examples}"
        )

    result = df[["id", "tma_mdpl"]].copy()
    result["tma_mdpl"] = pd.to_numeric(
        result["tma_mdpl"],
        errors="coerce",
    )
    result = pd.concat([result, parsed], axis=1)

    if result["tma_mdpl"].isna().any():
        examples = (
            result.loc[result["tma_mdpl"].isna(), "id"]
            .head(5)
            .tolist()
        )
        raise ValueError(
            "Ada nilai tma_mdpl submission yang kosong atau bukan angka. "
            f"Contoh ID: {examples}"
        )

    result["date"] = result["datetime"].dt.date
    result["hour"] = result["datetime"].dt.hour

    duplicate_mask = result.duplicated(
        subset=["datetime", "station_key"],
        keep=False,
    )

    if duplicate_mask.any():
        examples = result.loc[
            duplicate_mask,
            ["id", "datetime", "nama_pos"],
        ].head(10)

        raise ValueError(
            "Submission memiliki kombinasi datetime + pos duplikat:\n"
            + examples.to_string(index=False)
        )

    return result


def choose_target_column(df: pd.DataFrame) -> str:
    candidates = (
        "api_tma_mdpl",
        "target",
        "answer",
        "label",
        "actual",
        "actual_tma_mdpl",
        "tma_mdpl",
    )

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Tidak menemukan kolom nilai jawaban. Gunakan salah satu: "
        "api_tma_mdpl, target, answer, label, actual, "
        "actual_tma_mdpl, atau tma_mdpl."
    )


def read_answer_key(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"File kunci jawaban tidak ditemukan: {path}"
        )

    df = pd.read_csv(path)
    target_column = choose_target_column(df)

    # Format dengan ID kompetisi.
    if "id" in df.columns:
        parsed = parse_submission_id(df["id"])

        key = pd.concat(
            [
                df[["id", target_column]].rename(
                    columns={target_column: "actual_tma_mdpl"}
                ),
                parsed,
            ],
            axis=1,
        )

    # Format api_manual_reference.csv.
    elif "datetime" in df.columns:
        key = pd.DataFrame()
        key["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
        )

        if "station_key" in df.columns:
            station_source = df["station_key"]
        elif "nama_pos" in df.columns:
            station_source = df["nama_pos"]
        elif "api_station_name" in df.columns:
            station_source = df["api_station_name"]
        else:
            raise ValueError(
                "File referensi harus memiliki salah satu kolom nama pos: "
                "station_key, nama_pos, atau api_station_name."
            )

        key["station_key"] = (
            station_source.astype(str).map(normalize_station_name)
        )
        key["actual_tma_mdpl"] = pd.to_numeric(
            df[target_column],
            errors="coerce",
        )

    else:
        raise ValueError(
            "Format kunci tidak dikenali. Format yang didukung:\n"
            "  - id + tma_mdpl/api_tma_mdpl\n"
            "  - datetime + station_key/nama_pos/api_station_name "
            "+ api_tma_mdpl"
        )

    key["actual_tma_mdpl"] = pd.to_numeric(
        key["actual_tma_mdpl"],
        errors="coerce",
    )

    key = key.dropna(
        subset=["datetime", "station_key", "actual_tma_mdpl"]
    ).copy()

    duplicate_mask = key.duplicated(
        subset=["datetime", "station_key"],
        keep=False,
    )

    if duplicate_mask.any():
        examples = key.loc[
            duplicate_mask,
            ["datetime", "station_key", "actual_tma_mdpl"],
        ].head(10)

        raise ValueError(
            "Kunci memiliki kombinasi datetime + pos duplikat:\n"
            + examples.to_string(index=False)
        )

    return key[
        ["datetime", "station_key", "actual_tma_mdpl"]
    ].reset_index(drop=True)


# ============================================================
# METRIK
# ============================================================

def metrics_from_errors(errors: pd.Series) -> dict[str, float]:
    errors = pd.to_numeric(errors, errors="coerce").dropna()

    if errors.empty:
        return {
            "n": 0,
            "rmse": math.nan,
            "mae": math.nan,
            "bias": math.nan,
            "median_abs_error": math.nan,
            "max_abs_error": math.nan,
        }

    absolute_errors = errors.abs()

    return {
        "n": int(len(errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(absolute_errors)),
        "bias": float(np.mean(errors)),
        "median_abs_error": float(np.median(absolute_errors)),
        "max_abs_error": float(np.max(absolute_errors)),
    }


def grouped_metrics(
    df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    grouper: object
    if len(group_columns) == 1:
        grouper = group_columns[0]
    else:
        grouper = group_columns

    for keys, group in df.groupby(grouper, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_columns, keys))
        row.update(metrics_from_errors(group["error"]))
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# EVALUASI
# ============================================================

def evaluate(
    submission_path: Path,
    key_path: Path,
    output_dir: Path,
) -> None:
    submission = read_submission(submission_path)
    answer_key = read_answer_key(key_path)

    review = submission.merge(
        answer_key,
        on=["datetime", "station_key"],
        how="left",
        validate="one_to_one",
    )

    review["error"] = (
        review["tma_mdpl"] - review["actual_tma_mdpl"]
    )
    review["abs_error"] = review["error"].abs()
    review["squared_error"] = review["error"] ** 2

    matched = review.dropna(
        subset=["tma_mdpl", "actual_tma_mdpl"]
    ).copy()

    unmatched = review[
        review["actual_tma_mdpl"].isna()
    ].copy()

    if matched.empty:
        raise RuntimeError(
            "Tidak ada baris submission yang cocok dengan kunci jawaban."
        )

    overall = metrics_from_errors(matched["error"])
    coverage = len(matched) / len(submission)

    api_rmse = overall["rmse"]
    estimated_competition_rmse = estimate_competition_rmse(
        api_rmse
    )

    per_station = grouped_metrics(
        matched,
        ["nama_pos", "station_key"],
    ).sort_values(
        ["rmse", "n"],
        ascending=[False, False],
    )

    per_hour = grouped_metrics(
        matched,
        ["hour"],
    ).sort_values("hour")

    per_date = grouped_metrics(
        matched,
        ["date"],
    ).sort_values("date")

    output_dir.mkdir(parents=True, exist_ok=True)

    matched.to_csv(
        output_dir / "matched_rows.csv",
        index=False,
    )
    unmatched.to_csv(
        output_dir / "unmatched_submission_rows.csv",
        index=False,
    )
    per_station.to_csv(
        output_dir / "metrics_per_station.csv",
        index=False,
    )
    per_hour.to_csv(
        output_dir / "metrics_per_hour.csv",
        index=False,
    )
    per_date.to_csv(
        output_dir / "metrics_per_date.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "submission": str(submission_path),
                "answer_key": str(key_path),
                "total_rows": len(submission),
                "matched_rows": len(matched),
                "coverage": coverage,
                "api_rmse": api_rmse,
                "api_mae": overall["mae"],
                "api_bias": overall["bias"],
                "api_median_abs_error": (
                    overall["median_abs_error"]
                ),
                "api_max_abs_error": overall["max_abs_error"],
                "estimated_competition_rmse": (
                    estimated_competition_rmse
                ),
                "regression_intercept": REGRESSION_INTERCEPT,
                "regression_slope": REGRESSION_SLOPE,
                "regression_sample_count": (
                    REGRESSION_SAMPLE_COUNT
                ),
                "regression_residual_rmse": (
                    REGRESSION_RESIDUAL_RMSE
                ),
            }
        ]
    )

    summary.to_csv(
        output_dir / "evaluation_summary.csv",
        index=False,
    )

    print("\n=== HASIL EVALUASI ===")
    print(f"Submission              : {submission_path}")
    print(f"Kunci jawaban           : {key_path}")
    print(f"Total baris             : {len(submission):,}")
    print(f"Baris cocok             : {len(matched):,}")
    print(f"Coverage                : {coverage:.2%}")
    print(f"RMSE API                : {api_rmse:.6f} meter")
    print(
        "Estimasi RMSE kompetisi : "
        f"{estimated_competition_rmse:.6f}"
    )
    print(f"MAE API                 : {overall['mae']:.6f} meter")
    print(
        f"Bias prediksi           : "
        f"{overall['bias']:+.6f} meter"
    )
    print(
        "Median abs error        : "
        f"{overall['median_abs_error']:.6f} meter"
    )
    print(
        "Max abs error           : "
        f"{overall['max_abs_error']:.6f} meter"
    )

    print("\n=== FORMULA ESTIMASI ===")
    print(
        "RMSE kompetisi ≈ sqrt("
        f"{REGRESSION_INTERCEPT:.10f} + "
        f"{REGRESSION_SLOPE:.10f} × RMSE_API²)"
    )
    print(
        f"Kalibrasi: {REGRESSION_SAMPLE_COUNT} submission; "
        f"residual RMSE ≈ {REGRESSION_RESIDUAL_RMSE:.5f}"
    )

    print("\n=== PER JAM ===")
    print(
        per_hour[
            ["hour", "n", "rmse", "mae", "bias"]
        ].to_string(index=False)
    )

    print("\n=== 10 POS DENGAN RMSE TERTINGGI ===")
    print(
        per_station[
            ["nama_pos", "n", "rmse", "mae", "bias"]
        ].head(10).to_string(index=False)
    )

    if not unmatched.empty:
        print(
            f"\nPeringatan: {len(unmatched):,} baris tidak memiliki "
            "jawaban yang cocok."
        )

    print(
        f"\nFile detail disimpan di: {output_dir.resolve()}"
    )


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluasi submission tma_mdpl menggunakan kunci API "
            "dan estimasi RMSE kompetisi."
        )
    )

    parser.add_argument(
        "submission",
        type=Path,
        help="Path ke submission CSV.",
    )

    # Tetap mendukung cara lama:
    # python evaluate_submission.py submission.csv answer_key.csv
    parser.add_argument(
        "answer_key_positional",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Opsional: path kunci sebagai argumen posisi kedua."
        ),
    )

    parser.add_argument(
        "--answer-key",
        type=Path,
        default=None,
        help=(
            "Opsional: path kunci. Jika tidak diberikan, script "
            "mencari api_manual_reference_clean.csv otomatis."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Folder output. Jika tidak diberikan, dibuat otomatis "
            "dari nama submission."
        ),
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    submission_path = args.submission.expanduser()

    try:
        if args.answer_key is not None:
            key_path = args.answer_key.expanduser()
        elif args.answer_key_positional is not None:
            key_path = args.answer_key_positional.expanduser()
        else:
            key_path = find_default_answer_key(
                submission_path=submission_path,
                script_path=Path(__file__),
            )

        if args.output_dir is not None:
            output_dir = args.output_dir.expanduser()
        else:
            output_dir = (
                submission_path.resolve().parent
                / make_output_dir_name(submission_path)
            )

        evaluate(
            submission_path=submission_path,
            key_path=key_path,
            output_dir=output_dir,
        )

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    except Exception as exc:
        print(
            f"ERROR TAK TERDUGA: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
