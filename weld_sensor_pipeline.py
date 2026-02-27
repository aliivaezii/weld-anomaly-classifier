
#!/usr/bin/env python3
"""
Reproducible sensor-only welding anomaly pipeline.

What this script does
---------------------
1) Recursively discovers welding-run CSV files from the dataset roots.
2) Validates structure and sensor schema.
3) Extracts one feature vector per run (time-series -> tabular features).
4) Builds a leakage-aware split grouped by configuration folder.
5) Fits a deterministic preprocessing pipeline (median imputer + zero-variance filter + robust scaler).
6) Optionally trains baseline binary + multiclass models.
7) Optionally transforms new/test data and writes submission-ready predictions.

Designed for the folder structure described in the hackathon README:
    good_weld/<config>/<run_id>/<run_id>.csv
    defect_data_weld/<config>/<run_id>/<run_id>.csv
    test_data/sample_0001/sensor.csv

Works also on a flat folder of sample CSV files for debugging.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


RANDOM_STATE = 42
REQUIRED_NUMERIC_COLS = [
    "Pressure",
    "CO2 Weld Flow",
    "Feed",
    "Primary Weld Current",
    "Wire Consumed",
    "Secondary Weld Voltage",
]
OPTIONAL_TEXT_COLS = ["Date", "Time", "Part No", "Remarks"]
SKIP_CSV_NAMES = {
    "test_data_manifest.csv",
    "test_data_ground_truth.csv",
    "submission.csv",
    "features_raw.csv",
    "features_processed.csv",
    "train_features_processed.csv",
    "val_features_processed.csv",
    "manifest.csv",
    "split_manifest.csv",
}
RUN_ID_SUFFIX_RE = re.compile(r"-(\d{2})$")


# ----------------------------- utility helpers ----------------------------- #

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def dump_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=to_jsonable)


def safe_float(x) -> float:
    try:
        value = float(x)
        if math.isfinite(value):
            return value
        return np.nan
    except Exception:
        return np.nan


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.nanstd(aa) == 0 or np.nanstd(bb) == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapz(y[mask], x[mask]))


def slope_vs_time(x: np.ndarray, t: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(t)
    if mask.sum() < 2:
        return np.nan
    xx = x[mask]
    tt = t[mask]
    t_std = np.std(tt)
    if t_std == 0:
        return np.nan
    # Stable least-squares slope using centered values
    tt_centered = tt - tt.mean()
    xx_centered = xx - xx.mean()
    denom = np.sum(tt_centered ** 2)
    if denom == 0:
        return np.nan
    return float(np.sum(tt_centered * xx_centered) / denom)


def first_non_nan(series: np.ndarray) -> float:
    idx = np.flatnonzero(np.isfinite(series))
    return float(series[idx[0]]) if len(idx) else np.nan


def last_non_nan(series: np.ndarray) -> float:
    idx = np.flatnonzero(np.isfinite(series))
    return float(series[idx[-1]]) if len(idx) else np.nan


def peak_index_ratio(series: np.ndarray) -> float:
    mask = np.isfinite(series)
    if mask.sum() == 0:
        return np.nan
    valid = series[mask]
    if len(valid) == 1:
        return 0.0
    return float(np.nanargmax(valid) / (len(valid) - 1))


def infer_label_code_from_sample_id(sample_id: str) -> Optional[str]:
    m = RUN_ID_SUFFIX_RE.search(sample_id)
    if m:
        return m.group(1)
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_sample_id_from_path(csv_path: Path, df: Optional[pd.DataFrame] = None) -> str:
    if df is not None and "Part No" in df.columns:
        vals = (
            df["Part No"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace({"": np.nan})
            .dropna()
            .unique()
            .tolist()
        )
        if len(vals) == 1:
            return vals[0]
    if csv_path.name.lower() == "sensor.csv":
        return csv_path.parent.name
    return csv_path.stem


def expected_media_paths(csv_path: Path) -> Tuple[Path, Path]:
    if csv_path.name.lower() == "sensor.csv":
        return csv_path.with_name("weld.flac"), csv_path.with_name("weld.avi")
    stem = csv_path.stem
    return csv_path.with_name(f"{stem}.flac"), csv_path.with_name(f"{stem}.avi")


@dataclass
class RunRecord:
    sample_id: str
    csv_path: str
    source_root: str
    config_group: str
    run_folder: str
    label_code: Optional[str]
    has_audio: bool
    has_video: bool
    is_anonymized: bool


class ReplaceInfWithNaN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return X_df.replace([np.inf, -np.inf], np.nan)


# ----------------------------- discovery layer ----------------------------- #

def discover_sensor_csvs(roots: Sequence[Path]) -> List[Path]:
    csvs: List[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".csv":
            if root.name not in SKIP_CSV_NAMES:
                csvs.append(root)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if path.name in SKIP_CSV_NAMES:
                continue
            csvs.append(path)
    return sorted(set(csvs))


def classify_config_group(csv_path: Path, source_root: Path) -> str:
    rel = csv_path.relative_to(source_root) if csv_path.is_relative_to(source_root) else csv_path
    parts = rel.parts

    # Labeled structure: root/config/run/run.csv
    if len(parts) >= 3 and parts[-2] != parts[0]:
        if csv_path.name.lower() == "sensor.csv" and len(parts) >= 2:
            # test_data/sample_0001/sensor.csv -> sample_0001
            return parts[-2]
        return parts[-3] if len(parts) >= 3 else "flat"

    # Flat debug mode
    return "flat"


def build_manifest(roots: Sequence[Path]) -> pd.DataFrame:
    rows: List[Dict] = []
    for root in roots:
        root = root.resolve()
        if not root.exists():
            continue
        candidate_csvs = discover_sensor_csvs([root])
        for csv_path in candidate_csvs:
            try:
                preview = normalize_columns(pd.read_csv(csv_path, nrows=5))
            except Exception:
                preview = pd.DataFrame()
            sample_id = infer_sample_id_from_path(csv_path, preview if len(preview.columns) else None)
            label_code = None if sample_id.startswith("sample_") else infer_label_code_from_sample_id(sample_id)
            audio_path, video_path = expected_media_paths(csv_path)
            config_group = classify_config_group(csv_path, root)
            rows.append(
                {
                    "sample_id": sample_id,
                    "csv_path": str(csv_path.resolve()),
                    "source_root": str(root),
                    "config_group": config_group,
                    "run_folder": str(csv_path.parent.resolve()),
                    "label_code": label_code,
                    "has_audio": audio_path.exists(),
                    "has_video": video_path.exists(),
                    "is_anonymized": sample_id.startswith("sample_"),
                }
            )
    manifest = pd.DataFrame(rows).drop_duplicates(subset=["sample_id", "csv_path"]).reset_index(drop=True)
    if manifest.empty:
        raise FileNotFoundError("No candidate sensor CSV files were found under the provided roots.")
    return manifest


# ------------------------------ sensor loading ----------------------------- #

def load_sensor_csv(csv_path: Path) -> Tuple[pd.DataFrame, Dict]:
    raw = pd.read_csv(csv_path)
    df = normalize_columns(raw)

    required_missing = [c for c in REQUIRED_NUMERIC_COLS if c not in df.columns]
    has_date = "Date" in df.columns
    has_time = "Time" in df.columns

    qc = {
        "required_missing_columns": required_missing,
        "has_date": has_date,
        "has_time": has_time,
    }

    if required_missing:
        raise ValueError(f"Missing required numeric columns: {required_missing}")

    for col in REQUIRED_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if has_date and has_time:
        dt_str = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
        df["timestamp"] = pd.to_datetime(dt_str, errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    qc["timestamp_parse_rate"] = float(df["timestamp"].notna().mean())

    if df["timestamp"].notna().sum() >= 2:
        not_monotonic_before_sort = int((df["timestamp"].dropna().diff().dt.total_seconds().fillna(0) < 0).sum())
        qc["non_monotonic_steps_before_sort"] = not_monotonic_before_sort
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        elapsed_sec = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        duplicate_count = int(df["timestamp"].duplicated().sum())
        qc["duplicate_timestamp_count"] = duplicate_count
    else:
        qc["non_monotonic_steps_before_sort"] = np.nan
        qc["duplicate_timestamp_count"] = np.nan
        elapsed_sec = np.arange(len(df), dtype=float)

    df["elapsed_sec"] = elapsed_sec

    dt = np.diff(elapsed_sec)
    finite_dt = dt[np.isfinite(dt) & (dt >= 0)]
    median_dt = float(np.median(finite_dt)) if len(finite_dt) else np.nan
    qc["median_dt_sec"] = median_dt
    qc["row_count"] = int(len(df))
    qc["duration_sec"] = float(elapsed_sec[-1] - elapsed_sec[0]) if len(elapsed_sec) >= 2 else 0.0
    qc["all_numeric_nan_columns"] = [c for c in REQUIRED_NUMERIC_COLS if df[c].notna().sum() == 0]

    if "Part No" in df.columns:
        ids = df["Part No"].dropna().astype(str).str.strip().unique().tolist()
        qc["part_no_unique_count"] = len(ids)
        qc["part_no_example"] = ids[0] if ids else None
    else:
        qc["part_no_unique_count"] = 0
        qc["part_no_example"] = None

    return df, qc


# --------------------------- feature engineering --------------------------- #

def signal_features(signal: np.ndarray, t: np.ndarray, prefix: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    x = np.asarray(signal, dtype=float)
    tt = np.asarray(t, dtype=float)

    finite_mask = np.isfinite(x)
    valid = x[finite_mask]
    feats[f"{prefix}__missing_frac"] = float(1.0 - finite_mask.mean()) if len(x) else np.nan
    feats[f"{prefix}__n_valid"] = int(finite_mask.sum())

    if len(valid) == 0:
        keys = [
            "mean", "std", "min", "max", "median", "q05", "q25", "q75", "q95", "range", "iqr",
            "start", "end", "delta", "abs_mean", "rms", "slope", "diff_mean", "diff_std",
            "positive_frac", "zero_frac", "peak_index_ratio",
        ]
        for key in keys:
            feats[f"{prefix}__{key}"] = np.nan
        return feats

    diffs = np.diff(valid) if len(valid) >= 2 else np.array([np.nan])
    feats[f"{prefix}__mean"] = float(np.mean(valid))
    feats[f"{prefix}__std"] = float(np.std(valid, ddof=0))
    feats[f"{prefix}__min"] = float(np.min(valid))
    feats[f"{prefix}__max"] = float(np.max(valid))
    feats[f"{prefix}__median"] = float(np.median(valid))
    feats[f"{prefix}__q05"] = float(np.quantile(valid, 0.05))
    feats[f"{prefix}__q25"] = float(np.quantile(valid, 0.25))
    feats[f"{prefix}__q75"] = float(np.quantile(valid, 0.75))
    feats[f"{prefix}__q95"] = float(np.quantile(valid, 0.95))
    feats[f"{prefix}__range"] = feats[f"{prefix}__max"] - feats[f"{prefix}__min"]
    feats[f"{prefix}__iqr"] = feats[f"{prefix}__q75"] - feats[f"{prefix}__q25"]
    feats[f"{prefix}__start"] = first_non_nan(x)
    feats[f"{prefix}__end"] = last_non_nan(x)
    feats[f"{prefix}__delta"] = feats[f"{prefix}__end"] - feats[f"{prefix}__start"]
    feats[f"{prefix}__abs_mean"] = float(np.mean(np.abs(valid)))
    feats[f"{prefix}__rms"] = float(np.sqrt(np.mean(valid ** 2)))
    feats[f"{prefix}__slope"] = slope_vs_time(x, tt)
    feats[f"{prefix}__diff_mean"] = float(np.nanmean(diffs))
    feats[f"{prefix}__diff_std"] = float(np.nanstd(diffs))
    feats[f"{prefix}__positive_frac"] = float(np.mean(valid > 0))
    feats[f"{prefix}__zero_frac"] = float(np.mean(valid == 0))
    feats[f"{prefix}__peak_index_ratio"] = peak_index_ratio(x)
    return feats


def compute_active_mask(df: pd.DataFrame) -> np.ndarray:
    current = df["Primary Weld Current"].to_numpy(dtype=float)
    voltage = df["Secondary Weld Voltage"].to_numpy(dtype=float)
    feed = df["Feed"].to_numpy(dtype=float)

    cmax = np.nanmax(current) if np.isfinite(current).any() else 0.0
    vmax = np.nanmax(voltage) if np.isfinite(voltage).any() else 0.0
    fmax = np.nanmax(feed) if np.isfinite(feed).any() else 0.0

    thr_current = max(5.0, 0.05 * cmax)
    thr_voltage = max(1.0, 0.05 * vmax)
    thr_feed = max(1.0, 0.05 * fmax)

    active = (
        (np.nan_to_num(current, nan=0.0) > thr_current)
        | (np.nan_to_num(voltage, nan=0.0) > thr_voltage)
        | (np.nan_to_num(feed, nan=0.0) > thr_feed)
    )
    return active


def extract_run_features(df: pd.DataFrame, sample_id: str, qc: Dict) -> Dict:
    feats: Dict[str, float] = {"sample_id": sample_id}
    t = df["elapsed_sec"].to_numpy(dtype=float)
    duration_sec = float(t[-1] - t[0]) if len(t) >= 2 else 0.0

    feats["n_rows"] = int(len(df))
    feats["duration_sec"] = duration_sec
    feats["median_dt_sec"] = safe_float(qc.get("median_dt_sec"))
    feats["timestamp_parse_rate"] = safe_float(qc.get("timestamp_parse_rate"))
    feats["duplicate_timestamp_count"] = safe_float(qc.get("duplicate_timestamp_count"))
    feats["non_monotonic_steps_before_sort"] = safe_float(qc.get("non_monotonic_steps_before_sort"))
    feats["has_date_time"] = int(bool(qc.get("has_date")) and bool(qc.get("has_time")))

    for col in REQUIRED_NUMERIC_COLS:
        feats.update(signal_features(df[col].to_numpy(dtype=float), t, col.replace(" ", "_").lower()))

    active = compute_active_mask(df)
    active_idx = np.flatnonzero(active)
    feats["active_fraction"] = float(active.mean()) if len(active) else np.nan

    if len(active_idx):
        start_i = int(active_idx[0])
        end_i = int(active_idx[-1])
        feats["active_start_ratio"] = float(start_i / max(1, len(df) - 1))
        feats["active_end_ratio"] = float(end_i / max(1, len(df) - 1))
        feats["active_duration_sec"] = float(t[end_i] - t[start_i]) if end_i > start_i else 0.0
        active_df = df.iloc[start_i : end_i + 1].reset_index(drop=True)
        active_t = active_df["elapsed_sec"].to_numpy(dtype=float)
        for col in REQUIRED_NUMERIC_COLS:
            prefix = f"active__{col.replace(' ', '_').lower()}"
            feats.update(signal_features(active_df[col].to_numpy(dtype=float), active_t, prefix))
    else:
        feats["active_start_ratio"] = np.nan
        feats["active_end_ratio"] = np.nan
        feats["active_duration_sec"] = 0.0

    pressure = df["Pressure"].to_numpy(dtype=float)
    flow = df["CO2 Weld Flow"].to_numpy(dtype=float)
    feed = df["Feed"].to_numpy(dtype=float)
    current = df["Primary Weld Current"].to_numpy(dtype=float)
    wire = df["Wire Consumed"].to_numpy(dtype=float)
    voltage = df["Secondary Weld Voltage"].to_numpy(dtype=float)

    feats["corr_current_voltage"] = safe_corr(current, voltage)
    feats["corr_feed_current"] = safe_corr(feed, current)
    feats["corr_flow_current"] = safe_corr(flow, current)
    feats["corr_pressure_current"] = safe_corr(pressure, current)
    feats["wire_total_increment"] = last_non_nan(wire) - first_non_nan(wire)
    feats["wire_increment_per_sec"] = (
        feats["wire_total_increment"] / duration_sec if duration_sec and np.isfinite(duration_sec) else np.nan
    )
    feats["current_integral"] = trapz(current, t)
    feats["voltage_integral"] = trapz(voltage, t)
    feats["feed_integral"] = trapz(feed, t)
    feats["flow_integral"] = trapz(flow, t)
    feats["current_x_voltage_integral"] = trapz(np.nan_to_num(current) * np.nan_to_num(voltage), t)

    mean_current = np.nanmean(current)
    mean_voltage = np.nanmean(voltage)
    mean_feed = np.nanmean(feed)
    feats["mean_voltage_to_current"] = mean_voltage / mean_current if np.isfinite(mean_current) and mean_current != 0 else np.nan
    feats["mean_feed_to_current"] = mean_feed / mean_current if np.isfinite(mean_current) and mean_current != 0 else np.nan

    # Sanity flags
    feats["flag_any_negative_pressure"] = int(np.nanmin(pressure) < 0) if np.isfinite(pressure).any() else 0
    feats["flag_voltage_spike_gt_60"] = int(np.nanmax(voltage) > 60) if np.isfinite(voltage).any() else 0
    feats["flag_zero_current_entire_run"] = int(np.nanmax(current) <= 0) if np.isfinite(current).any() else 1

    return feats


def extract_features_from_manifest(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: List[Dict] = []
    validation_rows: List[Dict] = []

    for row in manifest.itertuples(index=False):
        csv_path = Path(row.csv_path)
        record = {
            "sample_id": row.sample_id,
            "csv_path": row.csv_path,
            "source_root": row.source_root,
            "config_group": row.config_group,
            "run_folder": row.run_folder,
            "label_code": row.label_code,
            "has_audio": int(row.has_audio),
            "has_video": int(row.has_video),
            "is_anonymized": int(row.is_anonymized),
            "status": "ok",
            "error": None,
        }
        try:
            df, qc = load_sensor_csv(csv_path)
            sample_id_detected = infer_sample_id_from_path(csv_path, df)
            record["part_no_matches_sample_id"] = int(sample_id_detected == row.sample_id)
            record.update({f"qc__{k}": v for k, v in qc.items()})
            feats = extract_run_features(df, row.sample_id, qc)
            feats.update(
                {
                    "csv_path": row.csv_path,
                    "source_root": row.source_root,
                    "config_group": row.config_group,
                    "run_folder": row.run_folder,
                    "label_code": row.label_code,
                    "has_audio": int(row.has_audio),
                    "has_video": int(row.has_video),
                    "is_anonymized": int(row.is_anonymized),
                }
            )
            feature_rows.append(feats)
        except Exception as e:
            record["status"] = "failed"
            record["error"] = f"{type(e).__name__}: {e}"
        validation_rows.append(record)

    features = pd.DataFrame(feature_rows).sort_values("sample_id").reset_index(drop=True)
    validation = pd.DataFrame(validation_rows).sort_values("sample_id").reset_index(drop=True)
    return features, validation


# ------------------------------ split + fit ------------------------------- #

META_COLUMNS = {
    "sample_id", "csv_path", "source_root", "config_group", "run_folder", "label_code",
    "has_audio", "has_video", "is_anonymized", "split"
}


def feature_columns_from_df(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in META_COLUMNS]
    return cols


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["binary_target"] = (out["label_code"].astype(str) != "00").astype(int)
    return out


def make_group_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    out = df.copy()
    labeled = out["label_code"].notna()
    out["split"] = np.where(labeled, "train", "unlabeled")

    labeled_df = out.loc[labeled].copy()
    if labeled_df.empty:
        return out

    groups = labeled_df["config_group"].fillna("missing_group")
    unique_groups = pd.Series(groups).nunique()
    if len(labeled_df) < 2 or unique_groups < 2:
        # Safe fallback for tiny debug sets: keep everything in train
        out.loc[labeled_df.index, "split"] = "train"
        return out

    try:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        train_idx, val_idx = next(splitter.split(labeled_df, groups=groups))
    except ValueError:
        out.loc[labeled_df.index, "split"] = "train"
        return out

    labeled_df.iloc[train_idx, labeled_df.columns.get_loc("split")] = "train"
    labeled_df.iloc[val_idx, labeled_df.columns.get_loc("split")] = "val"

    out.loc[labeled_df.index, "split"] = labeled_df["split"]
    return out


def build_preprocessor(
    X_train: pd.DataFrame,
    max_missing_frac: float = 0.95,
) -> Tuple[Pipeline, List[str], List[str]]:
    feature_cols = feature_columns_from_df(X_train)
    if not feature_cols:
        raise ValueError("No engineered feature columns were found.")

    selected_before_preproc = [
        c for c in feature_cols
        if pd.to_numeric(X_train[c], errors="coerce").isna().mean() <= max_missing_frac
    ]
    if not selected_before_preproc:
        raise ValueError("All feature columns were removed by the missing-value filter.")

    preprocessor = Pipeline(
        steps=[
            ("replace_inf", ReplaceInfWithNaN()),
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
        ]
    )
    preprocessor.fit(X_train[selected_before_preproc])

    variance_mask = preprocessor.named_steps["variance"].get_support()
    selected_after_preproc = [c for c, keep in zip(selected_before_preproc, variance_mask) if keep]
    return preprocessor, selected_before_preproc, selected_after_preproc


def transform_with_preprocessor(
    df: pd.DataFrame,
    preprocessor: Pipeline,
    selected_before_preproc: Sequence[str],
    selected_after_preproc: Sequence[str],
) -> pd.DataFrame:
    X = df.copy()
    Xt = preprocessor.transform(X[list(selected_before_preproc)])
    out = pd.DataFrame(Xt, columns=list(selected_after_preproc), index=df.index)
    out.insert(0, "sample_id", df["sample_id"].values)
    if "label_code" in df.columns:
        out["label_code"] = df["label_code"].values
    if "split" in df.columns:
        out["split"] = df["split"].values
    if "config_group" in df.columns:
        out["config_group"] = df["config_group"].values
    return out


def compute_binary_metrics(y_true: np.ndarray, p_defect: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (p_defect >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, p_defect))
    return metrics


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=np.unique(y_true), zero_division=0
    )
    labels = list(np.unique(y_true))
    per_class = {
        str(lbl): {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for lbl, p, r, f, s in zip(labels, precision, recall, f1, support)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": per_class,
    }


def fit_models(
    features_df: pd.DataFrame,
    artifacts_dir: Path,
    threshold: float = 0.5,
) -> Dict:
    ensure_dir(artifacts_dir)
    df = add_binary_target(features_df)
    df = make_group_split(df, val_size=0.2, random_state=RANDOM_STATE)

    labeled_df = df[df["label_code"].notna()].copy()
    train_df = labeled_df[labeled_df["split"] == "train"].copy()
    val_df = labeled_df[labeled_df["split"] == "val"].copy()

    y_train_binary = train_df["binary_target"].to_numpy(dtype=int)
    if len(np.unique(y_train_binary)) < 2:
        raise ValueError(
            "Training data does not contain both binary classes. "
            "You need at least one good run (00) and one defect run (!=00)."
        )

    preprocessor, selected_before, selected_after = build_preprocessor(train_df)
    joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")
    dump_json(selected_before, artifacts_dir / "selected_features_before_variance.json")
    dump_json(selected_after, artifacts_dir / "selected_features_final.json")

    train_processed = transform_with_preprocessor(train_df, preprocessor, selected_before, selected_after)
    val_processed = transform_with_preprocessor(val_df, preprocessor, selected_before, selected_after)

    train_processed.to_csv(artifacts_dir / "train_features_processed.csv", index=False)
    val_processed.to_csv(artifacts_dir / "val_features_processed.csv", index=False)
    labeled_df.to_csv(artifacts_dir / "split_manifest.csv", index=False)

    X_train = train_processed[selected_after]
    X_val = val_processed[selected_after] if not val_processed.empty else pd.DataFrame(columns=selected_after)

    # Binary defect model with calibrated probabilities
    binary_base = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    binary_model = CalibratedClassifierCV(binary_base, method="sigmoid", cv=3)
    binary_model.fit(X_train, y_train_binary)
    joblib.dump(binary_model, artifacts_dir / "binary_model.joblib")

    metrics: Dict = {"binary": {}, "multiclass": {}, "artifacts_dir": str(artifacts_dir)}

    if not val_processed.empty:
        p_defect = binary_model.predict_proba(X_val)[:, 1]
        metrics["binary"] = compute_binary_metrics(
            val_df["binary_target"].to_numpy(dtype=int), p_defect, threshold=threshold
        )

    # Multiclass defect-type model trained only on defect runs
    train_defect = train_df[train_df["binary_target"] == 1].copy()
    if train_defect["label_code"].nunique() >= 2:
        train_defect_processed = train_processed.loc[train_defect.index]
        multi_model = RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        multi_model.fit(train_defect_processed[selected_after], train_defect["label_code"].astype(str))
        joblib.dump(multi_model, artifacts_dir / "multiclass_model.joblib")

        val_defect = val_df[val_df["binary_target"] == 1].copy()
        if not val_defect.empty:
            val_defect_processed = val_processed.loc[val_defect.index]
            val_multi_pred = multi_model.predict(val_defect_processed[selected_after])
            metrics["multiclass"] = compute_multiclass_metrics(
                val_defect["label_code"].astype(str).to_numpy(), val_multi_pred
            )
    else:
        metrics["multiclass"] = {
            "warning": "Skipped multiclass training because the train split had fewer than 2 defect classes."
        }

    dump_json(metrics, artifacts_dir / "metrics.json")
    return metrics


def prepare_only(
    input_roots: Sequence[Path],
    output_dir: Path,
    fit_preprocessor: bool = True,
    max_missing_frac: float = 0.95,
) -> Dict:
    ensure_dir(output_dir)
    manifest = build_manifest(input_roots)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    features_df, validation_df = extract_features_from_manifest(manifest)
    validation_df.to_csv(output_dir / "validation_report.csv", index=False)
    features_df.to_csv(output_dir / "features_raw.csv", index=False)

    summary = {
        "n_manifest_rows": int(len(manifest)),
        "n_feature_rows": int(len(features_df)),
        "n_failed_rows": int((validation_df["status"] == "failed").sum()),
        "n_anonymized_rows": int(features_df["is_anonymized"].sum()) if "is_anonymized" in features_df.columns else 0,
        "label_distribution": (
            features_df["label_code"].fillna("unlabeled").value_counts(dropna=False).to_dict()
            if "label_code" in features_df.columns else {}
        ),
    }

    # Fit preprocessor only when we have labeled training data
    can_fit = fit_preprocessor and not features_df.empty and features_df["label_code"].notna().any()
    if can_fit:
        split_df = make_group_split(features_df.copy(), val_size=0.2, random_state=RANDOM_STATE)
        split_df.to_csv(output_dir / "split_manifest.csv", index=False)

        train_df = split_df[split_df["split"] == "train"].copy()
        if not train_df.empty:
            preprocessor, selected_before, selected_after = build_preprocessor(
                train_df, max_missing_frac=max_missing_frac
            )
            joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
            dump_json(selected_before, output_dir / "selected_features_before_variance.json")
            dump_json(selected_after, output_dir / "selected_features_final.json")

            processed_all = transform_with_preprocessor(split_df, preprocessor, selected_before, selected_after)
            processed_all.to_csv(output_dir / "features_processed.csv", index=False)

            summary["n_selected_features_before_variance"] = len(selected_before)
            summary["n_selected_features_final"] = len(selected_after)
        else:
            summary["warning"] = "Could not fit preprocessor because the train split was empty."
    else:
        summary["warning"] = "Preprocessor was not fit because no labeled rows were available."

    dump_json(summary, output_dir / "summary.json")
    return summary


def transform_only(
    input_roots: Sequence[Path],
    artifacts_dir: Path,
    output_dir: Path,
) -> Dict:
    ensure_dir(output_dir)
    preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")
    with (artifacts_dir / "selected_features_before_variance.json").open("r", encoding="utf-8") as f:
        selected_before = json.load(f)
    with (artifacts_dir / "selected_features_final.json").open("r", encoding="utf-8") as f:
        selected_after = json.load(f)

    manifest = build_manifest(input_roots)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    features_df, validation_df = extract_features_from_manifest(manifest)
    features_df.to_csv(output_dir / "features_raw.csv", index=False)
    validation_df.to_csv(output_dir / "validation_report.csv", index=False)

    processed_df = transform_with_preprocessor(features_df, preprocessor, selected_before, selected_after)
    processed_df.to_csv(output_dir / "features_processed.csv", index=False)

    summary = {
        "n_manifest_rows": int(len(manifest)),
        "n_feature_rows": int(len(features_df)),
        "n_failed_rows": int((validation_df["status"] == "failed").sum()),
        "n_output_features": int(len(selected_after)),
    }
    dump_json(summary, output_dir / "summary.json")
    return summary


def predict_with_trained_models(
    input_roots: Sequence[Path],
    artifacts_dir: Path,
    output_dir: Path,
    threshold: float = 0.5,
) -> Dict:
    ensure_dir(output_dir)
    transform_only(input_roots, artifacts_dir, output_dir)

    processed_df = pd.read_csv(output_dir / "features_processed.csv")
    with (artifacts_dir / "selected_features_final.json").open("r", encoding="utf-8") as f:
        selected_after = json.load(f)

    binary_model = joblib.load(artifacts_dir / "binary_model.joblib")
    multiclass_model_path = artifacts_dir / "multiclass_model.joblib"
    multiclass_model = joblib.load(multiclass_model_path) if multiclass_model_path.exists() else None

    X = processed_df[selected_after]
    p_defect = binary_model.predict_proba(X)[:, 1]
    pred_binary = (p_defect >= threshold).astype(int)

    if multiclass_model is not None:
        pred_label = multiclass_model.predict(X)
    else:
        pred_label = np.array(["01"] * len(X), dtype=object)

    final_label = np.where(pred_binary == 1, pred_label.astype(str), "00")
    submission = pd.DataFrame(
        {
            "sample_id": processed_df["sample_id"].astype(str),
            "pred_label_code": final_label.astype(str),
            "p_defect": p_defect.astype(float),
        }
    ).sort_values("sample_id").reset_index(drop=True)
    submission.to_csv(output_dir / "submission.csv", index=False)

    summary = {
        "n_predictions": int(len(submission)),
        "threshold": float(threshold),
        "submission_path": str((output_dir / "submission.csv").resolve()),
    }
    dump_json(summary, output_dir / "prediction_summary.json")
    return summary


# --------------------------------- CLI ----------------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible sensor-only pipeline for welding anomaly detection."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Discover runs, validate files, extract run-level features, and optionally fit preprocessing artifacts.",
    )
    prepare_parser.add_argument(
        "--input-roots",
        nargs="+",
        required=True,
        help="One or more dataset roots or CSV files.",
    )
    prepare_parser.add_argument("--output-dir", required=True, help="Directory to write pipeline outputs.")
    prepare_parser.add_argument(
        "--skip-fit-preprocessor",
        action="store_true",
        help="Only extract raw features; do not fit a preprocessor.",
    )
    prepare_parser.add_argument(
        "--max-missing-frac",
        type=float,
        default=0.95,
        help="Drop engineered features whose train-split missing fraction exceeds this value.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Prepare features, fit deterministic preprocessing artifacts, and train baseline models.",
    )
    train_parser.add_argument(
        "--input-roots",
        nargs="+",
        required=True,
        help="One or more labeled training roots (e.g. good_weld defect_data_weld).",
    )
    train_parser.add_argument("--output-dir", required=True, help="Directory to write artifacts and metrics.")
    train_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary defect decision threshold used for validation and prediction.",
    )

    transform_parser = subparsers.add_parser(
        "transform",
        help="Use a previously fitted preprocessor to transform new runs into model-ready tabular features.",
    )
    transform_parser.add_argument(
        "--input-roots",
        nargs="+",
        required=True,
        help="One or more roots or CSVs to transform.",
    )
    transform_parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory containing preprocessor.joblib and selected feature JSON files.",
    )
    transform_parser.add_argument("--output-dir", required=True, help="Directory to write transformed outputs.")

    predict_parser = subparsers.add_parser(
        "predict",
        help="Transform new runs and produce submission-ready predictions with the baseline models.",
    )
    predict_parser.add_argument(
        "--input-roots",
        nargs="+",
        required=True,
        help="One or more roots or CSVs to predict on.",
    )
    predict_parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory containing fitted preprocessing and model artifacts.",
    )
    predict_parser.add_argument("--output-dir", required=True, help="Directory to write predictions.")
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary defect decision threshold.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    input_roots = [Path(p).resolve() for p in args.input_roots]
    output_dir = Path(args.output_dir).resolve()

    try:
        if args.command == "prepare":
            summary = prepare_only(
                input_roots=input_roots,
                output_dir=output_dir,
                fit_preprocessor=not args.skip_fit_preprocessor,
                max_missing_frac=float(args.max_missing_frac),
            )
        elif args.command == "train":
            ensure_dir(output_dir)
            prep_summary = prepare_only(
                input_roots=input_roots,
                output_dir=output_dir,
                fit_preprocessor=True,
                max_missing_frac=0.95,
            )
            features_df = pd.read_csv(output_dir / "features_raw.csv")
            metrics = fit_models(features_df, artifacts_dir=output_dir, threshold=float(args.threshold))
            summary = {"prepare": prep_summary, "train": metrics}
            dump_json(summary, output_dir / "run_summary.json")
        elif args.command == "transform":
            summary = transform_only(
                input_roots=input_roots,
                artifacts_dir=Path(args.artifacts_dir).resolve(),
                output_dir=output_dir,
            )
        elif args.command == "predict":
            summary = predict_with_trained_models(
                input_roots=input_roots,
                artifacts_dir=Path(args.artifacts_dir).resolve(),
                output_dir=output_dir,
                threshold=float(args.threshold),
            )
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as e:
        err = {"status": "failed", "error_type": type(e).__name__, "error": str(e)}
        ensure_dir(output_dir)
        dump_json(err, output_dir / "error.json")
        print(json.dumps(err, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, default=to_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
