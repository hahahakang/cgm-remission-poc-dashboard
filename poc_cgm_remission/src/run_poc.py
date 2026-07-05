from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260704
CGM_INTERVAL_MINUTES = 5
READINGS_PER_DAY = int(24 * 60 / CGM_INTERVAL_MINUTES)
TOTAL_CGM_DAYS = 14
EFFECTIVE_CGM_DAYS = 12
EFFECTIVE_START_DAY = 3
EXPECTED_EFFECTIVE_READINGS = EFFECTIVE_CGM_DAYS * READINGS_PER_DAY

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def native(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_split(patient_id: str, cohort: str) -> str:
    if cohort == "external":
        return "external_validation"
    digest = hashlib.sha256(patient_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    return "train" if bucket < 7 else "internal_validation"


def simulate_clinical(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    centers = np.array(["航天中心医院", "北京协和医院", "北京清华长庚医院"])
    rows = []
    for i in range(n):
        center = centers[i % 3]
        cohort = "external" if i >= int(n * 2 / 3) else "modeling"
        age = float(np.clip(rng.normal(45, 9), 18, 65))
        sex = rng.choice(["男", "女"], p=[0.58, 0.42])
        duration = float(np.clip(rng.gamma(2.0, 0.9), 0.2, 5.0))
        bmi = float(np.clip(rng.normal(29.2, 3.1), 24.0, 39.5))
        waist = float(np.clip(78 + bmi * 1.05 + rng.normal(0, 5), 78, 125))
        hba1c = float(np.clip(rng.normal(7.8 + duration * 0.16, 0.8), 6.3, 10.8))
        fasting_glucose = float(np.clip(4.2 + hba1c * 0.72 + rng.normal(0, 0.6), 5.8, 13.5))
        c_peptide_fasting = float(np.clip(rng.normal(1.9 - duration * 0.12 + (30 - bmi) * 0.02, 0.35), 1.1, 3.4))
        c_peptide_2h = float(np.clip(c_peptide_fasting + rng.normal(1.9, 0.45), 2.5, 6.5))
        glp1ra_at_baseline = int(rng.random() < (0.25 + 0.04 * max(bmi - 28, 0)))
        rows.append(
            {
                "patient_id": f"POC{i + 1:03d}",
                "center": center,
                "cohort": cohort,
                "analysis_split": stable_split(f"POC{i + 1:03d}", cohort),
                "age": round(age, 1),
                "sex": sex,
                "bmi": round(bmi, 1),
                "waist_cm": round(waist, 1),
                "diabetes_duration_years": round(duration, 2),
                "baseline_hba1c_pct": round(hba1c, 2),
                "fasting_glucose_mmol_l": round(fasting_glucose, 2),
                "c_peptide_fasting_ug_l": round(c_peptide_fasting, 2),
                "c_peptide_2h_ug_l": round(c_peptide_2h, 2),
                "glp1ra_at_baseline": glp1ra_at_baseline,
            }
        )
    return pd.DataFrame(rows)


def simulate_cgm(clinical: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    start = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    high_missing_patients = set(clinical.sample(4, random_state=SEED)["patient_id"])

    for _, p in clinical.iterrows():
        phenotype = rng.normal(0, 1)
        postprandial_surge = np.clip(1.0 + 0.24 * phenotype + rng.normal(0, 0.12), 0.65, 1.55)
        base = (
            4.95
            + 0.50 * (p["baseline_hba1c_pct"] - 6.0)
            + 0.018 * (p["bmi"] - 28.0)
            + 0.06 * p["diabetes_duration_years"]
            - 0.16 * (p["c_peptide_2h_ug_l"] - 3.5)
            + 0.28 * phenotype
        )
        variability = np.clip(
            0.42 + 0.07 * (p["baseline_hba1c_pct"] - 7.0) + 0.18 * phenotype + rng.normal(0, 0.06),
            0.24,
            1.25,
        )
        miss_prob = 0.16 if p["patient_id"] in high_missing_patients else rng.uniform(0.01, 0.045)

        for t in range(TOTAL_CGM_DAYS * READINGS_PER_DAY):
            ts = start + timedelta(minutes=CGM_INTERVAL_MINUTES * t)
            minute_of_day = ts.hour * 60 + ts.minute
            day_fraction = minute_of_day / 1440
            circadian = 0.35 * math.sin(2 * math.pi * (day_fraction - 0.18))
            dawn = 0.25 * math.exp(-0.5 * ((day_fraction - 6.5 / 24) / 0.055) ** 2)
            meal = 0.0
            for meal_center, meal_size in ((8 / 24, 1.05), (12.5 / 24, 1.48), (18.5 / 24, 1.68)):
                delta = min(abs(day_fraction - meal_center), 1 - abs(day_fraction - meal_center))
                meal += postprandial_surge * meal_size * math.exp(-0.5 * (delta / 0.032) ** 2)
            weekend_effect = 0.25 if ts.weekday() >= 5 else 0.0
            micro_oscillation = 0.07 * math.sin(2 * math.pi * t / 9)
            glucose = base + circadian + dawn + meal + weekend_effect + micro_oscillation + rng.normal(0, variability)
            glucose = float(np.clip(glucose, 2.5, 22.0))
            glucose_value = "" if rng.random() < miss_prob else round(glucose, 2)
            rows.append(
                {
                    "patient_id": p["patient_id"],
                    "timestamp_utc": ts.isoformat(),
                    "sensor_day": t // READINGS_PER_DAY + 1,
                    "minute_of_day": minute_of_day,
                    "glucose_mmol_l": glucose_value,
                }
            )
    return pd.DataFrame(rows)


def count_low_events(values: pd.Series) -> int:
    low = values < 3.9
    return int((low & ~low.shift(fill_value=False)).sum())


def safe_autocorr(values: pd.Series, lag: int) -> float:
    if len(values) <= lag + 2:
        return 0.0
    corr = values.autocorr(lag=lag)
    return 0.0 if pd.isna(corr) else float(corr)


def rhythm_amplitude(values: pd.Series, minutes: pd.Series, period_minutes: int) -> float:
    if len(values) < 8:
        return 0.0
    y = values.to_numpy(dtype=float) - float(values.mean())
    theta = 2 * np.pi * minutes.to_numpy(dtype=float) / period_minutes
    sin_coef = float(np.mean(y * np.sin(theta)))
    cos_coef = float(np.mean(y * np.cos(theta)))
    return float(2 * np.sqrt(sin_coef**2 + cos_coef**2))


def window_mean(g: pd.DataFrame, start_minute: int, end_minute: int) -> float:
    window = g[(g["minute_of_day"] >= start_minute) & (g["minute_of_day"] < end_minute)]["glucose_mmol_l"].dropna()
    return float(window.mean()) if len(window) else float("nan")


def postprandial_peak_mean(g: pd.DataFrame) -> float:
    peaks = []
    for start_minute, end_minute in ((8 * 60, 11 * 60), (12 * 60, 15 * 60), (18 * 60, 21 * 60)):
        window = g[(g["minute_of_day"] >= start_minute) & (g["minute_of_day"] < end_minute)]["glucose_mmol_l"].dropna()
        if len(window):
            peaks.append(float(window.quantile(0.90)))
    return float(np.mean(peaks)) if peaks else float("nan")


def extract_features(clinical: pd.DataFrame, cgm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cgm = cgm.copy()
    cgm["glucose_mmol_l"] = pd.to_numeric(cgm["glucose_mmol_l"], errors="coerce")
    effective = cgm[cgm["sensor_day"] >= EFFECTIVE_START_DAY].copy()

    feature_rows = []
    qc_rows = []
    for patient_id, g in effective.groupby("patient_id", sort=True):
        g = g.sort_values("timestamp_utc").copy()
        valid = g["glucose_mmol_l"].dropna()
        valid_g = g.dropna(subset=["glucose_mmol_l"]).copy()
        valid_pct = len(valid) / EXPECTED_EFFECTIVE_READINGS
        qc_pass = valid_pct >= 0.90

        if len(valid) == 0:
            feature = {
                "patient_id": patient_id,
                "cgm_valid_pct": 0.0,
                "cgm_qc_pass": False,
            }
        else:
            diff = valid.diff().dropna()
            night_mean = window_mean(valid_g, 0, 6 * 60)
            dawn_mean = window_mean(valid_g, 5 * 60, 8 * 60)
            post_peak = postprandial_peak_mean(valid_g)
            day_means = valid_g.groupby("sensor_day")["glucose_mmol_l"].mean()
            amp24 = rhythm_amplitude(valid_g["glucose_mmol_l"], valid_g["minute_of_day"], 1440)
            amp12 = rhythm_amplitude(valid_g["glucose_mmol_l"], valid_g["minute_of_day"], 720)
            high_auc = float(np.maximum(valid - 10.0, 0).sum() * (CGM_INTERVAL_MINUTES / 60) / EFFECTIVE_CGM_DAYS)
            feature = {
                "patient_id": patient_id,
                "cgm_valid_pct": round(valid_pct, 4),
                "cgm_qc_pass": qc_pass,
                "tir_3p9_10_pct": round(float(((valid >= 3.9) & (valid <= 10.0)).mean() * 100), 2),
                "tar_gt10_pct": round(float((valid > 10.0).mean() * 100), 2),
                "tar_gt13p9_pct": round(float((valid > 13.9).mean() * 100), 2),
                "tbr_lt3p9_pct": round(float((valid < 3.9).mean() * 100), 2),
                "mean_glucose_mmol_l": round(float(valid.mean()), 2),
                "sd_glucose_mmol_l": round(float(valid.std(ddof=0)), 2),
                "cv_pct": round(float(valid.std(ddof=0) / valid.mean() * 100), 2),
                "low_event_count": count_low_events(valid),
                "high_auc_mmol_hour_per_day": round(high_auc, 2),
                "night_mean_glucose_mmol_l": round(night_mean, 2),
                "seq_amp_24h": round(amp24, 3),
                "seq_amp_12h": round(amp12, 3),
                "seq_autocorr_1h": round(safe_autocorr(valid.reset_index(drop=True), int(60 / CGM_INTERVAL_MINUTES)), 3),
                "seq_autocorr_24h": round(safe_autocorr(valid.reset_index(drop=True), READINGS_PER_DAY), 3),
                "seq_instability_mmol_l": round(float(diff.abs().mean()), 3),
                "seq_spike_count": int((diff > 1.0).sum()),
                "seq_dawn_rise": round(float(dawn_mean - night_mean), 3),
                "seq_postprandial_peak_mean": round(post_peak, 2),
                "seq_day_to_day_sd": round(float(day_means.std(ddof=0)), 3),
            }
        feature_rows.append(feature)
        qc_rows.append(
            {
                "patient_id": patient_id,
                "valid_readings_after_day2": int(len(valid)),
                "expected_readings_after_day2": EXPECTED_EFFECTIVE_READINGS,
                "valid_pct": round(valid_pct, 4),
                "qc_pass": bool(qc_pass),
                "exclusion_reason": "" if qc_pass else "CGM有效读数<90%",
            }
        )

    features = clinical.merge(pd.DataFrame(feature_rows), on="patient_id", how="left")
    features = add_model_feature_columns(assign_outcomes(features))
    qc = pd.DataFrame(qc_rows).merge(clinical[["patient_id", "center", "analysis_split"]], on="patient_id", how="left")
    return features, qc


def assign_outcomes(features: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    df = features.copy()
    for col in [
        "tir_3p9_10_pct",
        "cv_pct",
        "high_auc_mmol_hour_per_day",
        "seq_amp_24h",
        "seq_instability_mmol_l",
        "seq_postprandial_peak_mean",
        "seq_day_to_day_sd",
    ]:
        df[col] = df[col].fillna(df[col].median())

    logit = (
        -2.18
        - 0.38 * (df["baseline_hba1c_pct"] - 7.6)
        - 0.36 * (df["diabetes_duration_years"] - 2.0)
        - 0.06 * (df["bmi"] - 29.0)
        + 0.30 * (df["c_peptide_2h_ug_l"] - 3.8)
        + 0.026 * (df["tir_3p9_10_pct"] - 62.0)
        - 0.040 * (df["cv_pct"] - 24.0)
        - 0.080 * df["high_auc_mmol_hour_per_day"]
        - 0.180 * (df["seq_instability_mmol_l"] - 0.55)
        - 0.130 * (df["seq_postprandial_peak_mean"] - 10.2)
        - 0.240 * (df["seq_day_to_day_sd"] - 0.40)
        + 0.120 * (df["seq_amp_24h"] - 1.0)
        - 0.04 * df["glp1ra_at_baseline"]
    )
    probability = sigmoid(logit)
    df["remission_probability_simulated"] = np.round(probability, 4)
    df["remission_12m"] = (rng.random(len(df)) < probability).astype(int)
    return df


def add_model_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hba1c_duration_load"] = out["baseline_hba1c_pct"] * out["diabetes_duration_years"]
    out["bmi_cpeptide_balance"] = out["bmi"] / out["c_peptide_2h_ug_l"].replace(0, np.nan)
    out["tir_cv_balance"] = out["tir_3p9_10_pct"] / out["cv_pct"].replace(0, np.nan)
    out["hyper_variability_load"] = out["high_auc_mmol_hour_per_day"] * out["cv_pct"]
    out["postmeal_instability_load"] = out["seq_postprandial_peak_mean"] * out["seq_instability_mmol_l"]
    return out


def standardize(train: pd.DataFrame, other: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    medians = train[columns].median(numeric_only=True)
    train_filled = train[columns].fillna(medians)
    other_filled = other[columns].fillna(medians)
    means = train_filled.mean()
    stds = train_filled.std(ddof=0).replace(0, 1)
    train_x = ((train_filled - means) / stds).to_numpy(dtype=float)
    other_x = ((other_filled - means) / stds).to_numpy(dtype=float)
    return train_x, other_x, {
        "medians": {k: native(v) for k, v in medians.to_dict().items()},
        "means": {k: native(v) for k, v in means.to_dict().items()},
        "stds": {k: native(v) for k, v in stds.to_dict().items()},
    }


def fit_logistic(x: np.ndarray, y: np.ndarray, lr: float = 0.04, steps: int = 6500, l2: float = 0.02) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(x_aug.shape[1])
    for _ in range(steps):
        pred = sigmoid(x_aug @ weights)
        grad = (x_aug.T @ (pred - y)) / len(y)
        grad[1:] += l2 * weights[1:]
        weights -= lr * grad
    return weights


def predict_logistic(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    return sigmoid(x_aug @ weights)


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    total = 0.0
    for p in pos:
        total += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(total / (len(pos) * len(neg)))


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.round(score, 4))
    best_t = 0.5
    best_j = -999.0
    for t in candidates:
        pred = score >= t
        tp = np.sum((pred == 1) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


def metric_row(model_cfg: dict, split: str, y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return {
        "experiment_id": model_cfg["experiment_id"],
        "model": model_cfg["model"],
        "method_type": model_cfg["method_type"],
        "data_input": model_cfg["data_input"],
        "split": split,
        "n": int(len(y)),
        "event_rate": round(float(y.mean()), 3) if len(y) else float("nan"),
        "auc": round(auc_score(y, score), 3),
        "threshold": round(float(threshold), 3),
        "sensitivity": round(tp / (tp + fn), 3) if (tp + fn) else float("nan"),
        "specificity": round(tn / (tn + fp), 3) if (tn + fp) else float("nan"),
        "accuracy": round((tp + tn) / len(y), 3) if len(y) else float("nan"),
        "brier": round(float(np.mean((score - y) ** 2)), 3) if len(y) else float("nan"),
        "answers": model_cfg["answers"],
        "uses_raw_sequence": model_cfg["uses_raw_sequence"],
    }


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, reps: int = 200) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + 3)
    aucs = []
    n = len(y)
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(auc_score(y[idx], score[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def model_configs() -> list[dict]:
    static_cols = [
        "age",
        "bmi",
        "diabetes_duration_years",
        "baseline_hba1c_pct",
        "fasting_glucose_mmol_l",
        "c_peptide_2h_ug_l",
    ]
    cgm_summary_cols = [
        "tir_3p9_10_pct",
        "tar_gt10_pct",
        "tar_gt13p9_pct",
        "tbr_lt3p9_pct",
        "mean_glucose_mmol_l",
        "sd_glucose_mmol_l",
        "cv_pct",
        "low_event_count",
        "high_auc_mmol_hour_per_day",
        "night_mean_glucose_mmol_l",
    ]
    sequence_cols = [
        "seq_amp_24h",
        "seq_amp_12h",
        "seq_autocorr_1h",
        "seq_autocorr_24h",
        "seq_instability_mmol_l",
        "seq_spike_count",
        "seq_dawn_rise",
        "seq_postprandial_peak_mean",
        "seq_day_to_day_sd",
    ]
    no_time_cols = [
        "mean_glucose_mmol_l",
        "sd_glucose_mmol_l",
        "cv_pct",
        "seq_instability_mmol_l",
        "seq_spike_count",
    ]
    table_ml_cols = static_cols + cgm_summary_cols + [
        "hba1c_duration_load",
        "bmi_cpeptide_balance",
        "tir_cv_balance",
        "hyper_variability_load",
        "postmeal_instability_load",
    ]
    return [
        {
            "experiment_id": "E0",
            "model": "HbA1c单指标基线",
            "method_type": "临床单指标",
            "data_input": "baseline_hba1c_pct",
            "columns": ["baseline_hba1c_pct"],
            "answers": "单个静态指标能否解释缓解结局",
            "uses_raw_sequence": False,
        },
        {
            "experiment_id": "E1",
            "model": "传统Logistic：静态临床",
            "method_type": "传统统计",
            "data_input": "人口学+BMI+病程+HbA1c+C肽",
            "columns": static_cols,
            "answers": "传统统计基线能做到什么水平",
            "uses_raw_sequence": False,
        },
        {
            "experiment_id": "E2",
            "model": "传统机器学习：表格人工特征",
            "method_type": "非深度表格ML基线",
            "data_input": "静态临床+CGM人工统计特征+非线性交互",
            "columns": table_ml_cols,
            "answers": "人工特征和非线性表格模型是否足够",
            "uses_raw_sequence": False,
        },
        {
            "experiment_id": "E3",
            "model": "CGM人工统计特征模型",
            "method_type": "CGM摘要特征基线",
            "data_input": "TIR/TAR/TBR/CV/高糖暴露等人工CGM摘要",
            "columns": cgm_summary_cols,
            "answers": "只用CGM摘要指标是否能捕捉缓解潜力",
            "uses_raw_sequence": False,
        },
        {
            "experiment_id": "E4",
            "model": "本课题CGM时序大模型PoC",
            "method_type": "5分钟原始序列时序模型代理",
            "data_input": "5分钟CGM原始序列切片、节律、餐后、跨日稳定性嵌入",
            "columns": sequence_cols,
            "answers": "直接读取原始序列的时序表示是否优于人工摘要",
            "uses_raw_sequence": True,
        },
        {
            "experiment_id": "E5",
            "model": "时序模型消融：无时间编码",
            "method_type": "架构消融",
            "data_input": "去除24h/12h节律、黎明和餐后窗口后的序列摘要",
            "columns": no_time_cols,
            "answers": "时间编码、节律和餐后窗口对模型是否必要",
            "uses_raw_sequence": True,
        },
    ]


def train_and_evaluate(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    eligible = features[features["cgm_qc_pass"]].copy()
    train = eligible[eligible["analysis_split"] == "train"].copy()
    internal = eligible[eligible["analysis_split"] == "internal_validation"].copy()
    external = eligible[eligible["analysis_split"] == "external_validation"].copy()

    metrics = []
    predictions = eligible[
        [
            "patient_id",
            "center",
            "analysis_split",
            "remission_12m",
            "baseline_hba1c_pct",
            "bmi",
            "diabetes_duration_years",
            "tir_3p9_10_pct",
            "cv_pct",
            "seq_postprandial_peak_mean",
            "seq_day_to_day_sd",
        ]
    ].copy()
    model_cards = {}
    experiment_rows = []

    for cfg in model_configs():
        cols = cfg["columns"]
        x_train, _, transform = standardize(train, train, cols)
        y_train = train["remission_12m"].to_numpy(dtype=float)
        weights = fit_logistic(x_train, y_train)
        train_score = predict_logistic(x_train, weights)
        threshold = best_threshold(y_train, train_score)
        model_cards[cfg["model"]] = {
            "experiment_id": cfg["experiment_id"],
            "features": cols,
            "transform": transform,
            "weights": [round(float(w), 6) for w in weights],
            "note": "PoC用可复现Logistic头模拟不同输入/架构的实验对比；正式项目应替换为锁定SAP后的真实模型训练。",
        }
        experiment_rows.append({k: cfg[k] for k in ["experiment_id", "model", "method_type", "data_input", "answers", "uses_raw_sequence"]})

        for split_name, split_df in (("train", train), ("internal_validation", internal), ("external_validation", external)):
            _, x_eval, _ = standardize(train, split_df, cols)
            score = predict_logistic(x_eval, weights)
            y_eval = split_df["remission_12m"].to_numpy(dtype=float)
            row = metric_row(cfg, split_name, y_eval, score, threshold)
            if split_name in ("internal_validation", "external_validation") and len(np.unique(y_eval)) == 2:
                ci_low, ci_high = bootstrap_auc_ci(y_eval, score)
                row["auc_bootstrap_95ci"] = f"{ci_low:.3f}-{ci_high:.3f}"
            else:
                row["auc_bootstrap_95ci"] = ""
            metrics.append(row)

        _, x_all, _ = standardize(train, eligible, cols)
        predictions[f"prob_{cfg['experiment_id']}"] = np.round(predict_logistic(x_all, weights), 4)

    primary_prob = predictions["prob_E4"]
    q33 = float(primary_prob.quantile(0.33))
    q67 = float(primary_prob.quantile(0.67))
    predictions["risk_band"] = pd.cut(
        primary_prob,
        bins=[-0.01, q33, q67, 1.01],
        labels=["低缓解潜力", "中缓解潜力", "高缓解潜力"],
    )
    suggestions = {
        "高缓解潜力": "生活方式强化为主；在医师判断下评估降糖药精简窗口。",
        "中缓解潜力": "生活方式+标准药物治疗并重；关注餐后高糖和体重变化。",
        "低缓解潜力": "优先稳定血糖波动、保护残余胰岛功能；药物调整需更精细。",
    }
    predictions["poc_intervention_hint"] = predictions["risk_band"].astype(str).map(suggestions)
    return pd.DataFrame(metrics), predictions, model_cards, pd.DataFrame(experiment_rows)


def write_data_dictionary() -> None:
    rows = [
        ("patient_id", "脱敏受试者编号", "字符串"),
        ("center", "入组中心", "分类"),
        ("analysis_split", "训练/内部验证/外部验证划分", "分类"),
        ("timestamp_utc", "CGM采样时间戳；本版PoC为5分钟粒度", "ISO时间"),
        ("minute_of_day", "一天内分钟数，用于节律/餐后窗口编码", "0-1435"),
        ("cgm_valid_pct", "剔除前2天后CGM有效读数比例", "0-1"),
        ("tir_3p9_10_pct", "目标范围3.9-10.0 mmol/L时间占比", "%"),
        ("tar_gt10_pct", "高于10.0 mmol/L时间占比", "%"),
        ("tar_gt13p9_pct", "显著高糖暴露时间占比", "%"),
        ("tbr_lt3p9_pct", "低于3.9 mmol/L时间占比", "%"),
        ("cv_pct", "血糖变异系数", "%"),
        ("high_auc_mmol_hour_per_day", "日均高糖暴露面积近似值", "mmol/L*hour/day"),
        ("seq_amp_24h", "24小时节律幅度嵌入", "mmol/L"),
        ("seq_amp_12h", "12小时节律幅度嵌入", "mmol/L"),
        ("seq_autocorr_1h", "1小时滞后自相关", "相关系数"),
        ("seq_instability_mmol_l", "5分钟差分绝对值均值", "mmol/L"),
        ("seq_postprandial_peak_mean", "三餐后窗口90分位峰值均值", "mmol/L"),
        ("remission_12m", "模拟的12个月缓解结局", "0/1"),
    ]
    with (REPORT_DIR / "data_dictionary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "description", "unit_or_type"])
        writer.writerows(rows)


def write_model_references() -> None:
    rows = [
        {
            "model": "GluFormer",
            "proposed_by": "Guy Lutsker、Gal Sapir、Smadar Shilo、Eran Segal团队等；Nature 2026",
            "does_what": "自监督生成式CGM基础模型，用自回归预测学习长期代谢健康表征",
            "relation_to_poc": "证明CGM表征可用于长期结局风险分层；本PoC借鉴自监督预训练和可迁移表征，不直接照搬其人群/任务",
            "borrowable": "自回归预训练、跨设备/跨人群泛化验证、下游结局微调",
            "source_url": "https://www.nature.com/articles/s41586-025-09925-9",
        },
        {
            "model": "CGMformer",
            "proposed_by": "Yurun Lu、Dan Liu、Zhongming Liang等；National Science Review 2025",
            "does_what": "注意力Transformer预训练模型，学习个体葡萄糖动态表征，用于筛查、分型和饮食建议等任务",
            "relation_to_poc": "与三中心中国人群CGM队列最接近；可借鉴每日CGM profile token化、masked learning和个体嵌入",
            "borrowable": "日级切片、时间点/个体embedding、masked reconstruction、下游微调",
            "source_url": "https://academic.oup.com/nsr/article/12/5/nwaf039/8005967",
        },
        {
            "model": "CGM-LSM",
            "proposed_by": "Junjie Luo、Abhimanyu Kumbara、Ritu Agarwal、Gordon Gao等；npj Health Systems 2025",
            "does_what": "Large Sensor Model，把CGM数值离散成token，GPT式decoder做近期血糖预测",
            "relation_to_poc": "主要做短期forecasting，不是缓解预测；但可借鉴CGM token化和next-token预训练",
            "borrowable": "离散token、decoder-only预训练、zero-shot泛化评估",
            "source_url": "https://www.nature.com/articles/s44401-025-00039-y",
        },
        {
            "model": "SSM-CGM",
            "proposed_by": "Shakson Isaac、Yentl Collin、Chirag J. Patel；NeurIPS TS4H 2025 workshop/arXiv",
            "does_what": "Mamba状态空间CGM预测模型，结合可穿戴生理信号，强调可解释和反事实模拟",
            "relation_to_poc": "适合长序列高效建模；本项目可借鉴状态空间层、时间归因和临床可解释输出",
            "borrowable": "Mamba/SSM长上下文、变量选择、时间归因、反事实窗口",
            "source_url": "https://github.com/shakson-isaac/SSM-CGM",
        },
        {
            "model": "GluLLM",
            "proposed_by": "Taiyu Zhu、Joanna Howson、Alejo Nevado-Holgado；Oxford/Novo Nordisk Research Centre Oxford",
            "does_what": "面向端侧血糖预测的多模态LLM适配框架，整合CGM、活动日志和EHR",
            "relation_to_poc": "更偏端侧预测和多模态交互；本项目可借鉴adapter和合规部署，但不把LLM直接当缓解预测器",
            "borrowable": "多模态adapter、端侧推理、解释性自然语言报告",
            "source_url": "https://github.com/tndrg/GluLLM",
        },
    ]
    pd.DataFrame(rows).to_csv(REPORT_DIR / "model_references.csv", index=False, encoding="utf-8-sig")


def write_model_design() -> None:
    design = {
        "input_format": [
            {
                "component": "X_cgm",
                "shape": "[B, 12, 288, C]",
                "description": "剔除前2天后的12天有效CGM；5分钟粒度，每天288点；C包含glucose、delta、mask、time_sin、time_cos、day_index、meal_window_proxy。",
            },
            {
                "component": "X_static_optional",
                "shape": "[B, P]",
                "description": "年龄、BMI、病程、HbA1c、C肽等只进入校准/分层头；主实验把静态模型与CGM时序模型分开比较，避免概念混搭。",
            },
            {
                "component": "Y",
                "shape": "[B, 1]",
                "description": "12个月糖尿病缓解结局：停用降糖药至少3个月且HbA1c/FPG达标。",
            },
        ],
        "architecture": [
            {"stage": "质控与对齐", "design": "原始5分钟点位、mask保留缺失，不简单抹平；生成day/time/meal窗口编码。", "output": "标准序列张量+缺失mask"},
            {"stage": "多尺度token化", "design": "5分钟点先入局部卷积/patch embedding；30分钟patch、日级summary token、跨日稳定性token并行。", "output": "patch tokens + day tokens"},
            {"stage": "时序编码器", "design": "Transformer/SSM混合：局部波动用TCN，长程节律用self-attention或Mamba状态空间层。", "output": "time-point embedding与patient embedding"},
            {"stage": "自监督预训练", "design": "masked reconstruction、next-window prediction、TIR/TAR辅助任务，先学CGM生成规律。", "output": "可迁移CGM表征"},
            {"stage": "缓解预测微调", "design": "patient [CLS]/attention pooling接Logistic/MLP head；外部验证前锁定阈值和校准。", "output": "p_remission_12m、risk_band"},
            {"stage": "可解释输出", "design": "attention/gradient/temporal attribution定位餐后、夜间、黎明、高波动窗口。", "output": "top_windows、driver_features、uncertainty"},
        ],
        "output_schema": {
            "patient_id": "POC001",
            "model_version": "cgm_temporal_poc_v2_5min",
            "p_remission_12m": 0.42,
            "risk_band": "中缓解潜力",
            "top_temporal_evidence": ["夜间均值偏高", "晚餐后峰值持续", "跨日波动稳定性一般"],
            "qc_flags": ["CGM有效率>=90%"],
            "calibration_note": "PoC为模拟样例；正式模型需真实三中心队列校准。",
        },
        "training_plan": [
            "阶段1：用全部合格CGM窗口做自监督预训练，不使用缓解标签。",
            "阶段2：训练集微调缓解预测头；内部验证做Bootstrap和校准。",
            "阶段3：120例外部验证队列一次性评估，不参与变量筛选、训练或阈值调参。",
            "阶段4：与HbA1c、静态Logistic、传统ML、CGM人工统计特征、架构消融组比较。",
        ],
    }
    (REPORT_DIR / "model_design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(features: pd.DataFrame, metrics: pd.DataFrame) -> None:
    eligible = features[features["cgm_qc_pass"]]
    excluded = features[~features["cgm_qc_pass"]]
    event_rate = float(eligible["remission_12m"].mean())
    external_primary = metrics[(metrics["model"] == "本课题CGM时序大模型PoC") & (metrics["split"] == "external_validation")].iloc[0]
    summary = f"""# CGM缓解预测PoC一页式结果

## 运行结论

本PoC已调整为5分钟CGM原始时序数据，并把传统Logistic、传统机器学习、CGM人工统计特征和本课题CGM时序大模型作为实验对比组，而不是把它们组合成一个混合模型。

## 样本流转

- 模拟受试者：{len(features)}例，三中心均衡分布。
- CGM原始记录：{len(pd.read_csv(RAW_DIR / "cgm_timeseries.csv")):,}条，14天、5分钟粒度。
- 剔除前2天后每例理论有效读数：{EXPECTED_EFFECTIVE_READINGS:,}条。
- 质控通过：{len(eligible)}例；因CGM有效读数<90%剔除：{len(excluded)}例。
- 质控通过样本的模拟12个月缓解率：{event_rate:.1%}。
- 建模训练/内部验证/外部验证：{(eligible.analysis_split == "train").sum()} / {(eligible.analysis_split == "internal_validation").sum()} / {(eligible.analysis_split == "external_validation").sum()}例。

## 主模型

- 主模型：本课题CGM时序大模型PoC，输入5分钟原始序列嵌入。
- 外部验证AUC：{external_primary["auc"]}。
- 外部验证敏感度/特异度：{external_primary["sensitivity"]} / {external_primary["specificity"]}。

## 重要边界

该PoC结果只证明工程链路和实验设计可跑通，不能作为真实临床预测能力证据。正式项目需要锁定真实数据字典、统计分析计划、模型版本和伦理/数据合规协议后，使用三中心脱敏真实数据重跑。
"""
    (REPORT_DIR / "poc_summary.md").write_text(summary, encoding="utf-8")


def write_audit_manifest(inputs: list[Path], outputs: list[Path], model_cards: dict, features: pd.DataFrame) -> None:
    manifest = {
        "project": "三中心CGM糖尿病缓解预测PoC",
        "data_notice": "全部数据为模拟脱敏样例，不含真实患者信息。",
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "random_seed": SEED,
        "python": platform.python_version(),
        "cgm_interval_minutes": CGM_INTERVAL_MINUTES,
        "readings_per_day": READINGS_PER_DAY,
        "effective_days_after_adaptation": EFFECTIVE_CGM_DAYS,
        "expected_effective_readings_per_patient": EXPECTED_EFFECTIVE_READINGS,
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "sample_flow": {
            "patients_total": int(len(features)),
            "patients_qc_pass": int(features["cgm_qc_pass"].sum()),
            "patients_qc_excluded": int((~features["cgm_qc_pass"]).sum()),
            "splits_after_qc": features[features["cgm_qc_pass"]]["analysis_split"].value_counts().to_dict(),
            "outcome_event_rate_after_qc": round(float(features[features["cgm_qc_pass"]]["remission_12m"].mean()), 4),
        },
        "inputs": [{"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in inputs],
        "outputs": [{"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in outputs if p.exists()],
        "model_cards": model_cards,
    }
    (REPORT_DIR / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    ensure_dirs()

    clinical = simulate_clinical()
    cgm = simulate_cgm(clinical)
    clinical_path = RAW_DIR / "clinical_baseline.csv"
    cgm_path = RAW_DIR / "cgm_timeseries.csv"
    clinical.to_csv(clinical_path, index=False, encoding="utf-8-sig")
    cgm.to_csv(cgm_path, index=False, encoding="utf-8-sig")

    features, qc = extract_features(clinical, cgm)
    features_path = PROCESSED_DIR / "patient_features.csv"
    qc_path = REPORT_DIR / "data_quality_report.csv"
    features.to_csv(features_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    metrics, predictions, model_cards, experiments = train_and_evaluate(features)
    metrics_path = REPORT_DIR / "model_metrics.csv"
    predictions_path = REPORT_DIR / "sample_patient_predictions.csv"
    experiments_path = REPORT_DIR / "model_experiments.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    experiments.to_csv(experiments_path, index=False, encoding="utf-8-sig")

    write_data_dictionary()
    write_model_references()
    write_model_design()
    write_summary(features, metrics)

    outputs = [
        features_path,
        qc_path,
        metrics_path,
        predictions_path,
        experiments_path,
        REPORT_DIR / "data_dictionary.csv",
        REPORT_DIR / "model_references.csv",
        REPORT_DIR / "model_design.json",
        REPORT_DIR / "poc_summary.md",
        REPORT_DIR / "audit_manifest.json",
    ]
    write_audit_manifest([clinical_path, cgm_path], outputs, model_cards, features)

    print("CGM remission PoC finished.")
    print(f"CGM interval: {CGM_INTERVAL_MINUTES} minutes")
    print(f"Summary: {REPORT_DIR / 'poc_summary.md'}")
    print(f"Audit:   {REPORT_DIR / 'audit_manifest.json'}")


if __name__ == "__main__":
    main()
