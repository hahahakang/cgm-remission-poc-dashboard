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
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


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
            0.48 + 0.08 * (p["baseline_hba1c_pct"] - 7.0) + 0.22 * phenotype + rng.normal(0, 0.08),
            0.30,
            1.35,
        )
        miss_prob = 0.16 if p["patient_id"] in high_missing_patients else rng.uniform(0.01, 0.05)

        for t in range(14 * 24 * 4):
            ts = start + timedelta(minutes=15 * t)
            day_fraction = (ts.hour * 60 + ts.minute) / 1440
            circadian = 0.35 * math.sin(2 * math.pi * (day_fraction - 0.18))
            meal = 0.0
            for meal_center, meal_size in ((8 / 24, 1.1), (12.5 / 24, 1.5), (18.5 / 24, 1.7)):
                delta = min(abs(day_fraction - meal_center), 1 - abs(day_fraction - meal_center))
                meal += postprandial_surge * meal_size * math.exp(-0.5 * (delta / 0.035) ** 2)
            weekend_effect = 0.25 if ts.weekday() >= 5 else 0.0
            glucose = base + circadian + meal + weekend_effect + rng.normal(0, variability)
            glucose = float(np.clip(glucose, 2.5, 22.0))
            if rng.random() < miss_prob:
                glucose_value = ""
            else:
                glucose_value = round(glucose, 2)
            rows.append(
                {
                    "patient_id": p["patient_id"],
                    "timestamp_utc": ts.isoformat(),
                    "sensor_day": t // 96 + 1,
                    "glucose_mmol_l": glucose_value,
                }
            )
    return pd.DataFrame(rows)


def count_low_events(values: pd.Series) -> int:
    low = values < 3.9
    return int((low & ~low.shift(fill_value=False)).sum())


def extract_features(clinical: pd.DataFrame, cgm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cgm = cgm.copy()
    cgm["glucose_mmol_l"] = pd.to_numeric(cgm["glucose_mmol_l"], errors="coerce")
    effective = cgm[cgm["sensor_day"] >= 3].copy()
    expected = 12 * 24 * 4

    feature_rows = []
    qc_rows = []
    for patient_id, g in effective.groupby("patient_id", sort=True):
        valid = g["glucose_mmol_l"].dropna()
        valid_pct = len(valid) / expected
        qc_pass = valid_pct >= 0.90
        if len(valid) == 0:
            feature = {
                "patient_id": patient_id,
                "cgm_valid_pct": 0.0,
                "cgm_qc_pass": False,
                "tir_3p9_10_pct": np.nan,
                "tar_gt10_pct": np.nan,
                "tbr_lt3p9_pct": np.nan,
                "mean_glucose_mmol_l": np.nan,
                "sd_glucose_mmol_l": np.nan,
                "cv_pct": np.nan,
                "low_event_count": np.nan,
                "high_auc_mmol_day": np.nan,
                "night_mean_glucose_mmol_l": np.nan,
            }
        else:
            night = g[g["timestamp_utc"].str[11:13].astype(int).between(0, 5)]["glucose_mmol_l"].dropna()
            feature = {
                "patient_id": patient_id,
                "cgm_valid_pct": round(valid_pct, 4),
                "cgm_qc_pass": qc_pass,
                "tir_3p9_10_pct": round(float(((valid >= 3.9) & (valid <= 10.0)).mean() * 100), 2),
                "tar_gt10_pct": round(float((valid > 10.0).mean() * 100), 2),
                "tbr_lt3p9_pct": round(float((valid < 3.9).mean() * 100), 2),
                "mean_glucose_mmol_l": round(float(valid.mean()), 2),
                "sd_glucose_mmol_l": round(float(valid.std(ddof=0)), 2),
                "cv_pct": round(float(valid.std(ddof=0) / valid.mean() * 100), 2),
                "low_event_count": count_low_events(valid),
                "high_auc_mmol_day": round(float(np.maximum(valid - 10.0, 0).sum() * 0.25 / 12), 2),
                "night_mean_glucose_mmol_l": round(float(night.mean()), 2),
            }
        feature_rows.append(feature)
        qc_rows.append(
            {
                "patient_id": patient_id,
                "valid_readings_after_day2": int(len(valid)),
                "expected_readings_after_day2": expected,
                "valid_pct": round(valid_pct, 4),
                "qc_pass": bool(qc_pass),
                "exclusion_reason": "" if qc_pass else "CGM有效读数<90%",
            }
        )

    features = clinical.merge(pd.DataFrame(feature_rows), on="patient_id", how="left")
    features = assign_outcomes(features)
    qc = pd.DataFrame(qc_rows).merge(clinical[["patient_id", "center", "analysis_split"]], on="patient_id", how="left")
    return features, qc


def assign_outcomes(features: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    df = features.copy()
    tir = df["tir_3p9_10_pct"].fillna(df["tir_3p9_10_pct"].median())
    cv = df["cv_pct"].fillna(df["cv_pct"].median())
    high_auc = df["high_auc_mmol_day"].fillna(df["high_auc_mmol_day"].median())
    logit = (
        -2.15
        - 0.42 * (df["baseline_hba1c_pct"] - 7.6)
        - 0.40 * (df["diabetes_duration_years"] - 2.0)
        - 0.08 * (df["bmi"] - 29.0)
        + 0.34 * (df["c_peptide_2h_ug_l"] - 3.8)
        + 0.030 * (tir - 62.0)
        - 0.045 * (cv - 24.0)
        - 0.10 * high_auc
        - 0.05 * df["glp1ra_at_baseline"]
    )
    probability = sigmoid(logit)
    df["remission_probability_simulated"] = np.round(probability, 4)
    df["remission_12m"] = (rng.random(len(df)) < probability).astype(int)
    return df


def standardize(train: pd.DataFrame, other: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    medians = train[columns].median(numeric_only=True)
    train_filled = train[columns].fillna(medians)
    other_filled = other[columns].fillna(medians)
    means = train_filled.mean()
    stds = train_filled.std(ddof=0).replace(0, 1)
    train_x = ((train_filled - means) / stds).to_numpy(dtype=float)
    other_x = ((other_filled - means) / stds).to_numpy(dtype=float)
    return train_x, other_x, {"medians": medians.to_dict(), "means": means.to_dict(), "stds": stds.to_dict()}


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


def metric_row(model: str, split: str, y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return {
        "model": model,
        "split": split,
        "n": int(len(y)),
        "event_rate": round(float(y.mean()), 3) if len(y) else float("nan"),
        "auc": round(auc_score(y, score), 3),
        "threshold": round(float(threshold), 3),
        "sensitivity": round(tp / (tp + fn), 3) if (tp + fn) else float("nan"),
        "specificity": round(tn / (tn + fp), 3) if (tn + fp) else float("nan"),
        "accuracy": round((tp + tn) / len(y), 3) if len(y) else float("nan"),
        "brier": round(float(np.mean((score - y) ** 2)), 3) if len(y) else float("nan"),
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


def train_and_evaluate(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eligible = features[features["cgm_qc_pass"]].copy()
    train = eligible[eligible["analysis_split"] == "train"].copy()
    internal = eligible[eligible["analysis_split"] == "internal_validation"].copy()
    external = eligible[eligible["analysis_split"] == "external_validation"].copy()

    feature_sets = {
        "静态临床Logistic": [
            "age",
            "bmi",
            "diabetes_duration_years",
            "baseline_hba1c_pct",
            "fasting_glucose_mmol_l",
            "c_peptide_2h_ug_l",
        ],
        "CGM特征模型": [
            "tir_3p9_10_pct",
            "tar_gt10_pct",
            "tbr_lt3p9_pct",
            "mean_glucose_mmol_l",
            "sd_glucose_mmol_l",
            "cv_pct",
            "low_event_count",
            "high_auc_mmol_day",
        ],
        "临床+CGM融合模型": [
            "age",
            "bmi",
            "diabetes_duration_years",
            "baseline_hba1c_pct",
            "fasting_glucose_mmol_l",
            "c_peptide_2h_ug_l",
            "tir_3p9_10_pct",
            "tar_gt10_pct",
            "mean_glucose_mmol_l",
            "cv_pct",
            "low_event_count",
            "high_auc_mmol_day",
        ],
    }

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
        ]
    ].copy()
    model_cards = {}

    for model_name, cols in feature_sets.items():
        x_train, _, transform = standardize(train, train, cols)
        y_train = train["remission_12m"].to_numpy(dtype=float)
        weights = fit_logistic(x_train, y_train)
        train_score = predict_logistic(x_train, weights)
        threshold = best_threshold(y_train, train_score)
        model_cards[model_name] = {"features": cols, "transform": transform, "weights": [round(float(w), 6) for w in weights]}

        for split_name, split_df in (("train", train), ("internal_validation", internal), ("external_validation", external)):
            _, x_eval, _ = standardize(train, split_df, cols)
            score = predict_logistic(x_eval, weights)
            y_eval = split_df["remission_12m"].to_numpy(dtype=float)
            row = metric_row(model_name, split_name, y_eval, score, threshold)
            if split_name in ("internal_validation", "external_validation") and len(np.unique(y_eval)) == 2:
                ci_low, ci_high = bootstrap_auc_ci(y_eval, score)
                row["auc_bootstrap_95ci"] = f"{ci_low:.3f}-{ci_high:.3f}"
            else:
                row["auc_bootstrap_95ci"] = ""
            metrics.append(row)

        _, x_all, _ = standardize(train, eligible, cols)
        predictions[f"prob_{model_name}"] = np.round(predict_logistic(x_all, weights), 4)

    fused = predictions["prob_临床+CGM融合模型"]
    q33 = float(fused.quantile(0.33))
    q67 = float(fused.quantile(0.67))
    predictions["risk_band"] = pd.cut(
        fused,
        bins=[-0.01, q33, q67, 1.01],
        labels=["低缓解潜力", "中缓解潜力", "高缓解潜力"],
    )
    suggestions = {
        "高缓解潜力": "生活方式强化为主；在医师判断下评估降糖药精简窗口。",
        "中缓解潜力": "生活方式+标准药物治疗并重；关注餐后高糖和体重变化。",
        "低缓解潜力": "优先稳定血糖波动、保护残余胰岛功能；药物调整需更精细。",
    }
    predictions["poc_intervention_hint"] = predictions["risk_band"].astype(str).map(suggestions)
    return pd.DataFrame(metrics), predictions, model_cards


def write_data_dictionary() -> None:
    rows = [
        ("patient_id", "脱敏受试者编号", "字符串"),
        ("center", "入组中心", "分类"),
        ("analysis_split", "训练/内部验证/外部验证划分", "分类"),
        ("cgm_valid_pct", "剔除前2天后CGM有效读数比例", "0-1"),
        ("tir_3p9_10_pct", "目标范围3.9-10.0 mmol/L时间占比", "%"),
        ("tar_gt10_pct", "高于10.0 mmol/L时间占比", "%"),
        ("tbr_lt3p9_pct", "低于3.9 mmol/L时间占比", "%"),
        ("cv_pct", "血糖变异系数", "%"),
        ("high_auc_mmol_day", "高糖暴露面积近似值", "mmol/L*day"),
        ("remission_12m", "模拟的12个月缓解结局", "0/1"),
    ]
    with (REPORT_DIR / "data_dictionary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "description", "unit_or_type"])
        writer.writerows(rows)


def write_summary(features: pd.DataFrame, qc: pd.DataFrame, metrics: pd.DataFrame) -> None:
    eligible = features[features["cgm_qc_pass"]]
    excluded = features[~features["cgm_qc_pass"]]
    event_rate = float(eligible["remission_12m"].mean())
    external_fused = metrics[(metrics["model"] == "临床+CGM融合模型") & (metrics["split"] == "external_validation")].iloc[0]
    external_static = metrics[(metrics["model"] == "静态临床Logistic") & (metrics["split"] == "external_validation")].iloc[0]
    summary = f"""# CGM缓解预测PoC一页式结果

## 运行结论

本PoC使用模拟脱敏样例数据，完整跑通了三中心CGM真实世界队列的最小工程闭环：原始数据落地、CGM质量门禁、特征提取、训练/内部验证/外部验证隔离、对照模型评估、个体风险分层、审计留痕。

## 样本流转

- 模拟受试者：{len(features)}例，三中心均衡分布。
- CGM原始记录：{len(pd.read_csv(RAW_DIR / "cgm_timeseries.csv")):,}条，14天、15分钟粒度。
- 质控通过：{len(eligible)}例；因CGM有效读数<90%剔除：{len(excluded)}例。
- 质控通过样本的模拟12个月缓解率：{event_rate:.1%}。
- 建模训练/内部验证/外部验证：{(eligible.analysis_split == "train").sum()} / {(eligible.analysis_split == "internal_validation").sum()} / {(eligible.analysis_split == "external_validation").sum()}例。

## 模型对照

- 外部验证静态临床模型AUC：{external_static["auc"]}。
- 外部验证临床+CGM融合模型AUC：{external_fused["auc"]}。
- 融合模型外部验证敏感度/特异度：{external_fused["sensitivity"]} / {external_fused["specificity"]}。

## 可审计交付件

- `data_quality_report.csv`记录每例CGM有效率和剔除原因。
- `model_metrics.csv`记录各模型、各验证集的AUC、敏感度、特异度、Brier分数。
- `sample_patient_predictions.csv`记录个体预测概率、风险分层和PoC干预提示。
- `audit_manifest.json`记录脚本hash、输入输出hash、随机种子和运行环境。

## 重要边界

该PoC结果仅证明工程链路可跑通，不能作为真实疗效或临床预测能力证据。正式项目需要锁定真实数据字典、统计分析计划、质控规则、模型版本和伦理/数据合规协议后，使用三中心脱敏真实数据重跑。
"""
    (REPORT_DIR / "poc_summary.md").write_text(summary, encoding="utf-8")


def write_audit_manifest(inputs: list[Path], outputs: list[Path], model_cards: dict, features: pd.DataFrame) -> None:
    manifest = {
        "project": "三中心CGM糖尿病缓解预测PoC",
        "data_notice": "全部数据为模拟脱敏样例，不含真实患者信息。",
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "random_seed": SEED,
        "python": platform.python_version(),
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

    metrics, predictions, model_cards = train_and_evaluate(features)
    metrics_path = REPORT_DIR / "model_metrics.csv"
    predictions_path = REPORT_DIR / "sample_patient_predictions.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    write_data_dictionary()
    write_summary(features, qc, metrics)

    outputs = [
        features_path,
        qc_path,
        metrics_path,
        predictions_path,
        REPORT_DIR / "data_dictionary.csv",
        REPORT_DIR / "poc_summary.md",
        REPORT_DIR / "audit_manifest.json",
    ]
    write_audit_manifest([clinical_path, cgm_path], outputs, model_cards, features)

    print("CGM remission PoC finished.")
    print(f"Summary: {REPORT_DIR / 'poc_summary.md'}")
    print(f"Audit:   {REPORT_DIR / 'audit_manifest.json'}")


if __name__ == "__main__":
    main()
