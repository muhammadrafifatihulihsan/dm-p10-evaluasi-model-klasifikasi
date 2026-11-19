"""Praktikum Data Mining 10: evaluasi model klasifikasi deteksi fraud.

Seluruh angka pada laporan berasal dari eksekusi skrip ini. Dataset dibuat
ulang secara deterministik dengan seed 42, lalu hasilnya disimpan ke folder
``hasil/``.
"""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

AKAR = Path(__file__).resolve().parent.parent
DATA = AKAR / "data"
HASIL = AKAR / "hasil"
DATA.mkdir(exist_ok=True)
HASIL.mkdir(exist_ok=True)

SEED_DATA = 42
SEED_SPLIT = 42
SEED_MODEL = 42


def metrik(tp, tn, fp, fn):
    """Hitung accuracy, precision, recall, dan F1 dari empat sel confusion matrix."""
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def bangkitkan_data():
    """Bangkitkan dataset transaksi pembayaran digital sesuai modul praktikum."""
    rng = np.random.default_rng(SEED_DATA)
    n = 500
    amount = rng.lognormal(mean=5.0, sigma=1.0, size=n)
    hour = rng.integers(0, 24, n)
    new_device = rng.binomial(1, 0.25, n)
    foreign_ip = rng.binomial(1, 0.12, n)
    score = (
        -4.5
        + 0.45 * (amount > 300)
        + 1.1 * new_device
        + 1.5 * foreign_ip
        + 0.7 * (hour < 5)
    )
    prob = 1 / (1 + np.exp(-score))
    is_fraud = rng.binomial(1, prob)
    df = pd.DataFrame(
        {
            "amount": amount.round(2),
            "hour": hour,
            "new_device": new_device,
            "foreign_ip": foreign_ip,
            "isFraud": is_fraud,
        }
    )
    df.to_csv(DATA / "dataset_fraud.csv", index=False)
    return df


def evaluasi():
    df = bangkitkan_data()
    X = df.drop(columns="isFraud")
    y = df["isFraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED_SPLIT, stratify=y
    )
    model = DecisionTreeClassifier(
        max_depth=4, class_weight="balanced", random_state=SEED_MODEL
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    manual = metrik(tp, tn, fp, fn)
    sklearn_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    baseline = np.zeros(len(y_test), dtype=int)
    cm_base = confusion_matrix(y_test, baseline, labels=[0, 1])
    btn, bfp, bfn, btp = [int(v) for v in cm_base.ravel()]
    base_manual = metrik(btp, btn, bfp, bfn)

    y_prob = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    threshold_rows = []
    for threshold in [0.2, 0.4, 0.5, 0.7]:
        pred = (y_prob >= threshold).astype(int)
        cm_t = confusion_matrix(y_test, pred, labels=[0, 1])
        ttn, tfp, tfn, ttp = [int(v) for v in cm_t.ravel()]
        m = metrik(ttp, ttn, tfp, tfn)
        threshold_rows.append(
            {
                "threshold": threshold,
                "tn": ttn,
                "fp": tfp,
                "fn": tfn,
                "tp": ttp,
                "accuracy": m["accuracy"],
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
            }
        )

    tugas1 = {"tp": 24, "tn": 156, "fp": 12, "fn": 8}
    tugas1.update(metrik(24, 156, 12, 8))

    hasil = {
        "lingkungan": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "dataset": {
            "n": int(len(df)),
            "fitur": list(X.columns),
            "kelas_0": int((y == 0).sum()),
            "kelas_1": int((y == 1).sum()),
            "fraud_rate": float(y.mean()),
        },
        "split": {
            "train": int(len(y_train)),
            "test": int(len(y_test)),
            "train_0": int((y_train == 0).sum()),
            "train_1": int((y_train == 1).sum()),
            "test_0": int((y_test == 0).sum()),
            "test_1": int((y_test == 1).sum()),
        },
        "model": {
            "nama": "DecisionTreeClassifier",
            "max_depth": 4,
            "class_weight": "balanced",
            "random_state": SEED_MODEL,
            "kelas": [int(c) for c in model.classes_],
        },
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "matrix": cm.tolist()},
        "metrics_manual": manual,
        "metrics_sklearn": sklearn_metrics,
        "baseline": {
            "tn": btn,
            "fp": bfp,
            "fn": bfn,
            "tp": btp,
            "matrix": cm_base.tolist(),
            **base_manual,
        },
        "threshold": threshold_rows,
        "tugas1": tugas1,
        "classification_report": {
            "baseline": classification_report(
                y_test, baseline, labels=[0, 1], zero_division=0
            ),
            "model": classification_report(
                y_test, y_pred, labels=[0, 1], zero_division=0
            ),
        },
    }

    (HASIL / "hasil_evaluasi.json").write_text(
        json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "kondisi": "Decision Tree",
                **manual,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            },
            {
                "kondisi": "Baseline semua normal",
                **base_manual,
                "tn": btn,
                "fp": bfp,
                "fn": bfn,
                "tp": btp,
            },
        ]
    ).to_csv(HASIL / "tabel_metrik.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(HASIL / "tabel_threshold.csv", index=False)

    print("=== 1. DATASET ===")
    print(df.head().to_string(index=False))
    print()
    print(df["isFraud"].value_counts().sort_index().to_string())
    print(f"Proporsi fraud: {y.mean():.4f}")
    print()
    print("=== 2. SPLIT DAN MODEL ===")
    print(f"Data latih: {len(y_train)} | Data uji: {len(y_test)}")
    print(f"Latih normal/fraud: {(y_train == 0).sum()}/{(y_train == 1).sum()}")
    print(f"Uji normal/fraud: {(y_test == 0).sum()}/{(y_test == 1).sum()}")
    print(model)
    print()
    print("=== 3. CONFUSION MATRIX ===")
    print("Confusion matrix (baris aktual, kolom prediksi):")
    print(cm)
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print()
    print("=== 4. METRIK MANUAL DAN SKLEARN ===")
    print(
        f"Manual   : accuracy={manual['accuracy']:.3f} precision={manual['precision']:.3f} "
        f"recall={manual['recall']:.3f} f1={manual['f1']:.3f}"
    )
    print(
        f"Sklearn  : accuracy={sklearn_metrics['accuracy']:.3f} "
        f"precision={sklearn_metrics['precision']:.3f} "
        f"recall={sklearn_metrics['recall']:.3f} f1={sklearn_metrics['f1']:.3f}"
    )
    print()
    print("=== 5. BASELINE SEMUA NORMAL ===")
    print(f"Baseline accuracy: {base_manual['accuracy']:.4f}")
    print("Classification report baseline:")
    print(hasil["classification_report"]["baseline"])
    print("Classification report Decision Tree:")
    print(hasil["classification_report"]["model"])
    print()
    print("=== 6. PENGARUH THRESHOLD ===")
    print("threshold | precision | recall | f1 | TN | FP | FN | TP")
    for r in threshold_rows:
        print(
            f"{r['threshold']:.1f}       | {r['precision']:.3f}     | {r['recall']:.3f}  "
            f"| {r['f1']:.3f} | {r['tn']} | {r['fp']} | {r['fn']} | {r['tp']}"
        )
    print()
    print("=== 7. TUGAS 1 MANUAL ===")
    print(
        f"TP=24 TN=156 FP=12 FN=8 -> accuracy={tugas1['accuracy']:.4f} "
        f"precision={tugas1['precision']:.4f} recall={tugas1['recall']:.4f} "
        f"f1={tugas1['f1']:.4f}"
    )
    return hasil


if __name__ == "__main__":
    evaluasi()
