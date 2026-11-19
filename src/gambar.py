"""Membuat gambar hasil evaluasi dari berkas JSON."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

AKAR = Path(__file__).resolve().parent.parent
HASIL = AKAR / "hasil"
GAMBAR = AKAR / "gambar"
GAMBAR.mkdir(exist_ok=True)
DATA = json.loads((HASIL / "hasil_evaluasi.json").read_text(encoding="utf-8"))

HITAM = "#1a1a1a"
ABU = "#555555"
ABU_MUDA = "#e8e8e8"
ABU_TUA = "#8c8c8c"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": HITAM,
    "text.color": HITAM,
    "savefig.facecolor": "white",
})


def simpan(fig, nama):
    fig.savefig(GAMBAR / nama, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("tulis", nama)


def distribusi():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    nilai = [DATA["dataset"]["kelas_0"], DATA["dataset"]["kelas_1"]]
    batang = ax.bar(["Normal (0)", "Fraud (1)"], nilai,
                    color=[ABU_MUDA, HITAM], edgecolor=HITAM, width=0.5)
    for b, n in zip(batang, nilai):
        ax.text(b.get_x() + b.get_width() / 2, n + 8, f"{n}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Jumlah transaksi")
    ax.set_ylim(0, max(nilai) * 1.18)
    ax.set_title("Distribusi kelas dataset (500 transaksi)", fontsize=11.5)
    ax.grid(axis="y", linestyle=":", color="#aaaaaa", alpha=0.6)
    ax.set_axisbelow(True)
    simpan(fig, "distribusi_kelas.png")


def confusion():
    cm = np.array(DATA["confusion"]["matrix"])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    cmap = LinearSegmentedColormap.from_list("flat", ["#ffffff", "#bdbdbd", "#4d4d4d"])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            warna = "white" if cm[i, j] > cm.max() / 2 else HITAM
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=warna)
    ax.set_xticks([0, 1], ["Prediksi normal", "Prediksi fraud"])
    ax.set_yticks([0, 1], ["Aktual normal", "Aktual fraud"])
    ax.set_title("Confusion matrix Decision Tree", fontsize=11.5)
    ax.set_xlabel("Label prediksi")
    ax.set_ylabel("Label aktual")
    simpan(fig, "confusion_matrix.png")


def threshold():
    rows = DATA["threshold"]
    x = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(x, [r["precision"] for r in rows], marker="o", color=HITAM,
            linewidth=1.8, label="Precision")
    ax.plot(x, [r["recall"] for r in rows], marker="s", color=ABU_TUA,
            linewidth=1.8, linestyle="--", label="Recall")
    ax.plot(x, [r["f1"] for r in rows], marker="^", color=ABU,
            linewidth=1.4, linestyle=":", label="F1-score")
    ax.set_xlabel("Ambang probabilitas fraud")
    ax.set_ylabel("Nilai metrik")
    ax.set_ylim(0, 0.55)
    ax.set_xticks(x)
    ax.set_title("Precision, recall, dan F1 pada empat ambang", fontsize=11.5)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(linestyle=":", color="#aaaaaa", alpha=0.6)
    ax.set_axisbelow(True)
    simpan(fig, "threshold.png")


def perbandingan():
    m = DATA["metrics_manual"]
    b = DATA["baseline"]
    label = ["Accuracy", "Precision", "Recall", "F1"]
    model = [m["accuracy"], m["precision"], m["recall"], m["f1"]]
    base = [b["accuracy"], b["precision"], b["recall"], b["f1"]]
    x = np.arange(len(label))
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - 0.18, base, 0.36, label="Baseline semua normal",
           color=ABU_MUDA, edgecolor=HITAM)
    ax.bar(x + 0.18, model, 0.36, label="Decision Tree",
           color="#4d4d4d", edgecolor=HITAM)
    ax.set_xticks(x, label)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Nilai metrik")
    ax.set_title("Perbandingan baseline dan Decision Tree", fontsize=11.5)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", linestyle=":", color="#aaaaaa", alpha=0.6)
    ax.set_axisbelow(True)
    simpan(fig, "perbandingan_metrik.png")


if __name__ == "__main__":
    distribusi()
    confusion()
    threshold()
    perbandingan()
