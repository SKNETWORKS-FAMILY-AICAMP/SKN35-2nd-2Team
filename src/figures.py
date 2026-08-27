# -*- coding: utf-8 -*-
"""
결과서·PPT 에 붙일 그림.

    reports/figures/01_낙폭비교.png    랜덤 vs 게임분할 — 무엇이 게임을 외웠나
    reports/figures/02_ROC.png         처음 보는 게임에서의 ROC 곡선
    reports/figures/03_글의효과.png     글을 더하면 어떻게 되는가

실행 (레포 루트에서)
    uv run python -m src.figures
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import roc_curve

from src.config import ENC_READ, FIGURES, RESULTS_CSV

# 한글이 네모로 깨지지 않게
for _f in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"):
    if _f in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140

C_RANDOM, C_GROUP, C_TEXT = "#94a3b8", "#2563eb", "#f97316"

LABEL = {
    "로지스틱(숫자만)": "로지스틱\n숫자만",
    "부스팅(숫자만)": "부스팅\n숫자만",
    "MLP(숫자만)": "MLP\n숫자만",
    "MLP(글만)": "MLP\n글만",
    "MLP(숫자+글PCA64)": "MLP\n숫자+글",
    "부스팅(숫자+글PCA64)": "부스팅\n숫자+글",
}
ORDER = list(LABEL)


def _table():
    df = pd.read_csv(RESULTS_CSV, encoding=ENC_READ)
    df = df.drop_duplicates(["모델명", "변수묶음", "분할방식"], keep="last")
    p = df.pivot_table(index="모델명", columns="분할방식", values="AUC")
    e = df[df.분할방식 != "랜덤"].set_index("모델명")["편차"]
    p = p.reindex([m for m in ORDER if m in p.index])
    return p, e.reindex(p.index)


# ── 1. 낙폭 비교 ────────────────────────────────────────────────
def fig_dropoff():
    p, err = _table()
    x = np.arange(len(p)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x - w/2, p["랜덤"], w, label="랜덤 분할 (아는 게임)", color=C_RANDOM)
    ax.bar(x + w/2, p["게임(5)"], w, yerr=err, capsize=3,
           label="게임 단위 분할 (처음 보는 게임)", color=C_GROUP)

    for i, (a, b) in enumerate(zip(p["랜덤"], p["게임(5)"])):
        ax.text(i - w/2, a + .012, f"{a:.3f}", ha="center", fontsize=8.5)
        ax.text(i + w/2, b + .045, f"{b:.3f}", ha="center", fontsize=8.5)
        ax.annotate(f"{b-a:+.3f}", xy=(i, 0.525), ha="center", fontsize=9,
                    color="#b91c1c" if b - a < -0.09 else "#334155",
                    fontweight="bold" if b - a < -0.09 else "normal")

    ax.axhline(0.5, color="#cbd5e1", lw=1, ls="--")
    ax.text(-0.45, 0.507, "동전 던지기", fontsize=8, color="#94a3b8", ha="left")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in p.index], fontsize=9)
    ax.set_ylabel("AUC"); ax.set_ylim(0.48, 0.90)
    ax.set_title("처음 보는 게임에서 얼마나 떨어지는가  —  숫자는 낙폭",
                 fontsize=13, pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=.95)
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES / "01_낙폭비교.png"); plt.close(fig)


# ── 2. ROC ──────────────────────────────────────────────────────
def fig_roc(curves):
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for (name, color, ls), (yt, yp, auc) in curves.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, color=color, ls=ls, lw=2, label=f"{name}  AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#cbd5e1", ls="--", lw=1, label="동전 던지기  0.500")
    ax.set_xlabel("잘못 이탈이라 한 비율 (FPR)")
    ax.set_ylabel("진짜 이탈을 잡은 비율 (TPR)")
    ax.set_title("처음 보는 게임에서의 ROC 곡선\n(게임 단위 분할 1조각)", fontsize=12, pad=12)
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=.25); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES / "02_ROC.png"); plt.close(fig)


# ── 3. 글의 효과 ────────────────────────────────────────────────
def fig_text_effect():
    p, _ = _table()
    pairs = [("부스팅", "부스팅(숫자만)", "부스팅(숫자+글PCA64)"),
             ("MLP", "MLP(숫자만)", "MLP(숫자+글PCA64)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)
    for ax, (title, a, b) in zip(axes, pairs):
        for i, split in enumerate(["랜덤", "게임(5)"]):
            v0, v1 = p.loc[a, split], p.loc[b, split]
            ax.plot([0, 1], [v0, v1], "-o", lw=2.4, ms=7,
                    color=C_RANDOM if i == 0 else C_TEXT,
                    label="랜덤 분할" if i == 0 else "게임 단위 분할")
            ax.annotate(f"{v1-v0:+.3f}", xy=(1.04, v1), fontsize=10, va="center",
                        color="#16a34a" if v1 > v0 else "#b91c1c", fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["숫자만", "숫자+글"])
        ax.set_xlim(-.25, 1.35); ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_ylabel("AUC"); axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("글을 더하면 —  아는 게임에선 그대로, 처음 보는 게임에선 오른다",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(FIGURES / "03_글의효과.png"); plt.close(fig)


if __name__ == "__main__":
    import pandas as _pd
    from sklearn.metrics import roc_auc_score

    from src.embed import load as load_emb
    from src.evaluate import features, load_english, make_splits
    from src.train_dl import (boosting, feature_cols, make_boosting_text,
                              make_mlp_text_pca, mlp_numeric, with_text)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_dropoff(); print("  01_낙폭비교.png")

    d, y, g = load_english(); sp = make_splits(d, g)
    X = features(d, feature_cols("B셋")); Xt = with_text(X, load_emb(d))
    tr, te = sp["group"][0]                      # 게임 단위 1조각
    curves = {}
    for (name, color, ls, Xd, fn) in [
        ("부스팅 · 숫자만",  "#94a3b8", "-",  X,  boosting),
        ("MLP · 숫자만",    "#64748b", "--", X,  mlp_numeric),
        ("MLP · 숫자+글",   "#2563eb", "-",  Xt, make_mlp_text_pca(64)),
        ("부스팅 · 숫자+글", "#f97316", "-",  Xt, make_boosting_text(64)),
    ]:
        pr = fn(Xd.iloc[tr], y[tr], Xd.iloc[te])
        curves[(name, color, ls)] = (y[te], pr, roc_auc_score(y[te], pr))
    fig_roc(curves); print("  02_ROC.png")

    fig_text_effect(); print("  03_글의효과.png")
    print(f"\n저장 위치: {FIGURES}")
