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


# ── 4. 글을 읽는 방식별 성능 ────────────────────────────────────
def fig_text_ladder():
    """
    글을 다루는 방식을 점점 좋게 하면서 성능이 어떻게 변하는지.
    "글이 도움이 되는가" 라는 질문의 최종 답이다.
    """
    import matplotlib.pyplot as plt

    names = ["글만\n(숫자 없이)", "숫자 + 글\n고정 임베딩",
             "숫자 + 글\n원문 직접 읽기", "숫자만\n(글 안 씀)"]
    rand  = [0.7047, 0.7770, 0.8074, 0.8129]
    grp   = [0.6611, 0.7295, 0.7593, 0.7504]
    x = np.arange(len(names)); w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.bar(x - w/2, rand, w, label="랜덤 분할 (아는 게임)", color=C_RANDOM)
    ax.bar(x + w/2, grp,  w, label="게임 단위 분할 (처음 보는 게임)", color=C_GROUP)
    for i, (a_, b_) in enumerate(zip(rand, grp)):
        ax.text(i - w/2, a_ + .004, f"{a_:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, b_ + .004, f"{b_:.3f}", ha="center", fontsize=9)

    # 고정 임베딩 -> 원문 직접 읽기 : 두 분할에서 같은 폭으로 오른다
    for off, vals, col in [(-w/2, rand, "#16a34a"), (w/2, grp, "#16a34a")]:
        ax.annotate("", xy=(2 + off, vals[2] - .004), xytext=(1 + off, vals[1] - .004),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.8))
    ax.text(1.5, 0.845, "제대로 읽으면  랜덤 +0.030 · 게임 +0.030",
            ha="center", fontsize=10, color="#16a34a", fontweight="bold")

    ax.axhline(0.8129, color="#94a3b8", ls="--", lw=1)
    ax.text(3.42, 0.8145, "숫자만 (랜덤)", fontsize=8, color="#94a3b8", ha="right")

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("AUC"); ax.set_ylim(0.63, 0.865)
    ax.set_title("글을 아무리 잘 읽혀도 숫자를 넘지 못했다  —  단, 처음 보는 게임에서는 도움이 된다",
                 fontsize=12.5, pad=14)
    ax.legend(loc="lower right", fontsize=9, framealpha=.95)
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIGURES / "05_글을읽는방식.png"); plt.close(fig)


# ── 6. 학습 곡선 — 데이터를 더 넣으면 오르나 ────────────────────
def fig_learning_curve():
    """
    데이터 양이 병목인지 확인한 결과.
    마지막 구간이 평평하면 "더 모아도 안 오른다" 는 뜻이다.
    """
    import matplotlib.pyplot as plt

    n    = [5368, 13422, 26844, 40266, 53689]
    auc  = [0.7373, 0.7564, 0.7657, 0.7752, 0.7770]
    # 팀원이 같은 모델을 2.1배 데이터로 돌린 결과 (게임분할 평균)
    full = {"영어 6.7만": 0.7389, "전체 13.9만": 0.7365}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                  gridspec_kw={"width_ratios": [1.7, 1]})

    ax.plot(n, auc, "-o", lw=2.4, ms=8, color=C_GROUP)
    for x, v in zip(n, auc):
        ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    for i in range(1, len(n)):
        d = auc[i] - auc[i - 1]
        ax.annotate(f"{d:+.4f}", ((n[i] + n[i-1]) / 2, (auc[i] + auc[i-1]) / 2),
                    textcoords="offset points", xytext=(6, -20), ha="center",
                    fontsize=9, color="#16a34a" if d > .005 else "#b91c1c",
                    fontweight="bold" if d <= .005 else "normal")
    ax.annotate("여기서 평평해진다", xy=(53689, 0.7770), xytext=(38000, 0.7885),
                fontsize=9.5, color="#b91c1c",
                arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.4))
    ax.set_xlabel("학습에 쓴 행 수"); ax.set_ylabel("AUC (랜덤 분할)")
    ax.set_title("데이터를 늘리면 계속 오르는가", fontsize=12, pad=10)
    ax.set_ylim(0.725, 0.795)
    ax.grid(alpha=.25); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax2.bar(list(full), list(full.values()), color=[C_GROUP, C_RANDOM], width=.55)
    for i, v in enumerate(full.values()):
        ax2.text(i, v + .0008, f"{v:.4f}", ha="center", fontsize=10)
    ax2.set_ylim(0.72, 0.75); ax2.set_ylabel("AUC (게임 단위, 4개 모델 평균)")
    ax2.set_title("데이터 2.1배로 늘리면", fontsize=12, pad=10)
    ax2.grid(axis="y", alpha=.25); ax2.set_axisbelow(True)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)

    fig.suptitle("데이터 양은 병목이 아니었다", fontsize=13.5)
    fig.tight_layout(); fig.savefig(FIGURES / "06_학습곡선.png"); plt.close(fig)


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
    fig_text_ladder(); print("  05_글을읽는방식.png")
    fig_learning_curve(); print("  06_학습곡선.png")
    print(f"\n저장 위치: {FIGURES}")
