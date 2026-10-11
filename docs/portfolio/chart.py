"""Portfolio image 05: measured results per stage, read from eval/results/runs/*.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # the project folder
RUNS = ROOT / "eval" / "results" / "runs"
OUT = ROOT / "docs" / "portfolio" / "05-measured-results.png"

stages = [
    ("Vector search", "baseline"),
    ("+ BM25 hybrid", "hybrid"),
    ("+ Reranker", "rerank"),
    ("+ Second try &\ndate-aware prompt", "a10"),
]
answer, hit5, labels = [], [], []
for label, run in stages:
    s = json.loads((RUNS / f"{run}.json").read_text(encoding="utf-8"))["summary"]
    labels.append(label)
    answer.append(round(s["score"] * 100))
    hit5.append(round(s["hit5"] * 100))
# A14 (eval/results/a14_langchain.md): the LangChain chain, same exam, same model, prompt and judge.
labels.append("LangChain\nout of the box")
answer.append(83)
hit5.append(83)

BG, FG, MUTED = "#0b0d10", "#f2f6fa", "#8fa0b4"
BLUE, GREY = "#2f7fd4", "#ccd7e4"
fig, ax = plt.subplots(figsize=(15.68, 6.98), dpi=100)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
x = range(len(labels))
w = 0.36
b1 = ax.bar([i - w / 2 for i in x], answer, w, color=BLUE, label="Answer score (LLM judge)")
b2 = ax.bar([i + w / 2 for i in x], hit5, w, color=GREY, label="Retrieval hit@5")
for bars in (b1, b2):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{bar.get_height():.0f}%",
            ha="center",
            va="bottom",
            color=FG,
            fontsize=15,
            fontweight="bold",
        )
for bars in (b1, b2):
    bars[-1].set_alpha(0.45)
ax.axvline(3.5, color=MUTED, lw=1, ls="--")
ax.set_xticks(list(x), labels, color=FG, fontsize=13)
ax.set_ylim(50, 104)
ax.set_yticks(
    [50, 60, 70, 80, 90, 100], [f"{v}%" for v in (50, 60, 70, 80, 90, 100)], color=MUTED, fontsize=11
)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=12, ncol=2)
fig.suptitle(
    "Same 55-question exam at every stage",
    color=FG,
    fontsize=22,
    fontweight="bold",
    x=0.06,
    ha="left",
    y=0.95,
)
fig.text(
    0.06,
    0.865,
    "46 answerable + 9 that must be refused. Refusals 9/9 and faithfulness 100% at every stage. "
    "Model: Gemini 3.5 Flash-Lite, judge: NVIDIA Nemotron.",
    color=MUTED,
    fontsize=12,
    ha="left",
)
fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.14)
fig.savefig(OUT, facecolor=BG)
print("saved", OUT, list(zip(labels, answer, hit5)))
