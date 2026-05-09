from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


# Slide/report display names. Keep CLI variant names in result files, but render
# the plot using the model names used in the presentation.
DISPLAY_LABELS = {
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid": "Dense + BM25",
    "dense_bm25": "Dense + BM25",
    "hybrid_ce": "Dense + BM25 + CE rerank",
    "dense_bm25_ce": "Dense + BM25 + CE rerank",
    "hybrid_lora": "Dense + BM25 + fine-tuned rerank",
    "dense_bm25_lora": "Dense + BM25 + fine-tuned rerank",
    "dense_graph": "Dense + Graph",
    "graph_hybrid": "Dense + BM25 + Graph",
    "graph_hybrid_ce": "Dense + BM25 + Graph + CE rerank",
    "graph_hybrid_lora": "Dense + BM25 + Graph + fine-tuned rerank",
}

# Colors/markers chosen to visually match the project slide:
# blue dense, teal hybrid, purple CE, orange LoRA, green graph.
STYLE_MAP = {
    "dense": {"color": "#2166ac", "marker": "o"},
    "bm25": {"color": "#8c8c8c", "marker": "s"},
    "hybrid": {"color": "#2fb3a3", "marker": "s"},
    "dense_bm25": {"color": "#2fb3a3", "marker": "s"},
    "hybrid_ce": {"color": "#7b3294", "marker": "D"},
    "dense_bm25_ce": {"color": "#7b3294", "marker": "D"},
    "hybrid_lora": {"color": "#f15a24", "marker": "*"},
    "dense_bm25_lora": {"color": "#f15a24", "marker": "*"},
    "dense_graph": {"color": "#00897b", "marker": "^"},
    "graph_hybrid": {"color": "#00897b", "marker": "^"},
    "graph_hybrid_ce": {"color": "#5e3c99", "marker": "P"},
    "graph_hybrid_lora": {"color": "#d73027", "marker": "X"},
}

CANONICAL_ORDER = [
    "dense",
    "dense_bm25",
    "dense_bm25_ce",
    "dense_bm25_lora",
    "dense_graph",
    "bm25",
    "hybrid",
    "hybrid_ce",
    "hybrid_lora",
    "graph_hybrid",
    "graph_hybrid_ce",
    "graph_hybrid_lora",
]

# Manual offsets avoid the worst label collisions in the standard five-point
# FinanceBench slide plot. Values are in points, not data coordinates.
LABEL_OFFSETS = {
    "dense": (8, -14),
    "dense_bm25": (8, 7),
    "hybrid": (8, 7),
    "dense_bm25_ce": (8, 7),
    "hybrid_ce": (8, 7),
    "dense_bm25_lora": (8, 7),
    "hybrid_lora": (8, 7),
    "dense_graph": (8, 7),
    "graph_hybrid": (8, 7),
    "graph_hybrid_ce": (8, 7),
    "graph_hybrid_lora": (8, 7),
}


def _variant_key(value: object) -> str:
    return str(value).strip()


def _display_label(variant: str) -> str:
    return DISPLAY_LABELS.get(variant, variant)


def _style(variant: str) -> dict[str, object]:
    return STYLE_MAP.get(variant, {"color": "#1f77b4", "marker": "o"})


def _format_accuracy(value: float) -> str:
    # Prefer whole percentages for slide readability. Use one decimal only when
    # the value is not close to an integer.
    if abs(value - round(value)) < 0.05:
        return f"{round(value):.0f}%"
    return f"{value:.1f}%"


def _prepare_frame(summary_files: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for p in summary_files:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Missing summary file: {path}")
        frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError("No summary files supplied")
    df = pd.concat(frames, ignore_index=True)
    required = {"variant", "latency_ms_mean", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary files are missing required columns: {sorted(missing)}")
    df = df.copy()
    df["variant"] = df["variant"].map(_variant_key)
    df["accuracy_pct"] = df["accuracy"].astype(float) * 100.0
    df["latency_ms_mean"] = df["latency_ms_mean"].astype(float)
    order = {v: i for i, v in enumerate(CANONICAL_ORDER)}
    df["_order"] = df["variant"].map(lambda v: order.get(v, len(order) + 1))
    return df.sort_values(["_order", "variant"]).reset_index(drop=True)


def plot_accuracy_latency_from_summary(summary_files: Iterable[str | Path], out_path: str | Path) -> None:
    """Create a presentation-style accuracy/latency scatter plot.

    The output intentionally mirrors the deck style: labeled points, variant-
    specific markers/colors, an ideal-direction arrow, bottom legend, and an
    optional dashed highlight around Dense + BM25 when present.
    """
    df = _prepare_frame(summary_files)

    fig = plt.figure(figsize=(10.8, 6.4))
    ax = fig.add_subplot(111)

    for _, row in df.iterrows():
        variant = str(row["variant"])
        style = _style(variant)
        x = float(row["latency_ms_mean"])
        y = float(row["accuracy_pct"])
        size = 145 if style["marker"] == "*" else 78
        ax.scatter(
            x,
            y,
            s=size,
            marker=str(style["marker"]),
            color=str(style["color"]),
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        dx, dy = LABEL_OFFSETS.get(variant, (8, 7))
        label = f"{_display_label(variant)}\n({_format_accuracy(y)}, {x:.0f} ms)"
        ax.annotate(
            label,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color="#14213d",
            zorder=4,
        )

    # Data-driven axis bounds with enough margin for labels. The plot keeps the
    # actual measured millisecond scale rather than forcing the example slide's
    # 250-800 ms range.
    xs = df["latency_ms_mean"].astype(float)
    ys = df["accuracy_pct"].astype(float)
    x_range = max(float(xs.max() - xs.min()), 1.0)
    y_range = max(float(ys.max() - ys.min()), 1.0)
    ax.set_xlim(max(0.0, float(xs.min()) - 0.10 * x_range), float(xs.max()) + 0.16 * x_range)
    ax.set_ylim(max(0.0, float(ys.min()) - max(1.0, 0.10 * y_range)), float(ys.max()) + max(1.0, 0.16 * y_range))

    # Ideal direction arrow: higher accuracy and lower latency.
    ax.annotate(
        "",
        xy=(0.06, 0.88),
        xytext=(0.20, 0.70),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#2fb3a3", lw=1.9),
        zorder=2,
    )
    ax.text(
        0.14,
        0.80,
        "Ideal direction:\nhigher accuracy,\nlower latency",
        transform=ax.transAxes,
        color="#208f86",
        fontsize=9,
        va="center",
    )

    # Highlight the vanilla hybrid point, matching the dashed orange callout in
    # the original slide. Prefer dense_bm25, then hybrid if that alias is used.
    highlight_rows = df[df["variant"].isin(["dense_bm25", "hybrid"])]
    if not highlight_rows.empty:
        h = highlight_rows.iloc[0]
        ellipse = Ellipse(
            (float(h["latency_ms_mean"]), float(h["accuracy_pct"])),
            width=max(x_range * 0.26, 80.0),
            height=max(y_range * 0.20, 0.65),
            edgecolor="#f15a24",
            facecolor="none",
            linestyle=(0, (5, 4)),
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(ellipse)

    ax.set_title("End-to-End Accuracy vs. Latency", fontsize=17, pad=12)
    ax.set_xlabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#14213d")
        spine.set_linewidth(1.0)

    legend_handles: list[Line2D] = []
    seen_labels: set[str] = set()
    for _, row in df.iterrows():
        variant = str(row["variant"])
        label = _display_label(variant)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        style = _style(variant)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=str(style["marker"]),
                color="none",
                label=label,
                markerfacecolor=str(style["color"]),
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=9 if style["marker"] != "*" else 12,
            )
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=min(len(legend_handles), 5),
            frameon=False,
            fontsize=9,
            handletextpad=0.45,
            columnspacing=1.0,
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
