#!/usr/bin/env python3
"""
Generate publication-grade figures for the Qwen38-Flash-Next KV cache capacity
experiment directly from raw experiment artifacts.

Outputs (saved to wiki/docs/public/img/qwen38/):
  1. qwen38_throughput_scaling.png      (2x2: throughput, gain vs 1:1, marginal ratio, reqs + R+O tok/s)
  2. qwen38_timeseries_dynamics.png     (3x1: 1800s KV usage, waiting queue, prefix cache hit rate)
  3. qwen38_mechanism_phase_diagram.png (1x3: KV P95 -> tok/s, PC hit -> tok/s, waiting -> tok/s)
  4. qwen38_latency_tail_breakdown.png  (1x3: TTFT P50/P95, ITL P50/P95, queue P95 share of TTFT P95)
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths & experiment arm manifest (9 verified formal arms)
# ---------------------------------------------------------------------------
ROOT = Path("/Users/a19892608905/Desktop/kv cache")
ARMS_DIR = ROOT / "A4_qwen38_capacity_20261122" / "results" / "arms"
OUT_DIR = ROOT / "wiki" / "docs" / "public" / "img" / "qwen38"

ARM_MANIFEST: list[tuple[int, int, str]] = [
    (16, 100, "formal-exact-stop-20261119T025631Z-b100"),
    (16, 110, "formal-resume-20261119T112248Z-b110"),
    (16, 120, "formal-resume-20261119T121334Z-b120"),
    (16, 130, "formal-resume-20261119T121334Z-b130"),
    (20, 100, "formal-conc20-20261120T063934Z-b100"),
    (20, 110, "formal-conc20-20261120T063934Z-b110"),
    (20, 120, "formal-conc20-20261120T063934Z-b120"),
    (20, 130, "formal-conc20-20261122T073835Z-b130"),
    (24, 100, "formal-conc24-20261122T131830Z-b100"),
]

# Palette (clean, high-contrast, colorblind-conscious)
C16 = "#2563EB"       # Royal / steel blue
C16_LIGHT = "#93C5FD"
C20 = "#EA580C"       # Vermilion orange
C20_LIGHT = "#FDBA74"
C20_B130 = "#0D9488"  # Teal (for c20-b130 in timeseries)
C24 = "#991B1B"       # Deep crimson
C24_LIGHT = "#FCA5A5"
SLATE = "#334155"
MUTED = "#64748B"
GRID = "#E2E8F0"

SERVER_LOG_RE = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[loggers\.py:\d+\] Engine 000: "
    r"Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, "
    r"GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%"
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "PingFang SC",
                "Hiragino Sans GB",
                "Heiti SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "axes.labelcolor": "#1E293B",
            "axes.titlecolor": "#0F172A",
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.8,
            "legend.frameon": True,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#E2E8F0",
            "legend.framealpha": 0.94,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "--",
            "savefig.dpi": 260,
            "savefig.facecolor": "#FFFFFF",
        }
    )


def clean_spines(ax: plt.Axes, keep_right: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    if not keep_right:
        ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_arm_metrics() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for conc, budget_pct, arm_id in ARM_MANIFEST:
        p = ARMS_DIR / arm_id
        with open(p / "run.json") as f:
            rj = json.load(f)
        with open(p / "normalization.json") as f:
            nm = json.load(f)
        with open(p / "agentx" / "profile_export_aiperf.json") as f:
            ap = json.load(f)
        with open(p / "agentx" / "server_metrics_export.csv") as f:
            lines = [ln for ln in f if not ln.startswith("#") and ln.strip()]
        sm: dict[str, dict[str, str]] = {}
        for row in csv.DictReader(lines):
            m = row["Metric"]
            if m not in sm:
                sm[m] = row

        pc_hits = float(sm["vllm:prefix_cache_hits"]["avg"])
        pc_queries = float(sm["vllm:prefix_cache_queries"]["avg"])
        pc_hit_pct = (pc_hits / pc_queries * 100.0) if pc_queries > 0 else 0.0

        records.append(
            {
                "conc": conc,
                "budget_pct": budget_pct,
                "budget_mult": budget_pct / 100.0,
                "budget_delta": budget_pct - 100,
                "budget_gb": rj["budget_bytes"] / (1024**3),
                "arm_id": arm_id,
                "label": f"c{conc}-b{budget_pct}",
                "start_s": nm["start_ns"] / 1e9,
                "end_s": nm["sent_end_ns"] / 1e9,
                "tok_s": nm["primary_output_tokens_s"],
                "ro_tok_s": nm["reasoning_plus_output_tokens_s"],
                "reqs": nm["successful_requests"],
                "req_s": nm["primary_requests_s"],
                "kv_avg": float(sm["vllm:kv_cache_usage_perc"]["avg"]) * 100.0,
                "kv_p50": float(sm["vllm:kv_cache_usage_perc"]["p50"]) * 100.0,
                "kv_p95": float(sm["vllm:kv_cache_usage_perc"]["p95"]) * 100.0,
                "kv_max": float(sm["vllm:kv_cache_usage_perc"]["max"]) * 100.0,
                "run_avg": float(sm["vllm:num_requests_running"]["avg"]),
                "wait_avg": float(sm["vllm:num_requests_waiting"]["avg"]),
                "wait_p95": float(sm["vllm:num_requests_waiting"]["p95"]),
                "wait_max": float(sm["vllm:num_requests_waiting"]["max"]),
                "pc_hit_pct": pc_hit_pct,
                "ttft_p50_s": ap["time_to_first_token"]["p50"] / 1000.0,
                "ttft_p95_s": ap["time_to_first_token"]["p95"] / 1000.0,
                "ttft_avg_s": ap["time_to_first_token"]["avg"] / 1000.0,
                "itl_p50_ms": ap["inter_token_latency"]["p50"],
                "itl_p95_ms": ap["inter_token_latency"]["p95"],
                "e2e_p50_s": ap["request_latency"]["p50"] / 1000.0,
                "e2e_p95_s": ap["request_latency"]["p95"] / 1000.0,
                "queue_p95_s": float(sm["vllm:request_queue_time_seconds"]["p95"]),
                "queue_p50_s": float(sm["vllm:request_queue_time_seconds"]["p50"]),
            }
        )

    # Compute relative gain vs each concurrency's b100 baseline
    baselines = {r["conc"]: r["tok_s"] for r in records if r["budget_pct"] == 100}
    for r in records:
        base_tok = baselines[r["conc"]]
        r["gain_pct"] = (r["tok_s"] / base_tok - 1.0) * 100.0
        r["conversion"] = (
            r["gain_pct"] / r["budget_delta"] if r["budget_delta"] > 0 else np.nan
        )
    return records


def load_timeseries(arm: dict[str, Any]) -> dict[str, np.ndarray]:
    """Parse 10-second vLLM engine logs aligned to [0, 1800s] profiling window."""
    log_path = ARMS_DIR / arm["arm_id"] / "server.log.gz"
    start_dt = datetime.fromtimestamp(arm["start_s"], tz=timezone.utc)
    year = start_dt.year

    t_min, kv, wait, run, pc_hit, gen_tok = [], [], [], [], [], []
    with gzip.open(log_path, "rt", errors="replace") as f:
        for line in f:
            m = SERVER_LOG_RE.search(line)
            if not m:
                continue
            ts_str = m.group(1)
            dt = datetime.strptime(f"{year}-{ts_str}", "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
            rel_s = dt.timestamp() - arm["start_s"]
            if -5.0 <= rel_s <= 1805.0:
                t_min.append(max(0.0, min(1800.0, rel_s)) / 60.0)
                gen_tok.append(float(m.group(3)))
                run.append(int(m.group(4)))
                wait.append(int(m.group(5)))
                kv.append(float(m.group(6)))
                pc_hit.append(float(m.group(7)))

    return {
        "t_min": np.array(t_min),
        "kv": np.array(kv),
        "wait": np.array(wait),
        "run": np.array(run),
        "pc_hit": np.array(pc_hit),
        "gen_tok": np.array(gen_tok),
    }


# ---------------------------------------------------------------------------
# Figure 1: Throughput Scaling & Marginal Conversion Efficiency (2x2)
# ---------------------------------------------------------------------------
def plot_fig1_throughput_scaling(records: list[dict[str, Any]]) -> None:
    c16 = [r for r in records if r["conc"] == 16]
    c20 = [r for r in records if r["conc"] == 20]
    c24 = [r for r in records if r["conc"] == 24][0]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=260)
    fig.subplots_adjust(left=0.075, right=0.935, top=0.905, bottom=0.085, wspace=0.25, hspace=0.35)
    fig.suptitle(
        "Qwen3.8-Flash-Next 多并发 KV Cache 预算扩展吞吐与边际效率分析 (1800s 正式窗口)",
        fontsize=14,
        fontweight="bold",
        color="#0F172A",
        y=0.965,
    )

    # --- Panel (a): Output throughput vs KV budget ---
    ax = axes[0, 0]
    clean_spines(ax)
    x16 = [r["budget_mult"] for r in c16]
    y16 = [r["tok_s"] for r in c16]
    x20 = [r["budget_mult"] for r in c20]
    y20 = [r["tok_s"] for r in c20]

    ax.axhline(271.0, color="#94A3B8", linestyle=":", linewidth=1.2, zorder=1)
    ax.text(0.97, 297.0, "可见输出平台 (~271 tok/s)", fontsize=8.3, color=MUTED, ha="left")

    ax.plot(x16, y16, "-o", color=C16, lw=2.3, ms=7.2, label="conc=16 (1.00×–1.30×)", zorder=4)
    ax.plot(x20, y20, "-s", color=C20, lw=2.3, ms=7.2, label="conc=20 (1.00×–1.30×)", zorder=5)
    ax.scatter(
        [1.0],
        [c24["tok_s"]],
        marker="D",
        s=75,
        color=C24,
        edgecolor="white",
        linewidth=0.9,
        zorder=6,
        label="conc=24 基线探索点 (1.00×)",
    )

    # Careful non-overlapping annotations
    offsets_16 = [(0, 10), (0, 10), (0, 10), (0, 10)]
    for (x, y), (dx, dy) in zip(zip(x16, y16), offsets_16):
        ax.annotate(
            f"{y:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=C16,
        )
    offsets_20 = [(0, -16), (4, -16), (6, -16), (0, -17)]
    for (x, y), (dx, dy) in zip(zip(x20, y20), offsets_20):
        ax.annotate(
            f"{y:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=C20,
        )
    ax.annotate(
        f"c24-b100: {c24['tok_s']:.1f} tok/s\n(KV 峰值 100%, 过载)",
        xy=(1.0, c24["tok_s"]),
        xytext=(1.07, 56),
        fontsize=8.3,
        fontweight="bold",
        color=C24,
        arrowprops=dict(arrowstyle="->", color=C24, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#FECACA", lw=0.8),
    )

    ax.set_title("(a) 可见输出吞吐随 KV 预算变化")
    ax.set_xlabel("KV Cache 预算倍数 (每卡容量)")
    ax.set_ylabel("可见输出吞吐 (tok/s)")
    ax.set_xticks([1.0, 1.1, 1.2, 1.3])
    ax.set_xticklabels(["1.00×\n(25.8G)", "1.10×\n(28.4G)", "1.20×\n(31.0G)", "1.30×\n(33.6G)"])
    ax.set_xlim(0.96, 1.35)
    ax.set_ylim(10, 316)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right")

    # --- Panel (b): Relative throughput gain vs 1:1 linear line ---
    ax = axes[0, 1]
    clean_spines(ax)
    dx_vals = np.array([0, 10, 20, 30])
    g16 = [r["gain_pct"] for r in c16]
    g20 = [r["gain_pct"] for r in c20]

    # Shaded regions above and below y=x
    x_line = np.linspace(0, 32, 100)
    ax.fill_between(x_line, x_line, 50, color="#FFF7ED", alpha=0.65, zorder=0)
    ax.fill_between(x_line, 0, x_line, color="#F8FAFC", alpha=0.85, zorder=0)
    ax.plot(x_line, x_line, "--", color=SLATE, lw=1.4, label="1:1 线性参考线 (斜率=1.0×)", zorder=2)

    ax.plot(dx_vals, g20, "-s", color=C20, lw=2.4, ms=7.2, label="conc=20 相对增益", zorder=5)
    ax.plot(dx_vals, g16, "-o", color=C16, lw=2.2, ms=7.0, label="conc=16 相对增益", zorder=4)

    for x, y in zip(dx_vals[1:], g20[1:]):
        ax.annotate(
            f"+{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(-2, 9),
            ha="center",
            fontsize=8.6,
            fontweight="bold",
            color=C20,
        )
    for x, y in zip(dx_vals[1:], g16[1:]):
        dy = 9 if x == 10 else -15
        ax.annotate(
            f"+{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, dy),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=C16,
        )

    ax.text(4.5, 42.5, "超线性收益区 (Δ吞吐% > ΔKV%)", fontsize=8.4, color="#9A3412", fontweight="bold")
    ax.text(17.5, 4.0, "亚线性饱和区 (Δ吞吐% < ΔKV%)", fontsize=8.4, color=MUTED)

    ax.set_title("(b) 相对吞吐增益与 1:1 线性基准对比")
    ax.set_xlabel("KV 预算相对增量 (+Δ%)")
    ax.set_ylabel("相对吞吐增益 (%)")
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels(["+0%\n(1.00×)", "+10%\n(1.10×)", "+20%\n(1.20×)", "+30%\n(1.30×)"])
    ax.set_xlim(-1.5, 33.0)
    ax.set_ylim(-2, 49)
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.84))

    # --- Panel (c): Marginal conversion ratio ---
    ax = axes[1, 0]
    clean_spines(ax)
    cats = np.arange(3)
    w = 0.34
    conv16 = [r["conversion"] for r in c16[1:]]
    conv20 = [r["conversion"] for r in c20[1:]]

    bars16 = ax.bar(cats - w / 2, conv16, width=w, color=C16, alpha=0.88, edgecolor="white", label="conc=16", zorder=3)
    bars20 = ax.bar(cats + w / 2, conv20, width=w, color=C20, alpha=0.90, edgecolor="white", label="conc=20", zorder=3)
    ax.axhline(1.0, color=SLATE, linestyle="--", lw=1.4, zorder=4)
    ax.text(
        1.52,
        1.07,
        "1.0× 线性平衡线",
        ha="center",
        fontsize=8.2,
        color=SLATE,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec="none", alpha=0.85),
        zorder=5,
    )

    for b in bars16:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.06, f"{h:.2f}×", ha="center", va="bottom", fontsize=8.6, fontweight="bold", color=C16)
    for b in bars20:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.06, f"{h:.2f}×", ha="center", va="bottom", fontsize=8.6, fontweight="bold", color=C20)

    ax.set_title("(c) 预算边际转换比 (Δ吞吐% / ΔKV预算%)")
    ax.set_xlabel("KV 预算扩容档位")
    ax.set_ylabel("转换比 (倍)")
    ax.set_xticks(cats)
    ax.set_xticklabels(["+10% (1.10×)", "+20% (1.20×)", "+30% (1.30×)"])
    ax.set_ylim(0, 3.05)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")

    # --- Panel (d): Completed Requests + Total (Reasoning + Output) tok/s ---
    ax = axes[1, 1]
    clean_spines(ax, keep_right=True)
    ax.spines["right"].set_color("#94A3B8")

    # Show all 9 arms cleanly grouped
    labels = ["b100", "b110", "b120", "b130", "b100", "b110", "b120", "b130", "b100"]
    xs = np.array([0, 1, 2, 3, 4.6, 5.6, 6.6, 7.6, 9.2])
    reqs = [r["reqs"] for r in records]
    ro_s = [r["ro_tok_s"] for r in records]
    bar_colors = [C16] * 4 + [C20] * 4 + [C24]

    bars = ax.bar(xs, reqs, width=0.66, color=bar_colors, alpha=0.82, edgecolor="white", zorder=3, label="完成请求数 (左轴)")
    for i, (x, req) in enumerate(zip(xs, reqs)):
        if i < 4:
            # Place c16 counts cleanly inside bar top so the overlaid right-axis line never touches them
            ax.text(x, req - 55, f"{req}", ha="center", va="top", fontsize=7.9, color="#FFFFFF", fontweight="bold", zorder=5)
        else:
            ax.text(x, req + 22, f"{req}", ha="center", va="bottom", fontsize=7.9, color="#1E293B", fontweight="bold", zorder=5)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color("#475569")

    # Plot reasoning+output tok/s separately per concurrency group so lines don't bridge gaps
    ax2.plot(xs[:4], ro_s[:4], "-o", color="#1E3A8A", lw=1.9, ms=5.5, zorder=6, label="推理+输出总吞吐 (右轴)")
    ax2.plot(xs[4:8], ro_s[4:8], "-s", color="#7C2D12", lw=1.9, ms=5.5, zorder=6)
    ax2.scatter([xs[8]], [ro_s[8]], marker="D", s=45, color="#450A0A", zorder=6)

    # Annotate key right-axis endpoints cleanly without colliding with bar labels
    ax2.annotate(f"{ro_s[3]:.0f}", (xs[3], ro_s[3]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.9, color="#1E3A8A", fontweight="bold")
    ax2.annotate(f"{ro_s[4]:.0f}", (xs[4], ro_s[4]), xytext=(-12, -13), textcoords="offset points", ha="center", fontsize=7.9, color="#7C2D12", fontweight="bold")
    ax2.annotate(f"{ro_s[7]:.0f}", (xs[7], ro_s[7]), xytext=(10, -14), textcoords="offset points", ha="center", fontsize=7.9, color="#7C2D12", fontweight="bold")
    ax2.annotate(f"{ro_s[8]:.0f}", (xs[8], ro_s[8]), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7.9, color="#450A0A", fontweight="bold")

    # Group brackets / labels under x-axis
    ax.text(1.5, -245, "conc=16", ha="center", fontsize=8.8, fontweight="bold", color=C16)
    ax.text(6.1, -245, "conc=20", ha="center", fontsize=8.8, fontweight="bold", color=C20)
    ax.text(9.2, -245, "c24", ha="center", fontsize=8.8, fontweight="bold", color=C24)

    ax.set_title("(d) 1800s 有效请求完成数与全量输出吞吐 (含推理)")
    ax.set_ylabel("完成请求数 (reqs / 1800s)")
    ax2.set_ylabel("推理+可见输出吞吐 (tok/s)", color="#334155")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_ylim(0, 1560)
    ax2.set_ylim(0, 760)
    ax.grid(True, axis="y")

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1[:1] + h2[:1], ["完成请求数 (左轴)", "推理+输出 tok/s (右轴)"], loc="upper left", fontsize=8.2)

    out_path = OUT_DIR / "qwen38_throughput_scaling.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: 1800s Internal Time-Series Dynamics (3x1)
# ---------------------------------------------------------------------------
def plot_fig2_timeseries(records: list[dict[str, Any]]) -> None:
    by_label = {r["label"]: r for r in records}
    selected = [
        ("c16-b100", "c16-b100 (236.1 tok/s, 轻压基线)", C16, "-", 1.6),
        ("c20-b100", "c20-b100 (190.3 tok/s, 中高压受限)", C20, "-", 1.9),
        ("c20-b130", "c20-b130 (270.1 tok/s, +30% 扩容后)", C20_B130, "-", 1.9),
        ("c24-b100", "c24-b100 (38.2 tok/s, 容量过载)", C24, "-", 2.0),
    ]

    ts_map = {lbl: load_timeseries(by_label[lbl]) for lbl, *_ in selected}

    fig, axes = plt.subplots(3, 1, figsize=(13.2, 9.4), dpi=260, sharex=True)
    fig.subplots_adjust(left=0.075, right=0.80, top=0.92, bottom=0.075, hspace=0.22)
    fig.suptitle(
        "1800s 正式压测窗口内部动态时序：KV 利用率、等待队列与前缀缓存命中率",
        fontsize=13.8,
        fontweight="bold",
        color="#0F172A",
        y=0.965,
    )

    # --- Panel (a): GPU KV Cache Usage (%) ---
    ax = axes[0]
    clean_spines(ax)
    ax.axhspan(80, 102, color="#FEF2F2", alpha=0.75, zorder=0)
    ax.axhline(80, color="#DC2626", linestyle="--", lw=1.0, alpha=0.7, zorder=1)
    ax.text(0.45, 84.5, "高负载警戒区 (≥80%)", ha="left", fontsize=8.3, color="#B91C1C", fontweight="bold")

    for lbl, legend_txt, color, ls, lw in selected:
        ts = ts_map[lbl]
        ax.plot(ts["t_min"], ts["kv"], color=color, linestyle=ls, lw=lw, alpha=0.92, label=legend_txt)

    ax.set_title("(a) GPU KV Cache 实时利用率 (%)", loc="left", fontsize=11.0)
    ax.set_ylabel("KV 利用率 (%)")
    ax.set_ylim(-2, 105)
    ax.grid(True)

    # Side summary box for Panel (a)
    summary_a = "\n".join(
        [
            "均值 / P95 / 峰值:",
            f"c24-b100: {by_label['c24-b100']['kv_avg']:.1f}% / {by_label['c24-b100']['kv_p95']:.1f}% / {by_label['c24-b100']['kv_max']:.0f}%",
            f"c20-b100: {by_label['c20-b100']['kv_avg']:.1f}% / {by_label['c20-b100']['kv_p95']:.1f}% / {by_label['c20-b100']['kv_max']:.1f}%",
            f"c20-b130: {by_label['c20-b130']['kv_avg']:.1f}% / {by_label['c20-b130']['kv_p95']:.1f}% / {by_label['c20-b130']['kv_max']:.1f}%",
            f"c16-b100: {by_label['c16-b100']['kv_avg']:.1f}% / {by_label['c16-b100']['kv_p95']:.1f}% / {by_label['c16-b100']['kv_max']:.1f}%",
        ]
    )
    ax.text(
        1.015,
        0.50,
        summary_a,
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=8.2,
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.45", fc="#F8FAFC", ec="#CBD5E1", lw=0.8),
    )

    # --- Panel (b): Waiting Queue Depth ---
    ax = axes[1]
    clean_spines(ax)
    for lbl, legend_txt, color, ls, lw in selected:
        ts = ts_map[lbl]
        ax.plot(ts["t_min"], ts["wait"], color=color, linestyle=ls, lw=lw, alpha=0.90)
        if lbl == "c24-b100":
            ax.fill_between(ts["t_min"], 0, ts["wait"], color=C24, alpha=0.10)
        elif lbl == "c20-b100":
            ax.fill_between(ts["t_min"], 0, ts["wait"], color=C20, alpha=0.10)

    ax.set_title("(b) 调度器等待队列深度 (Waiting Requests)", loc="left", fontsize=11.0)
    ax.set_ylabel("等待请求数 (reqs)")
    ax.set_ylim(-0.6, 19.5)
    ax.grid(True)

    summary_b = "\n".join(
        [
            "等待均值 / P95 / 峰值:",
            f"c24-b100: {by_label['c24-b100']['wait_avg']:.2f} / {by_label['c24-b100']['wait_p95']:.0f} / {by_label['c24-b100']['wait_max']:.0f} reqs",
            f"c20-b100: {by_label['c20-b100']['wait_avg']:.2f} / {by_label['c20-b100']['wait_p95']:.0f} / {by_label['c20-b100']['wait_max']:.0f} reqs",
            f"c20-b130: {by_label['c20-b130']['wait_avg']:.2f} / {by_label['c20-b130']['wait_p95']:.0f} / {by_label['c20-b130']['wait_max']:.0f} reqs",
            f"c16-b100: {by_label['c16-b100']['wait_avg']:.2f} / {by_label['c16-b100']['wait_p95']:.0f} / {by_label['c16-b100']['wait_max']:.0f} reqs",
        ]
    )
    ax.text(
        1.015,
        0.50,
        summary_b,
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=8.2,
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.45", fc="#F8FAFC", ec="#CBD5E1", lw=0.8),
    )

    # --- Panel (c): Prefix Cache Hit Rate (%) ---
    ax = axes[2]
    clean_spines(ax)
    for lbl, legend_txt, color, ls, lw in selected:
        ts = ts_map[lbl]
        ax.plot(ts["t_min"], ts["pc_hit"], color=color, linestyle=ls, lw=lw, alpha=0.92)

    ax.set_title("(c) 累积 Prefix Cache 命中率演化 (%)", loc="left", fontsize=11.0)
    ax.set_xlabel("压测窗口运行时间 (分钟)")
    ax.set_ylabel("缓存命中率 (%)")
    ax.set_xlim(0, 30)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_ylim(38, 100)
    ax.grid(True)

    summary_c = "\n".join(
        [
            "全窗口总命中率:",
            f"c16-b100: {by_label['c16-b100']['pc_hit_pct']:.1f}%",
            f"c20-b130: {by_label['c20-b130']['pc_hit_pct']:.1f}% (+7.8pp)",
            f"c20-b100: {by_label['c20-b100']['pc_hit_pct']:.1f}%",
            f"c24-b100: {by_label['c24-b100']['pc_hit_pct']:.1f}% (驱逐重算)",
        ]
    )
    ax.text(
        1.015,
        0.50,
        summary_c,
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=8.2,
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.45", fc="#F8FAFC", ec="#CBD5E1", lw=0.8),
    )

    # Shared top legend inside figure
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_leg,
        loc="upper center",
        bbox_to_anchor=(0.44, 0.903),
        ncol=2,
        fontsize=8.8,
        frameon=True,
    )
    # Adjust top slightly so legend has clean breathing room
    fig.subplots_adjust(left=0.07, right=0.79, top=0.825, bottom=0.075, hspace=0.25)

    out_path = OUT_DIR / "qwen38_timeseries_dynamics.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Mechanism Phase Diagram across all 9 arms (1x3)
# ---------------------------------------------------------------------------
def plot_fig3_mechanism_phase(records: list[dict[str, Any]]) -> None:
    c16 = [r for r in records if r["conc"] == 16]
    c20 = [r for r in records if r["conc"] == 20]
    c24 = [r for r in records if r["conc"] == 24][0]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), dpi=260)
    fig.subplots_adjust(left=0.058, right=0.98, top=0.85, bottom=0.145, wspace=0.22)
    fig.suptitle(
        "KV 预算影响吞吐的因果机制相图：9 组正式实验臂的全景映射",
        fontsize=13.8,
        fontweight="bold",
        color="#0F172A",
        y=0.955,
    )

    # --- Panel (a): KV P95 Utilization vs Throughput (with regime zones) ---
    ax = axes[0]
    clean_spines(ax)
    ax.axvspan(35, 60, color="#ECFDF5", alpha=0.65, zorder=0)
    ax.axvspan(60, 80, color="#FFFBEB", alpha=0.75, zorder=0)
    ax.axvspan(80, 102, color="#FEF2F2", alpha=0.80, zorder=0)
    ax.axvline(60, color="#10B981", ls="--", lw=0.9, alpha=0.7)
    ax.axvline(80, color="#EF4444", ls="--", lw=0.9, alpha=0.7)

    ax.text(48.5, 319, "KV 充裕区\n(近似平台)", ha="center", va="top", fontsize=8.1, color="#065F46", fontweight="bold")
    ax.text(70.0, 319, "容量受限区\n(扩容高弹性)", ha="center", va="top", fontsize=8.1, color="#92400E", fontweight="bold")
    ax.text(91.0, 319, "过载退化区\n(频繁排队驱逐)", ha="center", va="top", fontsize=8.1, color="#991B1B", fontweight="bold")

    # Trajectory arrows for c16 and c20
    for group, color, marker, name in [(c16, C16, "o", "conc=16"), (c20, C20, "s", "conc=20")]:
        xs = [r["kv_p95"] for r in group]
        ys = [r["tok_s"] for r in group]
        ax.plot(xs, ys, color=color, lw=1.6, alpha=0.55, zorder=3)
        ax.scatter(xs, ys, color=color, marker=marker, s=68, edgecolor="white", lw=0.8, label=name, zorder=5)
        # Arrow from b100 to b130 along segments
        for i in range(len(xs) - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4, shrinkA=6, shrinkB=6),
                zorder=4,
            )

    ax.scatter([c24["kv_p95"]], [c24["tok_s"]], color=C24, marker="D", s=82, edgecolor="white", lw=0.9, label="conc=24 (1.00×)", zorder=6)

    # Clean point labels
    lbl_pos_a = {
        "c16-b100": (4, -16),
        "c16-b110": (-8, -16),
        "c16-b120": (2, 10),
        "c16-b130": (-4, -16),
        "c20-b100": (-6, -16),
        "c20-b110": (-16, -14),
        "c20-b120": (15, 7),
        "c20-b130": (2, 10),
        "c24-b100": (-16, 12),
    }
    for r in records:
        dx, dy = lbl_pos_a[r["label"]]
        short = f"b{r['budget_pct']}" if r["conc"] != 24 else "c24-b100"
        col = C16 if r["conc"] == 16 else (C20 if r["conc"] == 20 else C24)
        ax.annotate(short, (r["kv_p95"], r["tok_s"]), xytext=(dx, dy), textcoords="offset points", ha="center", fontsize=8.0, fontweight="bold", color=col)

    ax.set_title("(a) KV P95 利用率 vs 有效吞吐")
    ax.set_xlabel("GPU KV Cache P95 利用率 (%)")
    ax.set_ylabel("可见输出吞吐 (tok/s)")
    ax.set_xlim(37, 101)
    ax.set_ylim(10, 330)
    ax.grid(True, axis="y")
    ax.legend(loc="lower left")

    # --- Panel (b): Prefix Cache Hit Rate vs Throughput ---
    ax = axes[1]
    clean_spines(ax)
    for group, color, marker, name in [(c16, C16, "o", "conc=16"), (c20, C20, "s", "conc=20")]:
        xs = [r["pc_hit_pct"] for r in group]
        ys = [r["tok_s"] for r in group]
        ax.plot(xs, ys, color=color, lw=1.6, alpha=0.55, zorder=3)
        ax.scatter(xs, ys, color=color, marker=marker, s=68, edgecolor="white", lw=0.8, label=name, zorder=5)
        for i in range(len(xs) - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4, shrinkA=6, shrinkB=6),
                zorder=4,
            )

    ax.scatter([c24["pc_hit_pct"]], [c24["tok_s"]], color=C24, marker="D", s=82, edgecolor="white", lw=0.9, label="conc=24 (1.00×)", zorder=6)

    # Annotate key points cleanly with zero overlap
    ax.annotate("c24-b100\n(54.8%, 38.2 tok/s)", (c24["pc_hit_pct"], c24["tok_s"]), xytext=(14, 14), textcoords="offset points", fontsize=8.1, fontweight="bold", color=C24)
    ax.annotate("c20-b100\n(81.9%)", (c20[0]["pc_hit_pct"], c20[0]["tok_s"]), xytext=(-44, -22), textcoords="offset points", fontsize=8.0, fontweight="bold", color=C20)
    ax.annotate(
        "c20-b130\n(89.7%)",
        (c20[3]["pc_hit_pct"], c20[3]["tok_s"]),
        xytext=(-68, 18),
        textcoords="offset points",
        fontsize=8.0,
        fontweight="bold",
        color=C20,
        arrowprops=dict(arrowstyle="->", color=C20, lw=0.9),
    )
    ax.annotate("c16-b100\n(89.5%)", (c16[0]["pc_hit_pct"], c16[0]["tok_s"]), xytext=(6, -28), textcoords="offset points", fontsize=8.0, fontweight="bold", color=C16)
    ax.annotate(
        "c16-b130\n(91.8%)",
        (c16[3]["pc_hit_pct"], c16[3]["tok_s"]),
        xytext=(-8, 18),
        textcoords="offset points",
        fontsize=8.0,
        fontweight="bold",
        color=C16,
    )

    ax.set_title("(b) Prefix Cache 命中率 vs 有效吞吐")
    ax.set_xlabel("全窗口 Prefix Cache 命中率 (%)")
    ax.set_ylabel("可见输出吞吐 (tok/s)")
    ax.set_xlim(50, 99)
    ax.set_ylim(10, 330)
    ax.grid(True)

    # --- Panel (c): Mean Waiting Queue Depth vs Throughput ---
    ax = axes[2]
    clean_spines(ax)
    for group, color, marker, name in [(c16, C16, "o", "conc=16"), (c20, C20, "s", "conc=20")]:
        xs = [r["wait_avg"] for r in group]
        ys = [r["tok_s"] for r in group]
        ax.plot(xs, ys, color=color, lw=1.6, alpha=0.55, zorder=3)
        ax.scatter(xs, ys, color=color, marker=marker, s=68, edgecolor="white", lw=0.8, label=name, zorder=5)

    ax.scatter([c24["wait_avg"]], [c24["tok_s"]], color=C24, marker="D", s=82, edgecolor="white", lw=0.9, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(0.08, 9.5)
    ax.set_ylim(10, 330)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))

    ax.annotate(f"c16-b130\n(wait=0.11, Q95={c16[3]['queue_p95_s']:.1f}s)", (c16[3]["wait_avg"], c16[3]["tok_s"]), xytext=(2, 16), textcoords="offset points", fontsize=7.9, color=C16, fontweight="bold")
    ax.annotate(f"c16-b100\n(0.26)", (c16[0]["wait_avg"], c16[0]["tok_s"]), xytext=(-14, -26), textcoords="offset points", fontsize=7.9, color=C16, fontweight="bold")
    ax.annotate(f"c20-b130\n(0.34, Q95={c20[3]['queue_p95_s']:.1f}s)", (c20[3]["wait_avg"], c20[3]["tok_s"]), xytext=(14, 10), textcoords="offset points", fontsize=7.9, color=C20, fontweight="bold")
    ax.annotate(f"c20-b100\n(0.93, Q95={c20[0]['queue_p95_s']:.1f}s)", (c20[0]["wait_avg"], c20[0]["tok_s"]), xytext=(8, -14), textcoords="offset points", fontsize=7.9, color=C20, fontweight="bold")
    ax.annotate(f"c24-b100\n(wait=5.65, Q95={c24['queue_p95_s']:.1f}s)", (c24["wait_avg"], c24["tok_s"]), xytext=(-102, 18), textcoords="offset points", fontsize=8.0, color=C24, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C24, lw=1.0))

    ax.set_title("(c) 平均等待队列深度 vs 有效吞吐")
    ax.set_xlabel("平均等待请求数 (reqs, 对数坐标)")
    ax.set_ylabel("可见输出吞吐 (tok/s)")
    ax.grid(True, which="major")

    out_path = OUT_DIR / "qwen38_mechanism_phase_diagram.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Latency Percentiles & Queueing Decomposition (1x3)
# ---------------------------------------------------------------------------
def plot_fig4_latency_breakdown(records: list[dict[str, Any]]) -> None:
    c16 = [r for r in records if r["conc"] == 16]
    c20 = [r for r in records if r["conc"] == 20]
    c24 = [r for r in records if r["conc"] == 24][0]

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.3), dpi=260)
    fig.subplots_adjust(left=0.058, right=0.98, top=0.85, bottom=0.16, wspace=0.25)
    fig.suptitle(
        "首字延迟 (TTFT)、解码间隔 (ITL) 与服务端排队延迟剖析 (全 9 臂)",
        fontsize=13.8,
        fontweight="bold",
        color="#0F172A",
        y=0.955,
    )

    xs = np.array([0, 1, 2, 3, 4.7, 5.7, 6.7, 7.7, 9.4])
    xlabels = ["b100", "b110", "b120", "b130", "b100", "b110", "b120", "b130", "b100"]

    def add_group_labels(ax: plt.Axes, y_frac: float = -0.155) -> None:
        ax.text(1.5, y_frac, "conc=16", transform=ax.get_xaxis_transform(), ha="center", fontsize=8.8, fontweight="bold", color=C16)
        ax.text(6.2, y_frac, "conc=20", transform=ax.get_xaxis_transform(), ha="center", fontsize=8.8, fontweight="bold", color=C20)
        ax.text(9.4, y_frac, "c24", transform=ax.get_xaxis_transform(), ha="center", fontsize=8.8, fontweight="bold", color=C24)

    # --- Panel (a): TTFT P50 & P95 (s, log scale) ---
    ax = axes[0]
    clean_spines(ax)
    ttft50 = [r["ttft_p50_s"] for r in records]
    ttft95 = [r["ttft_p95_s"] for r in records]
    w = 0.36

    ax.bar(xs - w / 2, ttft50, width=w, color="#64748B", alpha=0.78, edgecolor="white", label="TTFT P50 (中位数)")
    p95_colors = [C16] * 4 + [C20] * 4 + [C24]
    ax.bar(xs + w / 2, ttft95, width=w, color=p95_colors, alpha=0.90, edgecolor="white", label="TTFT P95 (尾部)")

    ax.set_yscale("log")
    ax.set_ylim(0.7, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

    for x, v in zip(xs, ttft95):
        ax.text(x + w / 2, v * 1.10, f"{v:.1f}s", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#0F172A")
    # Highlight c24 P50 cleanly shifted left
    ax.text(xs[8] - w / 2 - 0.25, ttft50[8] * 1.12, f"{ttft50[8]:.1f}s", ha="center", va="bottom", fontsize=7.5, color="#475569", fontweight="bold")

    # Reduction callouts
    ax.annotate("-58.1%", xy=(3.18, 4.1), xytext=(1.9, 2.2), fontsize=8.2, fontweight="bold", color=C16,
                bbox=dict(boxstyle="round,pad=0.22", fc="#EFF6FF", ec="#BFDBFE", lw=0.7))
    ax.annotate("-42.7%", xy=(7.88, 8.8), xytext=(6.2, 2.5), fontsize=8.2, fontweight="bold", color=C20,
                bbox=dict(boxstyle="round,pad=0.22", fc="#FFF7ED", ec="#FED7AA", lw=0.7))

    ax.set_title("(a) 首字延迟 TTFT (P50 与 P95)")
    ax.set_ylabel("TTFT (秒, 对数坐标)")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8.1)
    ax.grid(True, axis="y", which="major")
    ax.legend(loc="upper left", fontsize=8.3)
    add_group_labels(ax)

    # --- Panel (b): Inter-Token Latency ITL P50 & P95 (ms, log scale) ---
    ax = axes[1]
    clean_spines(ax)
    itl50 = [r["itl_p50_ms"] for r in records]
    itl95 = [r["itl_p95_ms"] for r in records]

    ax.bar(xs - w / 2, itl50, width=w, color="#64748B", alpha=0.78, edgecolor="white", label="ITL P50 (中位数)")
    ax.bar(xs + w / 2, itl95, width=w, color=p95_colors, alpha=0.90, edgecolor="white", label="ITL P95 (尾部)")

    ax.set_yscale("log")
    ax.set_ylim(8, 460)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

    for x, v in zip(xs, itl95):
        ax.text(x + w / 2, v * 1.09, f"{v:.0f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#0F172A")
    ax.text(xs[8] - w / 2, itl50[8] * 1.09, f"{itl50[8]:.0f}", ha="center", va="bottom", fontsize=7.5, color="#475569", fontweight="bold")

    ax.annotate("c20 P95: -51.0%", xy=(7.88, 60), xytext=(5.1, 155), fontsize=8.2, fontweight="bold", color=C20,
                arrowprops=dict(arrowstyle="->", color=C20, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFF7ED", ec="#FED7AA", lw=0.7))

    ax.set_title("(b) 令牌间延迟 ITL (P50 与 P95)")
    ax.set_ylabel("ITL (毫秒, 对数坐标)")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8.1)
    ax.grid(True, axis="y", which="major")
    ax.legend(loc="upper left", fontsize=8.3)
    add_group_labels(ax)

    # --- Panel (c): vLLM Queue Time P95 vs TTFT P95 ---
    ax = axes[2]
    clean_spines(ax)
    q95 = [r["queue_p95_s"] for r in records]

    ax.bar(xs - w / 2, q95, width=w, color="#0F766E", alpha=0.88, edgecolor="white", label="vLLM 排队延迟 P95")
    ax.bar(xs + w / 2, ttft95, width=w, color="#CBD5E1", alpha=0.92, edgecolor="#64748B", lw=0.7, label="端到端 TTFT P95")

    ax.set_yscale("log")
    ax.set_ylim(2.0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

    for i, (x, q, t) in enumerate(zip(xs, q95, ttft95)):
        dx_shift = -0.22 if i == 8 else 0.0
        ax.text(x - w / 2 + dx_shift, q * 1.10, f"{q:.1f}s", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#0F766E")
    ax.text(xs[8] + w / 2 + 0.08, ttft95[8] * 1.09, f"{ttft95[8]:.1f}s", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#334155")

    ax.annotate(
        "高负载下排队占\nTTFT P95 约 69%",
        xy=(9.15, 43.3),
        xytext=(4.9, 56.0),
        fontsize=8.1,
        fontweight="bold",
        color=C24,
        arrowprops=dict(arrowstyle="->", color=C24, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#FECACA", lw=0.8),
    )

    ax.set_title("(c) 服务端排队延迟 P95 与 TTFT P95 对比")
    ax.set_ylabel("延迟 (秒, 对数坐标)")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8.1)
    ax.grid(True, axis="y", which="major")
    ax.legend(loc="upper left", fontsize=8.3)
    add_group_labels(ax)

    out_path = OUT_DIR / "qwen38_latency_tail_breakdown.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove legacy copied charts if present
    for old_name in [
        "conc16_throughput_gain.png",
        "conc20_throughput_gain.png",
        "cross_concurrency.png",
    ]:
        old_p = OUT_DIR / old_name
        if old_p.exists():
            old_p.unlink()
            print(f"Removed legacy chart: {old_p}")

    records = load_arm_metrics()
    plot_fig1_throughput_scaling(records)
    plot_fig2_timeseries(records)
    plot_fig3_mechanism_phase(records)
    plot_fig4_latency_breakdown(records)


if __name__ == "__main__":
    main()
