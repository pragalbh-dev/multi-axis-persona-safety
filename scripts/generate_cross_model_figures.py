"""Generate cross-model figures for the public GitHub Pages writeup.

Reads per-model JSON artifacts from results/phase_a/ and results/phase_b/
(plus reference numbers from results/plan_b_gemma2_27b/metrics.json) and
emits PNG + Plotly HTML figures into docs/figures/.

Figures:
  cross_model_auc_lift          AUC(AA) vs AUC(AA+PCs) per model (4 models)
  cross_model_attack_recovery   baseline / defended / best-attack harm rate per model
  cross_model_v_harm_alignment  cos(v_harm, Assistant direction) per model
  cross_model_probe_depth       probe-layer depth (fraction) per model
  cross_model_pipeline          per-model pipeline-validation small multiples
  per_model_attack_pareto       lambda dose-response curves per model (3 panels)
  per_model_lasso_pcs           which PCs each model's LASSO selects (heatmap)

The figures use the page palette from docs/index.html :root:
  --accent      #4C72B0
  --accent-warn #C44E52
  --accent-ok   #55A868
  neutral grey  #999999
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = Path(__file__).resolve().parents[1]
DOCS_FIGURES = REPO / "docs" / "figures"

C_ACCENT = "#4C72B0"
C_WARN = "#C44E52"
C_OK = "#55A868"
C_GREY = "#999999"
C_DARK = "#1a1a1a"

MODELS = [
    {
        "key": "gemma_2_27b",
        "label": "Gemma 2 27B",
        "phase_a_path": "results/plan_b_gemma2_27b/metrics.json",
        "phase_b_headline_path": None,  # constructed inline
        "harm_direction_path": None,
        "lambda_pareto_path": None,
        "n_layers": 42,
        "L_star": 21,
        "color": C_ACCENT,
    },
    {
        "key": "qwen_3_32b",
        "label": "Qwen 3 32B",
        "phase_a_path": "results/phase_a/qwen_3_32b/metrics.json",
        "phase_b_headline_path": "results/phase_b/qwen_3_32b/headline.json",
        "harm_direction_path": "results/phase_b/qwen_3_32b/harm_direction.json",
        "lambda_pareto_path": "results/phase_b/qwen_3_32b/lambda_pareto.json",
        "n_layers": 64,
        "L_star": 11,
        "color": C_OK,
    },
    {
        "key": "gemma_4_31b_thinking_off",
        "label": "Gemma 4 31B (reasoning off)",
        "phase_a_path": "results/phase_a/gemma_4_31b_thinking_off/metrics.json",
        "phase_b_headline_path": "results/phase_b/gemma_4_31b_thinking_off/headline.json",
        "harm_direction_path": "results/phase_b/gemma_4_31b_thinking_off/harm_direction.json",
        "lambda_pareto_path": "results/phase_b/gemma_4_31b_thinking_off/lambda_pareto.json",
        "n_layers": 60,
        "L_star": 14,
        "color": C_WARN,
    },
    {
        "key": "gemma_4_31b_thinking_on",
        "label": "Gemma 4 31B (reasoning on)",
        "phase_a_path": "results/phase_a/gemma_4_31b_thinking_on/metrics.json",
        "phase_b_headline_path": "results/phase_b/gemma_4_31b_thinking_on/headline.json",
        "harm_direction_path": "results/phase_b/gemma_4_31b_thinking_on/harm_direction.json",
        "lambda_pareto_path": "results/phase_b/gemma_4_31b_thinking_on/lambda_pareto.json",
        "n_layers": 60,
        "L_star": 59,
        "color": C_ACCENT,
    },
]

# Gemma 2 27B numbers come from results/plan_b_gemma2_27b/metrics.json (Phase A)
# and from the writeup numbers (Phase B per-attack at λ=0.05/0.10 sign-corrected).
# The Phase B "best harm-aligned attack" at coherence-preserving λ is v_harm at
# λ=0.10 → harm 14.6%, coherence 76.7% (from the existing follow-up findings table).
G2_PHASE_B = {
    "baseline_harm": 0.148,
    "aa_cap_harm": 0.018,
    "aa_cap_delta_pp": -13.0,
    "best_attack_label": "v_harm @ λ=0.10",
    "best_attack_harm": 0.146,
    "best_attack_recovery_pp": +12.8,
    "best_attack_coherence": 0.767,
    "cos_v_harm_aa": -0.36,
}


def load_json(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


def _clean_attack_label(raw: str) -> str:
    """Translate internal attack-condition keys to public-facing labels.

    Examples:
      full_aa_capped_signmatched_pc3_pos0p25 -> "harm-aligned PC3 (λ=0.25)"
      full_aa_capped_v_harm_pos0p15          -> "empirical harm direction (λ=0.15)"
      full_aa_capped_adv_null_pos0p25        -> "null-space attack (λ=0.25)"
    """
    s = raw.replace("full_aa_capped_", "").replace("aa_capped_", "")
    # Extract λ from suffix like pos0p25 / pos0p15 / pos1
    lam_str = ""
    for token in s.split("_"):
        if token.startswith("pos"):
            lam_raw = token[3:].replace("p", ".")
            try:
                lam_val = float(lam_raw)
                lam_str = f" (λ={lam_val:g})"
            except ValueError:
                pass
    if "signmatched_pc2" in s:
        return f"harm-aligned PC2{lam_str}"
    if "signmatched_pc3" in s:
        return f"harm-aligned PC3{lam_str}"
    if "signmatched_pc4" in s:
        return f"harm-aligned PC4{lam_str}"
    if "v_harm" in s:
        return f"empirical harm direction{lam_str}"
    if "adv_null" in s:
        return f"null-space attack{lam_str}"
    if "multi" in s:
        return f"multi-axis composite{lam_str}"
    return raw


def collect() -> list[dict]:
    out: list[dict] = []
    for m in MODELS:
        a = load_json(m["phase_a_path"])
        head_a = a["headline"]
        bs = head_a["blind_spot_auc_lift"]
        baseline_harm = head_a["baseline_harm_rate"]
        refusal_baseline = head_a.get("refusal_rate_per_condition", {}).get("baseline", None)
        if m["key"] == "gemma_2_27b":
            aa_cap_harm = G2_PHASE_B["aa_cap_harm"]
            aa_cap_delta_pp = G2_PHASE_B["aa_cap_delta_pp"]
            cos_v_harm_aa = G2_PHASE_B["cos_v_harm_aa"]
            best_attack_label = G2_PHASE_B["best_attack_label"]
            best_attack_harm = G2_PHASE_B["best_attack_harm"]
            best_attack_recovery_pp = G2_PHASE_B["best_attack_recovery_pp"]
            best_attack_coherence = G2_PHASE_B["best_attack_coherence"]
            phase_b_full = None
        else:
            head_b = load_json(m["phase_b_headline_path"])
            aa_cap_harm = head_b["aa_cap_only_harm_rate"]
            aa_cap_delta_pp = head_b["aa_cap_only_harm_rate"] * 100 - baseline_harm * 100
            cos_v_harm_aa = head_b.get("cos_v_harm_aa")
            # Best harm-aligned attack = max recovery_pp_vs_aa_cap among non-random axes
            per = head_b["per_attack_full"]
            best_label, best_v = None, None
            for k, v in per.items():
                if "random" in k:
                    continue
                if best_v is None or v["recovery_pp_vs_aa_cap"] > best_v["recovery_pp_vs_aa_cap"]:
                    best_v = v
                    best_label = k
            best_attack_label = _clean_attack_label(best_label)
            best_attack_harm = best_v["harm_rate"]
            best_attack_recovery_pp = best_v["recovery_pp_vs_aa_cap"]
            best_attack_coherence = best_v["coherence_rate"]
            phase_b_full = per
        out.append(
            {
                **m,
                "auc_aa": bs["auc_aa_only"],
                "auc_full": bs["auc_with_pcs"],
                "auc_lift": bs["delta"],
                "auc_lift_lo": bs["ci_low"],
                "auc_lift_hi": bs["ci_high"],
                "selected_pcs": bs["selected_pcs"],
                "baseline_harm": baseline_harm,
                "aa_cap_harm": aa_cap_harm,
                "aa_cap_delta_pp": aa_cap_delta_pp,
                "cos_v_harm_aa": cos_v_harm_aa,
                "refusal_baseline": refusal_baseline,
                "best_attack_label": best_attack_label,
                "best_attack_harm": best_attack_harm,
                "best_attack_recovery_pp": best_attack_recovery_pp,
                "best_attack_coherence": best_attack_coherence,
                "phase_b_full": phase_b_full,
            }
        )
    return out


def fig_auc_lift(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    aa = [r["auc_aa"] for r in rows]
    full = [r["auc_full"] for r in rows]
    lift = [r["auc_lift"] for r in rows]
    lift_err_lo = [r["auc_lift"] - r["auc_lift_lo"] for r in rows]
    lift_err_hi = [r["auc_lift_hi"] - r["auc_lift"] for r in rows]
    pcs = [", ".join(r["selected_pcs"]) for r in rows]

    # Plotly grouped bar
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Assistant direction only",
            x=labels,
            y=aa,
            marker_color=C_GREY,
            text=[f"{v:.2f}" for v in aa],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>AUC(AA only) = %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Assistant direction + persona PCs",
            x=labels,
            y=full,
            marker_color=C_ACCENT,
            text=[f"{v:.2f}" for v in full],
            textposition="outside",
            customdata=list(zip(lift, lift_err_lo, lift_err_hi, pcs)),
            hovertemplate=(
                "<b>%{x}</b><br>AUC(AA + PCs) = %{y:.3f}"
                "<br>lift Δ = %{customdata[0]:+.3f}"
                "<br>95%% BCa CI [%{customdata[1]:+.3f}, %{customdata[2]:+.3f}]"
                "<br>LASSO selects: %{customdata[3]}"
                "<extra></extra>"
            ),
        )
    )
    for i, r in enumerate(rows):
        fig.add_annotation(
            x=labels[i],
            y=full[i] + 0.04,
            text=f"Δ = {lift[i]:+.2f}",
            showarrow=False,
            font=dict(size=11, color=C_DARK),
            xshift=18,
        )
    fig.update_layout(
        barmode="group",
        title=(
            "Harm-prediction AUC: Assistant direction alone vs. Assistant direction + persona principal components"
            "<br><sub>Across all four models, adding the leading persona PCs lifts AUC by +0.10 to +0.31; every 95% BCa CI excludes zero.</sub>"
        ),
        yaxis=dict(title="AUC (binary harm prediction)", range=[0.5, 1.05]),
        template="plotly_white",
        legend=dict(orientation="h", x=0.0, y=-0.18),
        margin=dict(t=110, b=120, l=60, r=40),
        height=520,
    )
    fig.write_html(DOCS_FIGURES / "cross_model_auc_lift.html", include_plotlyjs="cdn")

    # Matplotlib PNG
    x = np.arange(len(labels))
    w = 0.38
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, aa, w, label="Assistant direction only", color=C_GREY)
    ax.bar(x + w / 2, full, w, label="Assistant direction + persona PCs", color=C_ACCENT)
    for xi, (a_v, f_v, lift_v, lo, hi) in enumerate(
        zip(aa, full, lift, lift_err_lo, lift_err_hi)
    ):
        ax.text(xi - w / 2, a_v + 0.01, f"{a_v:.2f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, f_v + 0.01, f"{f_v:.2f}", ha="center", fontsize=9)
        ax.errorbar([xi + w / 2], [f_v], yerr=[[lo], [hi]], fmt="none", ecolor=C_DARK, capsize=3)
        ax.text(xi, 0.55, f"Δ {lift_v:+.2f}", ha="center", fontsize=9, color=C_DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("AUC (binary harm prediction)")
    ax.set_title("Cross-model harm-prediction AUC: Assistant direction vs. + persona PCs")
    ax.legend(loc="lower right")
    fig2.tight_layout()
    fig2.savefig(DOCS_FIGURES / "cross_model_auc_lift.png", dpi=150)
    plt.close(fig2)


def fig_attack_recovery(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    baseline = [r["baseline_harm"] * 100 for r in rows]
    defended = [r["aa_cap_harm"] * 100 for r in rows]
    attacked = [r["best_attack_harm"] * 100 for r in rows]
    attack_labels = [r["best_attack_label"] for r in rows]
    coherences = [r["best_attack_coherence"] * 100 for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Undefended (baseline)",
            x=labels,
            y=baseline,
            marker_color=C_GREY,
            text=[f"{v:.1f}%" for v in baseline],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>baseline harm = %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Defended (Assistant-direction cap)",
            x=labels,
            y=defended,
            marker_color=C_OK,
            text=[f"{v:.1f}%" for v in defended],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>defended harm = %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Best harm-aligned attack on the defended model",
            x=labels,
            y=attacked,
            marker_color=C_WARN,
            text=[f"{v:.1f}%" for v in attacked],
            textposition="outside",
            customdata=list(zip(attack_labels, coherences)),
            hovertemplate=(
                "<b>%{x}</b><br>attack: %{customdata[0]}"
                "<br>harm = %{y:.1f}%"
                "<br>coherence = %{customdata[1]:.1f}%"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        barmode="group",
        title=(
            "Cap defends in three of four models — and fails on Gemma 4 31B (reasoning off)"
            "<br><sub>For each model: undefended baseline, harm rate after applying Lu et al.'s Assistant-direction cap, and harm rate when the defended model is then steered along its strongest harm-aligned direction.</sub>"
        ),
        yaxis=dict(title="Harm rate (%)", range=[0, max(attacked + baseline) * 1.25]),
        template="plotly_white",
        legend=dict(orientation="h", x=0.0, y=-0.18),
        margin=dict(t=110, b=130, l=60, r=40),
        height=560,
    )
    # Annotate the failure model
    for i, r in enumerate(rows):
        if r["key"] == "gemma_4_31b_thinking_off":
            fig.add_annotation(
                x=labels[i],
                y=attacked[i] + 4,
                text=f"+{r['best_attack_recovery_pp']:.1f} pp at {coherences[i]:.1f}% coherence",
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-30,
                font=dict(size=11, color=C_WARN),
            )
    fig.write_html(DOCS_FIGURES / "cross_model_attack_recovery.html", include_plotlyjs="cdn")

    # PNG
    x = np.arange(len(labels))
    w = 0.27
    fig2, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w, baseline, w, label="Undefended baseline", color=C_GREY)
    ax.bar(x, defended, w, label="Defended (AA cap)", color=C_OK)
    ax.bar(x + w, attacked, w, label="Best harm-aligned attack on defended model", color=C_WARN)
    for xi, (b, d, a, lab, coh) in enumerate(
        zip(baseline, defended, attacked, attack_labels, coherences)
    ):
        ax.text(xi - w, b + 0.5, f"{b:.1f}%", ha="center", fontsize=8)
        ax.text(xi, d + 0.5, f"{d:.1f}%", ha="center", fontsize=8)
        ax.text(xi + w, a + 0.5, f"{a:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Harm rate (%)")
    ax.set_title("Cross-model attack recovery — defense fails on Gemma 4 31B (reasoning off)")
    ax.legend(loc="upper left")
    fig2.tight_layout()
    fig2.savefig(DOCS_FIGURES / "cross_model_attack_recovery.png", dpi=150)
    plt.close(fig2)


def fig_v_harm_alignment(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    cos = [r["cos_v_harm_aa"] for r in rows]
    colors = [C_ACCENT if abs(v) < 0.4 else C_WARN if v > 0.4 else C_OK for v in cos]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(labels, cos, color=colors)
    ax.axvline(0, color=C_DARK, linewidth=0.6)
    ax.axvspan(-0.4, 0.4, alpha=0.05, color=C_GREY, label="near-orthogonal band (|cos| < 0.4)")
    for b, v in zip(bars, cos):
        ax.text(v + (0.02 if v >= 0 else -0.02), b.get_y() + b.get_height() / 2,
                f"{v:+.2f}", ha="left" if v >= 0 else "right", va="center", fontsize=10)
    ax.set_xlim(-0.7, 0.7)
    ax.set_xlabel("cos(empirical harm direction, Assistant direction)")
    ax.set_title(
        "Geometry of harm differs across models\n"
        "Gemma 4 31B with reasoning on substantially aligns the harm direction with the Assistant direction;\nthe other three are near-orthogonal."
    )
    fig.tight_layout()
    fig.savefig(DOCS_FIGURES / "cross_model_v_harm_alignment.png", dpi=150)
    plt.close(fig)


def fig_probe_depth(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    depth_pct = [100.0 * r["L_star"] / r["n_layers"] for r in rows]
    layer_text = [f"layer {r['L_star']} / {r['n_layers']}" for r in rows]
    colors = [C_WARN if d > 80 else C_ACCENT for d in depth_pct]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(labels, depth_pct, color=colors)
    for b, v, t in zip(bars, depth_pct, layer_text):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2,
                f"{v:.0f}%  ({t})", va="center", fontsize=10)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Probe layer as % of network depth")
    ax.set_title(
        "Reasoning shifts the probe layer to the network's final layer\n"
        "The layer where role information is most linearly separable is mid-depth on three of four models;\non Gemma 4 31B with reasoning on it sits at 98% depth."
    )
    fig.tight_layout()
    fig.savefig(DOCS_FIGURES / "cross_model_probe_depth.png", dpi=150)
    plt.close(fig)


def fig_pipeline_validation(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    baseline_harm = [r["baseline_harm"] * 100 for r in rows]
    refusal = [(r["refusal_baseline"] or 0) * 100 for r in rows]
    auc_aa = [r["auc_aa"] for r in rows]
    auc_full = [r["auc_full"] for r in rows]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Baseline harm rate (no defense)",
            "Refusal rate at baseline",
            "Harm-prediction AUC",
        ),
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Bar(x=labels, y=baseline_harm, marker_color=C_ACCENT, showlegend=False,
               text=[f"{v:.1f}%" for v in baseline_harm], textposition="outside",
               hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=refusal, marker_color=C_OK, showlegend=False,
               text=[f"{v:.1f}%" for v in refusal], textposition="outside",
               hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=labels, y=auc_aa, name="Assistant direction only",
               marker_color=C_GREY,
               text=[f"{v:.2f}" for v in auc_aa], textposition="outside",
               hovertemplate="<b>%{x}</b><br>AUC(AA)=%{y:.3f}<extra></extra>"),
        row=1, col=3,
    )
    fig.add_trace(
        go.Bar(x=labels, y=auc_full, name="Assistant direction + persona PCs",
               marker_color=C_ACCENT,
               text=[f"{v:.2f}" for v in auc_full], textposition="outside",
               hovertemplate="<b>%{x}</b><br>AUC(AA+PCs)=%{y:.3f}<extra></extra>"),
        row=1, col=3,
    )
    fig.update_yaxes(range=[0, max(baseline_harm) * 1.4], row=1, col=1, title_text="harm (%)")
    fig.update_yaxes(range=[0, 100], row=1, col=2, title_text="refusal (%)")
    fig.update_yaxes(range=[0.5, 1.05], row=1, col=3, title_text="AUC")
    fig.update_xaxes(tickangle=15)
    fig.update_layout(
        title="Pipeline validation across the four models",
        template="plotly_white",
        height=480,
        barmode="group",
        legend=dict(orientation="h", x=0.0, y=-0.20),
        margin=dict(t=80, b=130, l=60, r=40),
    )
    fig.write_html(DOCS_FIGURES / "cross_model_pipeline.html", include_plotlyjs="cdn")

    # PNG (3 panels)
    fig2, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(labels, baseline_harm, color=C_ACCENT)
    axes[0].set_title("Baseline harm rate")
    axes[0].set_ylabel("harm (%)")
    axes[1].bar(labels, refusal, color=C_OK)
    axes[1].set_title("Refusal rate at baseline")
    axes[1].set_ylabel("refusal (%)")
    x = np.arange(len(labels))
    w = 0.38
    axes[2].bar(x - w / 2, auc_aa, w, color=C_GREY, label="AA only")
    axes[2].bar(x + w / 2, auc_full, w, color=C_ACCENT, label="AA + PCs")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha="right")
    axes[2].set_ylim(0.5, 1.05)
    axes[2].set_title("Harm-prediction AUC")
    axes[2].set_ylabel("AUC")
    axes[2].legend(loc="lower right", fontsize=8)
    for ax in axes[:2]:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
    fig2.suptitle("Pipeline validation across the four models")
    fig2.tight_layout()
    fig2.savefig(DOCS_FIGURES / "cross_model_pipeline.png", dpi=150)
    plt.close(fig2)


def fig_per_model_pareto() -> None:
    panels = [m for m in MODELS if m["key"] != "gemma_2_27b"]
    fig = make_subplots(
        rows=1, cols=len(panels),
        subplot_titles=[m["label"] for m in panels],
        horizontal_spacing=0.08,
        shared_yaxes=True,
    )
    axis_color = {
        "signmatched_pc2": C_ACCENT,
        "signmatched_pc3": "#8E5CB6",
        "v_harm": C_WARN,
        "adv_null": "#D89C4E",
        "random_0": C_GREY,
        "random_1": "#BFBFBF",
    }
    axis_label = {
        "signmatched_pc2": "harm-aligned PC2",
        "signmatched_pc3": "harm-aligned PC3",
        "v_harm": "empirical harm direction",
        "adv_null": "null-space attack",
        "random_0": "random direction (1)",
        "random_1": "random direction (2)",
    }
    for col, m in enumerate(panels, start=1):
        d = load_json(m["lambda_pareto_path"])
        pareto = d["pareto"]
        for axis_key, points in pareto.items():
            xs = [p["lam"] for p in points]
            harm = [p["harm"] * 100 for p in points]
            coh = [p["coherence"] * 100 for p in points]
            color = axis_color.get(axis_key, "#777")
            label = axis_label.get(axis_key, axis_key)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=harm, name=label, legendgroup=axis_key,
                    showlegend=(col == 1),
                    mode="lines+markers",
                    line=dict(color=color),
                    customdata=coh,
                    hovertemplate=(
                        f"<b>{m['label']}</b><br>{label}"
                        "<br>λ = %{x}<br>harm = %{y:.1f}%"
                        "<br>coherence = %{customdata:.1f}%<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )
        fig.update_xaxes(title_text="steering magnitude λ", row=1, col=col)
    fig.update_yaxes(title_text="harm rate (%)", range=[0, 50], row=1, col=1)
    fig.update_layout(
        title=(
            "Per-direction dose-response on the defended model"
            "<br><sub>Each line is a distinct attack direction. The empirical harm direction and harm-aligned principal components are the genuine attacks; "
            "random directions and the null-space attack are diagnostics.</sub>"
        ),
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", x=0.0, y=-0.18),
        margin=dict(t=100, b=130, l=70, r=40),
    )
    fig.write_html(DOCS_FIGURES / "per_model_attack_pareto.html", include_plotlyjs="cdn")

    # Static PNG version
    fig2, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), sharey=True)
    for ax, m in zip(axes, panels):
        d = load_json(m["lambda_pareto_path"])
        pareto = d["pareto"]
        for axis_key, points in pareto.items():
            xs = [p["lam"] for p in points]
            harm = [p["harm"] * 100 for p in points]
            color = axis_color.get(axis_key, "#777")
            label = axis_label.get(axis_key, axis_key)
            ax.plot(xs, harm, marker="o", color=color, label=label)
        ax.set_title(m["label"])
        ax.set_xlabel("λ")
        ax.set_ylim(0, 50)
    axes[0].set_ylabel("harm rate (%)")
    axes[-1].legend(loc="upper right", fontsize=7)
    fig2.suptitle("Per-direction dose-response on the defended model")
    fig2.tight_layout()
    fig2.savefig(DOCS_FIGURES / "per_model_attack_pareto.png", dpi=150)
    plt.close(fig2)


def fig_lasso_pcs(rows: list[dict]) -> None:
    pcs = [f"pc{i}" for i in range(1, 11)]
    M = np.zeros((len(rows), len(pcs)))
    for i, r in enumerate(rows):
        for pc in r["selected_pcs"]:
            j = pcs.index(pc)
            M[i, j] = 1.0
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(pcs)))
    ax.set_xticklabels([p.upper() for p in pcs])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows])
    for i in range(len(rows)):
        for j in range(len(pcs)):
            txt = "" if M[i, j] == 0 else "selected"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.5 else C_GREY)
    ax.set_title(
        "Which persona principal components does the LASSO select per model?\n"
        "Reasoning-off concentrates on a single PC; reasoning-on selects nine — yet defense robustness reverses that pattern."
    )
    fig.tight_layout()
    fig.savefig(DOCS_FIGURES / "per_model_lasso_pcs.png", dpi=150)
    plt.close(fig)


def fig_thinking_off_breakdown(rows: list[dict]) -> None:
    """Detailed per-attack chart for the failure model."""
    r = next(r for r in rows if r["key"] == "gemma_4_31b_thinking_off")
    per = r["phase_b_full"]
    if per is None:
        return
    label_map = {
        "full_aa_capped_signmatched_pc2_pos0p25": "harm-aligned PC2 (λ=0.25)",
        "full_aa_capped_signmatched_pc3_pos0p25": "harm-aligned PC3 (λ=0.25)",
        "full_aa_capped_v_harm_pos0p25": "empirical harm direction (λ=0.25)",
        "full_aa_capped_adv_null_pos0p25": "null-space attack (λ=0.25)",
        "full_aa_capped_multi_signmatched_pos1": "multi-axis composite",
    }
    items = []
    for key, label in label_map.items():
        if key in per:
            v = per[key]
            items.append((label, v["harm_rate"] * 100, v["coherence_rate"] * 100))
    items.sort(key=lambda x: x[1], reverse=True)
    labels, harms, cohs = zip(*items)
    baseline = r["baseline_harm"] * 100
    defended = r["aa_cap_harm"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(labels),
            y=list(harms),
            marker_color=[C_WARN if h > defended else C_ACCENT for h in harms],
            text=[f"{h:.1f}%<br><sub>coh {c:.1f}%</sub>" for h, c in zip(harms, cohs)],
            textposition="outside",
            customdata=cohs,
            hovertemplate=(
                "<b>%{x}</b><br>harm = %{y:.1f}%"
                "<br>coherence = %{customdata:.1f}%<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=defended,
        line=dict(color=C_OK, dash="dash"),
        annotation_text=f"defended baseline = {defended:.1f}%",
        annotation_position="top right",
    )
    fig.add_hline(
        y=baseline,
        line=dict(color=C_GREY, dash="dot"),
        annotation_text=f"undefended baseline = {baseline:.1f}%",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=(
            "Gemma 4 31B (reasoning off): every attack class on the defended model"
            "<br><sub>The defense (Assistant-direction cap) sits above the undefended baseline at 11.2% — i.e. the cap slightly increases harm at zero attack. "
            "A single harm-aligned PC3 attack at coherence-preserving λ then drives harm to 23.6% at 99.8% coherence.</sub>"
        ),
        yaxis=dict(title="harm rate (%)", range=[0, max(harms) * 1.25]),
        template="plotly_white",
        margin=dict(t=110, b=110, l=70, r=40),
        height=520,
    )
    fig.write_html(DOCS_FIGURES / "thinking_off_attack_breakdown.html", include_plotlyjs="cdn")

    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, harms, color=[C_WARN if h > defended else C_ACCENT for h in harms])
    ax.axhline(defended, color=C_OK, linestyle="--", label=f"defended baseline ({defended:.1f}%)")
    ax.axhline(baseline, color=C_GREY, linestyle=":", label=f"undefended baseline ({baseline:.1f}%)")
    for xi, (h, c) in enumerate(zip(harms, cohs)):
        ax.text(xi, h + 0.5, f"{h:.1f}%\n(coh {c:.0f}%)", ha="center", fontsize=8)
    ax.set_ylabel("harm rate (%)")
    ax.set_title("Gemma 4 31B (reasoning off) — per-attack harm vs the defended baseline")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    fig2.tight_layout()
    fig2.savefig(DOCS_FIGURES / "thinking_off_attack_breakdown.png", dpi=150)
    plt.close(fig2)


def main() -> None:
    DOCS_FIGURES.mkdir(parents=True, exist_ok=True)
    rows = collect()
    fig_auc_lift(rows)
    fig_attack_recovery(rows)
    fig_v_harm_alignment(rows)
    fig_probe_depth(rows)
    fig_pipeline_validation(rows)
    fig_per_model_pareto()
    fig_lasso_pcs(rows)
    fig_thinking_off_breakdown(rows)
    print("Wrote figures to", DOCS_FIGURES)
    for r in rows:
        print(
            f"  {r['label']:36s}  AUC(AA)={r['auc_aa']:.3f}  AUC(full)={r['auc_full']:.3f}  "
            f"baseline={r['baseline_harm']*100:.1f}%  capped={r['aa_cap_harm']*100:.1f}%  "
            f"best-attack={r['best_attack_harm']*100:.1f}% ({r['best_attack_label']})"
        )


if __name__ == "__main__":
    main()
