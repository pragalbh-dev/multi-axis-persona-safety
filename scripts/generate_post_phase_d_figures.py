"""Generate Phase D / Phase E / Ext B figures for the GitHub Pages writeup.

Produces three new figure groups in docs/figures/ to complement the existing
cross_model_*.png/html set written by `scripts/generate_cross_model_figures.py`:

  phase_d_multi_axis_closure        — AA-only vs AA+PC2 vs AA+PC2+PC3 cap on g4_off,
                                      across {no_attack, adv_null, PC3-attack}
  phase_e_capability                — IFEval / GSM8k / EQ-Bench × {unsteered, AA-cap,
                                      multi-axis-cap (g4_off only)} × 4 subjects
  ext_b_v_harm_causal               — two-panel per subject: gross harm vs
                                      coherence-conditioned harm × λ ∈ {0.10, 0.25, 0.50},
                                      with the +5 pp threshold band and coherence-collapse
                                      zone shaded out

CPU-only. Reads JSON / parquet artifacts; writes PNG (matplotlib) + HTML (plotly)
into docs/figures/. Page palette matches docs/index.html.

Usage: uv run python -m scripts.generate_post_phase_d_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "figures"
DOCS.mkdir(parents=True, exist_ok=True)

C_ACCENT = "#4C72B0"
C_WARN = "#C44E52"
C_OK = "#55A868"
C_GREY = "#999999"
C_DARK = "#1a1a1a"
C_BAND = "#fff3b0"  # threshold-band shading


# ============================================================================
# Phase D — Multi-axis closure on Gemma 4 31B (reasoning off)
# ============================================================================


def fig_phase_d_closure() -> None:
    h = json.loads((REPO / "results/phase_d/gemma_4_31b_thinking_off/headline.json").read_text())
    M = h["matrix"]
    defences = ["aa_only", "aa_pc2", "aa_pc2_pc3"]
    defence_labels = ["AA cap only", "AA + PC2 cap", "AA + PC2 + PC3 cap"]
    attacks = ["zero", "adv_null_pos0p25", "signmatched_pc3_pos0p25"]
    attack_labels = ["No attack", "Null-space attack (λ=0.25)", "Harm-aligned PC3 attack (λ=0.25)"]
    colors = [C_OK, "#D8A546", C_WARN]

    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=140)
    x = np.arange(len(attacks))
    w = 0.27

    for i, (d, dlabel) in enumerate(zip(defences, defence_labels)):
        harm = [M[d][a]["harm_rate"] * 100 for a in attacks]
        bars = ax.bar(x + (i - 1) * w, harm, w, label=dlabel, color=colors[i], edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, harm):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.4, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color=C_DARK)

    # Phase A baseline reference line (unsteered = 10.2%). The annotation is
    # surfaced in the title rather than in-plot — placing it inline collides
    # with whichever bar happens to be near 10.2% (and there's always one in
    # the "No attack" column).
    ax.axhline(10.2, color=C_GREY, linestyle=":", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(attack_labels, fontsize=10)
    ax.set_ylabel("Harm rate (%)", fontsize=11)
    # Bump y-limit by ~4 pp above the tallest label (the headroom needs to hold
    # the value annotation AND keep clear of the title).
    ax.set_ylim(0, max(M["aa_only"]["signmatched_pc3_pos0p25"]["harm_rate"] * 100,
                       M["aa_pc2"]["signmatched_pc3_pos0p25"]["harm_rate"] * 100,
                       25) + 6)
    ax.set_title("Multi-axis cap closes ~12% of the PC3 blind spot on Gemma 4 31B (reasoning off, n=508/cell)\n"
                 "Dotted line = unsteered baseline (10.2%)",
                 fontsize=11.5, color=C_DARK, pad=12)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "phase_d_multi_axis_closure.png", bbox_inches="tight")
    plt.close(fig)

    # Plotly version
    pf = go.Figure()
    for i, (d, dlabel) in enumerate(zip(defences, defence_labels)):
        harm = [M[d][a]["harm_rate"] * 100 for a in attacks]
        coh  = [M[d][a]["coherence_rate"] * 100 for a in attacks]
        pf.add_trace(go.Bar(
            name=dlabel, x=attack_labels, y=harm,
            marker=dict(color=colors[i], line=dict(color="white", width=0.6)),
            text=[f"{v:.1f}%" for v in harm], textposition="outside",
            customdata=np.array(coh).reshape(-1, 1),
            hovertemplate=("%{x}<br>" + dlabel + ": %{y:.2f}%<br>coherence: %{customdata[0]:.1f}%<extra></extra>"),
        ))
    pf.add_hline(y=10.2, line=dict(color=C_GREY, dash="dot"),
                 annotation=dict(text="unsteered baseline (10.2%)", font=dict(color=C_GREY, size=10)),
                 annotation_position="top right")
    pf.update_layout(
        title="Multi-axis cap closes ~12% of the PC3 blind spot (g4 reasoning-off, n=508/cell)",
        yaxis_title="Harm rate (%)", barmode="group",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="-apple-system, Helvetica, Arial, sans-serif", size=12, color=C_DARK),
        height=460, margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    )
    pf.update_yaxes(gridcolor="#eee")
    pf.write_html(str(DOCS / "phase_d_multi_axis_closure.html"), include_plotlyjs="cdn", full_html=True)


# ============================================================================
# Phase E — Capability cost of AA-cap and multi-axis-cap
# ============================================================================


_SUBJECT_LABEL = {
    "gemma_2_27b": "Gemma 2 27B",
    "qwen_3_32b": "Qwen 3 32B",
    "gemma_4_31b_thinking_off": "Gemma 4 31B (reasoning off)",
    "gemma_4_31b_thinking_on": "Gemma 4 31B (reasoning on)",
}
_SUBJECT_ORDER = ["gemma_2_27b", "qwen_3_32b", "gemma_4_31b_thinking_off", "gemma_4_31b_thinking_on"]


def fig_phase_e_capability() -> None:
    h = json.loads((REPO / "results/phase_e/headline.json").read_text())
    M = h["matrix"]
    benches = ["ifeval", "gsm8k", "eq_bench"]
    bench_labels = ["IFEval", "GSM8k", "EQ-Bench"]

    # Build per-(subject, condition, bench) δ vs unsteered (pp) and absolute scores
    rows = []
    for sub in _SUBJECT_ORDER:
        for cond, benches_data in M[sub].items():
            for b in benches:
                if b in benches_data:
                    rows.append({
                        "subject": sub,
                        "subject_label": _SUBJECT_LABEL[sub],
                        "condition": cond,
                        "bench": b,
                        "score": benches_data[b]["score"] * 100,
                    })
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["subject", "condition"], columns="bench", values="score").reset_index()
    print("Phase E capability table:")
    print(pivot)

    # --- Matplotlib: 3 grouped bar panels (one per bench), 4 subjects × 3 conditions ---
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), dpi=140, sharey=True)
    cond_colors = {"unsteered": C_GREY, "aa_cap": C_ACCENT, "multi_axis_cap": C_WARN}
    cond_labels = {"unsteered": "Unsteered", "aa_cap": "AA cap", "multi_axis_cap": "AA + PC2 + PC3 cap"}
    condition_present = ["unsteered", "aa_cap", "multi_axis_cap"]
    for j, (b, blabel) in enumerate(zip(benches, bench_labels)):
        ax = axes[j]
        x = np.arange(len(_SUBJECT_ORDER))
        widths = 0.27
        for k, cond in enumerate(condition_present):
            vals = []
            for sub in _SUBJECT_ORDER:
                d = M[sub].get(cond, {}).get(b)
                vals.append(d["score"] * 100 if d else np.nan)
            offsets = (k - 1) * widths
            bars = ax.bar(x + offsets, vals, widths, label=cond_labels[cond],
                          color=cond_colors[cond], edgecolor="white", linewidth=0.6)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.1f}",
                            ha="center", va="bottom", fontsize=8, color=C_DARK)
                else:
                    # Mark the absent slot with italic "n/a" lifted high enough that
                    # it can't be misread as a bar of height ~5.
                    ax.text(bar.get_x() + bar.get_width() / 2, 14, "n/a", ha="center",
                            va="bottom", fontsize=8.5, color=C_GREY, style="italic")
        ax.set_xticks(x)
        ax.set_xticklabels(["G2 27B", "Qwen 3", "G4 off", "G4 on"], fontsize=9.5)
        ax.set_title(blabel, fontsize=11)
        ax.set_ylim(0, 105)  # uniform across the three panels (sharey=True covers it)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.set_ylabel("Score (%)", fontsize=11)
    # Single shared legend below the panels (avoids the asymmetric per-panel legend).
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Capability cost of AA cap and multi-axis cap (Phase E, all 4 subjects)",
                 fontsize=12, color=C_DARK, y=1.02)
    fig.text(0.5, -0.10,
             "Multi-axis cap (red) was scoped to Gemma 4 31B reasoning-off only — \"n/a\" marks subjects where it was not run.",
             ha="center", va="top", fontsize=9.5, color=C_DARK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(DOCS / "phase_e_capability.png", bbox_inches="tight")
    plt.close(fig)

    # --- Plotly: same shape ---
    pf = make_subplots(rows=1, cols=3, subplot_titles=bench_labels, shared_yaxes=True,
                       horizontal_spacing=0.06)
    for k, cond in enumerate(["unsteered", "aa_cap", "multi_axis_cap"]):
        for j, b in enumerate(benches):
            vals = []
            for sub in _SUBJECT_ORDER:
                d = M[sub].get(cond, {}).get(b)
                vals.append(d["score"] * 100 if d else None)
            pf.add_trace(go.Bar(
                x=["G2 27B", "Qwen 3", "G4 off", "G4 on"], y=vals,
                name=cond_labels[cond], legendgroup=cond,
                showlegend=(j == 0),
                marker=dict(color=cond_colors[cond]),
                text=[f"{v:.1f}" if v is not None else "" for v in vals],
                textposition="outside",
                hovertemplate=cond_labels[cond] + " — " + b + ": %{y:.2f}<extra></extra>",
            ), row=1, col=j + 1)
    pf.update_layout(
        title="Phase E: capability cost across 4 subjects × 3 benches",
        height=460, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="-apple-system, Helvetica, Arial, sans-serif", size=12, color=C_DARK),
        barmode="group", margin=dict(l=60, r=30, t=70, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.20, xanchor="center", x=0.5),
    )
    pf.update_yaxes(gridcolor="#eee", range=[0, 110])
    pf.write_html(str(DOCS / "phase_e_capability.html"), include_plotlyjs="cdn", full_html=True)


# ============================================================================
# Ext B — Coherence-conditioned v_harm causal test
# ============================================================================


def _binarize(label: str) -> int:
    return 1 if label in ("enough_info", "enough_info_and_follow_perfectly") else 0


def _coh(label: str) -> int:
    return 0 if label == "nonsensical" else 1


def _bca_ci_diff_pseudo(arr: np.ndarray, baseline: float, n_resamples: int, rng) -> tuple[float, float]:
    """BCa-style CI on (mean(arr) − baseline) using a length-matched binomial pseudo-vector."""
    n = len(arr)
    pseudo = np.zeros(n, dtype=np.int32)
    pseudo[: int(round(baseline * n))] = 1
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        ai = rng.integers(0, n, size=n)
        bi = rng.integers(0, len(pseudo), size=len(pseudo))
        diffs[i] = arr[ai].mean() - pseudo[bi].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def fig_ext_b_v_harm_causal() -> None:
    spec = [
        ("gemma_4_31b_thinking_off", "Gemma 4 31B reasoning off  ·  cos(v_harm, AA)=0.05", 0.102),
        ("gemma_4_31b_thinking_on",  "Gemma 4 31B reasoning on  ·  cos(v_harm, AA)=0.56", 0.150),
    ]
    rng = np.random.default_rng(42)

    # Pre-compute all per-(subject, λ) stats so we can build harmonized y-axis ranges.
    all_data = []
    for sub, label, baseline in spec:
        path = REPO / f"results/phase_b/{sub}/extensions/v_harm_causal/rollouts/v_harm_clean_judged.parquet"
        df = pd.read_parquet(path)
        df["harm_binary"] = df["harm_label_primary"].map(_binarize).astype(int)
        df["coherent"] = df["harm_label_primary"].map(_coh).astype(int)
        rows = []
        for cond, sub_df in df.groupby("condition_id"):
            lam = float(cond.split("_pos")[-1].replace("p", "."))
            n = len(sub_df); n_coh = int(sub_df["coherent"].sum())
            gross = float(sub_df["harm_binary"].mean())
            arr_g = sub_df["harm_binary"].to_numpy()
            ci_g_lo, ci_g_hi = _bca_ci_diff_pseudo(arr_g, baseline, 5000, rng)
            if n_coh > 0:
                coh_only = sub_df[sub_df["coherent"] == 1]
                harm_coh = float(coh_only["harm_binary"].mean())
                arr_c = coh_only["harm_binary"].to_numpy()
                ci_c_lo, ci_c_hi = _bca_ci_diff_pseudo(arr_c, baseline, 5000, rng)
            else:
                harm_coh = np.nan; ci_c_lo = ci_c_hi = np.nan
            rows.append({"lam": lam, "n": n, "n_coh": n_coh, "coherence": n_coh / n,
                         "harm_gross": gross,
                         "lift_g_pp": (gross - baseline) * 100,
                         "lift_g_lo_pp": ci_g_lo * 100, "lift_g_hi_pp": ci_g_hi * 100,
                         "harm_coh": harm_coh,
                         "lift_c_pp": (harm_coh - baseline) * 100 if not np.isnan(harm_coh) else np.nan,
                         "lift_c_lo_pp": ci_c_lo * 100, "lift_c_hi_pp": ci_c_hi * 100})
        all_data.append((sub, label, baseline, pd.DataFrame(rows).sort_values("lam").reset_index(drop=True)))

    # Harmonize y-axis ranges per COLUMN (gross vs coherence-conditioned) so the
    # two rows are vertically comparable. Right-column ignores uninformative (coh<0.5)
    # and bottom-row dominates the negative range; left-column has the coherence-collapse
    # outlier so it gets its own scale per row.
    all_g_rows = pd.concat([d for _, _, _, d in all_data])
    g_lo = float(min(all_g_rows["lift_g_lo_pp"].min(), -2)) - 3
    g_hi = float(max(all_g_rows["lift_g_hi_pp"].max(), 8)) + 5
    coh_only_rows = all_g_rows[all_g_rows["coherence"] >= 0.5]
    c_lo = float(min(coh_only_rows["lift_c_lo_pp"].min(), -2)) - 3
    c_hi = float(max(coh_only_rows["lift_c_hi_pp"].max(), 8)) + 4

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), dpi=140, sharex="col")
    for row, (sub, label, baseline, d) in enumerate(all_data):
        print(f"{sub}:"); print(d)

        # --- Left panel: gross harm rate ---
        ax_g = axes[row, 0]
        # Coherence-collapse zone shading + legend proxy.
        coh_zone_drawn = False
        for _, r_ in d.iterrows():
            if r_["coherence"] < 0.5:
                ax_g.axvspan(r_["lam"] - 0.04, r_["lam"] + 0.04, color="#f6d7d7",
                             alpha=0.55, zorder=0,
                             label=("coherence collapse (coh<0.5)" if not coh_zone_drawn else None))
                coh_zone_drawn = True
        ax_g.errorbar(d["lam"], d["lift_g_pp"],
                      yerr=[d["lift_g_pp"] - d["lift_g_lo_pp"], d["lift_g_hi_pp"] - d["lift_g_pp"]],
                      fmt="o-", color=C_ACCENT, ecolor=C_ACCENT, capsize=4, lw=2, ms=7,
                      label="Gross harm lift", zorder=3)
        ax_g.axhline(0, color=C_DARK, lw=0.7, zorder=1)
        ax_g.axhline(5, color=C_WARN, lw=1.0, linestyle="--", label="+5 pp threshold", zorder=1)

        # Coherence labels: 30 pt offset above the marker so they clear the
        # error-bar caps and don't crowd the +5 pp threshold line.
        for _, r_ in d.iterrows():
            ax_g.annotate(f"coh={r_['coherence']:.2f}",
                          xy=(r_["lam"], r_["lift_g_hi_pp"]),
                          xytext=(0, 8), textcoords="offset points",
                          ha="center", va="bottom", fontsize=8.5, color=C_GREY)

        # Subplot title (NOT inside the plot frame).
        ax_g.set_title(f"{label}  —  gross harm" if row == 0 else f"{label}  —  gross harm",
                       fontsize=10.5, color=C_DARK, pad=8, loc="left")

        ax_g.set_xlim(0.05, 0.55); ax_g.set_xticks([0.10, 0.25, 0.50])
        ax_g.set_ylim(g_lo, g_hi)
        ax_g.set_ylabel("Harm lift vs baseline (pp)", fontsize=10)
        ax_g.grid(axis="y", alpha=0.25, linestyle="--")
        ax_g.spines[["top", "right"]].set_visible(False)
        if row == 1:
            ax_g.set_xlabel("Steering λ", fontsize=10)
        # Legend in the upper-right corner where it has empty space (g_hi headroom).
        ax_g.legend(loc="upper right", fontsize=9, frameon=False)

        # --- Right panel: coherence-conditioned harm ---
        ax_c = axes[row, 1]
        mask_inf = d["coherence"] >= 0.5
        d_inf = d[mask_inf]; d_uninf = d[~mask_inf]
        ax_c.errorbar(d_inf["lam"], d_inf["lift_c_pp"],
                      yerr=[d_inf["lift_c_pp"] - d_inf["lift_c_lo_pp"], d_inf["lift_c_hi_pp"] - d_inf["lift_c_pp"]],
                      fmt="s-", color=C_OK, ecolor=C_OK, capsize=4, lw=2, ms=7,
                      label="Harm among coherent", zorder=3)
        ax_c.axhline(0, color=C_DARK, lw=0.7, zorder=1)
        ax_c.axhline(5, color=C_WARN, lw=1.0, linestyle="--", label="+5 pp threshold", zorder=1)
        for _, r_ in d_uninf.iterrows():
            # Plot the X marker slightly below 0 so it doesn't read as a real
            # data point at exactly zero on the y=0 reference line.
            ax_c.scatter(r_["lam"], -2.2, marker="x", color=C_GREY, s=80, zorder=5)
            ax_c.annotate(f"n_coh={r_['n_coh']}\nuninformative",
                          xy=(r_["lam"], -2.2),
                          xytext=(0, -10), textcoords="offset points",
                          ha="center", va="top", fontsize=8.5, color=C_GREY)
        ax_c.set_title(f"{label}  —  harm among coherent",
                       fontsize=10.5, color=C_DARK, pad=8, loc="left")
        ax_c.set_xlim(0.05, 0.55); ax_c.set_xticks([0.10, 0.25, 0.50])
        ax_c.set_ylim(c_lo, c_hi)
        ax_c.grid(axis="y", alpha=0.25, linestyle="--")
        ax_c.spines[["top", "right"]].set_visible(False)
        if row == 1:
            ax_c.set_xlabel("Steering λ", fontsize=10)
        ax_c.legend(loc="upper right", fontsize=9, frameon=False)

    fig.suptitle("Ext B: clean v_harm steering (no cap, no DAN attack) on baseline DAN prompts",
                 fontsize=12.5, color=C_DARK, y=1.00)
    fig.tight_layout()
    fig.savefig(DOCS / "ext_b_v_harm_causal.png", bbox_inches="tight")
    plt.close(fig)

    # Plotly companion (single 2x2 panel)
    pf = make_subplots(rows=2, cols=2, shared_xaxes="columns",
                       subplot_titles=["Gross harm", "Harm among coherent",
                                       "Gross harm", "Harm among coherent"],
                       horizontal_spacing=0.10, vertical_spacing=0.18)
    for row, (sub, label, baseline) in enumerate(spec, start=1):
        path = REPO / f"results/phase_b/{sub}/extensions/v_harm_causal/rollouts/v_harm_clean_judged.parquet"
        df = pd.read_parquet(path)
        df["harm_binary"] = df["harm_label_primary"].map(_binarize).astype(int)
        df["coherent"] = df["harm_label_primary"].map(_coh).astype(int)
        rows = []
        for cond, sub_df in df.groupby("condition_id"):
            lam = float(cond.split("_pos")[-1].replace("p", "."))
            n = len(sub_df); n_coh = int(sub_df["coherent"].sum())
            gross = float(sub_df["harm_binary"].mean())
            harm_coh = float(sub_df[sub_df["coherent"] == 1]["harm_binary"].mean()) if n_coh else np.nan
            rows.append({"lam": lam, "n": n, "n_coh": n_coh, "coherence": n_coh / n,
                         "lift_g_pp": (gross - baseline) * 100,
                         "lift_c_pp": (harm_coh - baseline) * 100 if not np.isnan(harm_coh) else None})
        d = pd.DataFrame(rows).sort_values("lam").reset_index(drop=True)
        pf.add_trace(go.Scatter(
            x=d["lam"], y=d["lift_g_pp"], mode="lines+markers+text",
            line=dict(color=C_ACCENT, width=2.5), marker=dict(size=10),
            text=[f"coh={c:.2f}" for c in d["coherence"]], textposition="bottom center",
            name=f"{label} — gross", legendgroup=sub, showlegend=False,
            hovertemplate="λ=%{x}<br>lift=%{y:.2f} pp<extra></extra>",
        ), row=row, col=1)
        d_inf = d[d["coherence"] >= 0.5]
        pf.add_trace(go.Scatter(
            x=d_inf["lam"], y=d_inf["lift_c_pp"], mode="lines+markers",
            line=dict(color=C_OK, width=2.5, dash="solid"), marker=dict(size=10, symbol="square"),
            name=f"{label} — coherent only", legendgroup=sub, showlegend=False,
            hovertemplate="λ=%{x}<br>lift_among_coherent=%{y:.2f} pp<extra></extra>",
        ), row=row, col=2)
        for col in (1, 2):
            pf.add_hline(y=0, line=dict(color=C_DARK, width=0.7), row=row, col=col)
            pf.add_hline(y=5, line=dict(color=C_WARN, width=1.0, dash="dash"), row=row, col=col)
        pf.add_annotation(x=0.50, y=d["lift_g_pp"].iloc[0],
                          text=label, xref=f"x{(row-1)*2+1}", yref=f"y{(row-1)*2+1}",
                          showarrow=False, xanchor="right", yanchor="bottom",
                          font=dict(size=10, color=C_DARK))

    pf.update_xaxes(tickvals=[0.10, 0.25, 0.50])
    pf.update_yaxes(title_text="lift vs baseline (pp)", gridcolor="#eee")
    pf.update_layout(
        height=620, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="-apple-system, Helvetica, Arial, sans-serif", size=12, color=C_DARK),
        title="Ext B — clean v_harm steering on baseline DAN (no cap, no DAN attack)",
        showlegend=False, margin=dict(l=70, r=30, t=80, b=50),
    )
    pf.write_html(str(DOCS / "ext_b_v_harm_causal.html"), include_plotlyjs="cdn", full_html=True)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    fig_phase_d_closure()
    fig_phase_e_capability()
    fig_ext_b_v_harm_causal()
    print(f"\nWrote 3 figure groups to {DOCS}")


if __name__ == "__main__":
    main()
