# Project page (`docs/`)

Single-page GitHub Pages site for the Plan B fellowship-application demo. Source: `index.html`.

## Publishing

In the GitHub repo settings → Pages → Source: `Deploy from a branch` → Branch: `main`, folder: `/docs`. Public URL will be `https://pragalbh-dev.github.io/multi-axis-persona-safety/`.

## Filling in numbers after the Plan B run

Every `<span class="placeholder">{KEY}</span>` in `index.html` reads from `results/plan_b_gemma2_27b/metrics.json`. Map:

| Placeholder | metrics.json key |
| --- | --- |
| `{AA_DELTA_PP}` | `harm.aa_cap_delta_pp` |
| `{PC2_RECOVERY_PP}` | `harm.pc2_recovery_pp` |
| `{PC3_RECOVERY_PP}` | `harm.pc3_recovery_pp` |
| `{RANDOM_MAX_PP}` | `harm.random_recovery_pp_max` |
| `{BLIND_SPOT_AUC_DELTA}` | `blind_spot.auc_delta` |
| `{CI_LOW}` / `{CI_HIGH}` | `blind_spot.ci_low` / `blind_spot.ci_high` |
| `{L_STAR}` | `extraction.l_star` |
| `{PC1_AA_COSSIM}` | `extraction.pc1_aa_cossim` |
| `{JUDGE_AGREEMENT_PCT}` | `judges.qwen_vs_gpt55_agreement_pct` |

A small fill script (post-Plan-B, ~10 lines of Python with `re.sub` over `metrics.json`) can do this automatically — write it after the run when the exact key paths are confirmed.

## Embedding the figures

The three iframes expect:

- `docs/figures/harm_rate_per_condition.html`
- `docs/figures/scree_plot.html`
- `docs/figures/blind_spot_summary.html`

Plan B writes these to `results/plan_b_gemma2_27b/figures/`. After the run, copy or symlink:

```bash
cp results/plan_b_gemma2_27b/figures/*.html docs/figures/
```

The Plotly HTMLs already use `include_plotlyjs="cdn"`, so they render standalone in an iframe with no extra setup.
