# Dashboard wireframe — Persona Steering Playground

Stage 1 T1.7. Stage 8 builds this; Stages 3-6 can ignore it. The Dash app at
`dashboard/app.py` ships only a layout skeleton in Stage 1 — no callbacks.

## Top-line goal

A reader picks a model, a steering mode, a strength, a defense config, a
dataset, and a prompt; the page shows the resulting response, harm label,
and a small geometry view (where this rollout lives in PC1×PC2 space, where
the AA threshold sits).

## ASCII layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Multi-Axis Persona Safety — Steering Playground                          │
├──────────────────────┬─────────────────────────────┬─────────────────────┤
│ Selectors (left)     │ Output (center)             │ Geometry (right)    │
│ ──────────────────── │ ─────────────────────────── │ ─────────────────── │
│ Model:               │ INPUT                       │ PC1 × PC2 scatter   │
│  [▼ Gemma 2 27B    ] │ ┌───────────────────────┐  │ (all rollouts in    │
│                      │ │ Pretend you are DAN…  │  │  current dataset,   │
│ Steering mode:       │ │                       │  │  current defense)   │
│  [▼ AA cap         ] │ └───────────────────────┘  │ ★ this rollout      │
│                      │                            │                     │
│ λ:  -2 ●━━━━━━○ +2   │ RESPONSE                    │ ┌─────────────┐     │
│                      │ ┌───────────────────────┐  │ │ AA proj bar │     │
│ Defense:             │ │ I cannot pretend to be│  │ │  ▓▓▓│  τ    │     │
│  ☑ AA cap            │ │ DAN. I am Gemma…      │  │ └─────────────┘     │
│  ☐ +PC2              │ │                       │  │                     │
│  ☐ +PC3              │ └───────────────────────┘  │ Capping markers:    │
│                      │                            │  AA τ = -1.42 (p25) │
│ Dataset:             │ HARM:  [ refusal ] safe    │  PC2 τ = … (off)    │
│  ◉ DAN               │ ASR baseline: 0.31         │                     │
│  ○ Shah-recon        │ ASR this cell: 0.07        │                     │
│                      │                            │                     │
│ Prompt:              │ Capability:                 │                     │
│  [search box]        │  IFEval = 0.72             │                     │
│  [▼ prompt list]     │                            │                     │
└──────────────────────┴─────────────────────────────┴─────────────────────┘
```

## Components

### Left rail — Selectors
- **Model dropdown** — from `dashboard/data/*.parquet` glob.
- **Steering mode dropdown** — `none | addition | ablation | capping | cap+steer`.
- **λ slider** — −2 to +2 in 0.5 steps, disabled when mode == capping.
- **Defense toggles** — checkboxes for AA cap / +PC2 / +PC3 / +all-LASSO.
- **Dataset radio** — DAN / Shah-reconstructed (per CONVENTIONS dual-set rule).
- **Prompt selector** — search box + dropdown (filtered by dataset).

### Center pane — Output
- **Input text** (disabled textarea).
- **Response text** (read-only; reflows on selection change).
- **Harm badge** — colored chip with the 9-category label + binary in tooltip.
- **ASR row** — baseline ASR (default Assistant) vs current cell ASR.
- **Capability row** — relevant capability bench score when condition has one.

### Right pane — Geometry
- **PC1×PC2 scatter** (Plotly) — every rollout for the current `(model,
  dataset, defense_config)` bucket; current selection highlighted with a
  star marker. Hovering shows the prompt_id and harm label.
- **AA projection bar** — single-axis bar showing this rollout's AA
  projection vs the τ marker (per `axes/{model}.json`).
- **Capping marker list** — for whichever PCs are currently capped under the
  active defense config, list τ values in human-readable form.

## Interaction rules

- Changing **model** reloads the parquet shard and resets all other selectors
  to their first valid value.
- Changing **defense** toggles re-loads the right-pane scatter (different
  rollout bucket).
- Changing **steering mode / λ** filters the same parquet to one row and
  updates center + right panes.
- All selections are URL-stateful so anyone can share a deep link to a
  specific cell.

## Stage 8 tasks (not Stage 1)

- `dashboard/build_shards.py` — write the per-model parquets + side artifacts.
- `dashboard/app.py` — fill in `@app.callback` decorators for each pane.
- `dashboard/Dockerfile` (or `space.yaml`) — HF Spaces deploy.

Stage 1 only ships `dashboard/app.py` as a layout-only skeleton (zero
callbacks) so the structure is committed and Stage 8 plugs values in.
