# Multi-Axis Persona Safety — Blog Draft

> Stage 1 wireframe; less formal than `paper.md`. Aimed at LessWrong /
> Twitter / community readers. Filled progressively as Stages 3-6 produce
> results.

## Hook

<!-- DATA NEEDED: headline AUC lift number (Exp 2) and ASR delta (Exp 3) -->

If you cap activations on PC1 of persona space, you reduce harmful-jailbreak
behavior by ~60%. But what about PC2, PC3, ...? We tested whether higher
PCs carry independent safety signal — and whether you can attack a
PC1-capped model by steering through them.

## Why this matters

- Persona jailbreaks work by activating a "role" subspace in the model's
  hidden state. Lu et al. showed there's a single dominant direction (the
  Assistant Axis) that explains most of it.
- "Most" is not "all." If the residual subspace also carries safety
  information, single-axis defenses miss attacks that live there.
- Open-weight models can be steered at the activation level. A defense
  that's robust to one direction is still vulnerable in `d_model − 1`.

## What we did

1. PCA the persona space for 4 subjects (Gemma 2 27B, Qwen 3 32B, Gemma 4
   31B in both thinking modes). 6 cross-model pairs.
2. Train logistic LASSO on `{AA, PC2, ..., PCk}` against binarized harm.
3. Check `AUC(AA + LASSO-selected PCs) − AUC(AA only)` — call this the
   "blind-spot lift."
4. Try to attack a model whose AA is capped at the paper's recommended τ,
   by steering through orthogonal PCs.
5. Test a multi-axis cap defense.

## Headline numbers

<!-- DATA NEEDED: results/exp2_safety_relevance/, results/exp4_blind_spot/, results/exp6_multi_axis_defense/ -->

## Caveats

- Activation-level threat model only — we don't claim these attacks are
  reachable from prompts (cf. the Non-Surjective paper).
- 2 Tier 1 subjects (paper had 3) due to early hardware constraints; Gemma 4
  31B thinking ON/OFF adds two reasoning-mode test conditions the paper
  doesn't cover.
- Llama 3.3 70B reproduction is gated on additional GPU availability
  (Stage 7 Ext 9).

## Try it

Interactive dashboard: `<HF Spaces URL when Stage 8 ships>`. Pick a model,
a steering mode, a strength, and watch the response + harm label change in
real time.

## Code + paper

- Repo: `<github URL>`
- Paper draft: `<arxiv URL>` (when ready)
- Pre-computed dashboard data: `<HF datasets URL>`
