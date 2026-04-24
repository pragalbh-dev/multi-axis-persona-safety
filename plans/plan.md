# Multi-Axis Persona Safety — Master Plan

## Agent Onboarding

If you're a new agent picking up this project:

1. Read `../CLAUDE.md` (auto-loaded) — project identity, key decisions, conventions
2. Read this file — stage overview, see where we are
3. Read `CONVENTIONS.md` — locked tooling/style decisions + log of in-flight decisions
4. Read `progress.md` — what's been done, plus the latest **Handoff block** for your stage
5. Read `decisions.md` — unplanned decisions made by earlier agents that may affect your work
6. Read the active stage plan (linked from the table below). Start with the "Required inputs" section at the top.
7. If you need research context: read `~/obsidian-vault/wiki/syntheses/multi-axis-persona-safety-scope.md`

Do NOT read the full scope doc unless you specifically need research context (hypotheses, positioning, methodology details). The CLAUDE.md has all the operational decisions.

---

## Stage-to-stage handoff protocol

Every stage passes context forward via a **Handoff block** in `progress.md`. The last task of every stage is to write this block.

**When you finish a stage:**
1. Check all task boxes in the stage plan.
2. Update "Current State" in `../CLAUDE.md` (active stage → next stage, date).
3. Append a Handoff block to `progress.md` (template at top of that file). It must list:
   - Artifacts produced (paths + what's in them)
   - Key decisions locked this stage (mirror the important ones into `CONVENTIONS.md` "Decide and log")
   - Gotchas discovered that the next stage needs to know
   - Open items the next stage must resolve

**When you start a stage:**
1. Open `progress.md` and find the Handoff block for your stage (e.g., "Stage 2 → Stage 3 Handoff").
2. That tells you where inputs live and what to watch for.
3. The stage plan's "Required inputs" section points to the same Handoff — read both.

This is the mechanism for a fresh agent (after context reset) to pick up without reading every prior stage plan.

---

## Unplanned-decision logging (mandatory)

If you make a decision during stage execution that was NOT in the pre-planned stage doc, **log it to `decisions.md`** using the template at the top of that file. This is separate from `progress.md` (which tracks work done) and from `CONVENTIONS.md` (which tracks policy, not execution decisions).

**When to log:**
- Picking a concrete integer / hyperparameter when the plan said "somewhere around X" (e.g., extraction layer argmax came in at L30 instead of paper's L32).
- Resolving an ambiguity in the paper when the plan didn't specify.
- Picking between viable library versions or implementations the plan didn't name.
- Taking a scope cut under time or compute pressure.
- Any choice you'd want a future agent (or the user) to be able to audit or reverse.

**When NOT to log:**
- Task completion (use the stage plan checkboxes + `progress.md`).
- Anything already in `CONVENTIONS.md` or a stage plan — that's pre-planned, not an unplanned decision.

**Template:** copy from the top of `decisions.md` — requires Decision, Alternatives considered, Reason, Source (paper line / file / URL / user instruction / own judgment), Reversibility (high/medium/low), How to revert, Downstream dependencies.

Why this protocol matters: experiments depend on each other. A small choice in Stage 3 (e.g., "I picked L31 instead of L32 because cos_sim was marginally higher") may show up as an anomaly in Stage 5 and we need to trace it back. Without the log, "this result looks off" is unanswerable.

---

## Project Summary

We extend the Assistant Axis paper to study persona space beyond PC1. Four sub-questions: structure (how many PCs matter), safety (do orthogonal PCs carry safety info), geometry (linear vs manifold), defense (does multi-axis capping help). Six core experiments, 5 analytical visualizations + 1 interactive demo, paper + blog post.

**Scope doc:** `~/obsidian-vault/wiki/syntheses/multi-axis-persona-safety-scope.md`

---

## Stage Overview

| # | Stage | Plan | Status | Tasks | Done | Depends on |
|---|-------|------|--------|-------|------|------------|
| 0 | Environment setup | [stage-0-environment.md](stage-0-environment.md) | pending | 8 | 0 | — |
| 1 | Architecture & wireframing | [stage-1-architecture.md](stage-1-architecture.md) | pending | 7 | 0 | Stage 0 |
| 2 | Core infrastructure | [stage-2-infrastructure.md](stage-2-infrastructure.md) | pending | 9 | 0 | Stage 1 |
| 3 | Foundation experiments (Exp 1+2) | [stage-3-foundation.md](stage-3-foundation.md) | pending | 10 | 0 | Stage 2 |
| 4 | Attack experiments (Exp 3+4) | [stage-4-attack.md](stage-4-attack.md) | pending | 8 | 0 | Stage 3 |
| 5 | Composition experiment (Exp 5) | [stage-5-composition.md](stage-5-composition.md) | pending | 7 | 0 | Stage 2 + Exp 1 |
| 6 | Defense experiment (Exp 6) | [stage-6-defense.md](stage-6-defense.md) | pending | 6 | 0 | Stage 4 |
| 7 | Extensions | [stage-7-extensions.md](stage-7-extensions.md) | pending | varies | 0 | Stage 6 |
| 8 | Presentation & dissemination | [stage-8-presentation.md](stage-8-presentation.md) | pending | 8 | 0 | Stages 3-6 |

**Note:** Stages 4 and 5 can run in parallel (different GPU workloads, no dependency between them). (Only if we have enough GPUs)

**Subjects across Stages 3/4/6 (revised 2026-04-24 for 2-GPU constraint):** Gemma 2 27B + Qwen 3 32B thinking OFF + **Gemma 4 31B dense** in both thinking ON and OFF modes = **4 subjects**. All run quantized per CONVENTIONS "Quantization policy." **Llama 3.3 70B moved to Stage 7 Ext 9** (hardware-gated on 4-GPU availability; cited from paper in the meantime). MoE (Qwen 3.6-35B-A3B) remains in Stage 7 Ext 1 (tooling-gated + hardware-gated). Gemma 4 31B was promoted from extension to core because it fills the paper's frontier+reasoning gap (§8.1) without extra tooling. Core focus: validating our setup against paper's reproducible results on 2 Tier 1 + extending the multi-axis safety story to 2 Tier 2 modes, which delivers H1-H4 claims independent of Llama.

---

## Experiment → Stage Mapping

| Experiment | Stage | What it tests |
|------------|-------|---------------|
| Exp 1: Persona Space Decomposition | Stage 3 | Structure — eigenspectrum, PC interpretation, effective dimensionality |
| Exp 2: Safety Relevance of Higher PCs | Stage 3 | Safety — per-PC harm correlation, blind spot fraction |
| Exp 3: Orthogonal Steering | Stage 4 | Attack — can PC2/PC3 steering degrade safety while PC1 stays safe? |
| Exp 4: Blind Spot Construction | Stage 4 | Attack — can adversarial null-space directions bypass PC1 capping? |
| Exp 5: Persona Composition | Stage 5 | Geometry — is persona arithmetic linear or nonlinear? Manifold evidence? |
| Exp 6: Multi-Axis Defense | Stage 6 | Defense — does multi-axis capping improve the Pareto frontier? |

---

## Visualization → Stage Mapping

| Visualization | Built in | Data from |
|---------------|----------|-----------|
| Viz 1: Persona Space Explorer (3D PCA) | Stage 3 | Exp 1 |
| Viz 2: Safety Heatmap Across PCs | Stage 3 | Exp 2 |
| Viz 3: Steering Effect Comparator | Stage 4 | Exp 3 |
| Viz 4: Layer-by-Layer Persona Signal | Stage 6 | Exp 1 + ablations |
| Viz 5: Persona Arithmetic | Stage 5 | Exp 5 |
| Viz 6: Persona Steering Playground | Stage 8 | All experiments |

---

## Report Structure

Filled progressively as experiments complete:

| Section | Filled after | Status |
|---------|-------------|--------|
| 1. Introduction | Stage 1 (draft) | pending |
| 2. Background & Related Work | Stage 1 (draft) | pending |
| 3. Persona Space Decomposition | Stage 3 (Exp 1) | pending |
| 4. Safety Relevance of Higher PCs | Stage 3 (Exp 2) | pending |
| 5. Orthogonal Steering | Stage 4 (Exp 3) | pending |
| 6. Blind Spot Construction | Stage 4 (Exp 4) | pending |
| 7. Persona Composition | Stage 5 (Exp 5) | pending |
| 8. Multi-Axis Defense | Stage 6 (Exp 6) | pending |
| 9. Discussion | Stage 8 (final) | pending |
| 10. Conclusion | Stage 8 (final) | pending |

---

## Dissemination Plan

| Artifact | Format | Audience | Built in |
|----------|--------|----------|----------|
| Technical report / paper | PDF (LaTeX or Typst) | Reviewers, researchers | Progressive, finalized Stage 8 |
| Blog post | Markdown → web | Twitter/LessWrong/community | Stage 8 |
| Interactive dashboard | Plotly Dash on HF Spaces | Everyone | Progressive, deployed Stage 8 |
| Persona Steering Playground | Plotly Dash on HF Spaces | Everyone | Stage 8 |
| Code + pre-computed data | GitHub + HuggingFace | Reproducibility | Stage 8 |
| Fellowship application | 1-page proposal | OpenAI / Astra reviewers | Whenever ready (deadline May 3) |
