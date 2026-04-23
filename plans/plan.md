# Multi-Axis Persona Safety — Master Plan

## Agent Onboarding

If you're a new agent picking up this project:

1. Read `../CLAUDE.md` (auto-loaded) — project identity, key decisions, conventions
2. Read this file — stage overview, see where we are
3. Read `progress.md` — what's been done
4. Read the active stage plan (linked from the table below) — your current tasks
5. If you need research context: read `~/Documents/knowledge-base/obsidian-vault/wiki/syntheses/multi-axis-persona-safety-scope.md`

Do NOT read the full scope doc unless you specifically need research context (hypotheses, positioning, methodology details). The CLAUDE.md has all the operational decisions.

---

## Project Summary

We extend the Assistant Axis paper to study persona space beyond PC1. Four sub-questions: structure (how many PCs matter), safety (do orthogonal PCs carry safety info), geometry (linear vs manifold), defense (does multi-axis capping help). Six core experiments, 5 analytical visualizations + 1 interactive demo, paper + blog post.

**Scope doc:** `~/Documents/knowledge-base/obsidian-vault/wiki/syntheses/multi-axis-persona-safety-scope.md`

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

**Note:** Stages 4 and 5 can run in parallel (different GPU workloads, no dependency between them).

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
