# Progress Ledger

Append-only log. Each entry: `[YYYY-MM-DD HH:MM] Stage X, Task Y: what was done.`

At the end of every stage, append a **Handoff block** (template below) so the next stage has what it needs.

---

## Handoff block template

Copy, fill, append at the **end** of this file when a stage completes. Do not edit earlier handoff blocks.

```markdown
## Stage {N} → Stage {M} Handoff — {YYYY-MM-DD}

**Artifacts produced:**
- `path/to/thing` — what it contains, schema pointer (e.g., see `CONVENTIONS.md` "Activation cache parquet schema")
- ...

**Decisions locked this stage:**
- Decision 1 (e.g., "Chose vLLM 0.6.x — SGLang didn't support Gemma 4 31B-it at setup time"). Mirror important ones to `CONVENTIONS.md` "Decide and log".
- ...

**Gotchas / surprises:**
- Thing the next stage agent needs to watch for (e.g., "Gemma 4 31B OOMs at batch=8 on TP=2, use batch=4"). Include reproduction steps if non-trivial.
- ...

**Open items for next stage:**
- What next stage must decide or verify before starting real work.
- ...

**Pointers into CONVENTIONS.md updated:**
- List of sections under "Decide and log" that this stage filled in.
```

---

## Log

- [2026-04-22 23:30] Project initialized. Repo created at ~/Documents/personal_projects/multi-axis-persona-safety/. CLAUDE.md, plan.md, and all stage plans written.
- [2026-04-23 18:00] Plan reviewed and tightened: judge switched to self-hosted (Qwen 3.6-27B primary + Gemma 4 31B-it cross-check), Tier 2 framing updated (MoE + reasoning via thinking-mode toggle), new sections added to CLAUDE.md (Inference & Serving, Tooling Versions & Env, Data & Checkpointing), `CONVENTIONS.md` created, Handoff protocol added to `plan.md` and this file, stage plans updated with Required inputs sections and Stage 0 expanded with 5 new setup tasks.
- [2026-04-23 20:30] Phased topology correction: judges run as batch-processing steps (load → classify parquet → tear down), not always-on servers. Large subjects like Llama 3.3 70B now get all 4 GPUs. Stage 0 T0.8/T0.9/T0.11 and Stage 2 T2.4 rewritten.
- [2026-04-24 14:00] Line-by-line paper cross-check against `~/obsidian-vault/raw/papers/assistant-axis/extracted.md` exposed several issues in our plan. Applied the full revision:
  - **Judge correction:** paper uses deepseek-v3 (not gpt-4.1-mini) for harm eval, validated at 91.6% against humans (Appendix D.2.2). Judge prompt is released verbatim — we use it directly.
  - **Judge validation protocol:** Stage 0 T0.9.5 added — build a 200-sample pseudo-ground-truth using GPT-5.5 as strongest frontier model (paper's labels aren't released). One-time ~$5-10 API spend.
  - **Label binarization locked:** `enough_info` OR `enough_info_and_follow_perfectly` → harm=1, rest → harm=0. Matches paper's "rate of harmful responses" metric.
  - **Statistical framework corrected:** per-PC correlation = point-biserial (binary × continuous). Joint model = **logistic LASSO** (not standard LASSO). Quality metric = **ROC-AUC** (not R²). **Blind spot lift = AUC(all PCs) − AUC(PC1-only)** with bootstrap BCa CI.
  - **Layer selection:** do the paper's empirical sweep per model — pick layer with max cos_sim(PC1, Assistant contrast). Not a fixed "middle layer". Added Stage 3 T3.1.5.
  - **Activation aggregation:** locked to mean over response tokens at post-MLP residual stream.
  - **Steering/random baseline norm:** scaled to average post-MLP residual stream norm on lmsys-chat-1m at extraction layer (paper convention). Cached by Stage 0 T0.6.
  - **Activation cache format:** corrected from parquet → safetensors. Parquet only for result tuples.
  - **Tier 2 promotion:** Gemma 4 31B dense (both thinking modes) moved from Ext 1 into Stages 3/4/6 as core subject — fills frontier+reasoning gap without new tooling. MoE stays in Ext because it needs nnsight per-expert hooks. For reasoning: extract activations at BOTH thinking tokens AND answer tokens.
  - **Compute cuts:** Stage 4 T4.1 runs 500-prompt stratified subsample at 7 intermediate strengths, full 1,100 only at λ=0 and strongest attack. Capability eval at 5 strengths not 9. Stage 4 T4.6 Method 2 (projected gradient) cut — reduces to LASSO equivalently.
  - **Stage 5 T5.3:** α=β=0.5 locked for primary linearity test; variable-α,β fit deferred to Stage 7 Ext 8.
  - **Stage 6 T6.2:** cross-percentile sweep now **conditional** on Phase A showing ≥10% additive multi-axis gain. Default is 4 configurations.
  - **Judge prompt template:** paper's verbatim prompt (Appendix D.2.2) stored to `configs/judge_prompt.yaml` as Stage 2 T2.0, ahead of all other Stage 2 work.
  - **Config template:** promoted from Stage 2 to Stage 1 T1.6.5 so Stage 2 implementers don't each reinvent it.
  - **Attribute labeling (T3.4):** made an explicit task using primary judge, 1,375 calls.
  - **Stage 7 rewritten:** Ext 1 is now MoE-only, Ext 2 is reasoning subspace deep-dive (thinking-vs-answer geometry), Ext 8 added for variable-α,β composition fit.
  - **Scientific Conventions section** added to `CONVENTIONS.md` with all paper-derived conventions locked.
  - **Assistant Axis wiki page** + wiki/log.md corrected re: judges.
- [2026-04-24 15:00] Two follow-up corrections from user:
  - **Task ordering:** moved judge-validation set creation (200-sample GPT-5.5 pseudo-ground-truth) from Stage 0 T0.9.5 → **Stage 2 T2.4.5**. It depends on downloaded datasets + a running subject + the finalized judge prompt, none of which exist at end-of-Stage-0. Stage 0 now only installs and smoke-tests judge infra.
  - **Statistical framework clarified:** paper's judge outputs **9 ordinal categories** (not 4 as I wrote earlier). Primary analysis stays **logistic LASSO on binarized harm** because (a) paper's headline metric is binary, (b) Stage 4 adversarial direction wants one "maximize harm" direction, (c) simplest and cleanest for our H1 claim. Added a **secondary ordinal LASSO** on 3-level collapse (refusal / partial / full-info) as a robustness check — reported if it disagrees with binary.
- [2026-04-24 17:00] Pre-implementation second sanity-pass applied (experiments × paper × hypotheses). Ten fixes:
  - **Role vector count fixed everywhere:** paper's fully/somewhat split yields n = 377-463 per model, not 275 (paper line 96). Stage 3 T3.1 now explicitly splits, T3.2 uses the actual per-model n for MP threshold γ=d/n. CONVENTIONS updated.
  - **Capping is multi-layer (paper line 676, 691):** 8 layers for Qwen, 16 for Llama. Stage 3 T3.1.5 extended to emit capping layer range (Tier 1 = paper verbatim, Tier 2 = width-sweep {4,8,16,24} on Pareto). Stage 2 T2.3 capper API updated to require layer ranges + all token positions by default. CONVENTIONS locked.
  - **τ-calibration distribution stated:** per-rollout mean-response-token PC projections at the target capping layer, cached as byproduct of Stage 3 T3.1 extraction — no separate pipeline. Default τ = 25th percentile (paper line 685). User confirmed: regenerate rollouts ourselves for Tier 1 if HF release doesn't include the raw data (conditional on Stage 0 T0.7 audit).
  - **Role-expression judge clarified — one model, two prompts:** Qwen 3.6-27B is the only judge model. It runs with `configs/judge_prompt.yaml` (9-cat) for safety experiments and `configs/role_expression_prompt.yaml` (3-label from paper Appendix A) only during Tier 2 role-vector extraction. No separate judge model; experiments are identical across tiers. Stage 2 T2.0 and CONVENTIONS clarified.
  - **Stage 0 T0.7 extended:** pulls 240 extraction questions + 5 default-Assistant system prompts + role-expression prompt template + (if released) per-model rollouts + paper's capping layer ranges from the assistant-axis repo.
  - **Default Assistant rollouts for Tier 2 now explicit** (Stage 3 T3.1): 300 default-Assistant rollouts per Tier 2 subject/mode, using paper's 5 system-prompt variants. Tier 1 reuses paper's if HF release provides them, else regenerate.
  - **Capability baselines:** new task Stage 3 T3.5.5 runs unsteered IFEval/MMLU Pro/GSM8k/EQ-Bench (paper's subset sizes) per subject. Sanity-checked against HF model cards (flag if >10% divergence). Cached at `results/baselines/capability_<subject>.parquet`; every downstream stage reads these for capability-delta computation. ~10 hours wall-clock total. User confirmed option (c) — run subsets on our own setup, not pull HF numbers, because HF card numbers use different harnesses and would confound deltas.
  - **PC2/PC3 cross-model pooling criterion locked** (paper line 294 shows PC2 partially stable, PC3 model-specific): cos_sim > 0.7 → pool; else per-model. Stage 3 T3.3 produces `configs/pc_pooling.yaml`; Stage 4 T4.1 respects it and refuses to average across non-pooled subspaces.
  - **Stage 4 T4.0 added** — locks PC1 capping config per subject (layer range + τ) and runs paper-reproduction sanity check (~60% harm reduction on Tier 1 via PC1 capping). Every downstream "PC1-capped" config reads from `configs/pc1_capping.yaml`.
  - **Stage 6 T6.1 orthogonality verification:** PCs are orthogonal at the extraction layer only; capping runs at ±4-8 layer range. T6.1 now verifies cos_sim between PCs at every capping layer; if >0.1 anywhere, switches to deterministic PC1→PC2→PC3 capping order. Records orthogonality matrix.
  - **Steering/capping token scope made explicit** (all token positions, prompt + response) — CONVENTIONS + Stage 2 T2.3.
- [2026-04-24 17:45] Layer-scope correction applied after re-reading paper §3.2.1 + §5.1.2 + Appendix G. My earlier 17:00 edits conflated extraction, steering, and capping layer scope. Paper's actual protocol (now correctly reflected in the plans):
  - **Extraction (single layer):** middle residual stream layer, validated by cos_sim(PC1, Assistant Axis) > 0.71 via per-layer sweep (paper line 96, 3426). Used for PCA + role vectors + per-prompt projections.
  - **Steering (single layer):** same as extraction — paper line 474 "at a middle layer", paper Appendix G.2 line 3438 "at a middle layer as well". My earlier Stage 2 T2.3 claim "capping AND steering operate on an adjacent range of layers" was wrong for steering. Fixed: steering API defaults to single layer, capping API defaults to multi-layer range.
  - **Capping (multi-layer range, INDEPENDENT of extraction layer):** paper §5.1.2 line 691 runs a 2D sweep (center × width) × τ percentile, NOT centered on extraction. Paper's optima: Qwen 3 32B center ≈ 49.5 width 8 (layers 46-53) vs extraction middle ≈ 32; Llama 3.3 70B center ≈ 63.5 width 16 (layers 56-71) vs extraction middle ≈ 40. Capping centers are deeper than extraction middle. My earlier Stage 3 T3.1.5 "sweep widths centered on extraction layer" was wrong. Fixed:
    - Stage 3 T3.1.5 now emits extraction layer ONLY (single layer, argmax cos_sim sweep). Capping range is no longer set here.
    - Stage 4 T4.0 now runs the 2D capping calibration: Tier 1 uses paper's verbatim ranges + τ = 25th percentile + reproduction sanity check; Tier 2 runs the full grid (centers 40-80% depth × widths {4,8,16,24} × τ {1,10,25,50,75}th) × Pareto-pick.
    - Stage 6 T6.2 extended: for each added PC (PC2, PC3, LASSO-selected) on Tier 2 subjects, run the same 2D grid; on Tier 1, default each added PC's layer range to PC1's and sweep τ only, fall back to 2D grid if Phase A fails to show gain.
    - CONVENTIONS "Layer-scope convention" rewritten with a clear extraction-vs-steering-vs-capping table.
  - Net effect: stage plans now correctly implement paper's "steering at extraction layer (middle, single); capping at middle-to-late depths (range, independently swept)" protocol.
- [2026-04-24 18:30] Two follow-ups per user:
  - **Extraction-layer statistic tightened** (Stage 3 T3.1.5): paper's text (line 96) just declares "the middle residual stream layer" — no explicit named statistic. The validation mechanism (Appendix G.1, Figure 27) is per-layer cos_sim(PC1, Assistant contrast); paper's chosen layers are approximately but not exactly `n_layers // 2`. Our operational statistic is argmax cos_sim over a per-layer sweep — this reproduces paper's implicit selection. Expect L* ≈ paper's reported integer ± a few layers; if we diverge >±3 for any Tier 1 model, log to `decisions.md` before proceeding. Capping-range mechanism (paper line 691, 2D sweep over center × width × τ) is already explicitly described in Stage 4 T4.0 — verified, no change needed there.
  - **Decisions-logging protocol added.** New file `plans/decisions.md` with template. Referenced from `plan.md` (agent onboarding + dedicated section) and `CONVENTIONS.md` (new "Unplanned-decision logging" rule). Purpose: any agent that makes a choice not in the pre-planned stage doc (hyperparameter pick, paper-ambiguity resolution, scope cut, etc.) must append a decision entry with reason + source + reversibility + how-to-revert. Separates work history (`progress.md`) from decision history (`decisions.md`) so later stages can audit and reverse upstream choices when anomalies surface.
- [2026-04-24 19:00] **Primary intervention direction switched from PC1 to Assistant Axis (AA) per paper §3.1 line ~468.** Paper explicitly recommends the contrast vector (AA) for reproducing their results because PC1 is not guaranteed to equal the Assistant direction in every model. Paper uses AA throughout main §3.2 steering and §5 capping; PC1 is only their Appendix G comparison. Our project had been using PC1 as the baseline — now corrected.
  - **AA definition:** `AA_L = mean(default Assistant at L) − mean(fully role-playing role vectors at L)`, L2-normalized. Computed per subject per layer during Stage 3 T3.1 extraction; cached to `data/cache/assistant_axis/<subject>.safetensors`.
  - **H1-H4 framing:** AA is the baseline "PC1-analog" direction. PC2, PC3, ... (PCA components of role space) remain the "higher PCs" probing multi-dimensional safety. They are orthogonal to PC1 by construction, ≈orthogonal to AA since cos_sim(PC1, AA) > 0.71 at L*.
  - **Per-subject PC1 ≈ AA check** added to Stage 3 T3.1.5 (threshold 0.7). If any subject falls below, drop PC1-based secondary analysis for that subject, log to `decisions.md`.
  - **Files updated:**
    - CONVENTIONS.md — new "Primary intervention direction" rule under Scientific conventions.
    - CLAUDE.md — new Key Decision section "Primary Intervention Direction (paper-aligned)".
    - Stage 3 T3.1 — adds per-layer AA computation + caching.
    - Stage 3 T3.1.5 — emits AA at L* to `configs/assistant_axis.yaml`; validates cos_sim(PC1, AA) > 0.7.
    - Stage 3 T3.8 — LASSO features now `{AA, PC2, ..., PCk}` (drop PC1 as redundant). Blind spot lift = `AUC(AA + selected PCs 2..k) − AUC(AA only)`.
    - Stage 4 T4.0 — renamed "PC1 capping config" → "Assistant Axis (AA) capping config". Output at `configs/aa_capping.yaml`. τ = 25th percentile of AA projections. Reproduction sanity check runs AA (not PC1), matches paper Figure 10 directly.
    - Stage 4 T4.5 — AA-capped + PC2/PC3 orthogonal steering. Blind spot severity = ASR(AA capped + PC_i steered) − ASR(AA capped baseline).
    - Stage 4 T4.6 — u_adv construction now includes explicit AA-projection step: `u_adv ← u_adv − (⟨u_adv, AA⟩/||AA||²) · AA`. Needed because PCs 2..k are orthogonal to PC1 but not strictly to AA.
    - Stage 6 T6.1 — multi-axis capping configurations: {AA}, {AA+PC2}, {AA+PC2+PC3}, {AA + all LASSO-selected PCs}. Orthogonality check extended to AA × PC pairs at each capping layer.
    - Stage 6 T6.2 — per-direction calibration now starts from AA (Stage 4 T4.0 output) + per-PC sweeps for added PCs.
  - **Stage 5 composition is unaffected** — it operates on role vectors in d_model activation space, independent of the AA vs PC1 choice.
- [2026-04-24 19:30] **Hardware-constrained pivot to 2 GPUs applied.** 2 of 4 RTX 5090s on the box are occupied by other workloads and not accessible for this project. Budget is May 3, so we execute on what we have.
  - **Subject cut:** Llama 3.3 70B dropped from core — at 70B it doesn't fit on 64 GB total even at aggressive quantization with KV cache. Deferred to new **Stage 7 Ext 9** (hardware-gated; unblocks if ≥4 GPUs available). For the cross-model claim, we cite paper's Llama result until then.
  - **Core subjects = 4:** Gemma 2 27B, Qwen 3 32B, Gemma 4 31B thinking-ON, Gemma 4 31B thinking-OFF. Cross-model PC1 stability claim now spans 2 architectures × 2 generations × 2 thinking modes (6 pairs) — broader in variation axes than paper's 3-same-generation Tier 1, even if fewer total subjects.
  - **All subjects + judges run quantized.** New CONVENTIONS "Quantization policy" + CLAUDE.md "Quantization policy" section lock preference order = official-fp8 → official/community-AWQ → Unsloth-fp8/AWQ → self-calibrated AWQ via `autoawq`. Avoid GGUF / bnb-nf4 / Dynamic Quants (vLLM compatibility + extraction fidelity). **fp8 ≠ GGUF Q8_0** — distinction called out explicitly so no future agent conflates them (fp8 = hardware-accelerated float-8 in vLLM; GGUF = llama.cpp int-8, incompatible with our throughput target).
  - **Quantization validity check** (new Stage 2 T2.1.5 utility + Stage 3 T3.1.0 gate): before any extraction, verify the quantized model produces sensible PC1 behavior. Tier 1 mode: project ~100 activations from Assistant-like + fantastical test roles onto paper's released bf16 PC1 direction (from HF, Stage 0 T0.7); require separation > 1.5 σ and default Assistant at the extreme. Tier 2 mode: perplexity within 5% of model-card bf16 + qualitative role-response sanity read. ~10 min per subject; gates Stage 3 T3.1 extraction. Cheaper than running full PCA to validate quant.
  - **Stage 0 updates:** T0.1 engine choice now hard-requires fp8 support on 5090 **Blackwell** (GB202, sm_120) tensor cores — 5090 is Blackwell, not Ada; engine release notes need to confirm Blackwell kernel support; T0.4/T0.5 use the preference-order selection + `decisions.md` logging per subject; T0.8/T0.9 judges are TP=2 only (no data_parallel); T0.11 phased-pipeline util test uses quantized models.
- [2026-04-24 20:00] **Corrected GPU architecture references: RTX 5090 is Blackwell (GB202, sm_120), not Ada Lovelace.** Ada = 4090 (AD102). Updates:
  - CLAUDE.md "Quantization policy" — fp8 hardware-accel attribution corrected to Blackwell (faster fp8 tensor cores than Ada + new fp4 generation).
  - CONVENTIONS.md "Quantization policy" — same correction; added note about fp4 (NVFP4) being Blackwell-native but bleeding-edge in vLLM, reserved as possible Ext 9 fallback for Llama 70B (70B × fp4 ≈ 35 GB weights might fit on 2× 5090 with tight KV). Not default.
  - Stage 0 T0.1 — engine-picking criterion now says "fp8 on Blackwell sm_120", flags that some vLLM/SGLang releases may have Blackwell-specific bugs and release notes need checking.
  - No methodology change from the correction; just accurate hardware attribution. Any agent reading plans/CLAUDE.md now knows we're on Blackwell, not Ada.
- [2026-04-24 20:30] **"Verify, don't guess" meta-rule added** — every agent must web-verify fast-moving facts (package versions, Blackwell kernel support, HF model IDs, fp8/AWQ availability, engine release notes) against PyPI / HuggingFace / GitHub releases / official docs rather than relying on training-cutoff knowledge. Cite URL + date/tag/commit in `plans/decisions.md`. Rule lives in:
  - CLAUDE.md "Do NOT" — top-level rule visible to every agent.
  - CONVENTIONS.md new section "Verify, don't guess — for fast-moving facts" — what to check, what NOT to check (stable project facts don't need re-verification), how to log the source.
  - plan.md agent onboarding — pointer to the CONVENTIONS section.
  - Stage 0 new "Stage 0 meta-rule" section — explicit checklist of what to web-verify during environment setup (vLLM/SGLang Blackwell support, Python version, HF model IDs, quant catalogs, uv/ruff/mypy/etc versions, assistant-axis repo state).
  - Intent: protect against the agent (including me in this session) writing `vllm>=0.x.y` or `repo_id` strings from memory when those are easily verifiable at source-of-truth.
  - **Stage 3 updates:** T3.1.0 new gating task (validity check); T3.3 extended to 4-subject cross-model matrix (6 pairs) with "broader-not-bigger" framing.
  - **Stage 7 Ext 9 added:** Llama 3.3 70B reproduction at fp8 TP=4, replay Stages 3/4/6, extend cross-model matrix to 5 subjects / 10 pairs. Scope locked so if GPUs free up mid-project we can execute immediately.
  - **plan.md** subject-count note updated; scope doc left aspirational (still lists 3 Tier 1). CLAUDE.md reflects hardware-constrained execution variant. If GPUs free up and Ext 9 runs, scope-doc framing stays correct without restructuring.
  - **Compute implications:** Stage 3 T3.5.5 capability baselines = 4 subjects × ~2 hr ≈ 8 hr. Stage 3 T3.6 safety eval baselines = 4 subjects × 1,100 prompts ≈ 6 hr. Stage 4 Tier-2 capping-range sweep (T4.0) = 2 Tier 2 subjects × 6-8 hr ≈ 16 hr. Total core-stage budget roughly ~30% smaller than the 5-subject plan.

## Stage 0 → Stage 1 Handoff — 2026-04-24

**Artifacts produced:**

*Env scaffold:*
- `pyproject.toml`, `uv.lock`, `.python-version` (3.12), `.venv/` — `vllm==0.19.1`, `torch==2.10.0+cu128`, `transformers==5.6.2`, `transformer-lens==3.0.0`, `nnsight==0.4+`, plus `hf-transfer`, `pynvml`, `pandas`, `pyarrow`, `scikit-learn`, `plotly`, `dash`, `dev: ruff/mypy/pytest`.
- `.env.example` committed; `.env` gitignored with `HF_TOKEN` (account `ub0001`, Gemma-2 license accepted).
- `src/utils/env.py` — hard-pins `CUDA_VISIBLE_DEVICES=2,3` at import; raises if anything widens.
- Dir skeleton: `src/{extraction,steering,evaluation,analysis,visualization,utils,experiments,data}/`, `configs/`, `data/{paper_artifacts,eval,cache/*}/`, `tests/{unit,integration}/`, `scripts/`, `external/`, `notebooks/`.

*Paper artifacts (`data/paper_artifacts/`, gitignored):*
- `extraction_questions.json` — 240 questions from paper's JSONL.
- `default_assistant_system_prompts.json` — 5 variants from `roles/instructions/default.json` (incl. empty string).
- `assistant_axis_vectors/` — full 1.2 GB HF dataset `lu-christina/assistant-axis-vectors`: all-layer AA direction `[n_layers, d_model]` bf16 + 275 role + 240 trait vectors + default vector + capping_config per Tier 1 subject. **Gemma 2 27B has no capping_config.pt** (must transcribe from paper Appendix F). **No raw rollouts released** — τ-calibration must be regenerated in Stage 3 T3.1.

*Eval datasets (`data/eval/`, gitignored):*
- `ifeval/` (541, train split, `google/IFEval@966cd89`)
- `mmlu_pro_1400/` (1400 seeded subsample, `TIGER-Lab/MMLU-Pro@54611cd`)
- `gsm8k_1000/` (1000 seeded subsample, `openai/gsm8k@740312a` main/test)
- `eq_bench/` (171 validation, `pbevan11/EQ-Bench@9ce8e5f`)
- `dan_jailbreak/` — primary safety set, **DAN in-the-wild 1,100 stratified** (13-cat, 84–85/cat) via subagent. HF + GitHub sources pinned in `manifest.json`.
- `reconstructed_jailbreak/smoke_26/` — smoke output from the Shah-reconstructor utility.

*Paper-derived configs (`configs/`):*
- `judge_prompt.yaml` — **NOT WRITTEN**. Deferred to Stage 2 T2.0 (9-category harm prompt needs paper Appendix D.2.2 transcription; not present in either starting repo).
- `role_expression_prompt.yaml` — 0-3 rubric template (paper Appendix A pattern, per-role `role_description` fill-in).
- `paper_capping_ranges.yaml` — Qwen 3 32B (46–53, center 49.5, width 8), Llama 3.3 70B (56–71, center 63.5, width 16) verbatim from paper §5.1.2. Gemma 2 27B capping layers: TODO from paper Appendix F before Stage 4 T4.0.
- `subjects.yaml` — all 4 subjects + judge with verified HF IDs, TP, attention backend, chat-template kwargs, per-model tuning.
- `eval_sizes.yaml` — p50/p95/p99 input lens × `max_in`/`max_out` per (dataset, family); 20 entries.
- `model_hooks.yaml` — post-MLP residual hook paths per family for TL + nnsight; Gemma 4 thinking-span extraction rule documented (empirical confirmation in Stage 3 T3.1).

*Eval infrastructure:*
- `src/evaluation/judge_batch.py` — template-agnostic phased judge driver (load → classify → tear down).
- `src/data/build_dan_jailbreak.py`, `src/data/reconstruct_shah_jailbreaks.py` — primary-set downloader + secondary-set synthesizer.
- `scripts/{download_eval_datasets,token_distribution_audit,smoke_load,phased_pipeline_smoke}.py`.

*Verification artifacts (`results/stage_0_smoke/`, gitignored):*
- Per-family smoke JSON × 5 (load seconds, tokens/sec, peak VRAM, sample output).
- `phase{1,2,3}_*.parquet` — end-to-end phased-pipeline dry-run output (50 prompts × Gemma 2 → 50 labels × Qwen 3.6 → 25 cross-check × Gemma 4). 100% parse rate, 100% inter-judge agreement (synthetic prompts are easy — real jailbreaks will vary).

**Decisions locked this stage (see `plans/decisions.md` for full entries):**
- **Engine = vLLM 0.19.1** (stable, PyPI 2026-04-18). Prerelease 0.20.0 tagged but not promoted; 0.19.1 has Qwen3_5 + Gemma4 arch classes + Blackwell sm_120 fp8 GEMM (from 0.19.0). Torch 2.10.0+cu128 transitively.
- **Python = 3.12.0** (pyenv).
- **Gemma 2 27B FP8 = `Infermatic/gemma-2-27b-it-FP8-Dynamic`** — no official Google fp8 exists; `nm-testing/` and `neuralmagic/` variants don't exist on HF.
- **Qwen 3 32B = `Qwen/Qwen3-32B-FP8`** (official; thinking OFF via chat_template_kwargs).
- **Gemma 4 31B = `RedHatAI/gemma-4-31B-it-FP8-block`** (compressed-tensors; `trust_remote_code=True`; **must use TRITON_ATTN** per vLLM #40677 — FLASHINFER breaks Gemma 4 on Blackwell).
- **Qwen 3.6-27B judge = `Qwen/Qwen3.6-27B-FP8`** — multimodal `Qwen3_5ForConditionalGeneration`, text-only use. For judge role: `gpu_memory_utilization=0.70` + `enforce_eager=True` + `max_model_len=1024` + `enable_thinking=False` (defaults to thinking ON — slow otherwise). 0.85 and 0.75 both OOMed during warmup.
- **Primary jailbreak set = DAN (TrustAIRLab/in-the-wild-jailbreak-prompts)** — Shah et al. 2311.03348 not publicly released; DAN is structurally equivalent and scientifically stronger (in-the-wild > synthetic). Reconstructor utility provided for secondary/confirmatory set.

**Gotchas / surprises:**
- **Qwen 3.6-27B and Gemma 4 31B are multimodal** (`*ForConditionalGeneration` with `vision_config`). Weights load even for text-only use → +~10 GB/GPU vs pure-text peers. Multimodal arch also appears not to TP-split vision weights symmetrically; Stage 3 T3.1 hooks need `model.language_model.layers[L].output` not `model.layers[L].output`.
- **vLLM TP=2 subprocess tear-down doesn't fully release VRAM** — sequential smoke loads OOMed on the 4th or 5th model even with `del llm; torch.cuda.empty_cache()`. Resource-tracker warns of 6 leaked semaphores per run. Solution for Stage 2+: spawn each model in a fresh subprocess (e.g., via `subprocess.run([sys.executable, "-c", ...])`) rather than looping in-process.
- **YAML `on`/`off` are reserved boolean aliases** — quote as `"on"`/`"off"` in configs if you want strings.
- **`torch.cuda.max_memory_allocated()` is invisible across vLLM TP subprocesses** — use pynvml (driver-level) for cross-process VRAM. Per-run baseline subtraction is polluted by leftover state; only first-in-batch or solo-process readings are accurate.
- **Qwen 3.6-27B defaults to thinking output** despite being listed as "non-thinking family" in some sources. `enable_thinking=False` required for judge use.
- **Gemma 2 27B has no pre-computed capping_config.pt** on the paper's HF release — we must transcribe from paper Appendix F before Stage 4 T4.0.
- **hf-transfer NOT installed by default** with a fresh uv env; added as runtime dep. Raw HF downloads were ~1 MB/s without it, ~27 MB/s with it. Set `HF_HUB_ENABLE_HF_TRANSFER=1` env var (or add to a shell rc).
- **GPUs 0 and 1 run a parallel `nvidia/Gemma-4-31B-IT-NVFP4` vLLM service on port 8000** for an unrelated LoRA-tuning workload. DO NOT TOUCH. Our project is scoped to GPUs 2,3 only via `src/utils/env.py`.

**Open items for Stage 1:**
- Stage 2 T2.0 must transcribe the 9-category harm judge prompt from paper Appendix D.2.2 (not present in any starting repo) → `configs/judge_prompt.yaml`.
- Stage 2 T2.0 must lock `enable_thinking=False` in the judge chat-template kwargs and ideally tune `compilation_config` to allow some CUDA graph capture (currently fully eager → slow). Target: ≥ 50 tok/s.
- Stage 4 T4.0 must transcribe Gemma 2 27B capping ranges from paper Appendix F — `configs/paper_capping_ranges.yaml` has it as explicit TODO.
- Stage 2 T2.1 factors the subject/judge loading pattern from `smoke_load.py` + `judge_batch.py` into a reusable service harness that spawns subprocesses per model (avoids the VRAM-leak between loads).
- Stage 3 T3.1 must confirm the hook-point YAML paths empirically on first extraction run — `configs/model_hooks.yaml` has the conventions but no empirical verification on our fp8 weights.
- Stage 2 T2.4.5 runs the 200-sample GPT-5.5 pseudo-ground-truth judge validation (only external API spend budgeted).

**Pointers into CONVENTIONS.md updated:**
- `Python version` — 3.12.0.
- `Inference engine` — vLLM 0.19.1 + torch 2.10.0+cu128.
- `Model IDs` — all 4 subjects + judge HF IDs with quant provenance.
- `Eval dataset IDs` — 4 datasets with revision SHAs; Shah deferred, DAN primary via subagent output.
- `Max input/output lengths per task` — audit table summarized.
- `Batch size & TP per model` — per-family load time / tokens/sec / VRAM / tuning flags.

- [2026-04-25 07:47] Stage 1 / T1.8: revert fp8/2-GPU → bf16/4-GPU. Updated `CLAUDE.md` (Models / Hardware / Quantization → Precision policy / Inference & Serving / Data & Checkpointing), `plans/CONVENTIONS.md` (Quantization → Precision policy; Model IDs section refreshed to bf16 base IDs; Batch size & TP block kept old fp8/TP=2 numbers as historical reference). Logged decision to `plans/decisions.md` 2026-04-25 entry.
- [2026-04-25 08:00] Stage 1 / T1.6.5: pinned `pydantic>=2,<3` in pyproject.toml. Wrote `configs/experiment_template.yaml` (4-GPU bf16 defaults). Built `src/utils/config.py::ExperimentConfig` with field validators (model_id ∈ subjects.yaml, layer-range bound, TP enum, paired eval_sizes, dataset/benchmark literals) and resolvers (`resolved_hook_point`, `resolved_eval_sizes`). Filled CONVENTIONS.md "Config schema per experiment".
- [2026-04-25 08:15] Stage 1 / T1.1: added `types.py` per module + `src/utils/manifest.py` + `src/utils/results.py`. Wrote `src/README.md` with module map + import-direction rule + reuse mandate (external/assistant-axis primitives). Updated all `__init__.py` re-exports.
- [2026-04-25 08:25] Stage 1 / T1.2: extraction-pipeline contracts. `src/extraction/extractor.py` (dispatcher), `backend_hf.py` (HF/TL stub wrapping external/assistant-axis ActivationExtractor), `backend_vllm.py` (Stage 3 T3.1 hand-off contract). Locked safetensors cache schema in CONVENTIONS.md "Activation cache safetensors schema".
- [2026-04-25 08:35] Stage 1 / T1.3: steering wrapper. `src/steering/steerer.py` re-exports `external.assistant_axis.steering.ActivationSteering` and adds `from_config(SteeringConfig)`, `cap_and_steer(...)` context-manager composition, `multi_axis_cap(...)` Stage 6 entry, `verify_orthogonality(...)`. Capping range expansion verified against a synthetic nn.Module (4 hooks for layer range [2,5]).
- [2026-04-25 08:45] Stage 1 / T1.4: eval harness contracts. `safety.py` (HARM_LABELS_9CAT + binarize_harm + dual-dataset driver stub), `capability.py` (per-bench dispatch stub keyed off configs/eval_sizes.yaml), `full.py` (phased driver: subject → primary judge → capability → cross-check, with self-preference skip). `evaluation/types.py` locks `PER_PROMPT_COLUMNS` (20 columns including aa_projection, pc_projections JSON, harm_binary).
- [2026-04-25 08:55] Stage 1 / T1.5: analysis + visualization stubs. `analysis/{bootstrap,correlation,lasso,blind_spot}.py` with implemented helpers (`bca_ci`, `bca_ci_difference`, `point_biserial`, `pearson_with_ci`, `kendall_tau`, `bh_fdr`, `auc_with_ci`) and Stage 3 T3.8 stubs (`logistic_lasso_cv`, `ordinal_lasso_cv`, `blind_spot_lift`). `visualization/figures.py` with `FIGURE_REGISTRY` (Fig 1-6 → renderer map) + `figure_paths` helper.
- [2026-04-25 09:05] Stage 1 / T1.6: report wireframes. `report/paper.md` (10 sections per plan.md, each with DATA NEEDED + FIGURES placeholders), `report/blog.md` (lighter mirror), `report/figures.md` (registry table mirroring src/visualization/figures.py).
- [2026-04-25 09:15] Stage 1 / T1.7: dashboard schema + wireframe + skeleton app. `dashboard/data/schema.md` (per-row columns superset of PER_PROMPT_COLUMNS + 2 dashboard-only fields), `dashboard/wireframe.md` (ASCII layout: left selectors / center prompt-response / right PC mini-plot + AA bar), `dashboard/app.py` (layout-only Dash skeleton, no callbacks).
- [2026-04-25 09:30] Stage 1 / T1.x: tests + gates. `tests/unit/{test_config,test_types,test_manifest,test_steerer_compose}.py` — 35 tests, all green. `ruff format` + `ruff check` clean (added `N803`, `N806` to ignore for scientific-Python naming convention). `mypy --strict src` green. Side-effect-free import probe: 2.0s for full module + dashboard layout.

## Stage 1 → Stage 2 Handoff — 2026-04-25

**Artifacts produced:**

*Module layer (src/):*
- `src/extraction/{__init__.py, types.py, extractor.py, backend_hf.py, backend_vllm.py}` — `ActivationCache` (safetensors+meta.json IO), `ExtractionConfig`, top-level `extract_activations(…)` dispatcher. Backends are Stage 1 stubs; Stage 2 T2.2 fills `backend_hf` (forward-hook over `external.assistant_axis.internals.activations.ActivationExtractor`); Stage 3 T3.1 fills `backend_vllm` (vLLM hidden-states or nnsight path).
- `src/steering/{__init__.py, types.py, steerer.py}` — `SteeringConfig` + thin wrappers around `external.assistant_axis.steering.ActivationSteering`: `from_config`, `cap_and_steer` (context-manager composition), `multi_axis_cap` (with `verify_orthogonality`).
- `src/evaluation/{__init__.py, types.py, safety.py, capability.py, full.py}` — `PER_PROMPT_COLUMNS` locked (20 cols), `SafetyResult`, `CapabilityResult`, `EvalResult`, `binarize_harm` + `HARM_LABELS_9CAT`. `eval_safety` / `eval_capability` / `eval_full` are Stage 2 T2.4-T2.6 stubs but signatures + contracts are pinned.
- `src/analysis/{__init__.py, types.py, bootstrap.py, correlation.py, lasso.py, blind_spot.py}` — `bca_ci`, `bca_ci_difference`, `point_biserial`, `pearson_with_ci`, `kendall_tau`, `bh_fdr`, `auc_with_ci` implemented; `logistic_lasso_cv` / `ordinal_lasso_cv` / `blind_spot_lift` are Stage 3 T3.8 contract stubs.
- `src/visualization/{__init__.py, types.py, figures.py}` — `FigureSpec`, `FigureKind` literal (6 figures), `FIGURE_REGISTRY` mapping each kind to `(fig_number, source_exp, viz_id, stage, renderer)`, `figure_paths(name, results_dir)` helper.
- `src/utils/{__init__.py, config.py, manifest.py, results.py}` — `ExperimentConfig` (pydantic v2), `Manifest` (dataclass + JSON IO + git_sha), `init_results_dir` enforcing the per-experiment output contract.
- `src/README.md` — one-page module map, import-direction rule, reuse mandate cribbing from external/assistant-axis.

*Configs:*
- `configs/experiment_template.yaml` — Stage 1 T1.6.5 schema, all fields documented. Defaults: dtype=bf16, tensor_parallel=4, datasets=[dan, shah_reconstructed], capability_benchmarks=[ifeval, mmlu_pro, gsm8k, eq_bench]. Loadable by `load_experiment_config('configs/experiment_template.yaml')`.

*Report wireframes:*
- `report/paper.md` — 10 sections (Introduction → Conclusion + Appendices A-D), each with `<!-- DATA NEEDED -->` + `<!-- FIGURES -->` placeholders pointing at `results/expN_*` dirs.
- `report/blog.md` — lighter LessWrong/Twitter-style mirror.
- `report/figures.md` — Fig 1..6 ↔ Viz 1..5 ↔ Stage/Exp registry table; mirrors `src/visualization/figures.py::FIGURE_REGISTRY`.

*Dashboard wireframe (Stage 8 will plug in callbacks):*
- `dashboard/wireframe.md` — ASCII layout + interaction rules.
- `dashboard/data/schema.md` — locked precomputed-tuple parquet schema (one shard per subject, two `★`-marked dashboard-only fields beyond `PER_PROMPT_COLUMNS`).
- `dashboard/app.py` — layout-only Dash skeleton; `app.layout` is a static `html.Div` with all selector / output / geometry components + `id`s ready for Stage 8 callbacks. Importing the module does not touch GPUs or load parquets.

*Tests + gates:*
- `tests/unit/test_config.py` — ExperimentConfig validator (template loads, unknown model_id, TP enum, layer range, partial eval-size pair, dataset enum, hook-point + eval-sizes resolvers).
- `tests/unit/test_types.py` — happy-path instantiation of every Stage 1 dataclass + ActivationCache safetensors round-trip + SteeringConfig negative paths.
- `tests/unit/test_manifest.py` — Manifest IO + `init_results_dir` fresh + resume.
- `tests/unit/test_steerer_compose.py` — capping-range expansion (4 hooks for layers [2,5]), `cap_and_steer` composition, `multi_axis_cap` orthogonality warning.
- 35/35 tests pass; `ruff format` + `ruff check` clean; `mypy --strict src` clean (33 source files).

**Decisions locked this stage (mirrored to CONVENTIONS.md / decisions.md as relevant):**
- Reverted core stages from fp8/2-GPU (TP=2) to bf16/4-GPU (TP=4). Llama 3.3 70B stays at Ext 9 (140 GB at bf16 doesn't fit). Quant-validity gate moved from Stage 3 prelude to Ext 9 prerequisites only. See `plans/decisions.md` 2026-04-25 entry.
- Pydantic v2 for config validation; pyproject.toml gains `pydantic>=2,<3`. Ruff lint adds N803 + N806 to ignore (scientific-Python convention).
- Activation cache layout = safetensors + sibling .meta.json, one file pair per (model, dataset, layer); aggregation applied at extract time so caches are `(n_prompts, d_model)`.
- Per-prompt result row (`PER_PROMPT_COLUMNS`) — 20 columns; superset is the dashboard-shard schema. Both Stage 4-6 details.parquet writers and Stage 8 dashboard build pipeline target this contract.
- Result-dir contract: `results/expN_*/{config.yaml, manifest.json, metrics.json, details.parquet, figures/}`. Enforced by `init_results_dir(cfg)`.
- Figure numbering 1..6 ↔ Viz 1..5 ↔ Stage/Exp matrix in `src/visualization/figures.py::FIGURE_REGISTRY` + `report/figures.md`.
- Markdown-only report scaffold (per user); LaTeX deferred to Stage 8 if needed.

**Gotchas / surprises:**
- The vendored `external/assistant-axis/assistant_axis/steering.py::ActivationSteering` is one-`intervention_type`-per-instance. Composing capping + steering requires nesting two context managers (`cap_and_steer` does this with `ExitStack`), not merging into one steerer. PyTorch fires hooks in registration order, so cap fires before steer at every layer where both apply — that ordering matches the paper's §5 forward pass.
- `external/assistant-axis` is added to `sys.path` at import time by `src/steering/steerer.py` (it isn't installed as a package). This works for our usage but a future agent who installs the upstream as a real package will need to drop the sys.path hack.
- `eval_sizes.yaml` keys are full subject IDs (`ifeval::gemma_2_27b`), NOT family keys — `model_hooks.yaml` uses family keys (`gemma_2`). `cfg.resolved_eval_sizes(dataset)` uses model_id; `cfg.resolved_hook_point(layer)` uses family. Easy to confuse.
- Pydantic v2 + Literal-typed list defaults need explicit `lambda: list[Literal...](...)` casts in `default_factory` to satisfy mypy strict — see `ExperimentConfig.datasets` / `capability_benchmarks`.
- The `_template_` sentinel in `configs/experiment_template.yaml` for `experiment_id` and `output_dir` is just a placeholder so `load_experiment_config` works on the template. Real experiments override.
- `src/evaluation/judge_batch.py` (Stage 0 deliverable) had two pre-existing mypy errors — one fixed by typing `chat_template_kwargs: dict[str, Any] | None`, the other by asserting `tokenize=False` returns `str`. No behavior change.
- Pre-existing `src/data/build_dan_jailbreak.py` had one untyped `dict` annotation; fixed to `dict[str, Any]` so mypy stays clean.

**Open items for Stage 2 (the implementation stage):**
- T2.0 — transcribe paper Appendix D.2.2 9-category harm prompt to `configs/judge_prompt.yaml`. Stage 1 evaluation/safety.py contract is ready to consume it via `judge_prompt_path`.
- T2.1 — wrap subject-load patterns from `scripts/smoke_load.py` + `judge_batch.py` into a reusable service harness that spawns subprocesses per model (avoids the resource-tracker leak documented in CONVENTIONS "Batch size & TP" note).
- T2.2 — fill `src/extraction/backend_hf.py::extract_via_hf` against `external/assistant-axis::ActivationExtractor` for the bf16 PCA-fitting path.
- T2.4 — fill `src/evaluation/safety.py::eval_safety` (compose `run_judge_batch` + `binarize_harm` + `bca_ci`).
- T2.4.5 — produce the 200-sample GPT-5.5 pseudo-ground-truth judge-validation set (the only external API spend budgeted).
- T2.5 — fill `src/evaluation/capability.py` adapters per benchmark.
- T2.6 — fill `src/evaluation/full.py::eval_full` phased driver.
- T2.x — Stage 2 should NOT modify the `PER_PROMPT_COLUMNS` schema, the result-dir contract, the `ExperimentConfig` shape, or the figure registry without explicit handoff back to Stage 1 design. Schema migrations require a `decisions.md` entry.

**Pointers into CONVENTIONS.md updated:**
- "Quantization policy" → renamed "Precision policy"; bf16 default for core stages; fp8 reserved for Ext 9. Quant-validity check moved to Ext 9 prerequisites.
- "Model IDs" — refreshed to bf16 base IDs (`google/gemma-2-27b-it`, `Qwen/Qwen3-32B`, `google/gemma-4-31B-it`, `Qwen/Qwen3.6-27B`) with TP=4. Historical fp8 IDs kept as Ext 9 reference.
- "Batch size & TP per model" — flagged Stage 0 fp8/TP=2 numbers as historical; live values live in `configs/subjects.yaml` per the Stage 1 prep grid-search commits.
- "Activation cache safetensors schema" — filled.
- "Config schema per experiment" — filled.
- [2026-04-30 14:30] SGLang `--forward-hooks` spike completed on Gemma 2 27B (4 hook patterns work; 1.85× per-token, ~2.8× wall-clock vs HF; capping overhead negligible) and partially on Gemma 4 31B (HF works after a 1-line `_POSSIBLE_LAYER_ATTRS` patch; SGLang 0.5.10 fails — no native `gemma4.py`, FlashInfer rmsnorm shape mismatch on `v_norm`). Verdict: mixed-backend rollout — wire SGLang for Gemma 2 + Qwen 3 steered conditions; keep Gemma 4 (both thinking modes) on HF. See `plans/decisions.md` 2026-04-30 entry. Spike artifacts: `src/steering/sglang_hook_factories.py`, `tests/integration/sglang_hooks_smoke.py`, `scripts/{bench_sglang_vs_hf,run_sglang_spike,run_throughput_bench}.{py,sh}`. Host setup: `cuda-toolkit-12-9` installed (required for sm_120 JIT). Hardware deviation noted: spike host is 1× RTX PRO 6000 Blackwell 96 GB.
- [2026-04-30 17:15] Gate 2 (`--subject` parameterization) complete. Amended `plans/may_3_directive.md` for the 2026-04-30 hardware deviation: 1× RTX PRO 6000 Blackwell 96 GB, TP=1, no co-hosting, cross-check judge dropped entirely. Added `gemma_4_31b_thinking_on` and `gemma_4_31b_thinking_off` subjects to `configs/subjects.yaml` (same hf_id as `gemma_4_31b`, distinct `chat_template_kwargs`). Refactored `src/experiments/plan_b.py::main` to take `--subject {gemma_2_27b,qwen_3_32b,gemma_4_31b_thinking_on,gemma_4_31b_thinking_off}`; resolves to `configs/plan_b_<subject>.yaml`. Authored 3 per-subject configs; output dir convention `results/phase_a/{subject_id}/`. Verified config + venv guards via dry-run with all steps skipped.

## Gate 2 → Phase A Handoff — 2026-04-30

**Artifacts produced:**

*Plan amendments:*
- `plans/may_3_directive.md` — 2026-04-30 amendment section appended; Hardware amendment paragraph at top; Cuts table tightened on cross-judge; Subject set adds explicit "no co-hosting" rule; Pre-execution gate 2 restated; Sequencing caveats add TP=1 swap budget + judge-phase standalone rule.

*Subject definitions:*
- `configs/subjects.yaml` — added `gemma_4_31b_thinking_on` (chat_template_kwargs.enable_thinking=true) and `gemma_4_31b_thinking_off` (false) entries. Both share `hf_id: google/gemma-4-31B-it`, `steered_backend: hf`, `layer_module_glob_prefix: model.language_model.layers`. `model_family_for()` resolves both to the `gemma_4` family via prefix-match → `model_hooks.yaml` lookups need no changes.
- TP auto-clamp via `src/utils/config.load_subjects` resolves all subjects to `tensor_parallel_size: 1` at runtime on the current 1-GPU host (configured value in YAML stays at 4).

*Per-subject experiment configs:*
- `configs/plan_b_qwen_3_32b.yaml` — model_id=qwen_3_32b, capping layers verbatim from paper [46,53] τ=p25, output_dir=results/phase_a/qwen_3_32b/, GPT-5.5 step disabled.
- `configs/plan_b_gemma_4_31b_thinking_on.yaml` — model_id=gemma_4_31b_thinking_on, capping [27,34] (center 50% of 60 layers, width 8, τ=p25), `extract_thinking_answer_split: true`, max_new_tokens bumped to 1024 for thinking traces.
- `configs/plan_b_gemma_4_31b_thinking_off.yaml` — model_id=gemma_4_31b_thinking_off, same capping pick as thinking-on, max_new_tokens=256.

*Orchestrator:*
- `src/experiments/plan_b.py::main` — `--subject` flag added; `--config` retained (mutually exclusive). Subject choices: `{gemma_2_27b, qwen_3_32b, gemma_4_31b_thinking_on, gemma_4_31b_thinking_off}`. Logs the resolved config path + model_id at startup. Venv guard fires before any GPU work.

**Decisions locked this gate (mirrored to `plans/decisions.md`):**
- Hardware deviation: 4× RTX 5090 / 128 GB → 1× RTX PRO 6000 Blackwell 96 GB. TP=1 across all subjects + judge.
- Co-hosting policy: never (CLAUDE.md "Inference & Serving" exception clause suspended for this window).
- Cross-check judge: dropped entirely (was a Cuts-table item with add-back path; now removed). Headline harm rests on Qwen 3.6-27B primary judge alone, validated to 93% vs GPT-5.5 on Plan B.
- GPT-5.5 step disabled in all 3 new per-subject configs (cross-judge cut + prior-spend exhausted).
- Gemma 4 31B thinking modes split into 2 subjects.yaml entries (was single `gemma_4_31b` entry with per-call ctk override). `paper_capping_ranges.yaml` already used the split keys → naming is consistent.
- Capping range for Gemma 4 31B (both thinking modes): single Phase A pick at center 50% of 60 layers, width 8, τ=p25 (layers [27,34]). Full 2D sweep per Stage 4 T4.0 deferred to Phase B.

**Gotchas / surprises:**
- The TP=4 measured-VRAM totals in `configs/inference_runtime.yaml` (e.g., Qwen 3 32B short = 116 GB total at gmu=0.75) are aggregates across 4 GPUs. They do NOT translate to single-GPU budgets; per-GPU budget on the 96 GB card is `~weights + (gmu × 96 GB − weights)` for KV. At gmu=0.90, Qwen 3 32B has ~17 GB for KV; Gemma 4 31B has ~14 GB. The Phase A agent must drop `max_model_len` accordingly (try 4096 first; fallback 2048).
- `assert_venv_for_subject(qwen_3_32b)` requires `.venv-sglang/bin/python` (sglang backend per spike). The 3 Gemma 4 modes use HF backend → either venv works.
- `eval_sizes.yaml` has no entries for `gemma_4_31b_thinking_on`/`_off` — Phase E (capability eval) will need them; copy the `gemma_4_31b` rows when Phase E starts. Phase A doesn't touch eval_sizes so this is non-blocking.
- Step 8 (GPT-5.5) gating: per-subject configs set `gpt55_validation.enabled: false` but `step_8_gpt55_validation` doesn't read that flag yet — pass `--skip 8` at runtime as belt + braces. Optional follow-up: wire `step_8` to honor `cfg["gpt55_validation"]["enabled"]`.
- The dry-run smoke (skip all steps) tries to read `extraction/L_star.txt` after step 2 is skipped. Real Phase A runs won't `--skip 2`; the apparent error is a resume-path artifact, not a bug.

**Open items for Phase A (the next agent):**
- Run `uv run python scripts/smoke_load.py --family qwen_3_32b` and equivalents for `gemma_4_31b_thinking_on`/`_off` to record actual peak VRAM at TP=1. Update `configs/inference_runtime.yaml` per-subject profiles with the TP=1 numbers (current entries are TP=4 measurements). If any subject OOMs at gmu=0.90, drop to gmu=0.85 and/or cut `max_model_len` to 2048.
- Per-subject Phase A pipeline: `uv run python -m src.experiments.plan_b --subject <subject_id>` (run from `.venv-sglang` for qwen_3_32b; from `.venv` for the Gemma 4 modes).
- Wall-clock budget per subject at TP=1 likely 3-4× the Gemma 2 27B Plan B figure (~8 hr → ~24-32 hr). Phase A across 3 new subjects is therefore ~3-4 days of compute. If too long, cut `n_role_rollouts_per_role` from 30 → 15 first.
- Phase A acceptance: per-subject `cos_sim(PC1, AA) > 0.7` at L*. If <0.7, log to `decisions.md` and drop PC1-secondary analysis for that subject (per CLAUDE.md "Primary Intervention Direction").
- Phase A→B handoff: each subject's `extraction/aa.safetensors` + `extraction/pcs.safetensors` + `extraction/L_star.txt` + `extraction/lmsys_norms_L<L*>.json` are the Phase B inputs.

**Pointers into CONVENTIONS.md updated:**
- (none this gate — Gate 2 changed orchestration but no schemas / cache layouts / scientific conventions)

- [2026-04-30 17:50] Phase A multi-subject kickoff. Subject 1/3 (Qwen 3 32B) launched: `phase_a_qwen_3_32b` (PID 80682, log `logs/phase_a_qwen_3_32b_20260430_174955.log`). TP=1 smoke load passed at gmu=0.85 (61 GiB weights + KV; 12 tok/s on 5-prompt batch — production batched throughput will be higher). Run skips steps 6 (steered → Phase B), 8 (GPT-5.5 dropped per amendment), 10 (figures → Phase F). Patched `src/experiments/plan_b.py` to be subject-aware: (a) `step_1b` resolves `n_layers` per family from `model_hooks.yaml` (was hardcoded 46 = Gemma 2); (b) `step_2` adds `_load_or_bootstrap_aa_and_roles` — paper-artifact path for Tier 1 (gemma_2_27b, qwen_3_32b), bootstrap-from-cached-step-1b path for Tier 2 (gemma_4_31b_*); (c) `step_3` honors `capping_layers_explicit` for paper-verbatim ranges; (d) `step_10` reads per-subject d_model + n_role_vectors from a new `extraction/pca_meta.json` sidecar; (e) venv guard lazy-skipped when `--skip 6` (Phase A doesn't touch SGLang). Bootstrap path validated against Gemma 2 27B paper artifacts: role vectors cos_sim 0.99, AA cos_sim 0.86 at L=22 — within reproduction range (gap explained by paper's 1200 rollouts/role + fully-vs-somewhat split vs our 30/role unsplit). Bug found + fixed during validation: bootstrap had filtered on `condition_id`, but `run_subject_rollouts` overwrites `condition_id` to "role_rollout" for all rows including default-Assistant; switched to `prompt_id` prefix split. Fix only affected the Tier 2 bootstrap path; Qwen 3 32B run is unaffected.

- [2026-05-01 13:16] **PHASE A COMPLETE — all 3 multi-subject runs finished.** Sentinel `results/phase_a/.chain_complete` touched. Cross-subject summary:
  | subject | L* (of n_layers) | cos_sim(PC1,AA) | baseline harm % | refusal % | AUC(AA) | AUC(AA+PCs) | blind-spot lift | 95% CI | LASSO PCs |
  |---|---|---|---|---|---|---|---|---|---|
  | qwen_3_32b | 11/64 | 0.903 | 8.6 | 71.2 | 0.628 | 0.930 | **0.312** | [0.214, 0.415] | 6 |
  | gemma_4_31b_thinking_off | 14/60 | 0.818 | 10.2 | 77.2 | 0.576 | 0.866 | **0.295** | [0.191, 0.406] | 1 |
  | gemma_4_31b_thinking_on | 59/60 | 0.822 | 15.0 | 77.6 | 0.813 | 0.905 | **0.101** | [0.051, 0.155] | 9 |

  Wall-clock: qwen 1h 34min; g4_off 1h 13min (resume after layer-discovery patch); g4_on 3h 21min (max_new=2048). All ran on single RTX PRO 6000 96 GB at TP=1, sequentially.

  **Three findings worth report-level treatment:**
  1. **L\* shifts dramatically with thinking mode on identical weights.** Gemma 4 31B thinking-off → L\*=14 (~23% depth); thinking-on → L\*=59 (~98% depth). Same model, same role-conditioning prompts, just `enable_thinking` toggle.
  2. **Thinking-on concentrates safety-relevant features along AA.** AUC(AA-only) rises from 0.576 (off) → 0.813 (on). Higher-PC blind spot shrinks correspondingly: lift drops from 0.295 to 0.101.
  3. **L\* in our runs systematically shallower than paper's "middle" for non-thinking subjects** (Qwen L\*=11/64 ~17%, g4_off L\*=14/60 ~23% vs paper's "middle ± few"). Worth a per_layer_cos_sim curve inspection — could be a surface artifact (flat plateau, argmax noise) or a real reproduction divergence.

  Operational gotchas surfaced + fixed during the runs:
  - `inference_runtime.yaml` was missing per-thinking-mode entries for gemma_4_31b — added.
  - `external/assistant-axis/internals/model.py::ProbingModel.get_layers()` had no path for Gemma4ForConditionalGeneration's nested `m.model.language_model.layers` — patched.
  - Bootstrap path in `_load_or_bootstrap_aa_and_roles` filtered on `condition_id` which is overwritten to "role_rollout" by run_subject_rollouts; switched to prompt_id prefix split.
  - `max_new_tokens=256` (inherited from Gemma 2 27B Plan B) caused 80–93% role-rollout truncation on Qwen + g4_off — methodology robust enough that the blind-spot signal still emerged, but for thinking-on we bumped to 2048 (validated 0% truncation: p99=847 role, p99=944 baseline).
  - lmsys-chat-1m gated dataset → falls back to extraction questions for residual norm calibration; consistent across all subjects so no cross-subject confound, but lmsys norm at L\*=59 is 6.21 vs L\*=14's 296.9 — Phase B will need per-layer norms for thinking-on, not just L\*-specific.
  - Phase A pipeline "completion notification" via `tail -F | grep PLAN B COMPLETE` was unreliable (cost 12 hr of GPU idle on the qwen→g4 boundary). Fixed with file-existence sentinels (`results/phase_a/.chain_complete`) + `scripts/chain_phase_a_subjects.sh` chain supervisor + `scripts/sanity_check_phase_a.py` programmatic gates between subjects.

  **Phase A → Phase B handoff inputs (per `may_3_directive.md`):**
  - For each subject, `results/phase_a/<subject>/extraction/{aa.safetensors, pcs.safetensors, L_star.txt, lmsys_norms_L<L*>.json, per_layer_cos_sim.json, pca_meta.json, tau_calibration.json}` are ready.
  - Phase B (Stage 4 attack arm + Ext E/F) is gated on user review of the above headline metrics. Not auto-launchable — Phase B consumes Phase A AA + PCs and cross-phase boundaries require human sign-off (per discussion 2026-05-01 ~07:20 UTC).

- [2026-05-03 09:45 → 20:44] Phase D thread A complete on Gemma 4 31B thinking-OFF. Two attempted runs:
  - **v1 (09:45 → 12:22, killed):** Initial design reused L*=14 PC2/PC3 unit at every AA capping layer 27-34 (defence_arm.py convention). All 5 PC2 calibration cells produced **0% coherence** (100% nonsensical "same same same..." token loops). Cap fired with ~80-180 magnitude excess at each of 8 layers; cumulative ~640 subtraction nuked the residual stream (norm ~300). Killed at step 4; data wiped. Failure mode + root cause logged in `plans/decisions.md` 2026-05-03 12:25 entry.
  - **v2 (12:26 → 20:44, COMPLETE):** PC2/PC3 caps moved to L*=14 single-layer only (the PCA extraction layer). AA cap unchanged at 8-layer range. Coherence preserved at ≥0.9665 across all 9 test cells.

  **Phase D test split headline (n=508/cell):**

  | Defence | No attack | adv_null λ=0.25 | **PC3 attack λ=0.25** |
  |---|---|---|---|
  | AA-cap only         | 11.22% | 18.70% | **23.62%** |
  | AA + PC2-cap (p10)  | 10.83% | 17.52% | **24.61%** |
  | AA + PC2 + PC3-cap (p25) | 10.63% | 17.52% | **22.05%** |

  **Multi-axis defence is partial, not transformative:**
  - PC3 attack: 23.62% (AA-only) → 22.05% (AA+PC2+PC3) — closes 1.57 pp of the 13.4 pp recovery (≈12%).
  - PC2-cap alone has zero/negative effect on PC3 attack (24.61% > 23.62%); PC3-cap is what carries the small reduction.
  - adv_null attack: 18.70% → 17.52% (−1.18 pp). Modest gain.
  - No attack: 11.22% → 10.63% (−0.59 pp). Baseline-of-baseline; AA-cap on g4_off is still net-negative vs unsteered 10.2%.

  **Mechanistic explanation (logged in `plans/decisions.md` 2026-05-03):** PC3 cap at L*=14 fires *before* the PC3 attack addition at the same layer (steerer.py registration order: cap → steer). The cap clamps pre-attack activations; the attack injects harm-positive perturbation immediately afterward; subsequent layers carry the full attack until AA cap at 27-34 catches some of it (same as Phase B's AA-only). Multi-axis cap as currently constructed cannot close the gap because the cap and attack occupy the same layer and the cap fires first. To fully close, the PC3 cap would need to fire *after* the attack at the same layer, OR at deeper layers where the perturbation has propagated — but the 8-layer-reuse alternative breaks coherence.

  **Wall-clock breakdown (v2):**
  - Step 1 (CPU setup): <1 min.
  - Step 2 (5 PC2 cells × 200 prompts): 90 min.
  - Step 3 (judge val_pc2, 1000 rows): 18 min.
  - Step 4 (τ_PC2 pick, p10): instant.
  - Step 5 (5 PC3 cells × 200 prompts, PC2 fixed at p10): 113 min.
  - Step 6 (judge val_pc3, 1000 rows): 18 min.
  - Step 7 (τ_PC3 pick, p25): instant.
  - Step 8 (6 test cells × 508 prompts, AA-only rows reused from Phase B): 239 min.
  - Step 9 (judge test, 3048 rows): 38 min.
  - Step 10 (assemble): instant.
  - Total v2 wall-clock: ~8 hr 18 min.

  **Phase D → Phase E/F handoff inputs:**
  - `results/phase_d/gemma_4_31b_thinking_off/headline.json` — 9-cell test matrix
  - `results/phase_d/gemma_4_31b_thinking_off/test_split.parquet` — full per-prompt judged rollouts (n=4572 = 6 new + 3 reused-from-Phase-B × 508)
  - `results/phase_d/gemma_4_31b_thinking_off/multi_axis_calibration.json` — τ pick tables + chosen percentiles
  - `results/phase_d/gemma_4_31b_thinking_off/test_split_summary.csv` — flat CSV for figure work
  - `results/phase_d/gemma_4_31b_thinking_off/extraction/cap_vectors/` — 2 single-layer L*=14 cap vector files for PC2 + PC3 (+8 reused-from-Phase-B AA cap vectors).
  - `results/phase_d/gemma_4_31b_thinking_off/extraction/tau_calibration_pc.json` — PC2/PC3 τ percentile tables at L*=14.

## Phase B → Phase D Handoff — 2026-05-03

**Artifacts produced (this Phase D run):**

*Configs + orchestration (committed):*
- `configs/phase_d_gemma_4_31b_thinking_off.yaml` — Phase D config: validation 200 prompts, test 508 (reused from Phase B), τ candidates {p1, p10, p25, p50, p75}, calibration attack signmatched_pc3 λ=0.25, coherence floor 0.90.
- `src/experiments/phase_d.py` — 10-step orchestrator (setup → val_pc2 → judge → pick_pc2 → val_pc3 → judge → pick_pc3 → test → judge → assemble); CLI: `uv run python -m src.experiments.phase_d --config configs/phase_d_gemma_4_31b_thinking_off.yaml`.

*Result artifacts (gitignored):*
- `results/phase_d/gemma_4_31b_thinking_off/{headline.json, multi_axis_calibration.json, test_split.parquet, test_split_summary.csv}` — final outputs.
- `results/phase_d/gemma_4_31b_thinking_off/rollouts/{val_pc2, val_pc2_judged.parquet, val_pc3, val_pc3_judged.parquet, test, test_judged.parquet, _val_subset.parquet, _test_subset.parquet}` — intermediate rollouts.
- `results/phase_d/gemma_4_31b_thinking_off/extraction/{cap_vectors, tau_calibration_pc.json}` — single-layer L*=14 PC2/PC3 cap inputs + τ tables.

**Decisions locked this stage (mirrored to `plans/decisions.md`):**
- 2026-05-03 09:45 — Cap-vector sign convention for PC2/PC3: input = -signmatched_pc (harm-negative direction), τ = -p<X> of role projection on input direction. Mirrors Phase B AA convention exactly.
- 2026-05-03 09:45 — Test split reuses Phase B's 508-prompt subset (not a fresh 550) for apples-to-apples comparison with Phase B headlines.
- 2026-05-03 12:25 — PC2/PC3 cap layer scope = L*=14 single-layer only. AA cap stays at 8-layer range [27, 34]. The 8-layer PC reuse alternative was tried in v1 and produced 0% coherence at all percentiles.

**Headline test-split numbers:**

| Defence | No attack | adv_null λ=0.25 | PC3 attack λ=0.25 |
|---|---|---|---|
| AA-cap only         | 11.22% | 18.70% | **23.62%** (Phase B reproduction) |
| AA + PC2-cap (p10)  | 10.83% | 17.52% | 24.61% |
| AA + PC2 + PC3-cap (p25) | 10.63% | 17.52% | **22.05%** (best multi-axis) |

τ_PC2 = 10th percentile of role projection at L*=14, τ_PC3 = 25th percentile (both at 100% coherence on validation).

**Gotchas / surprises:**
- **L*=14 PC unit at layer 27-34 has projection magnitudes of ~135 (vs per-layer AA at ~100 with std ~5).** Capping with τ derived from this distribution leads to ~80-180 excess per layer, which compounds across 8 layers into catastrophic clamping. Single-layer cap at the extraction layer is the workaround.
- **PC3 cap doesn't close the PC3-attack gap because of cap-before-attack hook ordering at the same layer.** Steerer registers cap hooks first (ActivationSteering with intervention=capping), then steer hooks (ActivationSteering with intervention=addition). PyTorch fires hooks in registration order at each layer. So at L*=14: cap fires → attack adds → activation propagates with full attack perturbation through L*+1 ... L26 (no further cap) → AA cap at L27-34 catches some of it. To close fully, PC3 cap would need to fire at L*=15+ (after attack injection). But that requires either (a) per-layer PCA refit at later layers, or (b) propagating the L*=14 unit forward — which broke coherence in v1.
- **AA cap on g4_off is a net-negative defence by itself** (Phase B already showed: 10.2% baseline → 11.22% AA-capped). Multi-axis cap brings it back to roughly the unsteered baseline (10.63%), but the slight harm increase from capping persists — the cap's "subtract excess in role-positive direction" hurts non-role activations in some cases.
- **Adv_null attack:** Multi-axis cap reduces 18.70% → 17.52% — small but consistent gain. Adv_null is constructed to be AA-orthogonal (LASSO PCs with AA projected out), so PC2+PC3 cap CAN catch its harm-aligned components. The fact that the gain is small suggests adv_null also injects mostly at L*=14, hitting the same cap-before-attack mechanism.

**Open items:**
- The "cap fires after attack" alternative — register attack hook, then cap hook, at the same layer. Would test whether cap-after-attack gives a better defence. Code change ~10 lines in `steerer.py::cap_and_steer` (swap context manager order). Worth a 2-cell follow-up: AA+PC2+PC3-cap × PC3-attack with reversed hook order, n=508 prompts, ~45 min.
- Per-layer PC PCA refit (extensions Ext I follow-on, but for capping rather than extraction). Would refit PCA at L=15..34 and use the per-layer top-k components as cap directions. ~30 min CPU + a fresh Phase D-style sweep (~6 hr GPU).
- Phase E (capability eval) — next thread per directive. 4 subjects × 2-3 conditions × 3 benches = 24-36 cells. 20-25 hr per directive.
- Ext B causal v_harm test on g4_off + g4_on (5-7 hr).
- Ext D bypass interpretation (CPU only, 2-3 hr).
- Phase F report figures + writeup.

**Pointers into CONVENTIONS.md updated:**
- (none — Phase D used existing conventions; no schema or layer-scope rules added.)

## Phase D → Phase E Handoff — 2026-05-04

**Scope (per `plans/may_3_directive.md` 2026-05-03 thread C, trimmed):**
4 subjects × {unsteered, AA-cap} × {IFEval, GSM8k, EQ-Bench} + multi-axis-cap × 3 benches on `gemma_4_31b_thinking_off` only. **9 rollout cells total.** MMLU-Pro dropped per directive (revivable on demand).

**Artifacts produced (this Phase E setup):**

*Configs + orchestration (committed):*
- `configs/phase_e.yaml` — 4 subjects, 3 benches, 9 cells; vllm profile=long; max_input_len=1024, max_new_tokens=1024.
- `src/experiments/phase_e.py` — 4-step orchestrator (setup → rollout → score → assemble); CLI: `uv run python -m src.experiments.phase_e --config configs/phase_e.yaml`.
- `src/evaluation/capability_score.py` — three scorers: IFEval via josejg `instruction_following_eval`, GSM8k strict numeric extract+match, EQ-Bench v2 fullscale parsed-MAE rubric.

*Result artifacts (gitignored):*
- `results/phase_e/{prompts/combined.parquet, vectors/<subject>/{aa,multi_axis}/, rollouts/<cell>.parquet, metrics/<cell>.json, headline.json}`.
- Combined prompts parquet n=1712 (541 IFEval + 1000 GSM8k + 171 EQ-Bench), tagged by `dataset` column for post-hoc per-bench scoring.

**Decisions locked this stage (mirrored to `plans/decisions.md`):**
- 2026-05-04 04:15 — Combined-bench rollout per (subject, condition) cell. One model load per cell instead of per (cell, bench) saves ~9× model-load wall-clock; max_new_tokens=1024 over-allocates GSM8k/EQ-Bench but doesn't change actual wall-clock (gen stops at EOS).
- 2026-05-04 04:18 — IFEval kwargs/instruction_id_list serialized to JSON strings in the prompts parquet to defeat pandas/pyarrow column-schema unioning (which inflated each kwargs entry with None placeholders for keys belonging to other rows' instruction types and broke the `instruction_following_eval` scorer's per-row dict expectations).
- 2026-05-04 04:20 — Cap vector files normalized to safetensors `vector` key under `results/phase_e/vectors/<subject>/...` because Plan B Gemma 2 27B's AA caps were saved with key `v` (HF backend's `_load_vec` accepts both, but SGLang's `capping_factory` defaults to `vector` and would otherwise fail on G2 cells).
- 2026-05-04 04:21 — Per-backend venv routing in orchestrator: `vllm`/`hf` cells → `.venv/bin/python` (has vllm + accelerate + transformer_lens), `sglang` cells → `.venv-sglang/bin/python` (has sglang). Parent stays venv-agnostic so a single Phase E run covers mixed-backend subjects.

**Phase E → Phase F handoff inputs (expected once sweep completes):**
- `results/phase_e/headline.json` — matrix[subject][condition][bench] of {n, score, extra} + deltas vs unsteered + per-(subject, condition) aggregates.
- `results/phase_e/rollouts/<cell>.parquet` — full per-prompt generations (preserved for ad-hoc re-scoring or qualitative inspection).

**Open items:**
- Phase E sweep currently running (detached via `nohup setsid`); 9 rollout cells per directive estimate of 20-25 hr.
- Ext B causal v_harm test on g4_off + g4_on (5-7 hr).
- Ext D bypass interpretation (CPU only, 2-3 hr).
- Phase F report figures + writeup.

**Pointers into CONVENTIONS.md updated:**
- (none — Phase E used existing conventions; capability scorer is a new file but doesn't add a global schema rule.)

## Phase E → Ext B (causal v_harm) Kickoff — 2026-05-05

**Scope (per `plans/may_3_directive.md` 2026-05-03 thread B):**
2 subjects × 3 λ ∈ {0.10, 0.25, 0.50} × 508 baseline DAN prompts (Phase B's `_full_subset.parquet`). Steering = `addition` mode at L*, **no cap, no DAN-style attack**. Subjects: g4_off (cos(v_harm,AA)=0.048, L*=14, lmsys_norm=296.91) and g4_on (cos(v_harm,AA)=0.562, L*=59, lmsys_norm=6.21). Qwen 3.6 27B judges; BCa 95% CI (10K resamples) on per-λ harm rate and on (harm_steered − Phase A baseline).

**Artifacts written this stage:**

*Configs + orchestration (committed):*
- `configs/ext_b_v_harm_causal_gemma_4_31b_thinking_off.yaml`, `configs/ext_b_v_harm_causal_gemma_4_31b_thinking_on.yaml` — per-subject configs; `causal_lift_threshold_pp=5.0`, `bootstrap_n=10000`.
- `src/experiments/ext_b_v_harm_causal.py` — 4-step orchestrator (setup → rollout → judge → assemble); CLI: `uv run python -m src.experiments.ext_b_v_harm_causal --subject {gemma_4_31b_thinking_off, gemma_4_31b_thinking_on}`.

*Result artifacts (gitignored, expected layout):*
- `results/phase_b/<subject>/extensions/v_harm_causal/{headline.json, harm_curve.parquet, rollouts/v_harm_clean_*.parquet, rollouts/v_harm_clean_judged.parquet}`.

**Decisions locked at kickoff:**
- 2026-05-05 03:38 — Re-use Phase B's already-saved `v_harm.safetensors` (norm = lmsys_norm at L*, bf16). The addition-mode coefficient = λ matches the Phase B convention exactly, so the 3 λ values translate directly to "add λ × lmsys_norm × v_harm_unit at L*".
- 2026-05-05 03:38 — Re-use Phase B's 508-prompt `_full_subset.parquet` for apples-to-apples comparison vs the AA-capped + v_harm runs already in Phase B's `lambda_sweep.parquet`. The clean-vs-AA-capped contrast is informative even though Phase A baseline is the headline reference.
- 2026-05-05 03:38 — Use Phase A `metrics.json::headline.baseline_harm_rate` as the reference for the lift CI (g4_off: 10.2%; g4_on: 15.0%). Per-prompt baseline harm_binary is not available at this layer, so the diff CI uses a length-matched binomial-equivalent pseudo-vector — slightly understates baseline variance but the steered samples are the dominant uncertainty.
- 2026-05-05 03:38 — Skip g4_on multi-axis composite + adv_null analogues; the directive explicitly excludes them. The clean v_harm sweep is the only Ext B experiment.

**Wall-clock estimate at launch (sequential, single PID via `nohup setsid`):**
- g4_off: 3 conditions × ~2780 s (Phase B v_harm full timing reference, max_new_tokens=256) ≈ 2.3 hr + judge ~25 min ≈ 2.7 hr.
- g4_on: 3 conditions × ~9300 s (Phase B v_harm full, max_new_tokens=1024) ≈ 7.8 hr + judge ~25 min ≈ 8.2 hr.
- Total ≈ 10.9 hr (over the directive's 5-7 hr estimate; root cause = HF Gemma 4 thinking-ON throughput is ~3.4× thinking-OFF, larger than the directive's 2× assumption).

**Run state at handoff:** detached via `nohup setsid`; log `logs/ext_b_v_harm_causal_20260505_033846.log`. Step 1 reuses cached `.step1.done` markers (CPU-only verification); step 2 step is loading the bf16 weights for the first λ as of 03:38 UTC.

**Open items:**
- Ext B sweep running; await `headline.json` for both subjects before Ext D / Phase F.
- Ext D bypass interpretation (CPU only, 2-3 hr) — can run in parallel since GPU isn't needed.
- Phase F report figures + writeup.

**Pointers into CONVENTIONS.md updated:**
- (none — Ext B reuses Phase B vectors + subset; no schema additions.)

