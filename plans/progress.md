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
