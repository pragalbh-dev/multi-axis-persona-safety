# Dashboard precomputed-data schema

Stage 1 T1.7. Stage 8 wires this up; until then we lock the schema so
Stages 3-6 can write rows in this shape directly from their `details.parquet`
(via `src.evaluation.types.PER_PROMPT_COLUMNS`) with a small
add-condition-id-and-shard step.

## Storage layout

One parquet per subject at `dashboard/data/{model_id}.parquet`. Sharding by
model lets the Dash app filter by model in O(1) without loading the whole
corpus. Estimated total size: ~7,500 rows × ~2 KB ≈ 15 MB for 4 subjects;
trivially HF-Spaces-hostable.

## Per-row schema

Superset of `PER_PROMPT_COLUMNS` with dashboard-specific additions marked **★**.

| Column | Type | Source | Notes |
| --- | --- | --- | --- |
| `model` | str | subject `model_id` | "gemma_2_27b" \| "qwen_3_32b" \| "gemma_4_31b_thinking_on" \| "gemma_4_31b_thinking_off" |
| `steering_mode` | str | `condition_id` decode | "none" \| "addition" \| "ablation" \| "capping" \| "mean_ablation" \| "cap+steer" |
| `lambda` | float / NaN | from row | steering coefficient; NaN for none/cap-only |
| `defense_config` | str / null | from row | "aa_cap_p25" \| "multi_axis_aa+pc2" \| ... \| null |
| `prompt_id` | str | from row | stable across datasets |
| `dataset` | str | from row | "dan" \| "shah_reconstructed" |
| `input_text` | str | from row |  |
| `response_text` | str | from row |  |
| `aa_projection` | float | from row | per-rollout AA projection at extraction layer |
| `pc_projections` | str (JSON list[float]) | from row | PC1..PCk per-rollout |
| `harm_binary` | int (0/1) | from row | binarized 9-cat label |
| `harm_label` | str | from row | full 9-cat label (display) |
| `capability_score` | float / null | from row | only set for capability-bench rows |
| `condition_id` | str | from row | hash(model, steering_mode, lambda, defense_config) |
| `git_sha` | str | from row | provenance |
| `subject_family` ★ | str | derived | `model_family_for(model)` for fast dropdown grouping |
| `is_default_assistant` ★ | bool | derived | true when `condition_id` is the no-steering, no-defense baseline |

The two **★** columns are added during the dashboard-shard build in Stage 8;
they are NOT in `PER_PROMPT_COLUMNS` because they're presentation-only.

## Build step (Stage 8)

```python
# pseudo-code; actual writer lives at dashboard/build_shards.py (Stage 8)
df = pd.concat([read_details(exp) for exp in EXPERIMENTS])
df["subject_family"] = df["model_id"].map(model_family_for)
df["is_default_assistant"] = (
    (df["steering_mode"] == "none") & df["defense_config"].isna()
)
for model, sub in df.groupby("model_id"):
    sub.to_parquet(f"dashboard/data/{model}.parquet", index=False)
```

## Query patterns the Dash app must support

1. **Filter by model**: load one parquet by `dashboard/data/{model}.parquet`.
2. **Filter by steering_mode + lambda**: pandas selection, near-instant.
3. **Find rollout for current selection**: `(model, steering_mode, lambda,
   defense_config, dataset, prompt_id)` → at most one row.
4. **PC mini-plot data**: all rows for the current `(model, dataset, defense_config)`
   bucket; project them onto PC1×PC2 using the cached PC vectors loaded
   separately from `data/cache/`. The dashboard does NOT recompute PCs.

## Cached side artifacts (also under `dashboard/data/`)

| File | Contents |
| --- | --- |
| `axes/{model}.json` | `{aa, pc1, pc2, pc3, ..., extraction_layer, capping_range, tau}` for each subject. Lets the right-pane PC plot draw cap thresholds. |
| `eval_aggregates.json` | Per-(model, condition) harm rate + capability score + BCa CIs. Used for the "headline" badges. |
