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
