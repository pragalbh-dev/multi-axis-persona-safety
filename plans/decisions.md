# Decisions Ledger

Append-only log of **unplanned decisions** — any choice made during stage execution that was NOT in the pre-written stage plan.

**What belongs here:**
- Picking a specific integer / hyperparameter when the plan said "something around X" (e.g., "extraction layer L* landed at 30, not the paper's reported 32 — difference of 2, log why").
- Resolving an ambiguity in the paper's convention (e.g., paper says "25th percentile" — of which distribution? of which activations? if we chose differently from the plan's default).
- Picking between two viable library versions or implementations.
- Small scope cuts taken under time pressure ("skipped the cross-check judge on 50 prompts because judge server was GPU-starved").
- Anything you'd want a future agent (or the user) to be able to audit, question, and potentially reverse.

**What does NOT belong here:**
- Progress updates (those go in `progress.md`).
- Task completion checkboxes (those go in the stage plan).
- Decisions already specified in the stage plan or `CONVENTIONS.md` (those were pre-planned — no need to log).
- Transient state during implementation (only the final decision goes here).

**Rule of thumb:** if you caught yourself saying "the plan didn't cover this, so I'm going with X" — that's a decision. Log it.

---

## Template

Copy this block, fill in, append at the **end** of this file. Do not edit earlier entries.

```markdown
## [YYYY-MM-DD HH:MM] Stage N / T{k.m} — {short title}

**Decision:** {what was decided in 1-2 sentences}

**Alternatives considered:**
- Option A — {why rejected}
- Option B — {why rejected}

**Reason:** {why this option won. Cite the constraint that forced the choice.}

**Source:** {pick one or more}
- Paper line / appendix reference (e.g., "paper line 691" or "Appendix D.2.2")
- File path (e.g., "configs/paper_capping_ranges.yaml")
- External URL
- User instruction ("user said in chat on 2026-04-24 to prefer X over Y")
- Own judgment ("no source; judgment call because …")

**Reversibility:** {high / medium / low}
- **High** — decision can be flipped by changing one config value and re-running one task.
- **Medium** — flipping requires re-running a stage's worth of work (hours of GPU).
- **Low** — flipping invalidates downstream experiments or requires restarting multiple stages.

**How to revert:** {concrete steps. Which file(s) to edit? Which artifacts need regeneration? Any handoff blocks to update in `progress.md`?}

**Downstream dependencies:** {what later stages / tasks are built on top of this decision?}
```

---

## Log

(append entries below this line)
