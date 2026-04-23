# Stage 8: Presentation & Dissemination

**Objective:** Finalize the report, build the interactive demo, write the blog post, deploy everything, and prepare fellowship application materials.

**Prerequisites:** Stages 3-6 complete (all core results in hand). Stage 7 results included if available.

**Completion criteria:** Paper finalized, blog post published, interactive dashboard + playground deployed on HF Spaces, code on GitHub, fellowship materials ready.

---

## Required inputs

- `progress.md` — **every Handoff from Stage 3 onward**: artifact paths, headline numbers, figures per stage.
- `CONVENTIONS.md` — file layout, figure-naming conventions.
- Precomputed tuples for Viz 6 from Stages 3-6 (must have been saved per T2.6 rule).
- Report skeleton in `report/paper.md` drafted progressively during prior stages.

**Last task of this stage: append Final Handoff block to `progress.md` with deployment URLs (HF Spaces, GitHub, blog post, arXiv ID if applicable) so future-you can find everything.**

---

## Tasks

- [ ] T8.1: Build Viz 6 (Persona Steering Playground)
  - Aggregate all precomputed output tuples from Stages 3-6 into `dashboard/data/`
  - Implement the Dash app: model selector, steering mode dropdown, strength slider, defense toggle, prompt selector
  - Display: response text, mini PCA plot, safety score, side-by-side diff with baseline
  - Test locally with full dataset
  - Estimate: ~7,500 precomputed entries, ~15MB data

- [ ] T8.2: Finalize the analytical dashboard (Viz 1-5)
  - Polish all 5 visualizations built during Stages 3-6
  - Ensure consistent styling, labeling, color schemes
  - Add navigation: sidebar with tabs for each visualization
  - Add an "About" page explaining the project and linking to the paper

- [ ] T8.3: Deploy dashboard to HuggingFace Spaces
  - Package as a Gradio or Streamlit app (simpler HF Spaces deployment) or Plotly Dash
  - Upload precomputed data
  - Test: works from external browser, loads within 5 seconds
  - Get a clean URL

- [ ] T8.4: Finalize technical report / paper
  - Complete Discussion (section 9): threat model caveats, relationship to Hidden Dimensions / PERSONA / Non-Surjective, limitations, future work
  - Complete Conclusion (section 10): summary of findings, recommendations for labs
  - Polish all sections, ensure consistent notation
  - Finalize figures: numbered, captioned, high-resolution
  - Format: decide on LaTeX (for arXiv submission) or Typst (faster to write) or just clean Markdown
  - Add references list

- [ ] T8.5: Write blog post
  - Target: 5-minute read, accessible to safety-interested non-experts
  - Structure: hook (the problem) → what we found → interactive demo link → implications
  - Include 2-3 key figures from the paper (embedded, not just linked)
  - Platform: LessWrong post, personal blog, or Alignment Forum

- [ ] T8.6: Prepare fellowship application materials
  - 1-page research proposal referencing results
  - Highlight: the methodology, key finding, interactive demo, open-source release
  - Frame for OpenAI Safety Fellowship: safety evaluation, robustness
  - Frame for Astra Fellowship: interpretability, activation steering
  - Tailor each application separately

- [ ] T8.7: Open-source release
  - Clean up code: remove debugging cruft, add docstrings to public interfaces
  - Create GitHub repo README: project overview, installation, quickstart, results summary
  - Upload pre-computed persona space decompositions to HuggingFace
  - License: MIT or Apache 2.0

- [ ] T8.8: Final review and submission
  - Proofread paper
  - Test all interactive demos one more time
  - Submit to arXiv (if paper-length) or post as technical report
  - Submit fellowship applications
  - Share blog post + demo link on Twitter, LessWrong, relevant Discord/Slack channels

---

## Expected Outputs

- Deployed interactive dashboard + Persona Steering Playground on HF Spaces
- Finalized paper (PDF)
- Blog post (published)
- GitHub repo with code + README
- HuggingFace dataset with pre-computed decompositions
- Submitted fellowship applications

---

## Notes

- The blog post should be written AFTER the paper is finalized — it's a distillation, not a parallel effort.
- The interactive demo is the most memorable artifact. Prioritize getting Viz 6 (Playground) working well over polishing Viz 1-5.
- For the fellowship application: lead with the finding, not the methodology. "We discovered that current persona-based safety defenses have blind spots in X% of persona space" is stronger than "We performed PCA on 275 character archetypes."
- ArXiv submission needs a clean LaTeX or PDF. If time is tight, a well-formatted Markdown document hosted on the project website works too.
- Timeline: fellowship deadline May 3. This stage should be substantially complete by May 2.
