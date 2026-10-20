# AI usage log

Record of work done by AI agents in this project, kept for journal AI-use disclosure.
One row per task. "Human" is what a person decided, wrote, or checked.

| Date | Tool / version | Model | Task | Output | Human |
|---|---|---|---|---|---|
| 2026-10-19 | Kiro Crew (kiro-cli) | claude-opus-5.5 | Review and rewrite of the training plan | docs/plan.md | Janghoon Lee set goals, constraints, and approved the plan |
| 2026-10-19 | Kiro Crew (kiro-cli) | claude-opus-5.5 | Repository layout, .gitignore, public-repo check script | .gitignore, scripts/check_public.py | - |
| 2026-10-19 | Kiro Crew (kiro-cli) | claude-opus-5.5 | T1 research: Z-Image model facts, dataset sources and AI Hub / NIKL terms from HF API, configs, arXiv and the AI Hub policy and FAQ pages | docs/model-facts.md, docs/datasets.md | - |
| 2026-10-19 | Kiro Crew (kiro-cli) | claude-opus-5.5 | T3: eval prompts, OCR engine comparison, Turbo and Base baseline runs on the GPU box | eval/, scripts/eval_baseline.py, scripts/compare_ocr.py, paper/runs/, paper/ocr/ | Janghoon Lee chose the box and approved running it |
| 2026-10-19 | Kiro Crew (kiro-cli) | claude-opus-5.5 | T6: synthetic Korean text generator, font and background sourcing, 10k render | synth/, paper/data-stats/synth-10k.* | - |
| 2026-10-19 | Kiro Crew (kiro-cli) | claude-opus-5.5 | AI Hub 105 pilot download and extraction scripts | infra/aihub_*.{sh,py} | Janghoon Lee applied for AI Hub access, issued the key and ran the Korean exit node |
| 2026-10-20 | Kiro Crew (kiro-cli) | claude-opus-5.5 | T4: AI Hub 105 parser (pairing, filters, crop search, shards) and stats report | scripts/aihub105_*.py, paper/data-stats/aihub105-pilot.* | - |
