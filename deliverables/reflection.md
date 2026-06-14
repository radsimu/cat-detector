# Reflection Document

## What Worked Well

**End-to-end story workflow** was the biggest productivity win. The `implement-story` skill orchestrated a full cycle — fetch from ADO, plan, get approval, implement, run quality gates, create PR, update story status — in a single conversation turn. What would typically involve switching between IDE, browser, terminal, and ADO became a continuous flow.

**Multi-agent code review** caught real issues that I would have shipped. The QA specialist agent identified that `cat_count` was semantically lying to API consumers by counting dogs, that the report-missed endpoint was opening a duplicate camera handle (which fails on Mac), and that concurrent `cap.read()` calls were a race condition. These weren't theoretical concerns — they were bugs that would have broken in production.

**CLAUDE.md as persistent context** eliminated the "explain the project again" problem. Every new conversation or agent started with full awareness of the tech stack, file locations, and design decisions. The ADRs reinforced this by documenting *why* choices were made, not just what they were.

**Backlog review with AI** surfaced issues a human PM would catch but a developer might miss. The review identified US-6 (fine-tuning) as XL and recommended splitting it into three stories, found unclear acceptance criteria in US-4 and US-5, and proposed a dependency-based priority ordering that made the implementation sequence obvious.

**Real-time testing with the webcam** provided immediate feedback. Running the detection on a live webcam stream, seeing YOLOv8 draw bounding boxes in real-time, and discovering that `yolov8n` misses an occluded garden cat while `yolov8m` catches it at 85.7% confidence — that kind of empirical validation would have taken much longer without the AI handling the integration plumbing.

## Challenges & Solutions

**Python 3.14 compatibility** was the most persistent friction. The `ultralytics` library depends on `torchvision`, which depends on `_lzma` — a C extension missing from pyenv's Python 3.14 build. Tests had to stub `ultralytics` before import, and the app itself required Python 3.11 via `pyenv`. The lesson: always test against the Python version you'll deploy on, not just the latest available.

**In-memory SQLite in tests** initially failed because each connection gets a separate database. The fix (`StaticPool`) was a one-liner but took multiple iterations to diagnose. The pattern (pool sharing for test isolation) is now embedded in the test infrastructure.

**Starlette 1.0 API change** broke the index page silently — `TemplateResponse` changed its signature, but the error manifested as an unhelpful Jinja2 cache `TypeError`. The fix was trivial (`request` moves from context dict to first positional arg), but finding it required reading the Starlette changelog rather than trusting the error message.

**Branch management with ADO policies** created friction when PRs couldn't be merged without an external approver. The workaround — creating a unified feature branch that merged all PRs locally — worked but added complexity. For training projects, lighter branch policies would reduce this overhead.

**Camera lifecycle on Mac** was a real-world issue that no amount of unit testing would have caught. The original open/read/close pattern caused the camera indicator to flash every 5 seconds and auto-exposure to reset, producing dark frames. Moving to a persistent capture with a threading lock for concurrent access was the right pattern but required understanding Mac-specific hardware behavior.

## Best Practices Discovered

**Plan before implementing** — The two-phase approach (plan in conversation, implement after approval) consistently produced better results than jumping straight to code. When I let Claude plan freely and reviewed the plan, I caught scope issues and design mismatches *before* any code was written.

**Let the AI do the boring plumbing, review the architecture** — Claude excels at wiring up FastAPI endpoints, writing test fixtures, generating Terraform configs, and handling ADO API calls. It's less reliable on architecture decisions (like whether to use a JSON column vs. a related table). My job was reviewing *what* to build, not *how* to wire it up.

**CLAUDE.md should document decisions, not code** — The most useful CLAUDE.md content was the "Key design decisions" section and the file location table. Documenting that "YOLOv8 model size is configurable" saved more time than documenting the Python class hierarchy.

**Test with real hardware early** — The webcam integration test revealed three bugs (camera flashing, concurrent reads, dark frames) that unit tests couldn't have found. Running the app against actual hardware should happen in the first sprint, not the last.

**Address PR feedback immediately** — The code review feedback (dog counting, camera duplication, threading race) was addressed in individual commits that could be traced back to specific concerns. This made the reviewer's job easier and kept the PR history clean.

## Client Application Ideas

**Internal tool prototyping** — The workflow demonstrated here (stories in ADO, AI-assisted implementation, quality gates, PR creation) maps directly to Zartis client engagements. A consultant could scaffold an internal tool (dashboard, monitoring system, automation) in hours rather than days, with tests and documentation included from the start.

**Legacy codebase onboarding** — CLAUDE.md + ADRs as a pattern for documenting existing client systems would dramatically reduce onboarding time for new team members. The AI reads the docs, understands the context, and can answer questions about architecture decisions that would otherwise require asking the original developer.

**Quality gate automation** — The multi-agent review pattern (code-reviewer + qa-specialist + ux-reviewer) could be packaged as a standard quality gate for client PRs. Each agent catches different classes of issues: the code reviewer finds semantic problems, the QA specialist finds edge cases, the UX reviewer finds usability issues.

**Detection/ML pipeline pattern** — The cat-detector architecture (camera capture, model inference, webhook notification, web UI, labeling, fine-tuning stub) is a reusable template for any real-time detection system. Swap YOLO for another model and the pipeline still works. Clients with manufacturing defect detection, security monitoring, or agricultural monitoring needs could start from this scaffold.
