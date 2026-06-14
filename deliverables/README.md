# Deliverables Index

## Custom Skills

| Skill | Path | Description |
|-------|------|-------------|
| `implement-story` | `.claude/skills/implement-story/SKILL.md` | Multi-phase story implementation workflow with ADO integration, planning, quality gates, and PR creation |
| `auto-implement` | `.claude/skills/auto-implement/SKILL.md` | Lightweight story implementation: plan, approve, code, test, commit |
| `handle-pr-feedback` | `.claude/skills/handle-pr-feedback/SKILL.md` | Fetches PR review comments from ADO, implements fixes, pushes, and summarizes changes |

## Custom Agents

| Agent | Path | Description |
|-------|------|-------------|
| `qa-specialist` | `.claude/agents/qa-specialist.md` | QA engineer persona that tests happy paths, edge cases, error scenarios, and accessibility |
| `researcher` | `.claude/agents/researcher.md` | Domain knowledge researcher that only uses verified sources |
| `ux-reviewer` | `.claude/agents/ux-reviewer.md` | UX specialist reviewing user interface quality and usability |

## Hooks

| Hook | Trigger | Path | Description |
|------|---------|------|-------------|
| `block-dangerous-ops` | PreToolUse (Bash) | `.claude/hooks/block-dangerous-ops.sh` | Blocks `rm -rf`, `--force`, and force push commands, requiring explicit approval |
| `warn-debug-statements` | PostToolUse (Write/Edit) | `.claude/hooks/warn-debug-statements.sh` | Warns when debug statements (print, breakpoint, pdb) are left in Python files |

## Automated Workflows

| Workflow | Description |
|----------|-------------|
| `implement-story` | 5-phase orchestration: fetch story from ADO, plan with architect agent, implement, run quality gates (code-reviewer + qa-specialist + ux-reviewer agents), create PR and update story status |

## Project Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| ADRs | `docs/adr/` | 4 Architecture Decision Records (tech stack, data model, API, auth) |
| PRD | `docs/prd.md` | Product Requirements Document with personas, metrics, solution overview |
| Terraform | `infra/` | Azure deployment config (App Service, SQL, Key Vault, App Insights) |
| CI/CD | `azure-pipelines.yml` | Azure DevOps pipeline (lint, test, build, deploy stages) |
| E2E scenarios | `tests/e2e_scenarios.feature` | Gherkin test scenarios for camera monitoring and configuration |
| Test frame | `tests/frames/cat_on_chair_occluded.jpg` | Real-world regression test image (occluded cat that yolov8n missed) |

## Azure DevOps Work Items

| ID | Story | Status |
|----|-------|--------|
| #6104 | US-1: Live Camera Monitoring | Resolved |
| #6105 | US-2: Cat Detection with Human Guard | Resolved |
| #6106 | US-3: Webhook Alert on Cat Detection | New |
| #6107 | US-4: Detection History Web UI | New |
| #6108 | US-5: Detection Labeling for Fine-Tuning | New |
| #6109 | US-6: Model Fine-Tuning Trigger | New |
| #6110 | US-7: Environment-Based Configuration | Resolved |
| #6303 | US-8: Live Camera View with Detection Overlay | Resolved |
| #6304 | US-9: False Negative Reporting | Resolved |

## Pull Requests

| PR | Title | Status |
|----|-------|--------|
| #3791 | QA audit: add AGPL-3.0 license and fix test suite | Open |
| #3795 | US-2: Cat Detection with Human Guard | Open |
| #3796 | US-8: Live Camera View with Detection Overlay | Open |
| #3797 | US-9: False Negative Reporting | Open |
| #3798 | All features unified | Open |
