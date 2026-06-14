---
name: implement-story
description: Implement story with full workflow including quality gates
---

Implement story with full workflow.

## Phase 1: Preparation
1. Fetch story details from Azure DevOps
2. Check for dependency conflicts with story-conflict-checker agent
3. If conflicts exist, STOP and report them
4. Move story to "In Progress" state

## Phase 2: Planning
1. Consult the architect agent for approach
2. If architecture changes needed, create/update ADRs
3. Present implementation plan
4. WAIT for user approval before proceeding

## Phase 3: Implementation
1. Implement the feature following the approved plan
2. Add appropriate error handling
3. Write unit tests for new code
4. Ensure all existing tests still pass

## Phase 4: Quality Gates
1. Run code-reviewer agent on changes
2. Address any Critical or Major issues
3. Run qa-specialist agent
4. Run ux-reviewer agent with Chrome DevTools
5. Fix any blocking issues found

## Phase 5: Completion
1. Create commit with descriptive message
2. Push to feature branch
3. Create pull request in Azure DevOps
4. Provide summary of changes

If any phase fails, STOP and report the issue.
