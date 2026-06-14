---
name: handle-pr-feedback
description: Handle feedback on pull request
---

Handle feedback on pull request (provide PR number).

1. Fetch PR details and comments from Azure DevOps
2. Identify actionable feedback items
3. For each feedback item:
   a. Analyze the request
   b. Implement the fix
   c. Add tests if needed
4. Commit changes with message: "fix: address PR feedback"
5. Push to the same branch
6. Comment on PR summarizing changes made

If feedback is unclear, ask for clarification before proceeding.
