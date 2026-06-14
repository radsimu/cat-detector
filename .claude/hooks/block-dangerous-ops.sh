#!/bin/bash
COMMAND=$(jq -r '.tool_input.command')
if echo "$COMMAND" | grep -qiE 'rm\s+-rf|--force|force\s+push'; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: "Dangerous operation requires explicit approval"
    }
  }'
else
  exit 0
fi