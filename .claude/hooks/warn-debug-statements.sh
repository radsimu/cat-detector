#!/bin/bash
# PostToolUse hook for Write|Edit — warn if debug statements are left in code
FILE_PATH=$(echo "$TOOL_INPUT" | jq -r '.file_path // empty')
if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
  exit 0
fi
# Check for common debug artifacts in Python files
if echo "$FILE_PATH" | grep -qE '\.py$'; then
  MATCHES=$(grep -nE '^\s*(print\(|breakpoint\(\)|import pdb|pdb\.set_trace)' "$FILE_PATH" 2>/dev/null || true)
  if [ -n "$MATCHES" ]; then
    echo "Warning: possible debug statements found in $FILE_PATH:"
    echo "$MATCHES"
  fi
fi
exit 0
