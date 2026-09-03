#!/bin/bash
# PostToolUse hook -- runs after every Edit/Write.
#
# Lints ONLY the file that was just written. The tree has ~250 pre-existing
# findings; linting all of them on every edit would make this hook noise and
# noise gets ignored. Touching a file is what puts it in scope -- the tree
# gets clean as it gets worked on, and no single commit has to do it all.
#
# Exit 2 is the contract: stderr goes back to Claude, which then has to fix it
# before continuing. Exit 0 means silent pass.

input=$(cat)
file=$(printf '%s' "$input" | python3 -c \
  'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)

case "$file" in *.py) ;; *) exit 0 ;; esac
[ -f "$file" ] || exit 0

root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root" || exit 0

if ! out=$(uvx ruff check "$file" --output-format=concise 2>&1); then
    printf 'ruff found problems in %s:\n\n%s\n' "$file" "$out" >&2
    exit 2
fi
exit 0
