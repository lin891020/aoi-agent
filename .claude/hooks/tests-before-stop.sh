#!/bin/bash
# Stop hook -- runs when Claude is about to end its turn.
#
# This is the one that matters. It means Claude cannot say "done" while the
# suite is red: exit 2 refuses the stop and hands the failures back.
#
# The suite is 1421 tests / ~46s, too slow to run on every edit, but once at
# the end of a turn is fine -- and only when Python actually changed, so
# conversations that touched nothing but docs end instantly.
#
# `stop_hook_active` guards the loop: if Claude is already here because this
# hook blocked once, do not block again. Otherwise a genuinely broken test
# would trap the session.

input=$(cat)
active=$(printf '%s' "$input" | python3 -c \
  'import json,sys; print(json.load(sys.stdin).get("stop_hook_active", False))' 2>/dev/null)
[ "$active" = "True" ] && exit 0

root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root" || exit 0

changed=$(git status --porcelain -- '*.py' 2>/dev/null)
[ -z "$changed" ] && exit 0

# -p no:warnings matters: the first run of this hook buried the one real
# failure under 20 lines of torch deprecation notices. Feedback that has to be
# searched is feedback that gets skimmed.
if ! out=$(uv run pytest -m "not dataset" -q --no-header -p no:warnings --tb=short 2>&1); then
    printf 'The suite is red. You changed Python and cannot stop here.\n\n%s\n' \
        "$(printf '%s' "$out" | tail -30)" >&2
    exit 2
fi
exit 0
