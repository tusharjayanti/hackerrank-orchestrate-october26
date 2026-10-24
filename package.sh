#!/usr/bin/env bash
# Build code.zip from tracked files only (git archive), so .env, caches,
# virtual environments, run outputs and the dataset can never ship.
# Organizer files (AGENTS.md, CLAUDE.md, problem_statement.md, ...) stay in
# the repo but never ship: only the paths listed below are archived.
# Usage: ./package.sh               (from any directory; writes code.zip at the repo root)
set -euo pipefail

cd "$(dirname "$0")"
cd "$(git rev-parse --show-toplevel)"

required=(README.md code)
optional=(pyproject.toml uv.lock requirements.txt .env.example)

paths=()
for p in "${required[@]}"; do
  if ! git cat-file -e "HEAD:$p" 2>/dev/null; then
    echo "refusing: $p is not in the committed tree" >&2
    exit 1
  fi
  paths+=("$p")
done
for p in "${optional[@]}"; do
  if git cat-file -e "HEAD:$p" 2>/dev/null; then
    paths+=("$p")
  else
    echo "warning: $p is not committed and will not ship" >&2
  fi
done

if [ -n "$(git status --porcelain --untracked-files=no -- "${paths[@]}")" ]; then
  echo "refusing: uncommitted changes in shipped paths; commit first so the zip matches a commit" >&2
  exit 1
fi

tracked="$(git ls-files -- "${paths[@]}")"
if grep -Eq '(^|/)\.env$' <<<"$tracked"; then
  echo "refusing: a .env file is tracked" >&2
  exit 1
fi
if grep -Eiq '\.(zip|tar|gz|tgz|7z|rar)$' <<<"$tracked"; then
  echo "refusing: an archive is tracked inside the shipped paths:" >&2
  grep -Ei '\.(zip|tar|gz|tgz|7z|rar)$' <<<"$tracked" >&2
  exit 1
fi
if grep -Eq '(^|/)(AGENTS|CLAUDE|HACKERRANK_README|problem_statement|agent_trace_instructions)\.md$' <<<"$tracked"; then
  echo "refusing: an organizer file would ship" >&2
  exit 1
fi
if grep -q '^dataset/' <<<"$tracked"; then
  echo "refusing: dataset files would ship" >&2
  exit 1
fi

rm -f code.zip
git archive --format=zip -o code.zip HEAD -- "${paths[@]}"
echo "code.zip from commit $(git rev-parse --short HEAD):"
unzip -Z1 code.zip
echo "$(unzip -Z1 code.zip | wc -l | tr -d ' ') files"
