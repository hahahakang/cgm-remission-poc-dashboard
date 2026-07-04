#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: ./publish_to_github.sh https://github.com/USER/REPO.git" >&2
  exit 1
fi

REPO_URL="$1"

git remote remove origin >/dev/null 2>&1 || true
git remote add origin "$REPO_URL"
git push -u origin main

cat <<'MSG'

After pushing, enable GitHub Pages:
  Settings -> Pages -> Deploy from a branch -> main -> /root

Your dashboard URL will be:
  https://USER.github.io/REPO/
MSG
