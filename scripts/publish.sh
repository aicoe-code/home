#!/usr/bin/env bash
set -euo pipefail

# Publish static site from ./Private to ./docs for GitHub Pages

usage() {
  cat <<USAGE
Usage: $(basename "$0") [-n] [-m "commit message"] [-b]

Options:
  -n               Dry-run (show what would change, no commit)
  -m "message"     Commit message to use when publishing
  -b               Bootstrap: copy current ./docs into ./Private (no delete)

Behavior:
  - Syncs ./Private -> ./docs using rsync
  - Excludes x_* files (work-in-progress), hidden files, _drafts/, and common junk
  - Uses --delete to remove files from docs that no longer exist in Private
USAGE
}

DRY_RUN=0
COMMIT_MSG="site: publish updates from Private/"
BOOTSTRAP=0
while getopts ":nm:bh" opt; do
  case $opt in
    n) DRY_RUN=1 ;;
    m) COMMIT_MSG=$OPTARG ;;
    b) BOOTSTRAP=1 ;;
    h) usage; exit 0 ;;
    :) echo "Error: -$OPTARG requires an argument" >&2; usage; exit 2 ;;
    \?) echo "Error: invalid option -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT_DIR"

if [ ! -d Private ]; then
  echo "./Private not found. Creating it now..."
  mkdir -p Private
fi

if [ ! -d docs ]; then
  mkdir -p docs
fi

if [ ! -f Private/index.html ]; then
  echo "Warning: ./Private/index.html not found. Pages may not have a homepage."
fi

RSYNC_FLAGS=("-av" "--delete" \
  "--exclude" ".DS_Store" \
  "--exclude" ".git*" \
  "--exclude" "x_*" \
  "--exclude" "_drafts/**" \
  "--exclude" "README.md")

if [ "$BOOTSTRAP" -eq 1 ]; then
  echo "> Bootstrapping: copying ./docs into ./Private (no deletions)"
  if [ -d docs ]; then
    rsync -av docs/ Private/
    echo "> Bootstrap complete. You can now edit files in ./Private."
    exit 0
  else
    echo "Warning: ./docs does not exist yet; nothing to bootstrap."
    exit 0
  fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "> Dry-run: showing planned changes from Private -> docs"
  rsync --dry-run "${RSYNC_FLAGS[@]}" Private/ docs/
  exit 0
fi

rsync "${RSYNC_FLAGS[@]}" Private/ docs/

if git -C "$ROOT_DIR" diff --quiet -- docs; then
  echo "No changes detected in docs/. Nothing to commit."
  exit 0
fi

git add docs
git commit -m "$COMMIT_MSG"
echo "> Committed. Pushing to origin/main..."
git push -u origin main
echo "> Publish complete. GitHub Pages will update shortly."
