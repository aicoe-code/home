#!/usr/bin/env bash
set -euo pipefail

# Publish static site from ./Private to ./docs for GitHub Pages

usage() {
  cat <<USAGE
Usage: $(basename "$0") [-n] [-m "commit message"] [-b] [-s <source_dir>]

Options:
  -n               Dry-run (show what would change, no commit)
  -m "message"     Commit message to use when publishing
  -b               Bootstrap: copy current ./docs into ./Private (no delete)
  -s <source_dir>  Source directory to publish from (default: auto-detect)

Behavior:
  - Syncs ./Private -> ./docs using rsync
  - Excludes x_* files (work-in-progress), hidden files, _drafts/, and common junk
  - Uses --delete to remove files from docs that no longer exist in Private
USAGE
}

DRY_RUN=0
COMMIT_MSG="site: publish updates from Private/"
BOOTSTRAP=0
SRC_DIR=""
while getopts ":nm:bs:h" opt; do
  case $opt in
    n) DRY_RUN=1 ;;
    m) COMMIT_MSG=$OPTARG ;;
    b) BOOTSTRAP=1 ;;
    s) SRC_DIR=$OPTARG ;;
    h) usage; exit 0 ;;
    :) echo "Error: -$OPTARG requires an argument" >&2; usage; exit 2 ;;
    \?) echo "Error: invalid option -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT_DIR"

if [ -z "$SRC_DIR" ]; then
  if [ -d Private/00-Public_site ]; then
    SRC_DIR="Private/00-Public_site"
  elif [ -d Private ]; then
    SRC_DIR="Private"
  else
    SRC_DIR="Private"
  fi
fi

if [ ! -d "$SRC_DIR" ]; then
  echo "Source directory '$SRC_DIR' not found. Creating it now..."
  mkdir -p "$SRC_DIR"
fi

if [ ! -d docs ]; then
  mkdir -p docs
fi

if [ ! -f "$SRC_DIR/index.html" ]; then
  echo "Warning: $SRC_DIR/index.html not found. Pages may not have a homepage."
fi

RSYNC_FLAGS=("-av" "--delete" \
  "--exclude" ".DS_Store" \
  "--exclude" ".git*" \
  "--exclude" "x_*" \
  "--exclude" "Private/**" \
  "--exclude" "_drafts/**" \
  "--exclude" "README.md")

if [ "$BOOTSTRAP" -eq 1 ]; then
  echo "> Bootstrapping: copying ./docs into $SRC_DIR (no deletions)"
  if [ -d docs ]; then
    rsync -av docs/ "$SRC_DIR"/
    echo "> Bootstrap complete. You can now edit files in $SRC_DIR."
    exit 0
  else
    echo "Warning: ./docs does not exist yet; nothing to bootstrap."
    exit 0
  fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "> Dry-run: showing planned changes from $SRC_DIR -> docs"
  rsync --dry-run "${RSYNC_FLAGS[@]}" "$SRC_DIR"/ docs/
  exit 0
fi

rsync "${RSYNC_FLAGS[@]}" "$SRC_DIR"/ docs/

if git -C "$ROOT_DIR" diff --quiet -- docs; then
  echo "No changes detected in docs/. Nothing to commit."
  exit 0
fi

git add docs
git commit -m "$COMMIT_MSG"
echo "> Committed. Pushing to origin/main..."
git push -u origin main
echo "> Publish complete. GitHub Pages will update shortly."
