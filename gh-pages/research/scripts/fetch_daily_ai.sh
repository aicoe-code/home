#!/bin/bash

# Daily arXiv AI Papers Fetcher Script
# This script fetches AI papers from arXiv for the previous day

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set Python path (adjust if needed)
PYTHON3="/usr/bin/python3"

# Get yesterday's date
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    YESTERDAY=$(date -v-1d "+%Y-%m-%d")
else
    # Linux
    YESTERDAY=$(date -d "yesterday" "+%Y-%m-%d")
fi

echo "Fetching AI papers from arXiv for $YESTERDAY..."

# Run the fetcher script
cd "$SCRIPT_DIR"
$PYTHON3 arxiv_daily_ai.py \
    --date "$YESTERDAY" \
    --output-dir "../daily" \
    --max-results 100 \
    --format all

echo "Done! Check the ../daily directory for output files."

# Optional: Add to index page
# You can uncomment this to automatically update an index
# echo "<li><a href='daily/arxiv-ai-papers-$YESTERDAY.html'>$YESTERDAY AI Papers</a></li>" >> ../daily/index.html