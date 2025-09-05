# arXiv AI Papers Daily Fetcher

## Overview
This collection of scripts fetches and organizes AI-related papers from arXiv on a daily basis. It monitors multiple AI categories and generates formatted outputs for easy browsing.

## Scripts

### 1. `arxiv_daily_ai.py`
Main Python script that fetches AI papers from arXiv using their API.

**Features:**
- Fetches papers from 9 AI-related categories (cs.AI, cs.LG, cs.CV, cs.CL, etc.)
- Filters papers by date (typically yesterday's papers)
- Calculates relevance scores based on AI keywords
- Generates multiple output formats (JSON, HTML, Markdown)
- Deduplicates papers that appear in multiple categories
- Rate limiting to respect arXiv API guidelines

**Usage:**
```bash
# Fetch yesterday's papers (default)
python3 arxiv_daily_ai.py

# Fetch papers from a specific date
python3 arxiv_daily_ai.py --date 2025-09-04

# Fetch only from specific categories
python3 arxiv_daily_ai.py --categories cs.AI cs.LG --max-results 50

# Generate only specific format
python3 arxiv_daily_ai.py --format html

# Custom output directory
python3 arxiv_daily_ai.py --output-dir /path/to/output
```

**Command-line Arguments:**
- `--date`: Date to fetch papers (YYYY-MM-DD format)
- `--output-dir`: Output directory (default: ../daily)
- `--max-results`: Maximum papers per category (default: 200)
- `--categories`: Specific categories to fetch
- `--format`: Output format (json/html/markdown/all)

### 2. `fetch_daily_ai.sh`
Bash wrapper script for daily execution.

**Usage:**
```bash
# Run daily fetch
./fetch_daily_ai.sh
```

This script automatically:
- Fetches yesterday's papers
- Saves to the `../daily` directory
- Generates all output formats

## Monitored Categories

- **cs.AI**: Artificial Intelligence
- **cs.LG**: Machine Learning
- **cs.CV**: Computer Vision and Pattern Recognition
- **cs.CL**: Computation and Language
- **cs.NE**: Neural and Evolutionary Computing
- **stat.ML**: Machine Learning (Statistics)
- **cs.RO**: Robotics
- **cs.HC**: Human-Computer Interaction
- **cs.IR**: Information Retrieval

## Relevance Scoring

Papers are scored based on the presence of AI-related keywords:
- LLM, transformer, GPT, BERT
- Neural networks, deep learning
- Computer vision, NLP
- Generative models, diffusion models
- And many more...

Keywords in titles receive higher weight than those in abstracts.

## Output Formats

### JSON
Structured data with all paper metadata, ideal for programmatic access.

### HTML
Styled webpage matching the AICOE research library design, with:
- Statistics dashboard
- Relevance-based sorting
- Color-coded priority badges
- Direct links to arXiv and PDF

### Markdown
Human-readable format with full abstracts and metadata.

## Scheduling Daily Runs

### Using cron (Linux/macOS)
Add to your crontab:
```bash
# Run at 2 AM daily (after arXiv updates at midnight EST)
0 2 * * * /path/to/fetch_daily_ai.sh
```

### Using GitHub Actions
Create `.github/workflows/fetch-arxiv.yml`:
```yaml
name: Fetch Daily arXiv Papers

on:
  schedule:
    - cron: '0 7 * * *'  # 7 AM UTC daily
  workflow_dispatch:  # Allow manual trigger

jobs:
  fetch:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Fetch papers
        run: |
          cd gh-pages/research/scripts
          python3 arxiv_daily_ai.py
      - name: Commit results
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add -A
          git commit -m "Update daily AI papers"
          git push
```

## API Rate Limiting

The script includes a 3-second delay between API calls to respect arXiv's rate limits. Please don't remove this delay.

## Troubleshooting

1. **No papers found**: arXiv updates at midnight EST. Run the script after this time.
2. **API errors**: Check your internet connection and ensure you're not exceeding rate limits.
3. **Missing categories**: Some categories may have no papers on certain days.

## Future Enhancements

Potential improvements:
- Email notifications for high-relevance papers
- RSS feed generation
- Integration with citation databases
- Automatic summarization using LLMs
- Topic clustering and visualization

## License

These scripts are provided as-is for research purposes. Please respect arXiv's terms of service when using their API.