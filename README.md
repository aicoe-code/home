# AICOE AI Paper Processing Pipeline

An automated pipeline for fetching, processing, and publishing AI research papers from arXiv.

## 🚀 Features

- **Automated Fetching**: Daily collection of AI papers from arXiv
- **Smart Filtering**: Relevance scoring based on keywords and categories
- **Deduplication**: Avoids processing the same papers multiple times
- **AI Processing**: Claude AI integration for generating summaries (optional)
- **Beautiful Output**: Clean, professional HTML website
- **GitHub Actions**: Fully automated with twice-daily runs
- **GitHub Pages**: Automatic deployment to your website

## 📁 Project Structure

```
.
├── pipeline/               # Core processing modules
│   ├── fetcher.py         # arXiv paper fetcher with deduplication
│   ├── processor.py       # Claude AI processor for summaries
│   ├── generator.py       # HTML generator (full-featured)
│   ├── simple_generator.py # HTML generator (no dependencies)
│   ├── run_pipeline.py    # Main orchestrator
│   └── config.yaml        # Configuration file
├── templates/             # Templates for summaries
│   └── summary-template.md
├── data/                  # Data storage
│   ├── raw/              # Fetched papers (JSON)
│   ├── processed/        # Processed summaries
│   └── archive/          # Archived old papers
├── docs/                  # GitHub Pages website
│   ├── index.html        # Main page
│   ├── papers/           # Individual paper pages
│   └── assets/           # CSS and other assets
└── .github/workflows/     # GitHub Actions
    └── daily-ai-papers.yml
```

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd home
```

### 2. Install Dependencies (Optional)

For full functionality with markdown processing:
```bash
pip install pyyaml markdown requests
```

For Claude AI integration:
```bash
pip install anthropic
```

### 3. Configure the Pipeline

Edit `pipeline/config.yaml` to customize:
- arXiv categories to monitor
- Keywords for relevance scoring
- Processing limits
- Output settings

### 4. Set Up Claude API (Optional)

Add your Claude API key as an environment variable:
```bash
export CLAUDE_API_KEY="your-api-key-here"
```

For GitHub Actions, add it as a repository secret:
- Go to Settings → Secrets → Actions
- Add new secret: `CLAUDE_API_KEY`

### 5. Enable GitHub Pages

1. Go to repository Settings → Pages
2. Source: Deploy from a branch
3. Branch: main, folder: /docs
4. Save

## 🎯 Usage

### Manual Run

Fetch papers from yesterday:
```bash
cd pipeline
python3 fetcher.py
```

Fetch papers from specific date:
```bash
python3 fetcher.py --date 2025-09-04
```

Generate HTML from fetched papers:
```bash
python3 simple_generator.py ../data/raw/arxiv_papers_*.json
```

Run complete pipeline:
```bash
python3 run_pipeline.py
```

### Automated Runs

The GitHub Actions workflow runs automatically:
- **Schedule**: 9 AM and 6 PM UTC daily
- **Manual trigger**: Actions tab → Run workflow

## 📊 Pipeline Components

### Fetcher (`fetcher.py`)
- Fetches papers from multiple arXiv categories
- Calculates relevance scores based on keywords
- Implements deduplication to avoid reprocessing
- Saves papers to JSON format

### Processor (`processor.py`)
- Processes papers with Claude AI (when API key configured)
- Generates structured summaries
- Falls back to basic summaries without API

### Generator (`generator.py` / `simple_generator.py`)
- Creates beautiful HTML pages
- Generates index with statistics
- Creates individual paper pages
- No external dependencies (simple version)

### Orchestrator (`run_pipeline.py`)
- Coordinates all components
- Handles configuration
- Manages data flow
- Provides logging and error handling

## 🎨 Customization

### Categories and Keywords

Edit `pipeline/config.yaml`:
```yaml
arxiv:
  categories:
    - cs.AI
    - cs.LG
    - cs.CV
  keywords:
    - transformer
    - neural network
    - deep learning
```

### Relevance Scoring

Adjust minimum relevance score:
```yaml
arxiv:
  min_relevance_score: 2.0  # Only process papers with score >= 2.0
```

### Processing Limits

Control how many papers to process:
```yaml
processing:
  max_papers_per_run: 50
```

## 📈 Output

The pipeline generates:
- **Website**: Clean HTML with all papers
- **Statistics**: Paper counts, relevance distribution
- **Archives**: Historical data preservation

View your website at: `https://<username>.github.io/<repo-name>/`

## 🔧 Troubleshooting

### No papers found
- Check if arXiv API is accessible
- Verify date parameter (papers may not exist for future dates)

### HTML not generating
- Ensure papers exist in data/raw/
- Check for Python errors in console

### GitHub Actions failing
- Verify secrets are configured
- Check workflow logs for specific errors

## 📝 License

MIT License - feel free to use and modify

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

Built with ❤️ for the AICOE Research Library