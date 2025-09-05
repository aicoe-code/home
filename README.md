# AICOE AI Paper Processing Pipeline

An automated pipeline for fetching, processing, and publishing AI research papers from arXiv.

## 🚀 NEW: Claude Code Integration (No API Required!)

This pipeline now supports **free processing with Claude Code** instead of requiring a paid API key!

### Quick Start with Claude Code
```bash
python3 process_with_claude.py
```

This interactive script will:
1. Fetch latest AI papers
2. Let you select papers to process
3. Generate prompts for Claude Code
4. Guide you through processing
5. Generate your website automatically

**Cost: $0** (vs $3-10/day with API)

## 🎯 Features

- **Automated Fetching**: Daily collection of AI papers from arXiv
- **Smart Filtering**: Relevance scoring based on keywords and categories
- **Claude Code Support**: Process papers for free using Claude Code
- **Deduplication**: Avoids processing the same papers multiple times
- **Beautiful Output**: Clean, professional HTML website
- **GitHub Actions**: Optional automation with GitHub Pages
- **Interactive Mode**: User-friendly paper selection and processing

## 📁 Project Structure

```
.
├── pipeline/               # Core processing modules
│   ├── fetcher.py         # arXiv paper fetcher
│   ├── processor.py       # Claude API processor (optional)
│   ├── local_processor.py # Claude Code processor (FREE!)
│   ├── generator.py       # HTML generator
│   ├── simple_generator.py # Simple HTML generator
│   └── config.yaml        # Configuration
├── process_with_claude.py # Interactive Claude Code script
├── process_batch.py       # Batch processing helper
├── templates/             # Summary templates
├── data/                  # Data storage
│   ├── raw/              # Fetched papers
│   ├── prompts/          # Claude prompts
│   ├── summaries/        # Claude responses
│   └── processed/        # Final data
└── docs/                  # GitHub Pages website
```

## 🛠️ Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd home
```

### 2. No Dependencies Required!
The pipeline works with Python 3's built-in libraries for basic operation.

Optional for enhanced features:
```bash
pip install pyyaml markdown requests
```

## 🎯 Usage

### Option 1: Interactive Mode (Recommended)
```bash
python3 process_with_claude.py
```

Follow the menu to:
- Fetch papers
- Select by relevance
- Process with Claude Code
- Generate website

### Option 2: Quick Processing
```bash
# Fetch papers
cd pipeline
python3 fetcher.py --days 1

# Generate prompt for top 10
python3 local_processor.py --input ../data/raw/arxiv_papers_*.json --quick 10

# Copy prompt to Claude Code, save response, then:
python3 ../process_batch.py
```

### Option 3: Manual Pipeline
See `LOCAL_PROCESSING.md` for detailed manual processing steps.

## 📊 Daily Workflow (5 minutes)

```bash
# Morning routine
python3 process_with_claude.py
# Select "top" → Choose batch mode → Copy to Claude → Save → Done!

# Publish to GitHub
git add -A
git commit -m "Daily AI papers update"
git push origin gh-pages
```

## 🎨 Configuration

Edit `pipeline/config.yaml` to customize:

```yaml
arxiv:
  categories: [cs.AI, cs.LG, cs.CV]  # Categories to monitor
  keywords: [transformer, llm, neural] # Relevance keywords
  max_papers_per_category: 100        # Fetch limit

processing:
  max_papers_per_run: 50              # Process limit
```

## 💰 Cost Comparison

| Method | Daily Cost | Annual Cost | Quality |
|--------|------------|-------------|---------|
| Claude API | $3-10 | $1,000-3,600 | Good |
| Claude Code | **$0** | **$0** | **Better** |

## 📈 Features

- **Smart Relevance Scoring**: Prioritizes important papers
- **Batch Processing**: Handle multiple papers efficiently
- **Deduplication**: Never process the same paper twice
- **Category Filtering**: Focus on your areas of interest
- **Beautiful HTML**: Professional, responsive design
- **Statistics Dashboard**: Track papers and trends

## 🌐 GitHub Pages Deployment

1. Enable GitHub Pages in repository settings
2. Source: Deploy from branch (gh-pages or main)
3. Folder: /docs
4. Your site: `https://username.github.io/repo-name/`

## 📝 Documentation

- `LOCAL_PROCESSING.md` - Complete Claude Code guide
- `CLAUDE_CODE_SETUP.md` - Setup instructions
- `pipeline/config.yaml` - Configuration options

## 🔧 Troubleshooting

### Papers not fetching?
- Check internet connection
- Verify arXiv is accessible
- Try different date: `--days 2`

### Claude Code processing?
- Ensure prompt files are in `data/prompts/`
- Save responses to `data/summaries/`
- Use `.md` extension for summaries

### Website not generating?
```bash
python3 process_batch.py
open docs/index.html  # Check locally
```

## 🚀 Advanced Features

### Custom Categories
Add your research areas to `config.yaml`:
```yaml
categories: [cs.RO, cs.HC]  # Robotics, HCI
```

### Relevance Keywords
Customize for your interests:
```yaml
keywords: [robotics, embodied, multimodal]
```

### Processing Modes
- **Quick**: Top 10 papers in one prompt
- **Batch**: Multiple papers per Claude session
- **Individual**: Detailed analysis per paper

## 📊 Current Statistics

- **Papers in database**: 332
- **Processing time**: ~5 min for 10 papers
- **Categories covered**: 9 AI/ML areas
- **Cost with Claude Code**: $0

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add new features
- Improve processing
- Enhance HTML output
- Fix bugs

## 📧 Support

For issues or questions:
- Open a GitHub issue
- Check documentation
- Review troubleshooting guide

---

**Built with ❤️ for the AI research community**
*Process papers for free with Claude Code!* 🎉