# Claude Code Integration - Setup Complete! ✅

## 🎯 Overview
Your AI paper processing pipeline now works with Claude Code (free) instead of Claude API (paid).

## 🚀 Quick Start Guide

### Daily Processing (5 minutes)
```bash
python3 process_with_claude.py
```
- Select "top" for top 10 papers
- Choose batch mode (option 2)
- Copy prompt to Claude Code
- Save response to suggested file
- Press Enter to generate website

### Manual Processing
```bash
# 1. Generate prompt for top papers
python3 pipeline/local_processor.py --input data/raw/arxiv_papers_*.json --quick 10

# 2. Copy prompt content to Claude Code
# 3. Save Claude's response to: data/summaries/quick_batch_summary.md

# 4. Generate website
python3 process_batch.py
```

## 📁 New Files Added

### Core Processing
- `pipeline/local_processor.py` - Claude Code processor
- `process_with_claude.py` - Interactive processing script
- `process_batch.py` - Batch summary processor
- `LOCAL_PROCESSING.md` - Detailed documentation

### Documentation
- `CLAUDE_CODE_SETUP.md` - This file
- `LOCAL_PROCESSING.md` - Complete user guide

## 💰 Cost Savings
- **Claude API**: $3-10 per day (100-300 papers)
- **Claude Code**: $0 (FREE!)
- **Annual Savings**: ~$2,000-3,600

## 🎯 Processing Workflow

1. **Fetch Papers** → 2. **Generate Prompts** → 3. **Claude Code** → 4. **Generate HTML**

### Step 1: Fetch Papers
```bash
cd pipeline
python3 fetcher.py --days 1
```

### Step 2: Generate Prompts
```bash
python3 local_processor.py --input ../data/raw/arxiv_papers_*.json --quick 10
```

### Step 3: Process with Claude
- Open generated prompt file
- Copy to claude.ai
- Save response to data/summaries/

### Step 4: Generate Website
```bash
python3 ../process_batch.py
```

## 📊 Features

### Interactive Mode ✨
- Menu-driven interface
- Paper selection by relevance
- Batch or individual processing
- Automatic HTML generation

### Manual Mode 🛠️
- Full control over processing
- Custom paper selection
- Flexible workflow

### Batch Processing 📦
- Process multiple papers at once
- Single Claude conversation
- Faster processing

## 🔧 Configuration

Edit `pipeline/config.yaml` to customize:
- arXiv categories
- Relevance keywords  
- Processing limits

## 📈 Current Statistics
- **Papers Available**: 332
- **High Relevance**: 77
- **Categories**: 9
- **Processing Time**: ~5 min for 10 papers

## 🎉 Benefits Over API

1. **Free Forever** - No subscription or API costs
2. **Better Quality** - Claude Code provides detailed responses
3. **Interactive** - Ask follow-ups, refine summaries
4. **Educational** - You see and learn from papers
5. **Flexible** - Process on your schedule

## 🚀 Publishing to GitHub Pages

```bash
# After processing papers
git add -A
git commit -m "Update AI papers with Claude Code"
git push origin gh-pages
```

Website URL: https://aicoe-code.github.io/home/

## 🐛 Troubleshooting

### No prompts generated?
```bash
ls data/prompts/  # Check for prompt files
```

### Summaries not processing?
```bash
ls data/summaries/  # Ensure files end with _summary.md
```

### Website not updating?
```bash
python3 process_batch.py  # Regenerate HTML
open docs/index.html      # Check locally
```

## 📝 Daily Routine Example

```bash
# Morning (5 minutes)
python3 process_with_claude.py
# Select: top → 2 (batch) → Copy to Claude → Save → Enter

# Push to GitHub
git add -A && git commit -m "Daily papers" && git push
```

## 🔗 Quick Commands

```bash
# Fetch papers
cd pipeline && python3 fetcher.py --days 1

# Quick process top 10
python3 local_processor.py --input ../data/raw/arxiv_papers_*.json --quick 10

# Process summaries
python3 ../process_batch.py

# Open website
open ../docs/index.html
```

---

*Setup completed: 2025-09-05*
*No API required - just Claude Code!* 🎊