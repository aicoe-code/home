# Processing AI Papers with Claude Code (No API Required!)

This guide explains how to process AI papers using Claude Code instead of the Claude API. This method is **completely free** and gives you more control over the summaries.

## 🎯 Why Use Claude Code?

- **✅ Free**: No API costs
- **✅ Interactive**: Review and refine summaries
- **✅ Better Quality**: Claude Code often provides more detailed responses
- **✅ No Setup**: No API keys or authentication needed
- **✅ Flexible**: Process papers at your own pace

## 🚀 Quick Start

### Option 1: Interactive Script (Easiest)

```bash
python3 process_with_claude.py
```

This interactive script will:
1. Fetch latest papers from arXiv
2. Let you select which papers to process
3. Generate prompts for Claude Code
4. Guide you through processing
5. Generate the HTML website

### Option 2: Manual Pipeline

```bash
# 1. Fetch papers
cd pipeline
python3 fetcher.py --days 1

# 2. Prepare prompts for Claude Code
python3 local_processor.py --input ../data/raw/arxiv_papers_*.json --prepare

# 3. Process with Claude (see instructions below)

# 4. Process completed summaries
python3 local_processor.py --process

# 5. Generate website
python3 simple_generator.py
```

## 📋 Step-by-Step Guide

### Step 1: Fetch Papers

Fetch the latest AI papers from arXiv:

```bash
cd pipeline
python3 fetcher.py --days 1  # Fetch from yesterday
```

This will:
- Download papers from 9 AI categories
- Calculate relevance scores
- Save to `data/raw/arxiv_papers_YYYY-MM-DD.json`

### Step 2: Select Papers to Process

You have several options:

**Process Top 10 Papers** (Recommended for daily processing):
```bash
python3 local_processor.py --input ../data/raw/arxiv_papers_*.json --quick 10
```

**Process All High-Relevance Papers**:
```bash
python3 local_processor.py --input ../data/raw/arxiv_papers_*.json --prepare
```

This creates prompt files in `data/prompts/`

### Step 3: Process with Claude Code

#### For Quick Processing (Batch Mode):

1. Open the generated prompt file from `data/prompts/quick_batch_*.txt`
2. Copy ALL content (Cmd+A, Cmd+C on Mac)
3. Open Claude Code (claude.ai)
4. Paste the prompt
5. Save Claude's response to `data/summaries/quick_batch_summary.md`

#### For Individual Processing:

1. Open each file from `data/prompts/`
2. Copy the content
3. Paste into Claude Code
4. Save response to `data/summaries/` (change `_prompt.txt` to `_summary.md`)

**💡 Tips for Claude Code:**
- You can process multiple papers in one conversation
- Ask Claude to format responses in markdown
- Request specific sections if needed
- You can ask follow-up questions about papers

### Step 4: Process Summaries

After saving Claude's responses:

```bash
cd pipeline
python3 local_processor.py --process
```

This will:
- Read all summaries from `data/summaries/`
- Create structured JSON files
- Move processed files to archive

### Step 5: Generate Website

```bash
python3 simple_generator.py
```

This creates your website in `docs/index.html`

### Step 6: Publish (Optional)

```bash
git add -A
git commit -m "Update papers with Claude Code"
git push origin gh-pages
```

## 🎮 Interactive Mode

The easiest way is using the interactive script:

```bash
python3 process_with_claude.py
```

This provides a menu-driven interface:
1. **Fetch papers** - Get latest from arXiv
2. **Select papers** - Choose by relevance or manually
3. **Process mode** - Individual or batch
4. **Claude processing** - Guided instructions
5. **Generate website** - Automatic HTML generation

## 📁 Directory Structure

```
data/
├── raw/           # Fetched papers from arXiv
├── prompts/       # Generated prompts for Claude
├── summaries/     # Your Claude responses
│   └── archive/   # Processed summaries
└── processed/     # Final structured data
```

## 🔄 Daily Workflow

Here's a typical daily workflow:

```bash
# Morning routine (5 minutes)
python3 process_with_claude.py

# Select "top" for top 10 papers
# Choose "2" for batch mode
# Copy prompt to Claude Code
# Save response
# Press Enter to generate website

# Push to GitHub
git add -A
git commit -m "Daily AI papers update"
git push
```

## 🎯 Processing Strategies

### Daily Processing (5-10 papers)
- Use `--quick 10` for top papers
- Single batch prompt
- 5-10 minutes total

### Weekly Deep Dive (20-50 papers)
- Process high-relevance papers
- Individual prompts for detailed summaries
- 30-60 minutes

### Selective Processing
- Manually select specific papers of interest
- Focus on your research areas
- Quality over quantity

## 💡 Pro Tips

1. **Batch Processing**: Process 5-10 papers in one Claude conversation for efficiency

2. **Custom Prompts**: Edit prompts before sending to Claude to focus on specific aspects

3. **Relevance Filtering**: Focus on papers with score > 5 for high-impact research

4. **Summary Quality**: You can ask Claude to expand on specific sections

5. **Categories**: Customize categories in `pipeline/config.yaml` to match your interests

## 🛠️ Troubleshooting

### No prompts generated?
- Check if papers exist in `data/raw/`
- Verify file paths in commands

### Summaries not processing?
- Ensure files are named `*_summary.md`
- Check they're in `data/summaries/`

### Website not updating?
- Run `python3 simple_generator.py` after processing
- Check `docs/index.html` was created

## 📊 Configuration

Edit `pipeline/config.yaml` to customize:

```yaml
# Categories to monitor
arxiv:
  categories:
    - cs.AI
    - cs.LG
    - cs.CV

# Keywords for relevance
keywords:
  - transformer
  - llm
  - neural network

# Processing limits
processing:
  max_papers_per_run: 20
```

## 🤖 Example Claude Prompt

Here's what the generated prompts look like:

```
Please analyze this academic paper and create a comprehensive summary...

Paper Title: Attention Is All You Need
Authors: Vaswani et al.
Categories: cs.CL, cs.LG

Abstract:
[Paper abstract here]

Please provide:
1. Executive Summary
2. Key Contributions
3. Methodology
4. Results
5. Strengths & Limitations
[etc...]
```

## 🎉 Benefits Over API

1. **Cost**: $0 vs $0.01-0.03 per paper with API
2. **Quality**: Claude Code often gives more detailed responses
3. **Control**: Review and refine each summary
4. **Learning**: You actually read the papers!
5. **Flexibility**: Process on your schedule

## 📚 Next Steps

- Customize categories in `config.yaml`
- Add your own keywords for relevance scoring
- Create templates for specific paper types
- Set up a weekly processing routine

---

Happy paper processing! 🎊 No API required, just you and Claude Code!