# AI Paper Processing Pipeline - Setup Instructions

## 🎉 Current Status
Your automated AI paper processing pipeline has been successfully deployed to GitHub!

- **Repository**: https://github.com/aicoe-code/home
- **Website**: https://aicoe-code.github.io/home/
- **Branch**: gh-pages
- **Status**: ✅ Live and operational

## 📋 Required Setup Steps

### 1. Add Claude API Key (For AI Summaries)

To enable AI-powered paper summaries:

1. Get your Claude API key from [Anthropic Console](https://console.anthropic.com/)
2. Go to your repository's secrets: https://github.com/aicoe-code/home/settings/secrets/actions
3. Click **"New repository secret"**
4. Add the following:
   - **Name**: `CLAUDE_API_KEY`
   - **Value**: Your Claude API key (starts with `sk-ant-...`)
5. Click **"Add secret"**

### 2. Verify GitHub Pages

1. Go to: https://github.com/aicoe-code/home/settings/pages
2. Ensure these settings:
   - **Source**: Deploy from a branch
   - **Branch**: `gh-pages`
   - **Folder**: `/ (root)`
3. Save if any changes were made

Your site should be live at: https://aicoe-code.github.io/home/

### 3. Test GitHub Actions Workflow

1. Go to: https://github.com/aicoe-code/home/actions
2. Click on **"Daily AI Papers Pipeline"**
3. Click **"Run workflow"** button
4. You can optionally specify:
   - Date to fetch papers
   - Force refetch option
   - Maximum papers to process
5. Click **"Run workflow"** to test

## 🔄 Automated Schedule

The pipeline runs automatically:
- **9:00 AM UTC** (4 AM EST / 1 AM PST)
- **6:00 PM UTC** (1 PM EST / 10 AM PST)

## 📊 What the Pipeline Does

1. **Fetches** AI papers from arXiv categories:
   - cs.AI (Artificial Intelligence)
   - cs.LG (Machine Learning)
   - cs.CV (Computer Vision)
   - cs.CL (Natural Language Processing)
   - And 5 more categories

2. **Scores** papers based on relevance keywords:
   - LLM, transformer, neural network
   - Deep learning, computer vision
   - And 20+ other keywords

3. **Processes** papers with Claude AI (when API key is set)
   - Generates structured summaries
   - Extracts key insights
   - Evaluates relevance to AICOE mission

4. **Publishes** to GitHub Pages
   - Clean, professional HTML
   - Statistics and visualizations
   - Mobile-responsive design

## 🛠️ Manual Operations

### Run Pipeline Locally

```bash
# Fetch papers from yesterday
cd pipeline
python3 fetcher.py

# Fetch papers from specific date
python3 fetcher.py --date 2025-09-04

# Generate HTML from fetched papers
python3 simple_generator.py ../data/raw/arxiv_papers_*.json

# Run complete pipeline
python3 run_pipeline.py
```

### Customize Configuration

Edit `pipeline/config.yaml` to:
- Add/remove arXiv categories
- Modify relevance keywords
- Adjust processing limits
- Change scoring thresholds

### Monitor Pipeline

Check pipeline status:
1. Go to [Actions tab](https://github.com/aicoe-code/home/actions)
2. View workflow runs and logs
3. Check for any errors or warnings

## 📁 Repository Structure

```
home/
├── pipeline/               # Core processing modules
│   ├── fetcher.py         # arXiv paper fetcher
│   ├── processor.py       # Claude AI processor
│   ├── generator.py       # HTML generator
│   ├── simple_generator.py # Simple HTML generator (no deps)
│   ├── run_pipeline.py    # Main orchestrator
│   └── config.yaml        # Configuration
├── templates/             # Summary templates
├── data/                  # Paper data storage
│   ├── raw/              # Fetched papers
│   ├── processed/        # AI summaries
│   └── archive/          # Old papers
├── docs/                  # GitHub Pages website
│   ├── index.html        # Main page
│   └── assets/           # CSS files
└── .github/workflows/     # GitHub Actions
    └── daily-ai-papers.yml
```

## 🐛 Troubleshooting

### Website not updating
- Check [Actions tab](https://github.com/aicoe-code/home/actions) for errors
- Verify GitHub Pages is enabled
- Wait 5-10 minutes for deployment

### No papers found
- Check if arXiv API is accessible
- Verify date parameter (no papers for future dates)
- Check workflow logs for specific errors

### Claude AI not working
- Verify API key is set correctly in secrets
- Check API key hasn't expired
- Monitor usage limits on Anthropic Console

### GitHub Actions failing
- Check secrets are configured
- Review workflow logs
- Ensure branch permissions are correct

## 📈 Current Statistics

- **Papers in database**: 332
- **High relevance papers**: 77
- **Categories covered**: 9
- **Last update**: 2025-09-05

## 🔗 Quick Links

- **Repository**: https://github.com/aicoe-code/home
- **Website**: https://aicoe-code.github.io/home/
- **Actions**: https://github.com/aicoe-code/home/actions
- **Settings**: https://github.com/aicoe-code/home/settings
- **Secrets**: https://github.com/aicoe-code/home/settings/secrets/actions
- **Pages Settings**: https://github.com/aicoe-code/home/settings/pages

## 💡 Tips

1. **Monitor daily runs**: Check Actions tab regularly
2. **Review high-relevance papers**: Focus on score > 5
3. **Customize keywords**: Add domain-specific terms
4. **Archive old data**: Pipeline auto-archives after 30 days
5. **Test changes locally**: Before pushing to GitHub

## 📝 Notes

- The pipeline is configured to handle up to 50 papers per run
- Deduplication prevents reprocessing of same papers
- Papers are archived after 30 days automatically
- The website updates within minutes of pipeline completion

---

*Setup completed on: 2025-09-05*
*Pipeline version: 1.0*
*For issues or questions, open a GitHub issue*