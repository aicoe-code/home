# GitHub Actions Workflows

## fetch-arxiv-daily.yml

This workflow automatically fetches AI-related papers from arXiv on a daily basis.

### Schedule
- **Automatic**: Runs daily at 7:00 AM UTC (2 AM EST / 3 AM EDT)
- **Manual**: Can be triggered manually from the Actions tab with custom parameters

### Features

#### Automatic Daily Fetch
- Fetches yesterday's papers from 9 AI categories
- Generates JSON, HTML, and Markdown outputs
- Commits results to the `gh-pages` branch
- Saves to `gh-pages/research/daily/` directory

#### Manual Trigger Options
When triggering manually, you can specify:
- **Date**: Specific date to fetch papers (YYYY-MM-DD format)
- **Max Results**: Maximum papers per category (default: 100)

### Workflow Steps

1. **Checkout Repository**: Fetches the gh-pages branch
2. **Setup Python**: Installs Python 3.11
3. **Determine Date**: Calculates yesterday's date or uses manual input
4. **Fetch Papers**: Runs the arxiv_daily_ai.py script
5. **Check Changes**: Detects if new papers were found
6. **Commit & Push**: Commits new files with descriptive message
7. **Create Summary**: Generates a summary in the Actions tab

### Monitored Categories

- cs.AI (Artificial Intelligence)
- cs.LG (Machine Learning)
- cs.CV (Computer Vision)
- cs.CL (Computation and Language)
- cs.NE (Neural and Evolutionary Computing)
- stat.ML (Machine Learning - Statistics)
- cs.RO (Robotics)
- cs.HC (Human-Computer Interaction)
- cs.IR (Information Retrieval)

### Output Files

For each run, the workflow generates:
- `arxiv-ai-papers-YYYY-MM-DD.json` - Structured data
- `arxiv-ai-papers-YYYY-MM-DD.html` - Styled webpage
- `arxiv-ai-papers-YYYY-MM-DD.md` - Markdown format

### Error Handling

The workflow includes:
- Error detection and reporting
- Summary generation even on failure
- Common issue troubleshooting in the summary

### Permissions

The workflow requires:
- `contents: write` - To commit and push changes

### Manual Trigger

To manually trigger the workflow:

1. Go to the Actions tab in your repository
2. Select "Fetch Daily arXiv AI Papers"
3. Click "Run workflow"
4. Optionally specify:
   - Date (leave empty for yesterday)
   - Max results per category
5. Click "Run workflow" button

### Viewing Results

After the workflow runs:
1. Check the Actions tab for the run summary
2. Browse to `https://[your-github-pages-url]/research/daily/`
3. Or check the `gh-pages/research/daily/` directory

### Troubleshooting

**No papers found:**
- arXiv may not have papers for the specified date
- Check if the date is in the future
- Verify arXiv API is accessible

**Workflow fails:**
- Check the Actions logs for specific errors
- Verify Python script syntax
- Ensure gh-pages branch exists

**Changes not appearing:**
- Clear browser cache
- Wait a few minutes for GitHub Pages to update
- Check if commits were pushed successfully

### Customization

To modify the workflow:

1. **Change schedule**: Edit the cron expression
   ```yaml
   - cron: '0 7 * * *'  # Current: 7 AM UTC daily
   ```

2. **Adjust categories**: Modify the script call
   ```yaml
   python3 arxiv_daily_ai.py --categories cs.AI cs.LG
   ```

3. **Change output format**: Modify the --format parameter
   ```yaml
   --format html  # Only HTML
   --format json  # Only JSON
   ```

### Monitoring

The workflow provides:
- Run history in the Actions tab
- Email notifications (if configured in GitHub settings)
- Detailed logs for each step
- Summary report after each run

### Cost Considerations

- GitHub Actions provides 2,000 free minutes/month for public repos
- Each run takes approximately 1-2 minutes
- Daily runs use about 30-60 minutes/month
- Well within free tier limits