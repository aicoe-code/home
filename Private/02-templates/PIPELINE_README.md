# PDF to Research HTML Pipeline

## Overview

This pipeline automates the process of converting research PDFs into formatted HTML pages for the AICOE research repository.

## Available Scripts

### 1. `pdf_research_pipeline.sh` (Recommended for Quick Use)
Simple bash script that handles the complete workflow.

### 2. `pdf_to_research_html.py` (Advanced)
Python script with PDF text extraction and AI integration capabilities.

## Installation

### Basic Requirements
```bash
# No additional requirements for the bash script
# Uses curl for downloading and standard Unix tools
```

### Advanced Requirements (for Python script)
```bash
pip install PyPDF2 requests
# Optional: pip install anthropic  # For AI summaries
```

## Usage

### Quick Start (Bash Script)

```bash
# Navigate to templates directory
cd /Users/srinivaskarri/Desktop/aicoe-home/home/Private/02-templates

# Run the pipeline
./pdf_research_pipeline.sh "https://arxiv.org/pdf/2309.12345.pdf" "Paper Title"

# Real example
./pdf_research_pipeline.sh "https://arxiv.org/pdf/1706.03762.pdf" "Attention Is All You Need"
```

### Advanced Usage (Python Script)

```bash
# With PDF text extraction
python3 pdf_to_research_html.py "https://arxiv.org/pdf/2309.12345.pdf" "Paper Title"

# The Python script can extract text from PDFs and generate summaries
```

## Workflow Steps

1. **Download PDF**: Downloads the PDF from the provided URL
2. **Create Research File**: 
   - Uses the template to create a markdown file
   - Adds proper date prefix (YYYY-MM-DD)
   - Sanitizes the filename
3. **Generate HTML**: Converts the markdown to HTML for web display

## Output Files

The pipeline creates two files:

1. **Markdown Research Summary**
   - Location: `/Private/03-research/YYYY-MM-DD-paper-name.md`
   - Contains structured research analysis
   - Editable for adding detailed insights

2. **HTML Page**
   - Location: `/gh-pages/research/YYYY-MM-DD-paper-name.html`
   - Ready for web deployment
   - Styled with the AICOE design system

## Examples

### Example 1: ArXiv Paper
```bash
./pdf_research_pipeline.sh \
  "https://arxiv.org/pdf/1706.03762.pdf" \
  "Attention Is All You Need"
```

Creates:
- `/Private/03-research/2025-09-05-attention-is-all-you-need.md`
- `/gh-pages/research/2025-09-05-attention-is-all-you-need.html`

### Example 2: Conference Paper
```bash
./pdf_research_pipeline.sh \
  "https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf" \
  "GPT-3 Language Models Are Few-Shot Learners"
```

### Example 3: Direct PDF Link
```bash
./pdf_research_pipeline.sh \
  "https://example.com/research/paper.pdf" \
  "Novel Machine Learning Approach"
```

## Post-Processing

After running the pipeline:

1. **Edit the Markdown File**
   ```bash
   # Open in your preferred editor
   code /Private/03-research/YYYY-MM-DD-paper-name.md
   ```

2. **Add Research Details**
   - Executive summary
   - Key contributions
   - Methodology
   - Results and findings
   - AICOE alignment analysis
   - Practical applications

3. **Update Metadata**
   - Author information
   - Publication details
   - Relevance scores
   - Priority level

4. **Regenerate HTML** (if needed)
   ```bash
   cd /gh-pages/research/scripts
   python3 simple_convert.py
   ```

## Troubleshooting

### PDF Download Fails
- Check the URL is accessible
- Try using a different PDF link
- Download manually and modify the script

### HTML Generation Issues
- Ensure the markdown file has valid frontmatter
- Check that all required templates exist
- Run the conversion script manually

### Missing Dependencies
```bash
# Install Python dependencies
pip install PyPDF2 requests

# Check Python version (3.6+ required)
python3 --version
```

## Advanced Features

### PDF Text Extraction (Python script only)
The Python script can extract text from PDFs for analysis:
```python
python3 pdf_to_research_html.py "url" "title"
```

### AI Summary Generation (Future Enhancement)
The Python script has placeholders for AI integration:
- OpenAI API
- Anthropic Claude
- Local LLMs

To enable, add your API keys and uncomment the relevant sections.

## File Naming Convention

All files follow this pattern:
```
YYYY-MM-DD-sanitized-paper-title.{md|html}
```

- Date prefix ensures chronological ordering
- Sanitized names are lowercase with hyphens
- Special characters are removed

## Integration with Existing Tools

This pipeline integrates with:
- `create_research_file.py`: For proper file naming
- `simple_convert.py`: For HTML generation
- AICOE templates: For consistent formatting

## Support

For issues or improvements:
1. Check this README first
2. Review the script comments
3. Ensure all paths are correct
4. Verify template files exist