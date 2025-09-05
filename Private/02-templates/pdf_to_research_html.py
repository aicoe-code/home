#!/usr/bin/env python3
"""
PDF to Research HTML Pipeline
Downloads a PDF, creates a research summary using AI, and generates an HTML page

Usage:
    python pdf_to_research_html.py <pdf_url> <paper_title>
    python pdf_to_research_html.py "https://arxiv.org/pdf/2309.12345.pdf" "Transformer Architecture Study"
"""

import os
import sys
import json
import re
import requests
import subprocess
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

# Try to import PDF processing libraries
try:
    import PyPDF2
    PDF_READER_AVAILABLE = True
except ImportError:
    PDF_READER_AVAILABLE = False
    print("Warning: PyPDF2 not installed. Install with: pip install PyPDF2")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Note: Anthropic API not available. Install with: pip install anthropic")

# Configuration paths
TEMPLATE_DIR = "/Users/srinivaskarri/Desktop/aicoe-home/home/Private/02-templates"
RESEARCH_DIR = "/Users/srinivaskarri/Desktop/aicoe-home/home/Private/03-research"
HTML_OUTPUT_DIR = "/Users/srinivaskarri/Desktop/aicoe-home/home/gh-pages/research"
TEMPLATE_PATH = f"{TEMPLATE_DIR}/academic-research-summary-template.md"
HTML_TEMPLATE_PATH = f"{TEMPLATE_DIR}/research-html-template.html"

def sanitize_filename(name):
    """Convert a name to a valid filename format"""
    name = re.sub(r'[^a-zA-Z0-9\s\-]', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.replace(' ', '-')
    name = name.lower()
    name = re.sub(r'-+', '-', name)
    name = name.strip('-')
    return name

def download_pdf(url, output_path):
    """Download PDF from URL"""
    print(f"📥 Downloading PDF from: {url}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✅ PDF downloaded successfully: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading PDF: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF"""
    if not PDF_READER_AVAILABLE:
        print("⚠️  PyPDF2 not available. Please install it or use manual extraction.")
        return None
    
    try:
        print(f"📄 Extracting text from PDF...")
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(f"   Found {num_pages} pages")
            
            # Extract text from first 10 pages for summary (to avoid token limits)
            pages_to_extract = min(10, num_pages)
            for page_num in range(pages_to_extract):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        print(f"✅ Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return None

def create_research_summary_prompt(paper_text, paper_title):
    """Create a prompt for AI to generate research summary"""
    prompt = f"""
Based on the following research paper, create a comprehensive research summary following the AICOE framework.

Paper Title: {paper_title}

Paper Content (excerpt):
{paper_text[:8000]}  # Limit to avoid token limits

Please provide a detailed analysis including:

1. **Executive Summary** (2-3 paragraphs)
2. **Key Contributions** (bullet points)
3. **Methodology** (if applicable)
4. **Key Findings/Results**
5. **Strengths** of the paper
6. **Limitations** and areas for improvement
7. **Relevance to AICOE Mission** (Understanding, Design, Deployment, Operation)
8. **Practical Applications**
9. **Future Research Directions**
10. **Critical Questions**

Format the response in Markdown with clear sections and subsections.
Include relevance scores (1-10) for:
- Overall Relevance to AICOE
- Reproducibility
- Potential Impact

Identify which AICOE areas this paper aligns with:
- Understanding AI systems
- Designing AI solutions
- Deploying AI systems
- Operating AI infrastructure
"""
    return prompt

def generate_summary_with_ai(paper_text, paper_title):
    """Use AI to generate research summary (placeholder for actual AI integration)"""
    # This is a placeholder function
    # In production, you would integrate with an AI API (OpenAI, Anthropic, etc.)
    print("🤖 Generating AI summary...")
    
    # For now, return a template summary
    summary_template = f"""
## Executive Summary

This paper presents research on {paper_title}. [AI-generated summary would go here]

## Key Contributions

- Contribution 1 related to {paper_title}
- Contribution 2 with novel approach
- Contribution 3 improving existing methods

## Methodology

The research employs [methodology details would be extracted from paper]

## Key Findings

1. Finding 1 with significant implications
2. Finding 2 demonstrating effectiveness
3. Finding 3 showing improvements

## Strengths

- Clear presentation of concepts
- Rigorous experimental validation
- Novel approach to problem

## Limitations

- Limited scope in certain areas
- Requires further validation
- Computational requirements

## AICOE Relevance

This research aligns with AICOE's mission in:
- **Understanding**: Provides insights into AI systems
- **Design**: Offers new design patterns
- **Deployment**: Includes deployment considerations
- **Operation**: Discusses operational aspects

## Practical Applications

- Application in industry settings
- Integration with existing systems
- Potential for automation

## Future Research Directions

- Extension to other domains
- Scaling considerations
- Integration opportunities

## Critical Questions

1. How does this compare to existing approaches?
2. What are the real-world implications?
3. How can this be integrated with current systems?
"""
    
    return summary_template

def create_research_markdown(paper_title, pdf_url, summary_content, output_path):
    """Create the research markdown file with proper formatting"""
    date_prefix = datetime.now().strftime('%Y-%m-%d')
    
    # Extract arXiv ID if present
    arxiv_match = re.search(r'arxiv\.org/pdf/(\d+\.\d+)', pdf_url.lower())
    arxiv_id = arxiv_match.group(1) if arxiv_match else ""
    
    # Read template
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Update template with actual content
    template = re.sub(r'title:\s*"[^"]*"', f'title: "{paper_title}"', template, count=1)
    template = re.sub(r'created_date:\s*"[^"]*"', f'created_date: "{date_prefix}"', template)
    template = re.sub(r'last_modified:\s*"[^"]*"', f'last_modified: "{date_prefix}"', template)
    template = re.sub(r'review_date:\s*"[^"]*"', f'review_date: "{date_prefix}"', template)
    template = re.sub(r'(paper:\s*\n\s*title:\s*)"[^"]*"', rf'\1"{paper_title}"', template)
    
    if arxiv_id:
        template = re.sub(r'arxiv_id:\s*"[^"]*"', f'arxiv_id: "{arxiv_id}"', template)
    
    # Add the AI-generated summary content
    # Find the end of frontmatter and insert content
    frontmatter_end = template.find('---', 3)
    if frontmatter_end != -1:
        frontmatter_end = template.find('\n', frontmatter_end) + 1
        template = template[:frontmatter_end] + "\n" + summary_content + "\n"
    
    # Write the file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"✅ Created research markdown: {output_path}")
    return True

def convert_markdown_to_html(markdown_path, html_output_path):
    """Convert markdown to HTML using existing conversion scripts"""
    # Check which conversion script is available
    convert_py = f"{HTML_OUTPUT_DIR}/scripts/convert.py"
    convert_js = f"{HTML_OUTPUT_DIR}/scripts/convert-md-to-html.js"
    simple_convert = f"{HTML_OUTPUT_DIR}/scripts/simple_convert.py"
    
    if os.path.exists(convert_py):
        print(f"🔄 Converting to HTML using Python converter...")
        # Modify the convert.py to accept input/output paths
        cmd = f"python3 {convert_py}"
        # We'll need to temporarily modify the script or create a wrapper
        # For now, we'll use a simple HTML generation
    elif os.path.exists(convert_js):
        print(f"🔄 Converting to HTML using Node.js converter...")
        cmd = f"node {convert_js} {markdown_path} {html_output_path}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"✅ HTML generated: {html_output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error converting to HTML: {e}")
    
    # Fallback: Simple HTML generation
    print(f"🔄 Generating HTML with simple converter...")
    return generate_simple_html(markdown_path, html_output_path)

def generate_simple_html(markdown_path, html_output_path):
    """Generate a simple HTML file from markdown"""
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse frontmatter
    if content.startswith('---'):
        frontmatter_end = content.find('---', 3)
        if frontmatter_end != -1:
            frontmatter = content[3:frontmatter_end]
            markdown_content = content[frontmatter_end+3:].strip()
            
            # Extract metadata
            title_match = re.search(r'title:\s*"([^"]*)"', frontmatter)
            title = title_match.group(1) if title_match else "Research Summary"
        else:
            title = "Research Summary"
            markdown_content = content
    else:
        title = "Research Summary"
        markdown_content = content
    
    # Simple markdown to HTML conversion
    html_content = markdown_content.replace('\n\n', '</p><p>')
    html_content = '<p>' + html_content + '</p>'
    html_content = re.sub(r'## (.*?)\n', r'<h2>\1</h2>\n', html_content)
    html_content = re.sub(r'# (.*?)\n', r'<h1>\1</h1>\n', html_content)
    html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)
    html_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_content)
    html_content = re.sub(r'- (.*?)\n', r'<li>\1</li>\n', html_content)
    html_content = re.sub(r'(<li>.*</li>\n)+', r'<ul>\g<0></ul>', html_content)
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - AICOE</title>
    <link rel="stylesheet" href="../styles.css">
</head>
<body>
    <div class="container">
        <nav class="breadcrumb">
            <a href="../index.html">Home</a> / 
            <a href="index.html">Research</a> / 
            <span>Current</span>
        </nav>
        
        <header class="research-header">
            <h1 class="research-title">{title}</h1>
        </header>
        
        <main class="research-content">
            {html_content}
        </main>
        
        <footer class="research-footer">
            <div class="navigation-links">
                <a href="index.html" class="btn btn-secondary">← Back to Research</a>
            </div>
        </footer>
    </div>
</body>
</html>"""
    
    with open(html_output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML generated: {html_output_path}")
    return True

def main(pdf_url, paper_title):
    """Main pipeline function"""
    print(f"\n🚀 Starting PDF to Research HTML Pipeline")
    print(f"=" * 50)
    print(f"📄 Paper: {paper_title}")
    print(f"🔗 URL: {pdf_url}")
    print(f"=" * 50)
    
    # Create directories if they don't exist
    os.makedirs(RESEARCH_DIR, exist_ok=True)
    os.makedirs(HTML_OUTPUT_DIR, exist_ok=True)
    
    # Generate filenames
    date_prefix = datetime.now().strftime('%Y-%m-%d')
    sanitized_name = sanitize_filename(paper_title)
    markdown_filename = f"{date_prefix}-{sanitized_name}.md"
    html_filename = f"{date_prefix}-{sanitized_name}.html"
    markdown_path = os.path.join(RESEARCH_DIR, markdown_filename)
    html_path = os.path.join(HTML_OUTPUT_DIR, html_filename)
    
    # Step 1: Download PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
        pdf_path = tmp_pdf.name
    
    if not download_pdf(pdf_url, pdf_path):
        print("Failed to download PDF. Exiting.")
        return False
    
    # Step 2: Extract text from PDF
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text:
        print("⚠️  Could not extract text from PDF. Creating template file instead.")
        paper_text = f"[PDF content from {pdf_url}]"
    
    # Step 3: Generate AI summary
    summary_content = generate_summary_with_ai(paper_text, paper_title)
    
    # Step 4: Create research markdown file
    if not create_research_markdown(paper_title, pdf_url, summary_content, markdown_path):
        print("Failed to create markdown file. Exiting.")
        return False
    
    # Step 5: Convert to HTML
    if not convert_markdown_to_html(markdown_path, html_path):
        print("Failed to generate HTML. Exiting.")
        return False
    
    # Clean up temporary PDF
    try:
        os.remove(pdf_path)
    except:
        pass
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📝 Markdown: {markdown_path}")
    print(f"🌐 HTML: {html_path}")
    print(f"\n📋 Next steps:")
    print(f"1. Review and edit the markdown file to add more details")
    print(f"2. Update scores and metadata in the frontmatter")
    print(f"3. Re-run HTML conversion if you make changes")
    print(f"4. The HTML is ready to view in the browser")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("PDF to Research HTML Pipeline")
        print("=" * 50)
        print("\nUsage:")
        print("  python pdf_to_research_html.py <pdf_url> <paper_title>")
        print("\nExamples:")
        print('  python pdf_to_research_html.py "https://arxiv.org/pdf/2309.12345.pdf" "Attention Is All You Need"')
        print('  python pdf_to_research_html.py "https://example.com/paper.pdf" "Novel AI Architecture"')
        print("\nRequirements:")
        print("  - PyPDF2: pip install PyPDF2")
        print("  - requests: pip install requests")
        print("  - (Optional) anthropic: pip install anthropic")
        print("\nThe script will:")
        print("  1. Download the PDF")
        print("  2. Extract text content")
        print("  3. Generate a research summary")
        print("  4. Create a markdown file in /Private/03-research/")
        print("  5. Convert to HTML in /gh-pages/research/")
        sys.exit(1)
    
    pdf_url = sys.argv[1]
    paper_title = sys.argv[2]
    
    main(pdf_url, paper_title)