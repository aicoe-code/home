#!/bin/bash

# PDF Research Pipeline Script
# Downloads PDF, creates research summary, and generates HTML

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEMPLATE_DIR="/Users/srinivaskarri/Desktop/aicoe-home/home/Private/02-templates"
RESEARCH_DIR="/Users/srinivaskarri/Desktop/aicoe-home/home/Private/03-research"
HTML_OUTPUT_DIR="/Users/srinivaskarri/Desktop/aicoe-home/home/gh-pages/research"
CONVERT_SCRIPT="${HTML_OUTPUT_DIR}/scripts/simple_convert.py"

# Function to display usage
usage() {
    echo -e "${YELLOW}PDF Research Pipeline${NC}"
    echo "====================="
    echo ""
    echo "Usage: $0 <pdf_url> <paper_title>"
    echo ""
    echo "Examples:"
    echo '  ./pdf_research_pipeline.sh "https://arxiv.org/pdf/2309.12345.pdf" "Attention Is All You Need"'
    echo '  ./pdf_research_pipeline.sh "https://example.com/paper.pdf" "Novel AI Architecture"'
    echo ""
    echo "This script will:"
    echo "  1. Download the PDF"
    echo "  2. Create a research summary template"
    echo "  3. Generate the HTML page"
    exit 1
}

# Function to sanitize filename
sanitize_filename() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g' | sed 's/-\+/-/g' | sed 's/^-//' | sed 's/-$//'
}

# Check arguments
if [ $# -lt 2 ]; then
    usage
fi

PDF_URL="$1"
PAPER_TITLE="$2"

echo -e "${BLUE}🚀 Starting PDF Research Pipeline${NC}"
echo "=================================="
echo -e "📄 Paper: ${GREEN}${PAPER_TITLE}${NC}"
echo -e "🔗 URL: ${BLUE}${PDF_URL}${NC}"
echo ""

# Generate filenames
DATE_PREFIX=$(date +%Y-%m-%d)
SANITIZED_NAME=$(sanitize_filename "$PAPER_TITLE")
MARKDOWN_FILE="${DATE_PREFIX}-${SANITIZED_NAME}.md"
HTML_FILE="${DATE_PREFIX}-${SANITIZED_NAME}.html"
MARKDOWN_PATH="${RESEARCH_DIR}/${MARKDOWN_FILE}"
HTML_PATH="${HTML_OUTPUT_DIR}/${HTML_FILE}"

# Create directories if they don't exist
mkdir -p "$RESEARCH_DIR"
mkdir -p "$HTML_OUTPUT_DIR"

# Step 1: Download PDF
echo -e "${YELLOW}Step 1: Downloading PDF...${NC}"
TEMP_PDF="/tmp/research_paper_${DATE_PREFIX}.pdf"
if curl -L -o "$TEMP_PDF" "$PDF_URL" 2>/dev/null; then
    echo -e "${GREEN}✅ PDF downloaded successfully${NC}"
    echo "   Size: $(ls -lh "$TEMP_PDF" | awk '{print $5}')"
else
    echo -e "${RED}❌ Failed to download PDF${NC}"
    exit 1
fi

# Step 2: Create research markdown using template
echo -e "\n${YELLOW}Step 2: Creating research summary...${NC}"

# Use the Python script to create the file with proper naming
cd "$TEMPLATE_DIR"
if python3 create_research_file.py "$PAPER_TITLE" "$DATE_PREFIX" >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Research markdown created${NC}"
else
    # Fallback: Create manually
    cp "${TEMPLATE_DIR}/academic-research-summary-template.md" "$MARKDOWN_PATH"
    
    # Update the template with basic information
    sed -i '' "s/title: \"\"/title: \"$PAPER_TITLE\"/" "$MARKDOWN_PATH"
    sed -i '' "s/created_date: \"[^\"]*\"/created_date: \"$DATE_PREFIX\"/" "$MARKDOWN_PATH"
    sed -i '' "s/last_modified: \"[^\"]*\"/last_modified: \"$DATE_PREFIX\"/" "$MARKDOWN_PATH"
    sed -i '' "s/review_date: \"\"/review_date: \"$DATE_PREFIX\"/" "$MARKDOWN_PATH"
    
    # Extract arXiv ID if present
    if [[ $PDF_URL == *"arxiv.org"* ]]; then
        ARXIV_ID=$(echo "$PDF_URL" | grep -oE '[0-9]+\.[0-9]+')
        sed -i '' "s/arxiv_id: \"\"/arxiv_id: \"$ARXIV_ID\"/" "$MARKDOWN_PATH"
    fi
    
    echo -e "${GREEN}✅ Research markdown created${NC}"
fi

echo "   File: $MARKDOWN_PATH"

# Add a note about the PDF source
echo -e "\n## Source Document\n\nThis research summary is based on the PDF available at: [$PDF_URL]($PDF_URL)\n\n## Executive Summary\n\n[To be completed after reviewing the paper]\n" >> "$MARKDOWN_PATH"

# Step 3: Open the markdown file for editing (optional)
echo -e "\n${YELLOW}Step 3: Review and Edit${NC}"
echo -e "The markdown template has been created at:"
echo -e "  ${BLUE}${MARKDOWN_PATH}${NC}"
echo ""
echo -e "${YELLOW}Please edit this file to add:${NC}"
echo "  • Executive summary"
echo "  • Key contributions"
echo "  • Methodology details"
echo "  • Findings and results"
echo "  • AICOE alignment analysis"
echo ""
read -p "Press Enter to continue to HTML generation (or Ctrl+C to stop and edit first)..."

# Step 4: Convert to HTML
echo -e "\n${YELLOW}Step 4: Generating HTML...${NC}"

# Check if conversion script exists
if [ -f "$CONVERT_SCRIPT" ]; then
    cd "$(dirname "$CONVERT_SCRIPT")"
    if python3 "$(basename "$CONVERT_SCRIPT")" 2>/dev/null; then
        echo -e "${GREEN}✅ HTML generated successfully${NC}"
    else
        echo -e "${YELLOW}⚠️  HTML generation needs manual intervention${NC}"
        echo "Run: python3 $CONVERT_SCRIPT"
    fi
else
    # Fallback: Create basic HTML
    cat > "$HTML_PATH" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${PAPER_TITLE} - AICOE</title>
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
            <h1 class="research-title">${PAPER_TITLE}</h1>
            <div class="paper-meta">
                <span class="meta-item">📅 ${DATE_PREFIX}</span>
                <span class="meta-item">🔗 <a href="${PDF_URL}" target="_blank">View PDF</a></span>
            </div>
        </header>
        
        <main class="research-content">
            <h2>Document Processing</h2>
            <p>This research summary is being prepared. Please check back later or view the <a href="${PDF_URL}">original PDF</a>.</p>
        </main>
        
        <footer class="research-footer">
            <div class="navigation-links">
                <a href="index.html" class="btn btn-secondary">← Back to Research</a>
            </div>
        </footer>
    </div>
</body>
</html>
EOF
    echo -e "${GREEN}✅ Basic HTML template created${NC}"
fi

# Clean up
rm -f "$TEMP_PDF"

# Summary
echo ""
echo -e "${GREEN}🎉 Pipeline completed!${NC}"
echo "========================"
echo -e "📝 Markdown: ${BLUE}${MARKDOWN_PATH}${NC}"
echo -e "🌐 HTML: ${BLUE}${HTML_PATH}${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Edit the markdown file to complete the research summary"
echo "2. Update metadata and scores in the frontmatter"
echo "3. Re-run the HTML conversion after editing:"
echo "   cd ${HTML_OUTPUT_DIR}/scripts"
echo "   python3 simple_convert.py"
echo ""
echo -e "${GREEN}The files are ready for your review!${NC}"