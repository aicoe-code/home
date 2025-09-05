#!/usr/bin/env python3
"""
Simple HTML Generator (no external dependencies)
"""

import json
import re
from datetime import datetime
from pathlib import Path
from html import escape

class SimpleHTMLGenerator:
    """Generate HTML without markdown dependency"""
    
    def __init__(self):
        self.output_dir = Path('docs')
        self.papers_dir = self.output_dir / 'papers'
        self.assets_dir = self.output_dir / 'assets'
        
        # Create directories
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
    
    def markdown_to_html(self, text):
        """Simple markdown to HTML conversion"""
        # Convert headers
        text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
        
        # Convert bold and italic
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        
        # Convert lists
        text = re.sub(r'^- (.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
        text = re.sub(r'(<li>.*?</li>\n)+', r'<ul>\g<0></ul>', text, flags=re.DOTALL)
        
        # Convert paragraphs
        paragraphs = text.split('\n\n')
        text = '\n'.join([f'<p>{p}</p>' if not p.startswith('<') else p for p in paragraphs])
        
        return text
    
    def generate_from_raw_papers(self, papers_file):
        """Generate HTML from raw papers JSON"""
        with open(papers_file, 'r') as f:
            data = json.load(f)
            papers = data.get('papers', [])
        
        print(f"Generating HTML for {len(papers)} papers...")
        
        # Generate CSS
        self._generate_css()
        
        # Generate index
        self._generate_index(papers)
        
        print(f"✅ HTML generated in {self.output_dir}")
    
    def _generate_css(self):
        """Generate CSS file"""
        css = """/* Simple Clean Styles */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: #f9f9f9;
}

h1, h2, h3 { color: #111; }
h1 { font-size: 2rem; margin-bottom: 1rem; }
h2 { font-size: 1.5rem; margin-top: 2rem; }
h3 { font-size: 1.2rem; margin-top: 1.5rem; }

.header {
    background: white;
    padding: 2rem;
    margin-bottom: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.stat-card {
    background: white;
    padding: 1rem;
    text-align: center;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.stat-number {
    font-size: 2rem;
    font-weight: bold;
    color: #2563eb;
}

.paper-card {
    background: white;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.paper-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.paper-meta {
    color: #666;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

.paper-summary {
    color: #555;
    margin-bottom: 1rem;
}

.paper-links a {
    color: #2563eb;
    text-decoration: none;
    margin-right: 1rem;
}

.paper-links a:hover {
    text-decoration: underline;
}

.category-tag {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    background: #e5e7eb;
    border-radius: 4px;
    font-size: 0.8rem;
    margin-right: 0.5rem;
}

.relevance-high { background: #fee2e2; color: #dc2626; }
.relevance-medium { background: #fef3c7; color: #d97706; }
.relevance-low { background: #dbeafe; color: #2563eb; }"""
        
        css_file = self.assets_dir / 'styles.css'
        with open(css_file, 'w') as f:
            f.write(css)
        print(f"  📝 Generated {css_file}")
    
    def _generate_index(self, papers):
        """Generate index page"""
        # Sort by relevance
        papers_sorted = sorted(papers, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Stats
        total = len(papers)
        high_rel = len([p for p in papers if p.get('relevance_score', 0) > 5])
        categories = set()
        for p in papers:
            categories.update(p.get('categories', []))
        
        # Generate paper cards
        papers_html = ""
        for paper in papers_sorted[:50]:  # Show top 50
            # Relevance badge
            score = paper.get('relevance_score', 0)
            if score > 5:
                rel_class, rel_text = 'relevance-high', 'High'
            elif score > 2:
                rel_class, rel_text = 'relevance-medium', 'Medium'
            else:
                rel_class, rel_text = 'relevance-low', 'Low'
            
            # Authors
            authors = paper.get('authors', [])[:3]
            if len(paper.get('authors', [])) > 3:
                authors.append('et al.')
            
            # Summary
            summary = paper.get('summary', '')[:300] + '...'
            
            # Categories
            cats = ' '.join([f'<span class="category-tag">{c}</span>' for c in paper.get('categories', [])[:5]])
            
            papers_html += f"""
        <div class="paper-card">
            <h3 class="paper-title">{escape(paper.get('title', 'Untitled'))}</h3>
            <div class="paper-meta">
                {escape(', '.join(authors))} | 
                {paper.get('published', '')[:10]} | 
                <span class="category-tag {rel_class}">{rel_text} ({score:.1f})</span>
            </div>
            <div>{cats}</div>
            <p class="paper-summary">{escape(summary)}</p>
            <div class="paper-links">
                <a href="{paper.get('arxiv_url', '#')}" target="_blank">📄 arXiv</a>
                <a href="{paper.get('pdf_url', '#')}" target="_blank">📥 PDF</a>
            </div>
        </div>"""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AICOE Research Library</title>
    <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
    <div class="header">
        <h1>AICOE Research Library</h1>
        <p>AI Papers - Daily Collection and Analysis</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-number">{total}</div>
            <div>Total Papers</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{high_rel}</div>
            <div>High Relevance</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(categories)}</div>
            <div>Categories</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{datetime.now().strftime('%b %d')}</div>
            <div>Last Updated</div>
        </div>
    </div>
    
    <h2>Recent Papers (Top 50 by Relevance)</h2>
    
    {papers_html}
    
    <footer style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #ddd; text-align: center; color: #666;">
        <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} | AICOE Research Library</p>
    </footer>
</body>
</html>"""
        
        index_file = self.output_dir / 'index.html'
        with open(index_file, 'w') as f:
            f.write(html)
        
        print(f"  📝 Generated {index_file}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        papers_file = sys.argv[1]
    else:
        # Try to find the most recent papers file
        raw_files = list(Path('data/raw').glob('*.json'))
        if raw_files:
            papers_file = str(raw_files[-1])
        else:
            print("No papers file found. Please provide a path.")
            sys.exit(1)
    
    generator = SimpleHTMLGenerator()
    generator.generate_from_raw_papers(papers_file)