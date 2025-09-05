#!/usr/bin/env python3
"""
HTML Generator for AI Paper Summaries
Generates clean, professional HTML pages from processed paper summaries
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from html import escape
import markdown

class HTMLGenerator:
    """Generate HTML pages for paper summaries"""
    
    def __init__(self, config: Dict = None):
        """Initialize generator with configuration"""
        self.config = config or self._default_config()
        self.output_dir = Path(self.config['output_dir'])
        self.papers_dir = self.output_dir / 'papers'
        self.assets_dir = self.output_dir / 'assets'
        
        # Create directories
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'output_dir': 'docs',
            'site_title': 'AICOE Research Library',
            'site_description': 'Daily AI Papers - Summaries and Analysis',
            'base_url': '',
            'papers_per_page': 20,
            'generate_rss': True
        }
    
    def generate_site(self, processed_papers: List[Dict]):
        """Generate complete website from processed papers"""
        print("🌐 Generating website...")
        
        # Generate CSS
        self._generate_css()
        
        # Generate individual paper pages
        for paper in processed_papers:
            self._generate_paper_page(paper)
        
        # Generate index page
        self._generate_index_page(processed_papers)
        
        # Generate archive page
        self._generate_archive_page(processed_papers)
        
        # Generate RSS feed
        if self.config['generate_rss']:
            self._generate_rss_feed(processed_papers)
        
        print(f"✅ Website generated in {self.output_dir}")
    
    def _generate_css(self):
        """Generate main CSS file"""
        css = """/* AICOE Research Library - Styles */
:root {
    --color-bg: #ffffff;
    --color-bg-subtle: #fafafa;
    --color-text: #09090b;
    --color-text-muted: #71717a;
    --color-border: #e4e4e7;
    --color-accent: #18181b;
    --color-link: #2563eb;
    --color-link-hover: #1d4ed8;
    
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: "SF Mono", Monaco, Consolas, monospace;
    
    --space-xs: 0.5rem;
    --space-sm: 0.75rem;
    --space-md: 1rem;
    --space-lg: 1.5rem;
    --space-xl: 2rem;
    --space-2xl: 3rem;
    
    --radius: 0.5rem;
    --max-width: 1200px;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: var(--font-sans);
    line-height: 1.6;
    color: var(--color-text);
    background: var(--color-bg);
}

.container {
    max-width: var(--max-width);
    margin: 0 auto;
    padding: var(--space-lg);
}

/* Header */
.header {
    border-bottom: 1px solid var(--color-border);
    margin-bottom: var(--space-2xl);
}

.header-content {
    max-width: var(--max-width);
    margin: 0 auto;
    padding: var(--space-xl) var(--space-lg);
}

.site-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: var(--space-xs);
}

.site-description {
    color: var(--color-text-muted);
}

/* Navigation */
.nav {
    display: flex;
    gap: var(--space-lg);
    margin-top: var(--space-md);
}

.nav a {
    color: var(--color-text);
    text-decoration: none;
    font-weight: 500;
    padding: var(--space-xs) 0;
    border-bottom: 2px solid transparent;
    transition: border-color 0.2s;
}

.nav a:hover,
.nav a.active {
    border-bottom-color: var(--color-accent);
}

/* Paper Cards */
.papers-grid {
    display: grid;
    gap: var(--space-lg);
}

.paper-card {
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    padding: var(--space-lg);
    transition: box-shadow 0.2s;
}

.paper-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.paper-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: var(--space-sm);
    line-height: 1.4;
}

.paper-title a {
    color: var(--color-text);
    text-decoration: none;
}

.paper-title a:hover {
    color: var(--color-link);
}

.paper-meta {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-md);
    margin-bottom: var(--space-md);
    font-size: 0.875rem;
    color: var(--color-text-muted);
}

.paper-authors {
    flex: 1;
}

.paper-date {
    white-space: nowrap;
}

.paper-categories {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-xs);
    margin-bottom: var(--space-md);
}

.category-tag {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    background: var(--color-bg-subtle);
    border: 1px solid var(--color-border);
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 500;
}

.paper-summary {
    color: var(--color-text-muted);
    line-height: 1.6;
    margin-bottom: var(--space-md);
}

.paper-links {
    display: flex;
    gap: var(--space-md);
}

.paper-links a {
    color: var(--color-link);
    text-decoration: none;
    font-weight: 500;
    font-size: 0.875rem;
}

.paper-links a:hover {
    color: var(--color-link-hover);
    text-decoration: underline;
}

/* Relevance Badge */
.relevance-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}

.relevance-high {
    background: #fee2e2;
    color: #dc2626;
}

.relevance-medium {
    background: #fef3c7;
    color: #d97706;
}

.relevance-low {
    background: #dbeafe;
    color: #2563eb;
}

/* Paper Page */
.paper-page {
    max-width: 900px;
    margin: 0 auto;
}

.paper-header {
    margin-bottom: var(--space-2xl);
    padding-bottom: var(--space-xl);
    border-bottom: 1px solid var(--color-border);
}

.paper-content {
    font-size: 1.0625rem;
    line-height: 1.8;
}

.paper-content h2 {
    margin-top: var(--space-2xl);
    margin-bottom: var(--space-md);
    font-size: 1.5rem;
    font-weight: 600;
}

.paper-content h3 {
    margin-top: var(--space-xl);
    margin-bottom: var(--space-sm);
    font-size: 1.25rem;
    font-weight: 600;
}

.paper-content p {
    margin-bottom: var(--space-md);
}

.paper-content ul,
.paper-content ol {
    margin-bottom: var(--space-md);
    padding-left: var(--space-lg);
}

.paper-content li {
    margin-bottom: var(--space-xs);
}

.paper-content blockquote {
    margin: var(--space-lg) 0;
    padding-left: var(--space-lg);
    border-left: 3px solid var(--color-border);
    color: var(--color-text-muted);
}

.paper-content code {
    font-family: var(--font-mono);
    font-size: 0.875em;
    padding: 0.125rem 0.25rem;
    background: var(--color-bg-subtle);
    border-radius: 0.25rem;
}

.paper-content pre {
    margin: var(--space-lg) 0;
    padding: var(--space-md);
    background: var(--color-bg-subtle);
    border-radius: var(--radius);
    overflow-x: auto;
}

.paper-content pre code {
    padding: 0;
    background: none;
}

/* Stats Section */
.stats-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: var(--space-lg);
    margin: var(--space-2xl) 0;
}

.stat-card {
    text-align: center;
    padding: var(--space-lg);
    background: var(--color-bg-subtle);
    border-radius: var(--radius);
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: var(--color-accent);
}

.stat-label {
    color: var(--color-text-muted);
    font-size: 0.875rem;
}

/* Footer */
.footer {
    margin-top: var(--space-2xl);
    padding-top: var(--space-xl);
    border-top: 1px solid var(--color-border);
    text-align: center;
    color: var(--color-text-muted);
    font-size: 0.875rem;
}

/* Responsive */
@media (max-width: 768px) {
    .container {
        padding: var(--space-md);
    }
    
    .header-content {
        padding: var(--space-lg) var(--space-md);
    }
    
    .site-title {
        font-size: 1.5rem;
    }
    
    .papers-grid {
        gap: var(--space-md);
    }
    
    .paper-card {
        padding: var(--space-md);
    }
    
    .stats-section {
        grid-template-columns: 1fr;
    }
}"""
        
        css_file = self.assets_dir / 'styles.css'
        with open(css_file, 'w') as f:
            f.write(css)
        print(f"  📝 Generated {css_file}")
    
    def _generate_paper_page(self, paper: Dict):
        """Generate individual paper HTML page"""
        paper_id = paper['paper_id'].replace('.', '_')
        filename = f"{paper_id}.html"
        filepath = self.papers_dir / filename
        
        # Convert markdown to HTML
        md = markdown.Markdown(extensions=['extra', 'codehilite', 'toc'])
        content_html = md.convert(paper['summary_markdown'])
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(paper['title'])} - AICOE Research Library</title>
    <link rel="stylesheet" href="../assets/styles.css">
</head>
<body>
    <header class="header">
        <div class="header-content">
            <h1 class="site-title">AICOE Research Library</h1>
            <p class="site-description">AI Papers - Summaries and Analysis</p>
            <nav class="nav">
                <a href="../index.html">Home</a>
                <a href="../archive.html">Archive</a>
            </nav>
        </div>
    </header>
    
    <div class="container paper-page">
        <article class="paper-header">
            <h1 class="paper-title">{escape(paper['title'])}</h1>
            <div class="paper-meta">
                <span class="paper-authors">
                    {escape(', '.join(paper['authors'][:5]))}
                    {' et al.' if len(paper['authors']) > 5 else ''}
                </span>
                <span class="paper-date">{paper['published_date'][:10]}</span>
            </div>
            <div class="paper-categories">
                {' '.join([f'<span class="category-tag">{cat}</span>' for cat in paper['categories']])}
            </div>
            <div class="paper-links">
                <a href="{paper['arxiv_url']}" target="_blank">📄 arXiv</a>
                <a href="{paper['pdf_url']}" target="_blank">📥 PDF</a>
            </div>
        </article>
        
        <div class="paper-content">
            {content_html}
        </div>
        
        <footer class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d')} | AICOE Research Library</p>
        </footer>
    </div>
</body>
</html>"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def _generate_index_page(self, papers: List[Dict]):
        """Generate main index page"""
        # Sort papers by date and relevance
        papers_sorted = sorted(papers, key=lambda x: (x['processed_date'], -x['relevance_score']), reverse=True)
        
        # Take most recent papers
        recent_papers = papers_sorted[:self.config['papers_per_page']]
        
        # Generate paper cards HTML
        papers_html = ""
        for paper in recent_papers:
            paper_id = paper['paper_id'].replace('.', '_')
            
            # Determine relevance class
            score = paper['relevance_score']
            if score > 5:
                relevance_class = 'relevance-high'
                relevance_text = 'High'
            elif score > 2:
                relevance_class = 'relevance-medium'
                relevance_text = 'Medium'
            else:
                relevance_class = 'relevance-low'
                relevance_text = 'Low'
            
            # Truncate summary
            summary = paper.get('summary_markdown', '')[:300] + '...'
            
            papers_html += f"""
        <article class="paper-card">
            <h2 class="paper-title">
                <a href="papers/{paper_id}.html">{escape(paper['title'])}</a>
            </h2>
            <div class="paper-meta">
                <span class="paper-authors">
                    {escape(', '.join(paper['authors'][:3]))}
                    {' et al.' if len(paper['authors']) > 3 else ''}
                </span>
                <span class="paper-date">{paper['published_date'][:10]}</span>
                <span class="relevance-badge {relevance_class}">{relevance_text}</span>
            </div>
            <div class="paper-categories">
                {' '.join([f'<span class="category-tag">{cat}</span>' for cat in paper['categories'][:5]])}
            </div>
            <p class="paper-summary">{escape(summary)}</p>
            <div class="paper-links">
                <a href="papers/{paper_id}.html">Read Summary →</a>
                <a href="{paper['arxiv_url']}" target="_blank">arXiv</a>
                <a href="{paper['pdf_url']}" target="_blank">PDF</a>
            </div>
        </article>"""
        
        # Calculate statistics
        total_papers = len(papers)
        high_relevance = len([p for p in papers if p['relevance_score'] > 5])
        categories = set()
        for p in papers:
            categories.update(p['categories'])
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AICOE Research Library - AI Paper Summaries</title>
    <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
    <header class="header">
        <div class="header-content">
            <h1 class="site-title">AICOE Research Library</h1>
            <p class="site-description">Daily AI Papers - Summaries and Analysis</p>
            <nav class="nav">
                <a href="index.html" class="active">Home</a>
                <a href="archive.html">Archive</a>
            </nav>
        </div>
    </header>
    
    <div class="container">
        <section class="stats-section">
            <div class="stat-card">
                <div class="stat-number">{total_papers}</div>
                <div class="stat-label">Total Papers</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{high_relevance}</div>
                <div class="stat-label">High Relevance</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(categories)}</div>
                <div class="stat-label">Categories</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{datetime.now().strftime('%b %d')}</div>
                <div class="stat-label">Last Updated</div>
            </div>
        </section>
        
        <h2 style="margin: 2rem 0 1rem; font-size: 1.5rem;">Recent Papers</h2>
        
        <div class="papers-grid">
            {papers_html}
        </div>
        
        <footer class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} | AICOE Research Library</p>
            <p>Automated AI paper processing and analysis pipeline</p>
        </footer>
    </div>
</body>
</html>"""
        
        index_file = self.output_dir / 'index.html'
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"  📝 Generated {index_file}")
    
    def _generate_archive_page(self, papers: List[Dict]):
        """Generate archive page with all papers"""
        # Group papers by date
        papers_by_date = {}
        for paper in papers:
            date = paper['published_date'][:10]
            if date not in papers_by_date:
                papers_by_date[date] = []
            papers_by_date[date].append(paper)
        
        # Sort dates
        sorted_dates = sorted(papers_by_date.keys(), reverse=True)
        
        # Generate HTML for papers
        archive_html = ""
        for date in sorted_dates:
            archive_html += f"""
        <h3 style="margin: 2rem 0 1rem; color: var(--color-text-muted);">{date}</h3>
        <ul style="list-style: none; padding: 0;">"""
            
            for paper in papers_by_date[date]:
                paper_id = paper['paper_id'].replace('.', '_')
                archive_html += f"""
            <li style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid var(--color-border);">
                <a href="papers/{paper_id}.html" style="color: var(--color-text); text-decoration: none; font-weight: 500;">
                    {escape(paper['title'])}
                </a>
                <div style="margin-top: 0.25rem; font-size: 0.875rem; color: var(--color-text-muted);">
                    {escape(', '.join(paper['authors'][:3]))}
                    {' et al.' if len(paper['authors']) > 3 else ''}
                </div>
            </li>"""
            
            archive_html += "\n        </ul>"
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Archive - AICOE Research Library</title>
    <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
    <header class="header">
        <div class="header-content">
            <h1 class="site-title">AICOE Research Library</h1>
            <p class="site-description">Daily AI Papers - Summaries and Analysis</p>
            <nav class="nav">
                <a href="index.html">Home</a>
                <a href="archive.html" class="active">Archive</a>
            </nav>
        </div>
    </header>
    
    <div class="container">
        <h2 style="margin-bottom: 2rem; font-size: 1.5rem;">Paper Archive</h2>
        <p style="margin-bottom: 2rem; color: var(--color-text-muted);">
            Complete archive of {len(papers)} processed papers
        </p>
        
        {archive_html}
        
        <footer class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} | AICOE Research Library</p>
        </footer>
    </div>
</body>
</html>"""
        
        archive_file = self.output_dir / 'archive.html'
        with open(archive_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"  📝 Generated {archive_file}")
    
    def _generate_rss_feed(self, papers: List[Dict]):
        """Generate RSS feed for papers"""
        # Sort papers by date
        papers_sorted = sorted(papers, key=lambda x: x['processed_date'], reverse=True)[:20]
        
        rss_items = ""
        for paper in papers_sorted:
            paper_id = paper['paper_id'].replace('.', '_')
            pub_date = datetime.fromisoformat(paper['processed_date']).strftime('%a, %d %b %Y %H:%M:%S +0000')
            
            rss_items += f"""
        <item>
            <title>{escape(paper['title'])}</title>
            <link>{self.config['base_url']}papers/{paper_id}.html</link>
            <description>{escape(paper.get('summary_markdown', '')[:500])}</description>
            <pubDate>{pub_date}</pubDate>
            <guid>{self.config['base_url']}papers/{paper_id}.html</guid>
        </item>"""
        
        rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>AICOE Research Library - AI Papers</title>
        <link>{self.config['base_url']}</link>
        <description>Daily AI paper summaries and analysis</description>
        <language>en-us</language>
        <lastBuildDate>{datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0000')}</lastBuildDate>
        {rss_items}
    </channel>
</rss>"""
        
        rss_file = self.output_dir / 'feed.xml'
        with open(rss_file, 'w', encoding='utf-8') as f:
            f.write(rss)
        
        print(f"  📝 Generated {rss_file}")


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate HTML site from processed papers')
    parser.add_argument('--input-dir', type=str, default='data/processed',
                       help='Directory with processed papers')
    parser.add_argument('--output-dir', type=str, default='docs',
                       help='Output directory for HTML')
    
    args = parser.parse_args()
    
    # Load all processed papers
    input_dir = Path(args.input_dir)
    papers = []
    
    for json_file in input_dir.glob('*.json'):
        with open(json_file, 'r') as f:
            papers.append(json.load(f))
    
    if not papers:
        print("No processed papers found")
        return
    
    print(f"📚 Generating site for {len(papers)} papers...")
    
    # Initialize generator
    generator = HTMLGenerator({'output_dir': args.output_dir})
    
    # Generate site
    generator.generate_site(papers)
    
    print(f"\n✅ Site generated successfully!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🌐 Open {args.output_dir}/index.html to view")


if __name__ == '__main__':
    main()