#!/usr/bin/env python3
"""
arXiv Daily AI Papers Fetcher
Fetches and organizes AI-related papers from arXiv daily
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import urllib.request
import xml.etree.ElementTree as ET
from html import escape

# AI-related arXiv categories
AI_CATEGORIES = [
    'cs.AI',  # Artificial Intelligence
    'cs.LG',  # Machine Learning
    'cs.CV',  # Computer Vision and Pattern Recognition
    'cs.CL',  # Computation and Language
    'cs.NE',  # Neural and Evolutionary Computing
    'stat.ML',  # Machine Learning (Statistics)
    'cs.RO',  # Robotics
    'cs.HC',  # Human-Computer Interaction
    'cs.IR',  # Information Retrieval
]

# Keywords to highlight/prioritize
AI_KEYWORDS = [
    'llm', 'large language model', 'transformer', 'gpt', 'bert', 
    'neural network', 'deep learning', 'reinforcement learning',
    'computer vision', 'nlp', 'natural language', 'generative',
    'diffusion', 'attention', 'multimodal', 'foundation model',
    'fine-tuning', 'prompt', 'chain-of-thought', 'in-context learning',
    'vision transformer', 'clip', 'stable diffusion', 'gan',
    'federated learning', 'transfer learning', 'few-shot', 'zero-shot'
]

class ArxivAIFetcher:
    def __init__(self, categories: List[str] = None, keywords: List[str] = None):
        self.categories = categories or AI_CATEGORIES
        self.keywords = [k.lower() for k in (keywords or AI_KEYWORDS)]
        self.base_url = 'http://export.arxiv.org/api/query'
        self.papers = []
        
    def fetch_papers_by_date(self, date: datetime, max_results: int = 500) -> List[Dict]:
        """Fetch papers updated on a specific date"""
        # Format date for arXiv API (YYYYMMDD format)
        date_str = date.strftime('%Y%m%d')
        next_date_str = (date + timedelta(days=1)).strftime('%Y%m%d')
        
        all_papers = []
        
        for category in self.categories:
            print(f"Fetching papers from {category}...")
            
            # Build query for papers updated in the date range
            query = f'cat:{category} AND lastUpdatedDate:[{date_str}0000 TO {next_date_str}0000]'
            
            # Build URL with parameters
            params = {
                'search_query': query,
                'max_results': str(max_results),
                'sortBy': 'lastUpdatedDate',
                'sortOrder': 'descending'
            }
            
            url = self.base_url + '?' + '&'.join([f'{k}={v.replace(" ", "+")}' for k, v in params.items()])
            
            try:
                # Fetch data from arXiv
                with urllib.request.urlopen(url) as response:
                    data = response.read().decode('utf-8')
                
                # Parse XML response
                papers = self._parse_arxiv_response(data, category)
                all_papers.extend(papers)
                
                # Rate limiting - wait 3 seconds between API calls
                time.sleep(10)
                
            except Exception as e:
                print(f"Error fetching papers from {category}: {e}")
                continue
        
        # Remove duplicates (papers can be in multiple categories)
        unique_papers = self._deduplicate_papers(all_papers)
        
        # Calculate relevance scores
        for paper in unique_papers:
            paper['relevance_score'] = self._calculate_relevance(paper)
        
        # Sort by relevance score
        unique_papers.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        self.papers = unique_papers
        return unique_papers
    
    def _parse_arxiv_response(self, xml_data: str, category: str) -> List[Dict]:
        """Parse arXiv API XML response"""
        root = ET.fromstring(xml_data)
        
        # Define namespaces
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        papers = []
        
        for entry in root.findall('atom:entry', ns):
            paper = {
                'id': entry.find('atom:id', ns).text.split('/')[-1],
                'title': entry.find('atom:title', ns).text.strip().replace('\n', ' '),
                'summary': entry.find('atom:summary', ns).text.strip(),
                'authors': [],
                'published': entry.find('atom:published', ns).text,
                'updated': entry.find('atom:updated', ns).text,
                'categories': [],
                'primary_category': category,
                'pdf_url': None,
                'arxiv_url': None,
                'comment': None
            }
            
            # Extract authors
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    paper['authors'].append(name.text)
            
            # Extract categories
            for cat in entry.findall('atom:category', ns):
                paper['categories'].append(cat.get('term'))
            
            # Extract links
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    paper['pdf_url'] = link.get('href')
                elif link.get('rel') == 'alternate':
                    paper['arxiv_url'] = link.get('href')
            
            # Extract comment if available
            comment = entry.find('arxiv:comment', ns)
            if comment is not None:
                paper['comment'] = comment.text
            
            papers.append(paper)
        
        return papers
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on arXiv ID"""
        seen = {}
        unique = []
        
        for paper in papers:
            arxiv_id = paper['id']
            if arxiv_id not in seen:
                seen[arxiv_id] = True
                unique.append(paper)
            else:
                # Merge categories if paper appears in multiple
                for existing in unique:
                    if existing['id'] == arxiv_id:
                        existing['categories'] = list(set(existing['categories'] + paper['categories']))
                        break
        
        return unique
    
    def _calculate_relevance(self, paper: Dict) -> float:
        """Calculate relevance score based on keywords"""
        score = 0.0
        
        # Check title and abstract for keywords
        text = (paper['title'] + ' ' + paper['summary']).lower()
        
        for keyword in self.keywords:
            if keyword in text:
                # Higher weight for title matches
                if keyword in paper['title'].lower():
                    score += 2.0
                else:
                    score += 1.0
        
        # Boost for papers in primary AI categories
        if paper['primary_category'] in ['cs.AI', 'cs.LG', 'cs.CV', 'cs.CL']:
            score += 0.5
        
        return score
    
    def save_json(self, filepath: str):
        """Save papers to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.papers, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.papers)} papers to {filepath}")
    
    def save_markdown(self, filepath: str):
        """Save papers to Markdown file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# arXiv AI Papers - {datetime.now().strftime('%Y-%m-%d')}\n\n")
            f.write(f"Total papers: {len(self.papers)}\n\n")
            
            for i, paper in enumerate(self.papers, 1):
                f.write(f"## {i}. {paper['title']}\n\n")
                f.write(f"**Authors:** {', '.join(paper['authors'])}\n\n")
                f.write(f"**Categories:** {', '.join(paper['categories'])}\n\n")
                f.write(f"**Links:** [arXiv]({paper['arxiv_url']}) | [PDF]({paper['pdf_url']})\n\n")
                f.write(f"**Abstract:** {paper['summary']}\n\n")
                if paper['comment']:
                    f.write(f"**Comment:** {paper['comment']}\n\n")
                f.write(f"**Relevance Score:** {paper['relevance_score']:.2f}\n\n")
                f.write("---\n\n")
        
        print(f"Saved markdown to {filepath}")
    
    def save_html(self, filepath: str):
        """Save papers to HTML file following AICOE design guidelines"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Embedded CSS following design guidelines
        css = '''
        /* AICOE Design System - Minimal, Clean, Professional */
        :root {
            /* Colors */
            --color-background: #FFFFFF;
            --color-background-subtle: #FAFAFA;
            --color-foreground: #09090B;
            --color-muted: #71717A;
            --color-border: #E4E4E7;
            --color-accent: #18181B;
            
            /* Typography */
            --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            
            /* Font Sizes */
            --text-xs: 0.75rem;
            --text-sm: 0.875rem;
            --text-base: 1rem;
            --text-lg: 1.125rem;
            --text-xl: 1.25rem;
            --text-2xl: 1.5rem;
            --text-3xl: 2rem;
            
            /* Spacing (8px grid) */
            --space-2: 0.5rem;
            --space-3: 0.75rem;
            --space-4: 1rem;
            --space-6: 1.5rem;
            --space-8: 2rem;
            --space-12: 3rem;
            
            /* Borders & Shadows */
            --radius-base: 0.25rem;
            --radius-md: 0.375rem;
            --radius-lg: 0.5rem;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-base: 0 1px 3px 0 rgb(0 0 0 / 0.1);
            
            /* Layout */
            --max-width: 1200px;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: var(--font-sans);
            font-size: var(--text-base);
            line-height: 1.6;
            color: var(--color-foreground);
            background-color: var(--color-background);
        }
        
        .container {
            max-width: var(--max-width);
            margin: 0 auto;
            padding: var(--space-6);
        }
        
        /* Typography */
        h1 {
            font-size: var(--text-3xl);
            font-weight: 600;
            line-height: 1.25;
            margin-bottom: var(--space-4);
        }
        
        h3 {
            font-size: var(--text-lg);
            font-weight: 600;
            line-height: 1.4;
            margin-bottom: var(--space-3);
        }
        
        p {
            color: var(--color-muted);
            line-height: 1.6;
        }
        
        a {
            color: var(--color-accent);
            text-decoration: none;
            transition: opacity 200ms ease;
        }
        
        a:hover {
            opacity: 0.8;
        }
        
        /* Navigation */
        .breadcrumb {
            font-size: var(--text-sm);
            color: var(--color-muted);
            margin-bottom: var(--space-8);
        }
        
        .breadcrumb a {
            color: var(--color-muted);
        }
        
        .breadcrumb a:hover {
            color: var(--color-foreground);
        }
        
        /* Hero Section */
        .hero {
            background: var(--color-background-subtle);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-lg);
            padding: var(--space-12) var(--space-8);
            margin-bottom: var(--space-8);
            text-align: center;
        }
        
        .hero p {
            font-size: var(--text-lg);
            max-width: 600px;
            margin: 0 auto;
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: var(--space-4);
            margin-bottom: var(--space-8);
        }
        
        .stat-card {
            background: var(--color-background);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-lg);
            padding: var(--space-6);
            text-align: center;
        }
        
        .stat-number {
            font-size: var(--text-2xl);
            font-weight: 600;
            color: var(--color-foreground);
            margin-bottom: var(--space-2);
        }
        
        .stat-label {
            font-size: var(--text-sm);
            color: var(--color-muted);
        }
        
        /* Paper Cards */
        .papers-list {
            display: flex;
            flex-direction: column;
            gap: var(--space-4);
        }
        
        .paper-card {
            background: var(--color-background);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-lg);
            padding: var(--space-6);
            transition: box-shadow 200ms ease;
        }
        
        .paper-card:hover {
            box-shadow: var(--shadow-base);
        }
        
        .paper-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: var(--space-4);
        }
        
        .paper-title {
            flex: 1;
            margin-right: var(--space-4);
        }
        
        .relevance-badge {
            font-size: var(--text-xs);
            font-weight: 600;
            padding: var(--space-2) var(--space-3);
            border-radius: var(--radius-base);
            white-space: nowrap;
        }
        
        .relevance-high {
            background: #FEE2E2;
            color: #DC2626;
            border: 1px solid #FCA5A5;
        }
        
        .relevance-medium {
            background: #FEF3C7;
            color: #D97706;
            border: 1px solid #FCD34D;
        }
        
        .relevance-low {
            background: #DBEAFE;
            color: #2563EB;
            border: 1px solid #93C5FD;
        }
        
        .paper-meta {
            display: flex;
            flex-direction: column;
            gap: var(--space-2);
            margin-bottom: var(--space-4);
            font-size: var(--text-sm);
            color: var(--color-muted);
        }
        
        .meta-row {
            display: flex;
            align-items: center;
            gap: var(--space-2);
        }
        
        .meta-label {
            font-weight: 600;
            min-width: 80px;
        }
        
        .paper-abstract {
            color: var(--color-muted);
            line-height: 1.6;
            margin-bottom: var(--space-3);
        }
        
        .paper-comment {
            font-size: var(--text-sm);
            color: var(--color-muted);
            font-style: italic;
            padding-top: var(--space-3);
            border-top: 1px solid var(--color-border);
        }
        
        /* Footer */
        .footer {
            margin-top: var(--space-12);
            padding-top: var(--space-6);
            border-top: 1px solid var(--color-border);
            text-align: center;
            font-size: var(--text-sm);
            color: var(--color-muted);
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .container {
                padding: var(--space-4);
            }
            
            .hero {
                padding: var(--space-8) var(--space-4);
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .paper-header {
                flex-direction: column;
            }
            
            .paper-title {
                margin-right: 0;
                margin-bottom: var(--space-3);
            }
        }
        '''
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Papers - {date_str} - AICOE</title>
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <nav class="breadcrumb">
            <a href="../../index.html">Home</a> / 
            <a href="../index.html">Research</a> / 
            <span>Daily AI Papers</span>
        </nav>
        
        <div class="hero">
            <h1>arXiv AI Papers - {date_str}</h1>
            <p>Daily collection of {len(self.papers)} AI-related papers from arXiv</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{len(self.papers)}</div>
                <div class="stat-label">Total Papers</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len([p for p in self.papers if p['relevance_score'] > 3])}</div>
                <div class="stat-label">High Relevance</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(set([cat for p in self.papers for cat in p['categories']]))}</div>
                <div class="stat-label">Categories</div>
            </div>
        </div>
        
        <div class="papers-list">
'''
        
        for i, paper in enumerate(self.papers, 1):
            # Determine relevance level
            if paper['relevance_score'] > 3:
                relevance_class = 'relevance-high'
                relevance_text = 'High'
            elif paper['relevance_score'] > 1:
                relevance_class = 'relevance-medium'
                relevance_text = 'Medium'
            else:
                relevance_class = 'relevance-low'
                relevance_text = 'Low'
            
            # Format authors
            authors = paper['authors'][:3]
            if len(paper['authors']) > 3:
                authors.append('et al.')
            authors_str = ', '.join([escape(a) for a in authors])
            
            # Truncate abstract
            abstract = paper['summary'][:500]
            if len(paper['summary']) > 500:
                abstract += '...'
            
            html += f'''
            <div class="paper-card">
                <div class="paper-header">
                    <h3 class="paper-title">{i}. {escape(paper['title'])}</h3>
                    <span class="relevance-badge {relevance_class}">{relevance_text} ({paper['relevance_score']:.1f})</span>
                </div>
                
                <div class="paper-meta">
                    <div class="meta-row">
                        <span class="meta-label">Authors:</span>
                        <span>{authors_str}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">Categories:</span>
                        <span>{', '.join(paper['categories'])}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">Links:</span>
                        <span>
                            <a href="{paper['arxiv_url']}" target="_blank">arXiv:{paper['id']}</a> | 
                            <a href="{paper['pdf_url']}" target="_blank">PDF</a>
                        </span>
                    </div>
                </div>
                
                <p class="paper-abstract">{escape(abstract)}</p>
                {f'<div class="paper-comment">Note: {escape(paper["comment"])}</div>' if paper.get('comment') else ''}
            </div>
'''
        
        html += f'''
        </div>
        
        <footer class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            <p>AICOE Research Library - AI Papers Daily Digest</p>
        </footer>
    </div>
</body>
</html>'''
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"Saved HTML to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Fetch daily AI papers from arXiv')
    parser.add_argument('--date', type=str, help='Date to fetch papers (YYYY-MM-DD). Default: yesterday')
    parser.add_argument('--output-dir', type=str, default='../daily', help='Output directory for files')
    parser.add_argument('--max-results', type=int, default=200, help='Maximum papers per category')
    parser.add_argument('--categories', nargs='+', help='Categories to fetch (e.g., cs.AI cs.LG)')
    parser.add_argument('--format', choices=['json', 'html', 'markdown', 'all'], default='all', 
                       help='Output format')
    
    args = parser.parse_args()
    
    # Determine date
    if args.date:
        fetch_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        # Default to yesterday (since arXiv updates at midnight)
        fetch_date = datetime.now() - timedelta(days=1)
    
    # Create fetcher
    categories = args.categories if args.categories else None
    fetcher = ArxivAIFetcher(categories=categories)
    
    # Fetch papers
    print(f"Fetching AI papers from {fetch_date.strftime('%Y-%m-%d')}...")
    papers = fetcher.fetch_papers_by_date(fetch_date, max_results=args.max_results)
    
    if not papers:
        print("No papers found for the specified date.")
        return
    
    print(f"Found {len(papers)} unique papers")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate filename base
    date_str = fetch_date.strftime('%Y-%m-%d')
    base_filename = os.path.join(args.output_dir, f'arxiv-ai-papers-{date_str}')
    
    # Save in requested formats
    if args.format in ['json', 'all']:
        fetcher.save_json(f'{base_filename}.json')
    
    if args.format in ['html', 'all']:
        fetcher.save_html(f'{base_filename}.html')
    
    if args.format in ['markdown', 'all']:
        fetcher.save_markdown(f'{base_filename}.md')
    
    # Print top papers
    print("\nTop 5 most relevant papers:")
    for i, paper in enumerate(papers[:5], 1):
        print(f"{i}. [{paper['relevance_score']:.2f}] {paper['title'][:80]}...")
        print(f"   Authors: {', '.join(paper['authors'][:3])}")
        print(f"   Categories: {', '.join(paper['categories'])}")
        print()

if __name__ == '__main__':
    main()