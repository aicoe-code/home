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
                time.sleep(3)
                
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
        """Save papers to HTML file matching existing style"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Papers - {date_str} - AICOE</title>
    <link rel="stylesheet" href="../../styles.css">
</head>
<body>
    <div class="container">
        <nav class="breadcrumb">
            <a href="../../index.html">Home</a> / 
            <a href="../index.html">Research</a> / 
            <span>Daily AI Papers</span>
        </nav>
        
        <div class="research-hero">
            <h1>arXiv AI Papers - {date_str}</h1>
            <p>Daily collection of AI-related papers from arXiv. Total papers: {len(self.papers)}</p>
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
        
        <div class="research-content">
'''
        
        for i, paper in enumerate(self.papers, 1):
            # Determine priority based on relevance score
            if paper['relevance_score'] > 3:
                priority_class = 'priority-high'
                priority_text = 'High Relevance'
            elif paper['relevance_score'] > 1:
                priority_class = 'priority-medium'
                priority_text = 'Medium Relevance'
            else:
                priority_class = 'priority-low'
                priority_text = 'Low Relevance'
            
            html += f'''
            <div class="paper-entry" style="margin-bottom: 2rem; padding: 1.5rem; background: #fafafa; border-radius: 8px; border: 1px solid #e5e5e5;">
                <h3>{i}. {escape(paper['title'])}</h3>
                <div class="priority-badge {priority_class}" style="display: inline-block; margin: 0.5rem 0;">
                    {priority_text} (Score: {paper['relevance_score']:.2f})
                </div>
                <div class="paper-meta" style="margin: 1rem 0;">
                    <span class="meta-item">👥 {', '.join([escape(a) for a in paper['authors'][:3]])}{' et al.' if len(paper['authors']) > 3 else ''}</span><br>
                    <span class="meta-item">📁 {', '.join(paper['categories'])}</span><br>
                    <span class="meta-item">🔗 <a href="{paper['arxiv_url']}" target="_blank">arXiv:{paper['id']}</a> | 
                    <a href="{paper['pdf_url']}" target="_blank">PDF</a></span>
                </div>
                <p style="color: #666; line-height: 1.6;">{escape(paper['summary'][:500])}{'...' if len(paper['summary']) > 500 else ''}</p>
                {f'<p style="font-size: 0.9em; color: #888; font-style: italic;">Note: {escape(paper["comment"])}</p>' if paper.get('comment') else ''}
            </div>
'''
        
        html += '''
        </div>
        
        <footer class="card text-center" style="margin-top: 2rem;">
            <p class="text-muted mb-0">Generated on ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
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