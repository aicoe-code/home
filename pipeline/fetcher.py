#!/usr/bin/env python3
"""
Enhanced arXiv Paper Fetcher
Fetches AI-related papers from arXiv with deduplication and relevance scoring
"""

import json
import os
import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

class ArxivPaperFetcher:
    """Enhanced fetcher for arXiv papers with deduplication and smart filtering"""
    
    def __init__(self, config: Dict = None):
        """Initialize fetcher with configuration"""
        self.config = config or self._default_config()
        self.base_url = 'http://export.arxiv.org/api/query'
        self.processed_papers_file = Path(self.config['data_dir']) / 'processed_papers.json'
        self.processed_ids = self._load_processed_papers()
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'categories': [
                'cs.AI', 'cs.LG', 'cs.CV', 'cs.CL', 'cs.NE',
                'stat.ML', 'cs.RO', 'cs.HC', 'cs.IR'
            ],
            'keywords': [
                'llm', 'large language model', 'transformer', 'gpt', 'bert',
                'neural network', 'deep learning', 'reinforcement learning',
                'computer vision', 'nlp', 'natural language', 'generative',
                'diffusion', 'attention', 'multimodal', 'foundation model',
                'fine-tuning', 'prompt', 'chain-of-thought', 'in-context learning',
                'vision transformer', 'clip', 'stable diffusion', 'gan',
                'federated learning', 'transfer learning', 'few-shot', 'zero-shot'
            ],
            'max_results_per_category': 100,
            'data_dir': 'data',
            'min_relevance_score': 1.0,
            'rate_limit_seconds': 3
        }
    
    def _load_processed_papers(self) -> Set[str]:
        """Load set of already processed paper IDs"""
        if self.processed_papers_file.exists():
            try:
                with open(self.processed_papers_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('processed_ids', []))
            except:
                return set()
        return set()
    
    def _save_processed_papers(self):
        """Save processed paper IDs"""
        self.processed_papers_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.processed_papers_file, 'w') as f:
            json.dump({
                'processed_ids': list(self.processed_ids),
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def fetch_papers_by_date(self, date: datetime = None, force_refetch: bool = False) -> List[Dict]:
        """
        Fetch papers for a specific date with deduplication
        
        Args:
            date: Date to fetch papers for (default: yesterday)
            force_refetch: If True, ignore deduplication and fetch all papers
        
        Returns:
            List of paper dictionaries
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)
        
        date_str = date.strftime('%Y%m%d')
        next_date_str = (date + timedelta(days=1)).strftime('%Y%m%d')
        
        print(f"📅 Fetching papers for {date.strftime('%Y-%m-%d')}")
        all_papers = []
        
        for category in self.config['categories']:
            print(f"  📚 Fetching from {category}...")
            
            # Build query
            query = f'cat:{category} AND lastUpdatedDate:[{date_str}0000 TO {next_date_str}0000]'
            params = {
                'search_query': query,
                'max_results': str(self.config['max_results_per_category']),
                'sortBy': 'lastUpdatedDate',
                'sortOrder': 'descending'
            }
            
            url = self.base_url + '?' + urllib.parse.urlencode(params)
            
            try:
                with urllib.request.urlopen(url) as response:
                    data = response.read().decode('utf-8')
                
                papers = self._parse_arxiv_response(data, category)
                
                # Filter out already processed papers unless force_refetch
                if not force_refetch:
                    new_papers = [p for p in papers if p['id'] not in self.processed_ids]
                    print(f"    Found {len(papers)} papers, {len(new_papers)} new")
                    papers = new_papers
                else:
                    print(f"    Found {len(papers)} papers")
                
                all_papers.extend(papers)
                time.sleep(self.config['rate_limit_seconds'])
                
            except Exception as e:
                print(f"  ❌ Error fetching {category}: {e}")
                continue
        
        # Remove duplicates
        unique_papers = self._deduplicate_papers(all_papers)
        
        # Calculate relevance scores
        for paper in unique_papers:
            paper['relevance_score'] = self._calculate_relevance(paper)
        
        # Filter by minimum relevance
        filtered_papers = [
            p for p in unique_papers 
            if p['relevance_score'] >= self.config['min_relevance_score']
        ]
        
        # Sort by relevance
        filtered_papers.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Update processed papers set
        for paper in filtered_papers:
            self.processed_ids.add(paper['id'])
        self._save_processed_papers()
        
        print(f"\n✅ Total: {len(filtered_papers)} relevant new papers")
        return filtered_papers
    
    def fetch_recent_papers(self, days: int = 7, force_refetch: bool = False) -> List[Dict]:
        """Fetch papers from the last N days"""
        all_papers = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i+1)
            papers = self.fetch_papers_by_date(date, force_refetch)
            all_papers.extend(papers)
        
        # Sort all by relevance
        all_papers.sort(key=lambda x: x['relevance_score'], reverse=True)
        return all_papers
    
    def _parse_arxiv_response(self, xml_data: str, category: str) -> List[Dict]:
        """Parse arXiv API XML response"""
        root = ET.fromstring(xml_data)
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
            
            # Extract comment
            comment = entry.find('arxiv:comment', ns)
            if comment is not None:
                paper['comment'] = comment.text
            
            # Generate unique hash for deduplication
            paper['hash'] = self._generate_paper_hash(paper)
            
            papers.append(paper)
        
        return papers
    
    def _generate_paper_hash(self, paper: Dict) -> str:
        """Generate unique hash for a paper"""
        # Use title and first author for uniqueness
        unique_str = f"{paper['title']}_{paper['authors'][0] if paper['authors'] else ''}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on ID and hash"""
        seen_ids = {}
        seen_hashes = {}
        unique = []
        
        for paper in papers:
            if paper['id'] not in seen_ids and paper['hash'] not in seen_hashes:
                seen_ids[paper['id']] = True
                seen_hashes[paper['hash']] = True
                unique.append(paper)
            elif paper['id'] in seen_ids:
                # Merge categories if duplicate
                for existing in unique:
                    if existing['id'] == paper['id']:
                        existing['categories'] = list(set(existing['categories'] + paper['categories']))
                        break
        
        return unique
    
    def _calculate_relevance(self, paper: Dict) -> float:
        """Calculate relevance score based on keywords and categories"""
        score = 0.0
        
        # Check title and abstract for keywords
        text = (paper['title'] + ' ' + paper['summary']).lower()
        
        for keyword in self.config['keywords']:
            keyword_lower = keyword.lower()
            if keyword_lower in text:
                # Higher weight for title matches
                if keyword_lower in paper['title'].lower():
                    score += 3.0
                else:
                    score += 1.0
        
        # Boost for papers in primary AI categories
        primary_categories = ['cs.AI', 'cs.LG', 'cs.CV', 'cs.CL']
        if paper['primary_category'] in primary_categories:
            score += 1.0
        
        # Boost for papers with code/implementation
        if paper.get('comment'):
            comment_lower = paper['comment'].lower()
            if any(term in comment_lower for term in ['github', 'code', 'implementation', 'dataset']):
                score += 2.0
        
        return score
    
    def save_papers(self, papers: List[Dict], output_dir: str = None):
        """Save papers to JSON file"""
        if output_dir is None:
            output_dir = Path(self.config['data_dir']) / 'raw'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        date_str = datetime.now().strftime('%Y-%m-%d')
        output_file = output_dir / f'arxiv_papers_{date_str}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'fetch_date': datetime.now().isoformat(),
                'total_papers': len(papers),
                'papers': papers
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(papers)} papers to {output_file}")
        return output_file
    
    def get_statistics(self, papers: List[Dict]) -> Dict:
        """Get statistics about fetched papers"""
        stats = {
            'total': len(papers),
            'by_category': {},
            'high_relevance': len([p for p in papers if p['relevance_score'] > 5]),
            'medium_relevance': len([p for p in papers if 2 < p['relevance_score'] <= 5]),
            'low_relevance': len([p for p in papers if p['relevance_score'] <= 2]),
            'with_code': len([p for p in papers if p.get('comment') and 'github' in p.get('comment', '').lower()])
        }
        
        for paper in papers:
            for cat in paper['categories']:
                stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
        
        return stats


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch AI papers from arXiv')
    parser.add_argument('--date', type=str, help='Date to fetch (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, help='Fetch papers from last N days')
    parser.add_argument('--force', action='store_true', help='Force refetch, ignore deduplication')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize fetcher
    fetcher = ArxivPaperFetcher()
    
    # Fetch papers
    if args.days:
        papers = fetcher.fetch_recent_papers(args.days, args.force)
    elif args.date:
        date = datetime.strptime(args.date, '%Y-%m-%d')
        papers = fetcher.fetch_papers_by_date(date, args.force)
    else:
        papers = fetcher.fetch_papers_by_date(force_refetch=args.force)
    
    # Save papers
    if papers:
        output_file = fetcher.save_papers(papers, args.output)
        
        # Print statistics
        stats = fetcher.get_statistics(papers)
        print("\n📊 Statistics:")
        print(f"  Total papers: {stats['total']}")
        print(f"  High relevance: {stats['high_relevance']}")
        print(f"  Medium relevance: {stats['medium_relevance']}")
        print(f"  Low relevance: {stats['low_relevance']}")
        print(f"  With code: {stats['with_code']}")
        
        # Print top 5 papers
        print("\n🏆 Top 5 papers by relevance:")
        for i, paper in enumerate(papers[:5], 1):
            print(f"{i}. [{paper['relevance_score']:.1f}] {paper['title'][:80]}...")
    else:
        print("No new papers found.")


if __name__ == '__main__':
    main()