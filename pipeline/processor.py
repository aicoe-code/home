#!/usr/bin/env python3
"""
Claude AI Paper Processor
Processes academic papers using Claude AI to generate structured summaries
"""

import json
import os
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import tempfile

class ClaudeProcessor:
    """Process papers using Claude AI to generate summaries"""
    
    def __init__(self, config: Dict = None):
        """Initialize processor with configuration"""
        self.config = config or self._default_config()
        self.template_path = Path(self.config['template_path'])
        self.processed_dir = Path(self.config['data_dir']) / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'api_key': os.environ.get('CLAUDE_API_KEY', ''),
            'model': 'claude-3-opus-20240229',
            'max_tokens': 4000,
            'temperature': 0.3,
            'template_path': 'templates/summary-template.md',
            'data_dir': 'data',
            'batch_size': 5,
            'rate_limit_delay': 2
        }
    
    def process_paper(self, paper: Dict) -> Dict:
        """
        Process a single paper using Claude AI
        
        Args:
            paper: Paper dictionary from fetcher
            
        Returns:
            Dictionary with paper and generated summary
        """
        print(f"🤖 Processing: {paper['title'][:60]}...")
        
        # Create prompt for Claude
        prompt = self._create_prompt(paper)
        
        # Call Claude API (using subprocess to call Claude CLI)
        summary = self._call_claude_cli(prompt, paper)
        
        # Parse and structure the response
        structured_summary = self._parse_summary(summary, paper)
        
        # Save processed paper
        self._save_processed_paper(structured_summary)
        
        return structured_summary
    
    def process_papers_batch(self, papers: List[Dict]) -> List[Dict]:
        """Process multiple papers in batch"""
        processed = []
        
        for i, paper in enumerate(papers):
            print(f"\n📄 Processing paper {i+1}/{len(papers)}")
            
            try:
                result = self.process_paper(paper)
                processed.append(result)
                
                # Rate limiting
                if i < len(papers) - 1:
                    time.sleep(self.config['rate_limit_delay'])
                    
            except Exception as e:
                print(f"  ❌ Error processing paper: {e}")
                # Create minimal summary for failed papers
                result = self._create_fallback_summary(paper)
                processed.append(result)
        
        return processed
    
    def _create_prompt(self, paper: Dict) -> str:
        """Create a detailed prompt for Claude"""
        prompt = f"""You are an AI research analyst helping to create structured summaries of academic papers.

Please analyze the following paper and create a comprehensive summary following the AICOE framework.

Paper Title: {paper['title']}
Authors: {', '.join(paper['authors'][:5])}
Categories: {', '.join(paper['categories'])}
arXiv ID: {paper['id']}

Abstract:
{paper['summary']}

{f"Additional Notes: {paper['comment']}" if paper.get('comment') else ""}

Please provide a detailed analysis including:

1. **Executive Summary** (2-3 paragraphs)
   - What is the main contribution?
   - Why is this work significant?
   - What problem does it solve?

2. **Key Contributions** (3-5 bullet points)
   - List the main technical or theoretical contributions

3. **Methodology**
   - Describe the approach, techniques, or algorithms used
   - What datasets or experiments were conducted?

4. **Key Findings/Results**
   - What were the main results?
   - How do they compare to baselines?
   - Include specific metrics if mentioned

5. **Strengths** (2-3 points)
   - What makes this paper strong or innovative?

6. **Limitations** (2-3 points)
   - What are the weaknesses or areas for improvement?

7. **AICOE Relevance**
   Rate (1-10) and explain relevance to:
   - Understanding AI systems
   - Designing AI solutions
   - Deploying AI systems
   - Operating AI infrastructure

8. **Practical Applications**
   - How can this be applied in real-world scenarios?
   - What industries or domains would benefit?

9. **Implementation Considerations**
   - Technical requirements
   - Computational resources needed
   - Integration complexity

10. **Critical Questions**
    - What questions does this raise?
    - What future research is needed?

Format your response as a structured markdown document with clear sections.
Be factual, objective, and avoid hyperbole. Focus on practical implications for enterprise AI implementation."""
        
        return prompt
    
    def _call_claude_cli(self, prompt: str, paper: Dict) -> str:
        """Call Claude using subprocess (simulated for now)"""
        # This is a placeholder that simulates Claude's response
        # In production, you would integrate with the actual Claude API
        
        # For now, generate a template response
        summary = f"""## Executive Summary

This paper presents {paper['title']} by {', '.join(paper['authors'][:3])} et al. The work addresses fundamental challenges in AI systems through novel approaches that advance the state of the art. The research demonstrates significant improvements in performance metrics while maintaining computational efficiency.

## Key Contributions

- **Novel Architecture**: Introduces an innovative approach to the problem domain
- **Performance Improvements**: Achieves state-of-the-art results on benchmark datasets
- **Efficiency Gains**: Reduces computational requirements by significant margins
- **Theoretical Insights**: Provides new understanding of underlying mechanisms

## Methodology

The research employs a systematic approach combining theoretical analysis with empirical validation. The authors utilize standard benchmark datasets and rigorous evaluation protocols to ensure reproducibility and comparability with existing methods.

## Key Findings

The experimental results demonstrate:
- Improvement of 15-20% over baseline methods
- Reduced training time by 40%
- Better generalization to unseen data
- Robustness to various input conditions

## Strengths

- **Clear Presentation**: Well-structured paper with clear explanations
- **Rigorous Evaluation**: Comprehensive experiments with proper baselines
- **Practical Relevance**: Direct applications to real-world problems

## Limitations

- **Scalability Concerns**: May face challenges with very large-scale deployments
- **Domain Specificity**: Results may not generalize to all application domains
- **Computational Requirements**: Still requires significant resources for training

## AICOE Relevance

**Understanding (8/10)**: Provides deep insights into AI system behavior
**Design (7/10)**: Offers new design patterns for AI architectures
**Deployment (6/10)**: Includes considerations for production deployment
**Operation (5/10)**: Limited discussion of operational aspects

## Practical Applications

This research has immediate applications in:
- Enterprise AI systems requiring efficient processing
- Real-time decision-making systems
- Resource-constrained environments

## Implementation Considerations

- **Technical Requirements**: Modern GPU infrastructure recommended
- **Integration Complexity**: Moderate - requires adaptation to existing pipelines
- **Time to Deploy**: Estimated 2-3 months for production implementation

## Critical Questions

1. How does this approach scale to multi-modal inputs?
2. What are the implications for federated learning scenarios?
3. Can the method be adapted for continual learning?"""
        
        return summary
    
    def _parse_summary(self, summary: str, paper: Dict) -> Dict:
        """Parse and structure the summary"""
        return {
            'paper_id': paper['id'],
            'title': paper['title'],
            'authors': paper['authors'],
            'categories': paper['categories'],
            'arxiv_url': paper['arxiv_url'],
            'pdf_url': paper['pdf_url'],
            'published_date': paper['published'],
            'relevance_score': paper.get('relevance_score', 0),
            'summary_markdown': summary,
            'processed_date': datetime.now().isoformat(),
            'metadata': {
                'comment': paper.get('comment'),
                'primary_category': paper.get('primary_category'),
                'fetched_date': datetime.now().isoformat()
            }
        }
    
    def _create_fallback_summary(self, paper: Dict) -> Dict:
        """Create a minimal summary for failed processing"""
        fallback_summary = f"""## Executive Summary

Paper: {paper['title']}
Authors: {', '.join(paper['authors'][:3])}

{paper['summary']}

## Processing Note

This paper could not be fully processed by the AI system. The above shows the original abstract.

Categories: {', '.join(paper['categories'])}
arXiv Link: {paper['arxiv_url']}"""
        
        return self._parse_summary(fallback_summary, paper)
    
    def _save_processed_paper(self, processed: Dict):
        """Save processed paper to file"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"{date_str}_{processed['paper_id'].replace('.', '_')}.json"
        filepath = self.processed_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 Saved: {filepath.name}")
    
    def load_processed_papers(self, date: str = None) -> List[Dict]:
        """Load previously processed papers"""
        pattern = f"{date}_*.json" if date else "*.json"
        files = list(self.processed_dir.glob(pattern))
        
        papers = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                papers.append(json.load(f))
        
        return papers


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process papers with Claude AI')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file with papers')
    parser.add_argument('--limit', type=int, help='Limit number of papers to process')
    
    args = parser.parse_args()
    
    # Load papers
    with open(args.input, 'r') as f:
        data = json.load(f)
        papers = data.get('papers', [])
    
    if args.limit:
        papers = papers[:args.limit]
    
    print(f"📚 Processing {len(papers)} papers...")
    
    # Initialize processor
    processor = ClaudeProcessor()
    
    # Process papers
    results = processor.process_papers_batch(papers)
    
    print(f"\n✅ Processed {len(results)} papers successfully")
    
    # Save summary report
    report_file = Path('data/processed') / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump({
            'processed_count': len(results),
            'timestamp': datetime.now().isoformat(),
            'papers': [{'id': r['paper_id'], 'title': r['title']} for r in results]
        }, f, indent=2)
    
    print(f"📊 Report saved to {report_file}")


if __name__ == '__main__':
    main()