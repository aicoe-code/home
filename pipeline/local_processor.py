#!/usr/bin/env python3
"""
Local Claude Code Processor
Processes papers using Claude Code (interactive) instead of API
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import textwrap

class LocalClaudeProcessor:
    """Process papers using Claude Code interactively"""
    
    def __init__(self, config: Dict = None):
        """Initialize processor with configuration"""
        self.config = config or self._default_config()
        self.prompts_dir = Path(self.config['data_dir']) / 'prompts'
        self.summaries_dir = Path(self.config['data_dir']) / 'summaries'
        self.processed_dir = Path(self.config['data_dir']) / 'processed'
        
        # Create directories
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'data_dir': 'data',
            'batch_size': 5,
            'template_path': 'templates/summary-template.md'
        }
    
    def prepare_prompts(self, papers: List[Dict]) -> List[str]:
        """
        Prepare prompts for Claude Code processing
        
        Returns:
            List of prompt filenames created
        """
        prompt_files = []
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n📝 Preparing prompts for {len(papers)} papers...")
        
        for i, paper in enumerate(papers):
            # Create filename
            paper_id = paper['id'].replace('.', '_')
            prompt_filename = f"{date_str}_{i+1:03d}_{paper_id}_prompt.txt"
            prompt_path = self.prompts_dir / prompt_filename
            
            # Generate prompt
            prompt = self._create_prompt(paper)
            
            # Save prompt
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            prompt_files.append(prompt_filename)
            
            # Also save paper metadata for later
            meta_path = self.prompts_dir / f"{prompt_filename}.meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(paper, f, indent=2)
            
            print(f"  ✅ Created prompt {i+1}/{len(papers)}: {prompt_filename}")
        
        return prompt_files
    
    def _create_prompt(self, paper: Dict) -> str:
        """Create a detailed prompt for Claude Code"""
        prompt = f"""Please analyze this academic paper and create a comprehensive summary following the AICOE framework.

# Paper Information

**Title:** {paper['title']}
**Authors:** {', '.join(paper['authors'][:5])}{'...' if len(paper['authors']) > 5 else ''}
**Categories:** {', '.join(paper['categories'])}
**arXiv ID:** {paper['id']}
**Link:** {paper.get('arxiv_url', 'N/A')}

# Abstract

{paper['summary']}

{f"# Additional Notes\n\n{paper['comment']}" if paper.get('comment') else ""}

# Required Analysis

Please provide a detailed analysis with the following sections:

## 1. Executive Summary (2-3 paragraphs)
- What is the main contribution?
- Why is this work significant?
- What problem does it solve?

## 2. Key Contributions (3-5 bullet points)
- List the main technical or theoretical contributions
- Be specific about what's novel

## 3. Methodology
- Describe the approach, techniques, or algorithms used
- What datasets or experiments were conducted?
- Key technical details

## 4. Key Findings/Results
- What were the main results?
- How do they compare to baselines?
- Include specific metrics if mentioned

## 5. Strengths (2-3 points)
- What makes this paper strong or innovative?
- Technical strengths
- Practical value

## 6. Limitations (2-3 points)
- What are the weaknesses or areas for improvement?
- What's missing or could be better?
- Scope limitations

## 7. AICOE Relevance
Rate (1-10) and explain relevance to:
- **Understanding AI systems:** How does this help understand AI?
- **Designing AI solutions:** What design insights does it provide?
- **Deploying AI systems:** How does it help with deployment?
- **Operating AI infrastructure:** What operational insights?

## 8. Practical Applications
- How can this be applied in real-world scenarios?
- What industries or domains would benefit?
- Specific use cases

## 9. Implementation Considerations
- Technical requirements
- Computational resources needed
- Integration complexity
- Estimated effort to implement

## 10. Critical Questions & Future Work
- What questions does this raise?
- What future research is needed?
- Gaps to be addressed

# Guidelines
- Be factual and objective
- Focus on practical implications for enterprise AI
- Avoid hyperbole
- Include specific details from the paper
- If information is not in the abstract, note it as "not specified in abstract"

Please format your response as clean markdown with clear section headers."""
        
        return prompt
    
    def process_with_claude(self, prompt_files: List[str]) -> Dict:
        """
        Guide user through processing with Claude Code
        
        Returns:
            Statistics about processing
        """
        total = len(prompt_files)
        
        print("\n" + "="*60)
        print("🤖 CLAUDE CODE PROCESSING")
        print("="*60)
        print(f"\n📋 {total} papers ready for processing\n")
        
        print("Instructions:")
        print("1. Open each prompt file from data/prompts/")
        print("2. Copy the entire content")
        print("3. Paste into Claude Code")
        print("4. Save Claude's response to data/summaries/ with same filename")
        print("   (change '_prompt.txt' to '_summary.md')")
        print("\nPrompt files are in: data/prompts/")
        print("Save summaries to: data/summaries/")
        
        # Show first few files
        print(f"\nFirst {min(5, total)} prompt files to process:")
        for i, filename in enumerate(prompt_files[:5], 1):
            print(f"  {i}. {filename}")
        
        if total > 5:
            print(f"  ... and {total - 5} more")
        
        return {
            'total_prompts': total,
            'prompts_dir': str(self.prompts_dir),
            'summaries_dir': str(self.summaries_dir)
        }
    
    def check_summaries(self) -> List[str]:
        """Check which summaries have been completed"""
        summary_files = list(self.summaries_dir.glob('*_summary.md'))
        return [f.name for f in summary_files]
    
    def process_completed_summaries(self) -> List[Dict]:
        """Process all completed summaries into final format"""
        processed = []
        summary_files = list(self.summaries_dir.glob('*_summary.md'))
        
        print(f"\n📚 Processing {len(summary_files)} completed summaries...")
        
        for summary_file in summary_files:
            # Find corresponding metadata
            meta_filename = summary_file.name.replace('_summary.md', '_prompt.txt.meta.json')
            meta_path = self.prompts_dir / meta_filename
            
            if not meta_path.exists():
                print(f"  ⚠️  No metadata for {summary_file.name}, skipping")
                continue
            
            # Load paper metadata
            with open(meta_path, 'r', encoding='utf-8') as f:
                paper = json.load(f)
            
            # Load summary
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
            
            # Create processed paper
            processed_paper = self._create_processed_paper(paper, summary)
            processed.append(processed_paper)
            
            # Save to processed directory
            self._save_processed_paper(processed_paper)
            
            # Move files to archive
            archive_dir = self.summaries_dir / 'archive'
            archive_dir.mkdir(exist_ok=True)
            summary_file.rename(archive_dir / summary_file.name)
            
            print(f"  ✅ Processed: {paper['title'][:50]}...")
        
        return processed
    
    def _create_processed_paper(self, paper: Dict, summary: str) -> Dict:
        """Create processed paper structure"""
        return {
            'paper_id': paper['id'],
            'title': paper['title'],
            'authors': paper['authors'],
            'categories': paper['categories'],
            'arxiv_url': paper.get('arxiv_url', ''),
            'pdf_url': paper.get('pdf_url', ''),
            'published_date': paper.get('published', ''),
            'relevance_score': paper.get('relevance_score', 0),
            'summary_markdown': summary,
            'processed_date': datetime.now().isoformat(),
            'processing_method': 'claude_code',
            'metadata': {
                'comment': paper.get('comment'),
                'primary_category': paper.get('primary_category'),
                'fetched_date': paper.get('updated', datetime.now().isoformat())
            }
        }
    
    def _save_processed_paper(self, processed: Dict):
        """Save processed paper to file"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"{date_str}_{processed['paper_id'].replace('.', '_')}.json"
        filepath = self.processed_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def create_batch_processor(self, papers: List[Dict], batch_size: int = 5) -> List[List[Dict]]:
        """Create batches for easier processing"""
        batches = []
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def generate_single_prompt(self, papers: List[Dict]) -> str:
        """Generate a single prompt for multiple papers (for efficiency)"""
        if len(papers) == 1:
            return self._create_prompt(papers[0])
        
        prompt = "Please analyze these academic papers and create comprehensive summaries following the AICOE framework.\n\n"
        prompt += "="*60 + "\n\n"
        
        for i, paper in enumerate(papers, 1):
            prompt += f"# PAPER {i} OF {len(papers)}\n\n"
            prompt += f"**Title:** {paper['title']}\n"
            prompt += f"**Authors:** {', '.join(paper['authors'][:3])}...\n"
            prompt += f"**arXiv ID:** {paper['id']}\n\n"
            prompt += f"**Abstract:**\n{paper['summary']}\n\n"
            prompt += "---\n\n"
        
        prompt += "\nFor EACH paper above, provide the standard analysis sections. Clearly label each paper's analysis."
        
        return prompt
    
    def quick_process(self, papers: List[Dict], max_papers: int = 10) -> str:
        """
        Generate a single prompt for quick processing of top papers
        """
        top_papers = papers[:max_papers]
        
        prompt_file = self.prompts_dir / f"quick_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        prompt = self.generate_single_prompt(top_papers)
        
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        # Save metadata
        meta_file = self.prompts_dir / f"{prompt_file.stem}_papers.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(top_papers, f, indent=2)
        
        print(f"\n✨ Quick processing prompt created!")
        print(f"📁 File: {prompt_file}")
        print(f"📊 Contains: {len(top_papers)} papers")
        print("\n1. Copy the content of this file")
        print("2. Paste into Claude Code")
        print("3. Save response as: data/summaries/quick_batch_summary.md")
        
        return str(prompt_file)


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process papers with Claude Code')
    parser.add_argument('--input', type=str, help='Input JSON file with papers')
    parser.add_argument('--prepare', action='store_true', help='Prepare prompts only')
    parser.add_argument('--process', action='store_true', help='Process completed summaries')
    parser.add_argument('--quick', type=int, help='Quick process top N papers')
    parser.add_argument('--batch-size', type=int, default=5, help='Papers per batch')
    
    args = parser.parse_args()
    
    processor = LocalClaudeProcessor()
    
    if args.input and (args.prepare or args.quick):
        # Load papers
        with open(args.input, 'r') as f:
            data = json.load(f)
            papers = data.get('papers', [])
        
        if args.quick:
            # Quick processing mode
            processor.quick_process(papers, args.quick)
        else:
            # Prepare individual prompts
            prompt_files = processor.prepare_prompts(papers[:args.batch_size])
            processor.process_with_claude(prompt_files)
    
    elif args.process:
        # Process completed summaries
        processed = processor.process_completed_summaries()
        print(f"\n✅ Processed {len(processed)} summaries")
        
        if processed:
            print("\nYou can now generate HTML with:")
            print("  python3 simple_generator.py")
    
    else:
        print("Usage:")
        print("  Prepare prompts:  python3 local_processor.py --input data/raw/papers.json --prepare")
        print("  Quick mode:       python3 local_processor.py --input data/raw/papers.json --quick 10")
        print("  Process summaries: python3 local_processor.py --process")


if __name__ == '__main__':
    main()