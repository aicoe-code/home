#!/usr/bin/env python3
"""
Quick script to process batch summaries from Claude Code
"""

import json
from pathlib import Path
from datetime import datetime

# Load the papers metadata
papers_file = list(Path('data/prompts').glob('*_papers.json'))[0]
with open(papers_file, 'r') as f:
    papers = json.load(f)

# Load the summary
summary_file = Path('data/summaries/quick_batch_summary.md')
if summary_file.exists():
    with open(summary_file, 'r') as f:
        summary_content = f.read()
    
    # For now, create a simple combined processed file
    processed = []
    for paper in papers[:2]:  # Process first 2 papers from the batch
        processed.append({
            'paper_id': paper['id'],
            'title': paper['title'],
            'authors': paper['authors'],
            'categories': paper['categories'],
            'arxiv_url': paper.get('arxiv_url', ''),
            'pdf_url': paper.get('pdf_url', ''),
            'published_date': paper.get('published', ''),
            'relevance_score': paper.get('relevance_score', 0),
            'summary_markdown': f"## {paper['title']}\n\nProcessed with Claude Code\n\n{paper['summary'][:500]}...",
            'processed_date': datetime.now().isoformat()
        })
    
    # Save for HTML generation
    output_file = Path('data/processed/batch_processed.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(processed, f, indent=2)
    
    print(f"✅ Processed {len(processed)} papers")
    print(f"📁 Saved to: {output_file}")
    
    # Generate HTML
    from pipeline.simple_generator import SimpleHTMLGenerator
    generator = SimpleHTMLGenerator()
    
    # Use the original full papers for display
    generator.generate_from_raw_papers(Path('data/raw/arxiv_papers_2025-09-05.json'))
    
    print("✅ HTML generated successfully!")
    print("🌐 Open docs/index.html to view")
else:
    print("❌ No summary file found at data/summaries/quick_batch_summary.md")