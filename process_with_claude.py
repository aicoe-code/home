#!/usr/bin/env python3
"""
Interactive Claude Code Processing Script
Main script for processing papers with Claude Code instead of API
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import subprocess
import time

# Add pipeline to path
sys.path.insert(0, 'pipeline')

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🤖 {title}")
    print("="*60)

def print_step(step_num, title):
    """Print a step header"""
    print(f"\n{'─'*50}")
    print(f"Step {step_num}: {title}")
    print('─'*50)

def fetch_papers():
    """Fetch latest papers from arXiv"""
    print_step(1, "Fetching Papers from arXiv")
    
    response = input("\n📅 Fetch papers from how many days ago? (default: 1): ").strip()
    days = int(response) if response else 1
    
    print(f"\n🔄 Fetching papers from {days} day(s) ago...")
    
    cmd = f"cd pipeline && python3 fetcher.py --days {days}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Parse output to find saved file
        for line in result.stdout.split('\n'):
            if 'Saved' in line and 'papers to' in line:
                print(f"✅ {line.strip()}")
            elif 'Total:' in line:
                print(f"📊 {line.strip()}")
        
        # Find the generated file
        raw_files = list(Path('data/raw').glob('*.json'))
        if raw_files:
            latest_file = max(raw_files, key=lambda f: f.stat().st_mtime)
            return str(latest_file)
    else:
        print(f"❌ Error fetching papers: {result.stderr}")
    
    return None

def load_papers(filepath):
    """Load papers from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('papers', [])

def select_papers_to_process(papers):
    """Let user select which papers to process"""
    print_step(2, "Select Papers to Process")
    
    # Sort by relevance
    papers_sorted = sorted(papers, key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    print(f"\n📚 Found {len(papers)} papers")
    print("\nTop 20 papers by relevance:\n")
    
    for i, paper in enumerate(papers_sorted[:20], 1):
        score = paper.get('relevance_score', 0)
        title = paper['title'][:70] + ('...' if len(paper['title']) > 70 else '')
        
        # Color code by relevance
        if score > 5:
            marker = "🔴"  # High
        elif score > 2:
            marker = "🟡"  # Medium
        else:
            marker = "⚪"  # Low
        
        print(f"{i:2}. {marker} [{score:4.1f}] {title}")
    
    print("\n" + "─"*50)
    print("Options:")
    print("  1-20: Select specific paper numbers (comma-separated)")
    print("  'top': Process top 10 papers")
    print("  'all': Process all papers")
    print("  'high': Process high relevance only (score > 5)")
    print("  'quit': Exit without processing")
    
    choice = input("\n📝 Your choice: ").strip().lower()
    
    if choice == 'quit':
        return []
    elif choice == 'all':
        return papers_sorted
    elif choice == 'top':
        return papers_sorted[:10]
    elif choice == 'high':
        return [p for p in papers_sorted if p.get('relevance_score', 0) > 5]
    else:
        # Parse specific numbers
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected = [papers_sorted[i] for i in indices if 0 <= i < len(papers_sorted)]
            return selected
        except:
            print("Invalid selection, using top 10")
            return papers_sorted[:10]

def prepare_prompts(papers):
    """Prepare prompts for Claude Code"""
    print_step(3, "Preparing Prompts for Claude Code")
    
    from pipeline.local_processor import LocalClaudeProcessor
    
    processor = LocalClaudeProcessor()
    
    print(f"\n📝 Preparing prompts for {len(papers)} papers...")
    
    # Ask for processing mode
    print("\nProcessing mode:")
    print("  1. Individual prompts (one file per paper)")
    print("  2. Batch prompt (all papers in one file)")
    
    mode = input("\n選択 Choice [1/2]: ").strip()
    
    if mode == '2':
        # Batch mode
        prompt_file = processor.quick_process(papers, len(papers))
        return 'batch', prompt_file
    else:
        # Individual mode
        prompt_files = processor.prepare_prompts(papers)
        return 'individual', prompt_files

def guide_claude_processing(mode, prompt_data):
    """Guide user through Claude Code processing"""
    print_step(4, "Process with Claude Code")
    
    print("\n" + "🤖 "*10)
    print("\n📋 CLAUDE CODE PROCESSING INSTRUCTIONS\n")
    
    if mode == 'batch':
        prompt_file = prompt_data
        print(f"1. Open this file: {prompt_file}")
        print("2. Select all content (Cmd+A) and copy (Cmd+C)")
        print("3. Paste into Claude Code")
        print("4. Save Claude's response to: data/summaries/batch_summary.md")
    else:
        prompt_files = prompt_data
        print(f"📁 {len(prompt_files)} prompt files created in: data/prompts/")
        print("\nFor each prompt file:")
        print("1. Open the file from data/prompts/")
        print("2. Copy entire content")
        print("3. Paste into Claude Code")
        print("4. Save response to data/summaries/ (change _prompt.txt to _summary.md)")
        
        if len(prompt_files) <= 5:
            print("\nFiles to process:")
            for f in prompt_files:
                print(f"  • {f}")
    
    print("\n" + "🤖 "*10)
    
    input("\n⏸️  Press Enter when you've completed processing with Claude...")

def process_summaries():
    """Process completed summaries"""
    print_step(5, "Processing Completed Summaries")
    
    from pipeline.local_processor import LocalClaudeProcessor
    
    processor = LocalClaudeProcessor()
    
    # Check for summaries
    summaries = processor.check_summaries()
    
    if not summaries:
        print("❌ No summaries found in data/summaries/")
        print("Please process papers with Claude Code first")
        return False
    
    print(f"\n✅ Found {len(summaries)} completed summaries")
    
    # Process them
    processed = processor.process_completed_summaries()
    
    if processed:
        print(f"\n✅ Successfully processed {len(processed)} papers")
        return True
    
    return False

def generate_html():
    """Generate HTML website"""
    print_step(6, "Generating HTML Website")
    
    print("\n🌐 Generating website...")
    
    cmd = "cd pipeline && python3 simple_generator.py"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Website generated successfully!")
        print("📁 Output: docs/index.html")
        
        # Ask if user wants to open
        if input("\n🌐 Open website in browser? [y/N]: ").strip().lower() == 'y':
            subprocess.run("open docs/index.html", shell=True)
        
        return True
    else:
        print(f"❌ Error generating HTML: {result.stderr}")
        return False

def main():
    """Main interactive processing flow"""
    print_header("AI PAPER PROCESSING WITH CLAUDE CODE")
    
    print("\n🎯 This script will help you:")
    print("  1. Fetch latest AI papers from arXiv")
    print("  2. Select papers to process")
    print("  3. Generate prompts for Claude Code")
    print("  4. Process with Claude (you do this)")
    print("  5. Generate HTML website")
    
    # Check for existing papers
    raw_files = list(Path('data/raw').glob('*.json'))
    
    if raw_files:
        latest = max(raw_files, key=lambda f: f.stat().st_mtime)
        age_days = (datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)).days
        
        print(f"\n📂 Found existing papers from {age_days} days ago")
        use_existing = input("Use existing papers? [Y/n]: ").strip().lower() != 'n'
        
        if use_existing:
            papers_file = str(latest)
        else:
            papers_file = fetch_papers()
    else:
        papers_file = fetch_papers()
    
    if not papers_file:
        print("❌ No papers to process")
        return
    
    # Load and select papers
    papers = load_papers(papers_file)
    selected_papers = select_papers_to_process(papers)
    
    if not selected_papers:
        print("👋 No papers selected. Goodbye!")
        return
    
    # Prepare prompts
    mode, prompt_data = prepare_prompts(selected_papers)
    
    # Guide through Claude processing
    guide_claude_processing(mode, prompt_data)
    
    # Process summaries
    if process_summaries():
        # Generate HTML
        if generate_html():
            print("\n" + "="*60)
            print("🎉 SUCCESS! Pipeline complete!")
            print("="*60)
            print("\n📊 Summary:")
            print(f"  • Papers processed: {len(selected_papers)}")
            print(f"  • Website updated: docs/index.html")
            
            print("\n📤 To publish to GitHub Pages:")
            print("  git add -A")
            print("  git commit -m 'Update papers with Claude Code'")
            print("  git push origin gh-pages")
    
    print("\n✨ Done!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()