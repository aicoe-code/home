#!/usr/bin/env python3

import os
import re
import yaml
from pathlib import Path

def parse_frontmatter(content):
    """Parse YAML frontmatter from markdown content"""
    if not content.startswith('---'):
        return {}, content
    
    parts = content.split('---', 2)
    if len(parts) < 3:
        return {}, content
    
    try:
        metadata = yaml.safe_load(parts[1])
    except:
        metadata = {}
    
    markdown_content = parts[2]
    return metadata, markdown_content

def markdown_to_html(markdown):
    """Convert markdown to HTML"""
    html = markdown
    
    # Headers with section types
    html = re.sub(r'^### (.*?)( \*\*\[.*?\]\*\*)?$', r'<h3>\1<span class="section-type">\2</span></h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.*)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.*)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^##### (.*)$', r'<h5>\1</h5>', html, flags=re.MULTILINE)
    html = re.sub(r'^###### (.*)$', r'<h6>\1</h6>', html, flags=re.MULTILINE)
    
    # Bold and italic
    html = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', html)
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    
    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
    
    # Checkboxes
    html = re.sub(r'^- \[x\] (.*)$', r'<li class="checkbox-item"><input type="checkbox" checked disabled> \1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^- \[ \] (.*)$', r'<li class="checkbox-item"><input type="checkbox" disabled> \1</li>', html, flags=re.MULTILINE)
    
    # Regular lists
    html = re.sub(r'^- (.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^\d+\. (.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # Blockquotes
    html = re.sub(r'^> (.*)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
    
    # Code blocks
    html = re.sub(r'```(.*?)\n(.*?)```', r'<pre><code class="language-\1">\2</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Tables - simple approach
    lines = html.split('\n')
    in_table = False
    new_lines = []
    table_html = []
    
    for line in lines:
        if '|' in line and not line.strip().startswith('<!--'):
            if not in_table:
                in_table = True
                table_html = ['<table>']
            
            # Skip separator rows
            if '---' in line:
                continue
                
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                row_html = '<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>'
                table_html.append(row_html)
        else:
            if in_table:
                table_html.append('</table>')
                new_lines.append('\n'.join(table_html))
                table_html = []
                in_table = False
            new_lines.append(line)
    
    if in_table:
        table_html.append('</table>')
        new_lines.append('\n'.join(table_html))
    
    html = '\n'.join(new_lines)
    
    # Wrap consecutive list items in <ul>
    html = re.sub(r'(<li class="checkbox-item">.*?</li>\s*)+', r'<ul class="checkbox-list">\n\g<0></ul>', html, flags=re.DOTALL)
    html = re.sub(r'(?<!<ul.*?>)(<li>.*?</li>\s*)+(?!</ul>)', r'<ul>\n\g<0></ul>', html, flags=re.DOTALL)
    
    # Paragraphs
    paragraphs = []
    for para in html.split('\n\n'):
        para = para.strip()
        if para and not para.startswith('<') and not para.startswith('---'):
            paragraphs.append(f'<p>{para}</p>')
        else:
            paragraphs.append(para)
    html = '\n\n'.join(paragraphs)
    
    # Horizontal rules
    html = re.sub(r'^---$', '<hr>', html, flags=re.MULTILINE)
    
    return html

def generate_score_badge(label, score, max_score=5):
    """Generate HTML for a score badge"""
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    if percentage >= 80:
        color = '#10b981'
    elif percentage >= 60:
        color = '#f59e0b'
    else:
        color = '#ef4444'
    
    return f'''
        <div class="score-badge">
            <span class="score-label">{label}</span>
            <div class="score-bar">
                <div class="score-fill" style="width: {percentage}%; background-color: {color};"></div>
            </div>
            <span class="score-value">{score}/{max_score}</span>
        </div>
    '''

def generate_html(metadata, content):
    """Generate complete HTML from metadata and content"""
    html_content = markdown_to_html(content)
    
    # Extract nested metadata
    paper = metadata.get('paper', {})
    review = metadata.get('review', {})
    aicoe = metadata.get('aicoe_alignment', {})
    tags = metadata.get('tags', [])
    
    # Generate tags HTML
    tags_html = ''.join([f'<span class="tag">{tag}</span>' for tag in tags])
    
    # Generate alignment indicators
    alignment_items = []
    for area in ['understanding', 'design', 'deployment', 'operation']:
        active = 'active' if aicoe.get(area, False) else ''
        alignment_items.append(f'''
            <div class="alignment-item {active}">
                <span class="indicator">✓</span> {area.capitalize()}
            </div>
        ''')
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata.get('title', 'Research Summary')} - AICOE</title>
    <link rel="stylesheet" href="../styles.css">
</head>
<body>
    <div class="container">
        <nav class="breadcrumb">
            <a href="../home.html">Home</a> / 
            <a href="index.html">Research</a> / 
            <span>Current</span>
        </nav>
        
        <header class="research-header">
            <div class="metadata-card">
                <h1 class="research-title">{metadata.get('title', 'Research Summary')}</h1>
                
                <div class="paper-meta">
                    <span class="meta-item">📅 {review.get('review_date', 'N/A')}</span>
                    <span class="meta-item">📄 {paper.get('paper_type', 'N/A')}</span>
                    <span class="meta-item">🔗 <a href="https://arxiv.org/abs/{paper.get('arxiv_id', '')}" target="_blank">arXiv:{paper.get('arxiv_id', 'N/A')}</a></span>
                </div>
                
                <div class="tags">
                    {tags_html}
                </div>
            </div>
            
            <div class="scores-container">
                {generate_score_badge('Relevance', review.get('relevance_score', 0))}
                {generate_score_badge('Reproducibility', review.get('reproducibility_score', 0))}
                {generate_score_badge('Impact', review.get('impact_score', 0))}
            </div>
            
            <div class="alignment-indicators">
                <h4>AICOE Mission Alignment</h4>
                <div class="alignment-grid">
                    {''.join(alignment_items)}
                </div>
            </div>
            
            <div class="priority-badge priority-{metadata.get('priority', 'medium')}">
                Priority: {metadata.get('priority', 'medium')}
            </div>
        </header>
        
        <main class="research-content">
            {html_content}
        </main>
        
        <footer class="research-footer">
            <div class="footer-meta">
                <p>Reviewed by: {review.get('reviewer', 'N/A')}</p>
                <p>Confidence Level: {review.get('confidence_level', 'N/A')}</p>
                <p>Template Version: {metadata.get('template_version', 'N/A')}</p>
            </div>
            
            <div class="navigation-links">
                <a href="index.html" class="btn btn-secondary">← Back to Research</a>
            </div>
        </footer>
    </div>
</body>
</html>'''
    
    return html

def convert_file(input_path, output_path):
    """Convert a markdown file to HTML"""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    metadata, markdown = parse_frontmatter(content)
    html = generate_html(metadata, markdown)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Converted {input_path} to {output_path}")

if __name__ == "__main__":
    # Convert the INGRID paper
    input_file = "/Users/srinivaskarri/Desktop/aicoe-home/home/Private/03-research/2025-09-04-INGRID-robotic-design.md"
    output_file = "/Users/srinivaskarri/Desktop/aicoe-home/home/gh-pages/research/2025-09-04-ingrid.html"
    
    if os.path.exists(input_file):
        convert_file(input_file, output_file)
    else:
        print(f"Error: {input_file} not found")