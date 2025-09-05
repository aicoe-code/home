#!/usr/bin/env python3

import os
import re
import json

def parse_yaml_simple(yaml_text):
    """Simple YAML parser for frontmatter - handles basic key-value pairs"""
    metadata = {}
    current_key = None
    current_indent = 0
    
    for line in yaml_text.split('\n'):
        if not line.strip() or line.strip().startswith('#'):
            continue
            
        # Count leading spaces
        indent = len(line) - len(line.lstrip())
        
        # Check if it's a key-value pair
        if ':' in line:
            parts = line.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ''
            
            # Remove quotes
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            
            if indent == 0:  # Top-level key
                current_key = key
                if value:
                    metadata[key] = value
                else:
                    metadata[key] = {}
            elif indent > 0 and current_key:  # Nested key
                if isinstance(metadata[current_key], dict):
                    metadata[current_key][key] = value
                    
        # Handle list items
        elif line.strip().startswith('- '):
            item = line.strip()[2:].strip()
            # Remove quotes
            if item.startswith('"') and item.endswith('"'):
                item = item[1:-1]
            elif item.startswith("'") and item.endswith("'"):
                item = item[1:-1]
                
            if current_key:
                if not isinstance(metadata[current_key], list):
                    metadata[current_key] = []
                metadata[current_key].append(item)
    
    return metadata

def parse_frontmatter(content):
    """Parse YAML frontmatter from markdown content"""
    if not content.startswith('---'):
        return {}, content
    
    lines = content.split('\n')
    end_index = -1
    
    for i in range(1, len(lines)):
        if lines[i].strip() == '---':
            end_index = i
            break
    
    if end_index == -1:
        return {}, content
    
    yaml_text = '\n'.join(lines[1:end_index])
    markdown_content = '\n'.join(lines[end_index + 1:])
    
    metadata = parse_yaml_simple(yaml_text)
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
    html = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<em>\1</em>', html)
    
    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
    
    # Checkboxes
    html = re.sub(r'^- \[x\] (.*)$', r'<li class="checkbox-item"><input type="checkbox" checked disabled> \1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^- \[ \] (.*)$', r'<li class="checkbox-item"><input type="checkbox" disabled> \1</li>', html, flags=re.MULTILINE)
    
    # Regular lists
    html = re.sub(r'^- (?!\[[ x]\])(.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^\d+\. (.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # Blockquotes  
    html = re.sub(r'^> ☐ (.*)$', r'<blockquote>☐ \1</blockquote>', html, flags=re.MULTILINE)
    html = re.sub(r'^> (.*)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
    
    # Code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Tables - simple approach
    lines = html.split('\n')
    in_table = False
    new_lines = []
    table_html = []
    header_done = False
    
    for line in lines:
        if '|' in line and '<' not in line:  # Avoid XML sections
            if not in_table:
                in_table = True
                table_html = ['<table>']
                header_done = False
            
            # Skip separator rows
            if '---' in line:
                continue
                
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                if not header_done:
                    row_html = '<tr>' + ''.join(f'<th>{cell}</th>' for cell in cells) + '</tr>'
                    header_done = True
                else:
                    row_html = '<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>'
                table_html.append(row_html)
        else:
            if in_table:
                table_html.append('</table>')
                new_lines.append('\n'.join(table_html))
                table_html = []
                in_table = False
                header_done = False
            new_lines.append(line)
    
    if in_table:
        table_html.append('</table>')
        new_lines.append('\n'.join(table_html))
    
    html = '\n'.join(new_lines)
    
    # Wrap consecutive list items
    html = re.sub(r'(<li class="checkbox-item">.*?</li>\s*)+', r'<ul class="checkbox-list">\n\g<0></ul>', html, flags=re.DOTALL)
    # Simplified regex for regular lists
    html = re.sub(r'((?:<li>(?!class).*?</li>\s*)+)', r'<ul>\n\1</ul>', html, flags=re.DOTALL)
    
    # Horizontal rules
    html = re.sub(r'^---$', '<hr>', html, flags=re.MULTILINE)
    
    # Paragraphs - wrap non-HTML content
    paragraphs = []
    for para in html.split('\n\n'):
        para = para.strip()
        if para and not para.startswith('<'):
            if not re.match(r'^(#|>|\||-)+ ', para):
                paragraphs.append(f'<p>{para}</p>')
            else:
                paragraphs.append(para)
        else:
            paragraphs.append(para)
    html = '\n\n'.join(paragraphs)
    
    return html

def generate_html(metadata, content):
    """Generate complete HTML from metadata and content"""
    # Convert markdown content
    html_content = markdown_to_html(content)
    
    # Extract values with defaults
    title = metadata.get('title', 'Research Summary')
    priority = metadata.get('priority', 'medium')
    
    # Extract nested metadata safely
    paper = metadata.get('paper', {})
    if isinstance(paper, dict):
        paper_type = paper.get('paper_type', 'N/A')
        arxiv_id = paper.get('arxiv_id', '')
    else:
        paper_type = 'N/A'
        arxiv_id = ''
    
    review = metadata.get('review', {})
    if isinstance(review, dict):
        reviewer = review.get('reviewer', 'N/A')
        review_date = review.get('review_date', 'N/A')
        confidence_level = review.get('confidence_level', 'N/A')
        relevance_score = int(review.get('relevance_score', 0))
        reproducibility_score = int(review.get('reproducibility_score', 0))
        impact_score = int(review.get('impact_score', 0))
    else:
        reviewer = review_date = confidence_level = 'N/A'
        relevance_score = reproducibility_score = impact_score = 0
    
    aicoe = metadata.get('aicoe_alignment', {})
    if isinstance(aicoe, dict):
        understanding = aicoe.get('understanding', 'false') == 'true'
        design = aicoe.get('design', 'false') == 'true'
        deployment = aicoe.get('deployment', 'false') == 'true'
        operation = aicoe.get('operation', 'false') == 'true'
    else:
        understanding = design = deployment = operation = False
    
    tags = metadata.get('tags', [])
    if isinstance(tags, list):
        tags_html = ''.join([f'<span class="tag">{tag}</span>' for tag in tags])
    else:
        tags_html = ''
    
    # Generate score badges
    def score_badge(label, score, max_score=5):
        percentage = (score / max_score * 100) if max_score > 0 else 0
        color = '#10b981' if percentage >= 80 else '#f59e0b' if percentage >= 60 else '#ef4444'
        return f'''
            <div class="score-badge">
                <span class="score-label">{label}</span>
                <div class="score-bar">
                    <div class="score-fill" style="width: {percentage}%; background-color: {color};"></div>
                </div>
                <span class="score-value">{score}/{max_score}</span>
            </div>'''
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - AICOE</title>
    <link rel="stylesheet" href="../styles.css">
    <link rel="stylesheet" href="research.css">
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
                <h1 class="research-title">{title}</h1>
                
                <div class="paper-meta">
                    <span class="meta-item">📅 {review_date}</span>
                    <span class="meta-item">📄 {paper_type}</span>
                    <span class="meta-item">🔗 <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank">arXiv:{arxiv_id}</a></span>
                </div>
                
                <div class="tags">{tags_html}</div>
            </div>
            
            <div class="scores-container">
                {score_badge('Relevance', relevance_score)}
                {score_badge('Reproducibility', reproducibility_score)}
                {score_badge('Impact', impact_score)}
            </div>
            
            <div class="alignment-indicators">
                <h4>AICOE Mission Alignment</h4>
                <div class="alignment-grid">
                    <div class="alignment-item {'active' if understanding else ''}">
                        <span class="indicator">✓</span> Understanding
                    </div>
                    <div class="alignment-item {'active' if design else ''}">
                        <span class="indicator">✓</span> Design
                    </div>
                    <div class="alignment-item {'active' if deployment else ''}">
                        <span class="indicator">✓</span> Deployment
                    </div>
                    <div class="alignment-item {'active' if operation else ''}">
                        <span class="indicator">✓</span> Operation
                    </div>
                </div>
            </div>
            
            <div class="priority-badge priority-{priority}">
                Priority: {priority}
            </div>
        </header>
        
        <main class="research-content">
            {html_content}
        </main>
        
        <footer class="research-footer">
            <div class="footer-meta">
                <p>Reviewed by: {reviewer}</p>
                <p>Confidence Level: {confidence_level}</p>
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
    
    print(f"✓ Converted {os.path.basename(input_path)} to {os.path.basename(output_path)}")

if __name__ == "__main__":
    # Convert the INGRID paper
    input_file = "/Users/srinivaskarri/Desktop/aicoe-home/home/Private/03-research/2025-09-04-INGRID-robotic-design.md"
    output_file = "/Users/srinivaskarri/Desktop/aicoe-home/home/gh-pages/research/2025-09-04-ingrid.html"
    
    if os.path.exists(input_file):
        convert_file(input_file, output_file)
    else:
        print(f"Error: {input_file} not found")