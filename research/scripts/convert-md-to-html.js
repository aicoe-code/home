#!/usr/bin/env node

/**
 * Markdown to HTML Converter for Research Summaries
 * Converts research summary markdown files with YAML frontmatter to beautiful HTML
 */

const fs = require('fs');
const path = require('path');

// Simple YAML parser for frontmatter
function parseYAMLFrontmatter(content) {
    const lines = content.split('\n');
    if (lines[0] !== '---') return { metadata: {}, content };
    
    let yamlEnd = -1;
    for (let i = 1; i < lines.length; i++) {
        if (lines[i] === '---') {
            yamlEnd = i;
            break;
        }
    }
    
    if (yamlEnd === -1) return { metadata: {}, content };
    
    const yamlContent = lines.slice(1, yamlEnd).join('\n');
    const markdownContent = lines.slice(yamlEnd + 1).join('\n');
    
    // Simple YAML parsing (basic implementation)
    const metadata = {};
    let currentKey = null;
    let currentIndent = 0;
    
    yamlContent.split('\n').forEach(line => {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) return;
        
        const colonIndex = trimmed.indexOf(':');
        if (colonIndex > 0) {
            const key = trimmed.substring(0, colonIndex).trim();
            const value = trimmed.substring(colonIndex + 1).trim();
            
            if (value) {
                metadata[key] = value.replace(/^["']|["']$/g, '');
            } else {
                currentKey = key;
                metadata[key] = [];
            }
        } else if (trimmed.startsWith('- ') && currentKey) {
            const value = trimmed.substring(2).replace(/^["']|["']$/g, '');
            if (Array.isArray(metadata[currentKey])) {
                metadata[currentKey].push(value);
            }
        }
    });
    
    return { metadata, content: markdownContent };
}

// Convert markdown to HTML (basic implementation)
function markdownToHTML(markdown) {
    let html = markdown;
    
    // Headers
    html = html.replace(/^### (.*?)( \*\*\[.*?\]\*\*)?$/gm, '<h3>$1<span class="section-type">$2</span></h3>');
    html = html.replace(/^## (.*)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*)$/gm, '<h1>$1</h1>');
    html = html.replace(/^#### (.*)$/gm, '<h4>$1</h4>');
    html = html.replace(/^##### (.*)$/gm, '<h5>$1</h5>');
    html = html.replace(/^###### (.*)$/gm, '<h6>$1</h6>');
    
    // Bold and italic
    html = html.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
    
    // Lists
    html = html.replace(/^- \[([ x])\] (.*)$/gm, (match, checked, text) => {
        const isChecked = checked === 'x' ? 'checked' : '';
        return `<li class="checkbox-item"><input type="checkbox" ${isChecked} disabled> ${text}</li>`;
    });
    html = html.replace(/^- (.*)$/gm, '<li>$1</li>');
    html = html.replace(/^\d+\. (.*)$/gm, '<li>$1</li>');
    
    // Blockquotes
    html = html.replace(/^> (.*)$/gm, '<blockquote>$1</blockquote>');
    
    // Code blocks
    html = html.replace(/```(.*?)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Tables
    html = html.replace(/\|(.+)\|/g, (match) => {
        const cells = match.split('|').filter(cell => cell.trim());
        const isHeader = cells.every(cell => cell.includes('---'));
        
        if (isHeader) return '';
        
        const cellTags = cells.map(cell => `<td>${cell.trim()}</td>`).join('');
        return `<tr>${cellTags}</tr>`;
    });
    
    // Wrap lists
    html = html.replace(/(<li.*?>.*?<\/li>\s*)+/g, (match) => {
        if (match.includes('checkbox-item')) {
            return `<ul class="checkbox-list">${match}</ul>`;
        }
        return `<ul>${match}</ul>`;
    });
    
    // Wrap tables
    html = html.replace(/(<tr>.*?<\/tr>\s*)+/g, '<table>$&</table>');
    
    // Paragraphs
    html = html.split('\n\n').map(para => {
        if (para.trim() && !para.startsWith('<')) {
            return `<p>${para.trim()}</p>`;
        }
        return para;
    }).join('\n\n');
    
    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr>');
    
    return html;
}

// Generate score badge HTML
function generateScoreBadge(label, score, maxScore = 5) {
    const percentage = (score / maxScore) * 100;
    const color = percentage >= 80 ? '#10b981' : percentage >= 60 ? '#f59e0b' : '#ef4444';
    
    return `
        <div class="score-badge">
            <span class="score-label">${label}</span>
            <div class="score-bar">
                <div class="score-fill" style="width: ${percentage}%; background-color: ${color};"></div>
            </div>
            <span class="score-value">${score}/${maxScore}</span>
        </div>
    `;
}

// Generate HTML from parsed content
function generateHTML(metadata, content) {
    const htmlContent = markdownToHTML(content);
    
    // Extract paper metadata
    const paper = metadata.paper || {};
    const review = metadata.review || {};
    const aicoe = metadata.aicoe_alignment || {};
    
    const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${metadata.title || 'Research Summary'} - AICOE</title>
    <link rel="stylesheet" href="../styles.css">
</head>
<body>
    <div class="container">
        <nav class="breadcrumb">
            <a href="../index.html">Home</a> / 
            <a href="index.html">Research</a> / 
            <span>Current</span>
        </nav>
        
        <header class="research-header">
            <div class="metadata-card">
                <h1 class="research-title">${metadata.title || 'Research Summary'}</h1>
                
                <div class="paper-meta">
                    <span class="meta-item">📅 ${review.review_date || 'N/A'}</span>
                    <span class="meta-item">📄 ${paper.paper_type || 'N/A'}</span>
                    <span class="meta-item">🔗 <a href="https://arxiv.org/abs/${paper.arxiv_id || ''}" target="_blank">arXiv:${paper.arxiv_id || 'N/A'}</a></span>
                </div>
                
                <div class="tags">
                    ${(metadata.tags || []).map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
            </div>
            
            <div class="scores-container">
                ${generateScoreBadge('Relevance', review.relevance_score || 0)}
                ${generateScoreBadge('Reproducibility', review.reproducibility_score || 0)}
                ${generateScoreBadge('Impact', review.impact_score || 0)}
            </div>
            
            <div class="alignment-indicators">
                <h4>AICOE Mission Alignment</h4>
                <div class="alignment-grid">
                    <div class="alignment-item ${aicoe.understanding ? 'active' : ''}">
                        <span class="indicator">✓</span> Understanding
                    </div>
                    <div class="alignment-item ${aicoe.design ? 'active' : ''}">
                        <span class="indicator">✓</span> Design
                    </div>
                    <div class="alignment-item ${aicoe.deployment ? 'active' : ''}">
                        <span class="indicator">✓</span> Deployment
                    </div>
                    <div class="alignment-item ${aicoe.operation ? 'active' : ''}">
                        <span class="indicator">✓</span> Operation
                    </div>
                </div>
            </div>
            
            <div class="priority-badge priority-${metadata.priority || 'medium'}">
                Priority: ${metadata.priority || 'medium'}
            </div>
        </header>
        
        <main class="research-content">
            ${htmlContent}
        </main>
        
        <footer class="research-footer">
            <div class="footer-meta">
                <p>Reviewed by: ${review.reviewer || 'N/A'}</p>
                <p>Confidence Level: ${review.confidence_level || 'N/A'}</p>
                <p>Template Version: ${metadata.template_version || 'N/A'}</p>
            </div>
            
            <div class="navigation-links">
                <a href="index.html" class="btn btn-secondary">← Back to Research</a>
            </div>
        </footer>
    </div>
</body>
</html>`;
    
    return html;
}

// Main conversion function
function convertMarkdownToHTML(inputPath, outputPath) {
    try {
        const content = fs.readFileSync(inputPath, 'utf-8');
        const { metadata, content: markdown } = parseYAMLFrontmatter(content);
        const html = generateHTML(metadata, markdown);
        
        fs.writeFileSync(outputPath, html);
        console.log(`✓ Converted ${path.basename(inputPath)} to ${path.basename(outputPath)}`);
    } catch (error) {
        console.error(`✗ Error converting ${inputPath}:`, error.message);
    }
}

// CLI usage
if (require.main === module) {
    const args = process.argv.slice(2);
    
    if (args.length < 1) {
        console.log('Usage: node convert-md-to-html.js <input.md> [output.html]');
        console.log('Example: node convert-md-to-html.js research-summary.md research-summary.html');
        process.exit(1);
    }
    
    const inputPath = args[0];
    const outputPath = args[1] || inputPath.replace(/\.md$/, '.html');
    
    convertMarkdownToHTML(inputPath, outputPath);
}

module.exports = { convertMarkdownToHTML, parseYAMLFrontmatter, markdownToHTML };