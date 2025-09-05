#!/usr/bin/env python3
"""
Research File Generator
Creates a new research file from template with proper date prefix naming convention

Usage:
    python create_research_file.py "Paper Title or Short Name"
    python create_research_file.py "INGRID robotic design"
"""

import os
import sys
import shutil
from datetime import datetime
import re

def sanitize_filename(name):
    """
    Convert a name to a valid filename format
    - Replace spaces with hyphens
    - Remove special characters
    - Convert to lowercase
    """
    # Remove special characters, keep only alphanumeric, spaces, and hyphens
    name = re.sub(r'[^a-zA-Z0-9\s\-]', '', name)
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    # Replace spaces with hyphens
    name = name.replace(' ', '-')
    # Convert to lowercase
    name = name.lower()
    # Remove multiple consecutive hyphens
    name = re.sub(r'-+', '-', name)
    # Strip leading/trailing hyphens
    name = name.strip('-')
    return name

def create_research_file(paper_name, custom_date=None):
    """
    Create a new research file from the template with proper naming
    
    Args:
        paper_name: The name/title of the paper
        custom_date: Optional custom date (YYYY-MM-DD format), defaults to today
    """
    # Paths
    template_path = "/Users/srinivaskarri/Desktop/aicoe-home/home/Private/02-templates/academic-research-summary-template.md"
    research_dir = "/Users/srinivaskarri/Desktop/aicoe-home/home/Private/03-research"
    
    # Check if template exists
    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}")
        return None
    
    # Create research directory if it doesn't exist
    os.makedirs(research_dir, exist_ok=True)
    
    # Generate date prefix
    if custom_date:
        try:
            # Validate date format
            datetime.strptime(custom_date, '%Y-%m-%d')
            date_prefix = custom_date
        except ValueError:
            print(f"Error: Invalid date format '{custom_date}'. Use YYYY-MM-DD format.")
            return None
    else:
        date_prefix = datetime.now().strftime('%Y-%m-%d')
    
    # Generate filename
    sanitized_name = sanitize_filename(paper_name)
    filename = f"{date_prefix}-{sanitized_name}.md"
    output_path = os.path.join(research_dir, filename)
    
    # Check if file already exists
    if os.path.exists(output_path):
        response = input(f"File {filename} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return None
    
    # Read template content
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update metadata in the template
    # Update the created_date in the frontmatter
    content = re.sub(
        r'created_date:\s*"[^"]*"',
        f'created_date: "{date_prefix}"',
        content
    )
    
    # Update the last_modified date
    content = re.sub(
        r'last_modified:\s*"[^"]*"',
        f'last_modified: "{date_prefix}"',
        content
    )
    
    # Update the review_date
    content = re.sub(
        r'review_date:\s*"[^"]*"',
        f'review_date: "{date_prefix}"',
        content
    )
    
    # If the paper name looks like a title, add it to the template
    if not sanitized_name.startswith('template') and not sanitized_name.startswith('test'):
        # Update the title in frontmatter
        content = re.sub(
            r'title:\s*"[^"]*"',
            f'title: "{paper_name}"',
            content,
            count=1
        )
        
        # Update the paper title
        content = re.sub(
            r'(paper:\s*\n\s*title:\s*)"[^"]*"',
            rf'\1"{paper_name}"',
            content
        )
    
    # Write the new file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Successfully created: {filename}")
    print(f"📁 Location: {output_path}")
    print(f"📅 Date prefix: {date_prefix}")
    print(f"📝 Paper name: {sanitized_name}")
    print("\nNext steps:")
    print("1. Open the file and fill in the research details")
    print("2. Update the metadata (authors, publication, DOI, etc.)")
    print("3. Complete the summary sections")
    print("4. Run the conversion script to generate HTML when done")
    
    return output_path

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Research File Generator")
        print("=" * 50)
        print("\nUsage:")
        print("  python create_research_file.py <paper-name> [date]")
        print("\nExamples:")
        print('  python create_research_file.py "INGRID robotic design"')
        print('  python create_research_file.py "Transformer architecture survey"')
        print('  python create_research_file.py "LLM evaluation metrics" "2025-01-10"')
        print("\nOptions:")
        print("  paper-name: The name or title of the paper (required)")
        print("  date: Optional date in YYYY-MM-DD format (defaults to today)")
        print("\nNaming Convention:")
        print("  Files will be saved as: YYYY-MM-DD-paper-name.md")
        print("  Example: 2025-09-05-ingrid-robotic-design.md")
        sys.exit(1)
    
    paper_name = sys.argv[1]
    custom_date = sys.argv[2] if len(sys.argv) > 2 else None
    
    create_research_file(paper_name, custom_date)

if __name__ == "__main__":
    main()