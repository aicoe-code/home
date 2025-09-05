# Research Template Usage Guide

## File Naming Convention

All research files MUST follow this naming convention:
```
YYYY-MM-DD-descriptive-name.md
```

Examples:
- `2025-09-04-INGRID-robotic-design.md`
- `2025-09-05-transformer-architecture-survey.md`
- `2025-09-06-llm-evaluation-metrics.md`

## Automated File Creation

### Method 1: Using the Python Script (Recommended)

We provide a script that automatically creates properly named files from the template:

```bash
# Basic usage - creates file with today's date
python create_research_file.py "Paper Title"

# Examples
python create_research_file.py "INGRID robotic design"
python create_research_file.py "Cognitively faithful AI decision making"

# With custom date
python create_research_file.py "Paper Title" "2025-09-04"
```

The script will:
1. Automatically add the date prefix (today's date or custom)
2. Sanitize the paper name (convert to lowercase, replace spaces with hyphens)
3. Create the file in `/Private/03-research/`
4. Update the metadata dates in the template
5. Add the paper title to the frontmatter

### Method 2: Manual Creation

If creating files manually:

1. **Copy the template:**
   ```bash
   cp academic-research-summary-template.md ../03-research/YYYY-MM-DD-paper-name.md
   ```

2. **Use today's date format:** `YYYY-MM-DD`
   - 2025-09-05 (correct)
   - 2025-9-5 (incorrect - always use two digits)

3. **Format the paper name:**
   - Use lowercase
   - Replace spaces with hyphens
   - Remove special characters
   - Keep it descriptive but concise

## File Organization

```
/Private/
├── 02-templates/
│   ├── academic-research-summary-template.md  # The template
│   ├── create_research_file.py                # Creation script
│   └── README.md                              # This file
├── 03-research/
│   ├── 2025-09-04-INGRID-robotic-design.md   # Properly named
│   ├── 2025-09-05-llm-evaluation.md          # Properly named
│   └── cognitively-faithful-ai.md            # ❌ Missing date prefix
```

## Common Mistakes to Avoid

1. **Missing date prefix:**
   - ❌ `cognitively-faithful-ai-decision-making-summary.md`
   - ✅ `2025-09-05-cognitively-faithful-ai-decision-making.md`

2. **Wrong date format:**
   - ❌ `09-05-2025-paper-name.md` (US format)
   - ❌ `2025-9-5-paper-name.md` (missing zeros)
   - ❌ `20250905-paper-name.md` (no hyphens)
   - ✅ `2025-09-05-paper-name.md` (ISO format)

3. **Spaces in filename:**
   - ❌ `2025-09-05-INGRID robotic design.md`
   - ✅ `2025-09-05-INGRID-robotic-design.md`

4. **Special characters:**
   - ❌ `2025-09-05-paper:new-approach!.md`
   - ✅ `2025-09-05-paper-new-approach.md`

## Converting to HTML

After completing your research summary:

```bash
# Navigate to scripts directory
cd /Users/srinivaskarri/Desktop/aicoe-home/home/gh-pages/research/scripts/

# Convert using Python
python convert.py

# Or using Node.js
node convert-md-to-html.js ../../Private/03-research/2025-09-05-paper-name.md
```

## Quick Checklist

Before saving a new research file:

- [ ] Date prefix is in YYYY-MM-DD format
- [ ] Date uses today's date or paper review date
- [ ] Filename uses hyphens, not spaces
- [ ] Filename is lowercase (except acronyms)
- [ ] No special characters in filename
- [ ] File is saved in `/Private/03-research/`
- [ ] Metadata dates are updated in frontmatter

## Support

If you accidentally created a file without the proper naming:

```bash
# Rename existing file (example)
mv ../03-research/cognitively-faithful-ai.md \
   ../03-research/2025-09-05-cognitively-faithful-ai.md
```

Or use the creation script to generate a properly named version and copy your content over.