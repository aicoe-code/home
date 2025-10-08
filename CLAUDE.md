# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the AI Center of Excellence (AICoE) public website repository - a static HTML site with a black and white minimal design system. The site is hosted on GitHub Pages and serves as the public face for the AI Center of Excellence organization.

**Repository:** `git@github.com:aicoe-code/home.git`

## Site Architecture

### Pure Static HTML
This is a **pure static HTML site** with no build process, bundlers, or frameworks. Each page is a self-contained HTML file with inline CSS and minimal JavaScript for icon rendering.

**Key architectural decisions:**
- No build step or dependencies
- Inline CSS in `<style>` tags for zero external requests
- Lucide icons loaded via CDN (`https://unpkg.com/lucide@latest`)
- CSS custom properties (CSS variables) for consistent theming
- Mobile-first responsive design

### Directory Structure

```
docs/                                   # GitHub Pages serves from this directory
├── index.html                          # Landing page with hero and service cards
├── mission.html                        # Centered mission statement page
├── about.html                          # About page with structured content
├── resources.html                      # Resources hub with 3 main sections
├── transformation-framework.html       # AI transformation framework details
├── ai-readiness-assessment.html       # Readiness assessment (placeholder)
├── implementation-roadmap.html        # Implementation roadmap (placeholder)
├── ai-architecture-patterns.html      # Architecture patterns (placeholder)
├── code-templates.html                # Code templates (placeholder)
├── best-practices.html                # Best practices guide (placeholder)
├── case-studies.html                  # Case studies (placeholder)
├── technical-documentation.html       # Technical docs (placeholder)
├── 404.html                           # Error page
├── design/                            # Design system (gitignored)
│   ├── design-guidelines.html         # Complete design system documentation
│   └── example-design.html            # Full-featured design example
├── templates/                         # Page templates (gitignored)
│   ├── README.md                      # Template usage documentation
│   ├── basic-page.html                # Simple content page
│   ├── content-page.html              # Cards and feature lists
│   ├── landing-page.html              # Hero with CTAs
│   └── resource-hub.html              # Filterable resource listing
└── resources/                         # Resource assets (gitignored)
```

## Design System

### Black & White Minimal Aesthetic
The entire site follows a strict black and white color palette with grayscale variations:

**CSS Variables (defined in every page):**
```css
--black: #000000
--white: #FFFFFF
--gray-100 through --gray-900 (9 shades)
```

### Typography Scale
- **Hero H1:** 5rem (80px), weight 300, letter-spacing -0.03em
- **Page H1:** 3.5rem (56px), weight 300, letter-spacing -0.03em
- **H2:** 2rem (32px), weight 400
- **Body:** 1rem (16px), weight 400
- **Font stack:** `-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif`

### Layout Constants
- **Max width:** 1400px for main containers
- **Section padding:** 6rem vertical, 3rem horizontal
- **Grid gap:** 3rem
- **Mobile breakpoint:** 768px

### Component Patterns

**Navigation:**
- Fixed position at top
- White background with 1px gray border-bottom
- Logo left, links right
- Consistent across all pages

**Cards:**
- White background, 1px gray border
- 2.5rem padding
- Hover: border changes to black, subtle shadow
- Black icon square at top (48px × 48px)

**Icons:**
- Uses Lucide Icons via CDN
- Rendered via `lucide.createIcons()` JavaScript
- Change icons by updating `data-lucide="icon-name"` attributes

## Creating New Pages

### Using Templates
1. Copy appropriate template from `docs/templates/`
2. Rename to your page name (e.g., `services.html`)
3. Update page `<title>` and navigation links
4. Replace placeholder content
5. Update Lucide icon names if needed

### Template Selection
- **basic-page.html** - Simple text content, headings
- **content-page.html** - Service pages, cards, feature lists
- **landing-page.html** - Hero sections, CTAs, stats
- **resource-hub.html** - Filterable lists, documentation hubs

### Navigation Updates
When adding new pages, update the navigation in ALL existing pages (13 pages total):

**Current Navigation Structure:**
```html
<ul class="nav-links">
    <li><a href="mission.html">Mission</a></li>
    <li><a href="resources.html">Resources</a></li>
    <li><a href="about.html">About</a></li>
</ul>
```

**Pages to update when changing navigation:**
- index.html
- mission.html
- about.html
- resources.html
- transformation-framework.html
- ai-readiness-assessment.html
- implementation-roadmap.html
- ai-architecture-patterns.html
- code-templates.html
- best-practices.html
- case-studies.html
- technical-documentation.html
- 404.html

## Design System Maintenance

### CSS Variables Location
All pages define CSS variables in the `:root` selector at the top of their `<style>` block. To maintain consistency:

1. Always use CSS variables instead of hardcoded colors
2. Copy the complete `:root` block when creating new pages
3. Never deviate from the defined color palette

### Icon System
**Adding new icons:**
1. Browse Lucide icons: https://lucide.dev/icons/
2. Add icon element: `<i data-lucide="icon-name"></i>`
3. Icons auto-render via included script

**Common icons used:**
- `arrow-right` - Links, CTAs
- `compass` - Navigation
- `code` - Technical content
- `book-open` - Learning/research
- `github` - GitHub links

## Deployment

### GitHub Pages
Site is automatically deployed from the `docs/` directory via GitHub Pages.

**To deploy changes:**
1. Make edits to HTML files in `docs/`
2. Commit changes
3. Push to `main` branch
4. GitHub Pages auto-deploys (2-3 minutes)

**Important:** There is no build process. HTML files are served directly as-is.

## Common Tasks

### Viewing Locally
Simply open any HTML file in a browser:
```bash
open docs/index.html
```

Or use a local server:
```bash
cd docs
python3 -m http.server 8000
# Visit http://localhost:8000
```

### Updating Design System
To change global design elements (colors, typography, spacing):
1. Update design guidelines: `docs/design/design-guidelines.html`
2. Update all page files individually (no global CSS file)
3. Update templates in `docs/templates/`
4. Test all pages for consistency

### Adding New Icons
No installation needed. Just add the `data-lucide` attribute:
```html
<i data-lucide="new-icon-name"></i>
```

## Page Structure Pattern

Every page follows this structure:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Title - AI Center of Excellence</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/lucide-static@0.16.29/font/lucide.min.css">
    <style>
        /* CSS variables */
        /* Component styles */
        /* Responsive breakpoints */
    </style>
</head>
<body>
    <!-- Navigation -->
    <!-- Content -->
    <!-- Footer (optional) -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <script>lucide.createIcons();</script>
</body>
</html>
```

## Design Philosophy

**Core principles:**
1. **Minimal & Modern:** Clean lines, generous whitespace, sophisticated simplicity
2. **Black & White Only:** Pure monochrome palette removes distraction
3. **No Build Complexity:** Static HTML for maximum simplicity and reliability
4. **Self-Contained Pages:** Each page has all CSS inline, no external dependencies
5. **Mobile-First:** Responsive design tested at 768px breakpoint

## File Naming Conventions

- Use kebab-case for filenames: `about.html`, `design-guidelines.html`
- Page names should be descriptive and lowercase
- No special characters except hyphens

## Current Site Status

### Complete Pages
- **index.html** - Landing page with hero section and service cards
- **mission.html** - Centered mission statement
- **about.html** - About page with structured content sections
- **resources.html** - Resources hub with 3 main sections (Frameworks, Tools, Documentation)
- **transformation-framework.html** - Full transformation framework content
- **404.html** - Minimal error page

### Placeholder Pages (Under Construction)
These pages have navigation, page structure, and placeholder content boxes:
- **ai-readiness-assessment.html**
- **implementation-roadmap.html**
- **ai-architecture-patterns.html**
- **code-templates.html**
- **best-practices.html**
- **case-studies.html**
- **technical-documentation.html**

All placeholder pages include back links to resources.html and follow the standard design system.

## Resources Section Structure

The resources.html page organizes content into three main sections:

1. **Frameworks & Methodologies** - Strategic approaches and assessments
2. **Tools & Code** - GitHub repos, architecture patterns, templates
3. **Documentation & Guides** - Best practices, case studies, technical docs

Each section uses a card grid layout with Lucide icons and hover effects.

## Important Notes

- **No JavaScript frameworks** - Site uses vanilla HTML/CSS/JS only
- **No CSS preprocessors** - Plain CSS with modern features (variables, grid, flexbox)
- **No package.json** - No dependencies, no npm, no build tools
- **Inline everything** - All CSS is inline, no external stylesheets (except Lucide)
- **Git workflow** - Direct commits to main, no build step needed
- **13 total pages** - 6 complete, 7 placeholders waiting for content
- **Gitignored directories** - design/, templates/, and resources/ excluded from repo

## Reference Documentation

- **Design Guidelines:** `docs/design/design-guidelines.html` - Complete design system specs
- **Template Docs:** `docs/templates/README.md` - How to use page templates
- **Example Page:** `docs/design/example-design.html` - Full-featured reference implementation
- **Lucide Icons:** https://lucide.dev/icons/ - Icon library documentation
