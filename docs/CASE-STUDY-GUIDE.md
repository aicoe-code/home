# Case Study Template - Usage Guide

## Overview

This guide explains how to use `case-study-template.html` to create compelling case studies that demonstrate AICoE's value creation.

## Quick Start

1. **Copy the template**:
   ```bash
   cp docs/case-study-template.html docs/case-study-[client-name].html
   ```

2. **Search for `[INSTRUCTIONS:`** markers in the HTML - these indicate sections to update

3. **Replace all `[Bracketed Placeholders]`** with actual content

4. **Update the page title** in the `<title>` tag

5. **Test locally** by opening in a browser

6. **Commit and push** to deploy via GitHub Pages

---

## Template Structure

### 1. Page Header Section
**Location:** Top of page after navigation

**Fields to update:**
- **Industry**: Healthcare, Manufacturing, Financial Services, Legal, Life Sciences, etc.
- **Company Size**: Enterprise (1000+ employees), Mid-Market (100-1000), Startup (<100)
- **Timeline**: How long the engagement took (e.g., "8 weeks", "3 months")
- **ROI**: The multiplier (e.g., "6X ROI", "10X ROI")
- **Title**: Format: `[Company/Project Name]: [Brief Value Statement]`
  - Example: "Fortune 500 Healthcare: 8X ROI Through AI-Powered Claims Processing"
- **Subtitle**: One sentence summarizing the transformation
  - Example: "Reduced claims processing time from 48 hours to 2 hours while improving accuracy by 35%"

---

### 2. Overview Section
**Purpose:** Set the context

**What to include:**
- Who is the client (can be anonymized: "A Fortune 500 healthcare company")
- Industry and market position
- Scale of operations (number of employees, customers, transactions, etc.)
- Strategic situation (growth phase, digital transformation initiative, etc.)
- Why they engaged with AICoE

**Length:** 2-3 paragraphs

**Example:**
> "A Fortune 500 healthcare insurance provider processing over 10 million claims annually was struggling with manual review processes that created bottlenecks and delayed customer reimbursements. With a strategic initiative to improve customer satisfaction and operational efficiency, they engaged AICoE to explore AI-powered automation solutions."

---

### 3. Challenge Section
**Purpose:** Paint the "before" picture with specific pain points

**Guidelines:**
- Use 2-4 challenge cards
- Each card should focus on ONE specific problem
- Include **metrics** where possible (time, cost, volume, error rates)
- Focus on **business impact**, not just technical issues
- Choose icons from [Lucide Icons](https://lucide.dev/icons/)

**Good Challenge Examples:**
- ❌ "Old technology stack" (too vague)
- ✅ "Processing 10,000+ insurance claims manually each day, resulting in 48-hour turnaround times and customer dissatisfaction"

- ❌ "Data problems" (too vague)
- ✅ "Inconsistent data across 12 legacy systems prevented unified customer view, causing duplicate work and errors"

**Icon Suggestions:**
- `alert-circle` - General problems/issues
- `trending-down` - Declining performance
- `clock` - Time/speed issues
- `users` - People/staffing challenges
- `x-circle` - Blockers/limitations

---

### 4. Solution Section
**Purpose:** Describe the approach and what was built

**What to include:**
- **Solution title**: Give the solution a name
- **Overall approach**: 1-2 paragraphs on strategy
- **Key Components**: 4-6 bullet points of what was built
  - Be specific but not overly technical
  - Focus on capabilities rather than code
- **Technologies**: List the tech stack

**Example Key Components:**
```
✅ "Custom NLP pipeline trained on 500K historical claims to auto-classify by complexity"
✅ "Real-time dashboard showing AI predictions with confidence scores for human review"
✅ "Automated routing system directing high-confidence claims for immediate processing"
```

❌ Avoid: "Built a machine learning model" (too generic)

**Technology Tags:**
Include languages, frameworks, platforms, and tools. Examples:
- Python, TensorFlow, PyTorch, Scikit-learn
- AWS, Azure, GCP
- Docker, Kubernetes
- React, FastAPI
- PostgreSQL, MongoDB

---

### 5. Implementation Journey
**Purpose:** Show the phases/timeline of delivery

**Guidelines:**
- Use 3-4 phase cards
- Show progression from discovery to deployment
- Include timeframes if possible
- Highlight quick wins or proof points

**Typical Phases:**
1. **Discovery & Assessment** (1-2 weeks)
   - Stakeholder interviews, data analysis, identifying quick wins

2. **Rapid Prototyping** (2-4 weeks)
   - Build working proof-of-concept, demonstrate initial value

3. **Production Development** (4-8 weeks)
   - Full implementation, integration, testing

4. **Deployment & Scale** (2-4 weeks)
   - Production rollout, monitoring, optimization

**Icon Suggestions:**
- `search` - Discovery/research
- `lightbulb` - Ideation/design
- `code` - Development
- `rocket` - Launch/deployment
- `trending-up` - Optimization/scale

---

### 6. Results Section ⭐ MOST IMPORTANT
**Purpose:** Prove the value created with hard metrics

**Guidelines:**
- **This is the most critical section** - be specific and quantitative
- Use the 4 metric cards for headline numbers
- Each metric should have a big number and clear label
- Follow with 3 cards explaining business/operational/strategic impact

**Metric Card Examples:**
```
6X  →  Return on Investment
85%  →  Reduction in Processing Time
$2.4M  →  Annual Cost Savings
99.2%  →  Accuracy Rate
10,000+  →  Hours Saved Annually
42%  →  Increase in Customer Satisfaction
```

**What makes strong metrics:**
- ✅ Quantifiable (numbers, percentages, dollar amounts)
- ✅ Business-relevant (tie to revenue, cost, time, quality)
- ✅ Verifiable (could be validated if asked)
- ❌ Vague ("significant improvement", "much faster")
- ❌ Only technical ("model accuracy" without business context)

**Impact Categories:**

1. **Business Impact**
   - Revenue increase/protection
   - Cost savings
   - ROI calculation
   - Market share or competitive advantage

2. **Operational Impact**
   - Time saved
   - Efficiency gains
   - Process improvements
   - People impacted

3. **Strategic Impact**
   - New capabilities unlocked
   - Scalability improvements
   - Future opportunities enabled
   - Risk reduction

---

### 7. Quote Section (Optional)
**Purpose:** Add credibility and human element

**Guidelines:**
- Use actual client testimonial if available
- Get written permission before publishing
- Keep it authentic - avoid marketing speak
- Can be attributed to role rather than name if needed

**Good Quote Format:**
> "The team delivered a working prototype in 10 days that immediately showed us the potential. Within 8 weeks we were processing claims 85% faster. This has fundamentally changed how we operate."
>
> **John Smith**, VP of Operations at [Company]

**If no quote available:** Remove this entire section - don't fake it!

---

### 8. Key Takeaways Section
**Purpose:** Extract lessons and insights

**Two Cards:**

1. **What Worked** (Success Factors)
   - 3-5 items that contributed to success
   - Example: "Starting with a focused 2-week pilot reduced risk and proved ROI before full investment"

2. **Lessons Learned** (Insights)
   - 3-5 learnings from the engagement
   - Example: "Clean, well-labeled training data was 10X more important than complex model architecture"

**These should be:**
- Specific to this case study
- Actionable or educational
- Honest (can mention challenges overcome)

---

## Writing Best Practices

### Tell a Story
Structure your case study as a narrative:
1. **Setup** (Overview) - Who and why
2. **Problem** (Challenge) - What was wrong
3. **Journey** (Solution + Implementation) - How we fixed it
4. **Resolution** (Results) - What changed
5. **Moral** (Takeaways) - What we learned

### Be Specific
- ❌ "Large enterprise"
- ✅ "Fortune 500 company with 50,000 employees processing 10M transactions annually"

- ❌ "Significant cost savings"
- ✅ "$2.4M annual cost savings (6X ROI in first year)"

- ❌ "Much faster processing"
- ✅ "Reduced from 48 hours to 2 hours (96% reduction)"

### Show Business Value
Always connect technical achievements to business outcomes:
- ❌ "Achieved 94% model accuracy"
- ✅ "Achieved 94% model accuracy, enabling automatic processing of 80% of claims without human review, saving 10,000 staff hours annually"

### Use Active Voice
- ❌ "A system was built that could process..."
- ✅ "We built a system that processes..."

### Maintain Confidentiality
If working with sensitive clients:
- Use "A Fortune 500 healthcare company" instead of naming
- Anonymize specific details that could identify
- Get approval before publishing anything specific

---

## Content Checklist

Before publishing, verify:

- [ ] All `[Placeholder Text]` replaced
- [ ] Page `<title>` tag updated
- [ ] Meta fields filled in (Industry, Size, Timeline, ROI)
- [ ] At least 3 specific metrics in Results section
- [ ] All icons rendering (check browser console)
- [ ] No broken links
- [ ] Responsive design works (test on mobile)
- [ ] Client approval obtained (if needed)
- [ ] Back link points to correct page
- [ ] Footer links work
- [ ] Lucide icons load correctly

---

## SEO Optimization (Optional)

To improve search visibility, consider adding:

```html
<meta name="description" content="[Case study summary with key metrics]">
<meta property="og:title" content="[Case Study Title]">
<meta property="og:description" content="[Summary]">
<meta property="og:type" content="article">
```

---

## Example: Complete Filled Template

See `/docs/case-study-healthcare-claims-processing.html` (when created) for a fully populated example.

---

## Publishing Workflow

1. **Draft**: Copy template and fill in all sections
2. **Review**: Check against this guide's checklist
3. **Test**: Open locally in browser, test all links and responsive design
4. **Approve**: Get internal/client sign-off if needed
5. **Commit**: Add to git
6. **Push**: Deploy to GitHub Pages
7. **Link**: Update `case-studies.html` to link to new case study

---

## Tips for Maximum Impact

### Lead with ROI
The ROI metric in the header should be compelling - this is often the first thing prospects see.

### Use Real Numbers
Even ranges are better than vague statements:
- "Reduced processing time by 80-90%" > "Significantly faster"

### Show the Journey
Implementation Journey section helps prospects understand the process and timeline to expect.

### Balance Technical and Business
Technical enough to show expertise, business-focused enough for executive readers.

### Update Regularly
As additional results come in (6 months, 1 year post-launch), update the metrics to show sustained value.

---

## Questions?

When filling out the template, ask yourself:

1. **Would I believe this?** (Specific metrics and details build credibility)
2. **Can I prove this?** (Only claim what can be verified)
3. **Would a prospect care?** (Focus on business outcomes, not technical trivia)
4. **Is it clear why this matters?** (Connect every point to business value)
5. **Does it tell a story?** (Should flow as a narrative, not just facts)

---

## File Naming Convention

**Format:** `case-study-[industry]-[focus-area].html`

**Examples:**
- `case-study-healthcare-claims-processing.html`
- `case-study-manufacturing-predictive-maintenance.html`
- `case-study-financial-fraud-detection.html`
- `case-study-legal-document-review.html`

Keep filenames lowercase with hyphens, descriptive but concise.
