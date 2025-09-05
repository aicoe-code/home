# AICOE Design Aesthetic

## Core Principles
Clean, professional, minimal - inspired by shadcn/ui design philosophy

## Visual Identity

### Design Philosophy
- **Minimalism First**: Remove everything unnecessary, keep only what adds value
- **Functional Beauty**: Every element serves a purpose
- **Clarity**: Information hierarchy is immediately apparent
- **Breathing Room**: Generous whitespace for visual comfort

### Color Palette
- **Background**: Pure white (#FFFFFF) or subtle gray (#FAFAFA)
- **Text Primary**: Near-black (#0A0A0A)
- **Text Secondary**: Muted gray (#71717A)
- **Accent**: Subtle purple gradient or single accent color
- **Borders**: Light gray (#E4E4E7)

### Typography
- **Font Stack**: System fonts for optimal performance
  - -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif
- **Font Sizes**: Constrained scale (14px, 16px, 20px, 24px, 32px)
- **Line Height**: 1.5-1.6 for body text
- **Font Weight**: Regular (400) and Semibold (600) only

### Components
- **Borders**: 1px solid, subtle radius (4-8px)
- **Shadows**: Minimal, only for elevation (0 1px 3px rgba(0,0,0,0.1))
- **Buttons**: Solid or ghost variants, clear hover states
- **Cards**: Clean containers with subtle borders
- **Spacing**: Consistent 8px grid system

### Interactions
- **Transitions**: Smooth, fast (200-300ms)
- **Hover States**: Subtle color shifts or slight elevation
- **Focus States**: Clear but not overwhelming
- **Feedback**: Immediate and understated

### Layout Principles
- **Grid System**: 12-column responsive grid
- **Max Width**: Content constrained (1200px typical)
- **Mobile First**: Design for small screens, enhance for larger
- **Consistency**: Repeated patterns and spacing throughout

## Implementation Guidelines
- Prefer CSS variables for theming
- Use semantic HTML elements
- Accessibility is non-negotiable (WCAG 2.1 AA minimum)
- Performance matters - minimize CSS, optimize assets
- Progressive enhancement approach