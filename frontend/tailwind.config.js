/** @type {import('tailwindcss').Config} */

// This Tailwind config exactly mirrors the design system tokens from the
// Stitch / Leison Lens design. Every color name here maps directly to a
// Material Design 3 token used in the components — do NOT rename them
// without updating all component classes too.
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      // ─── Material Design 3 Color Tokens ──────────────────────────────────
      colors: {
        // Primary: dusty violet — used for active states, buttons, accents
        primary: '#635789',
        'primary-container': '#9b8ec4',
        'primary-fixed': '#e8deff',
        'primary-fixed-dim': '#cdbef8',
        'on-primary': '#ffffff',
        'on-primary-container': '#322755',
        'on-primary-fixed': '#1e1341',
        'on-primary-fixed-variant': '#4b3f70',
        'inverse-primary': '#cdbef8',

        // Secondary: warm terracotta — used for warnings, emergency, accents
        secondary: '#904b37',
        'secondary-container': '#fea58b',
        'secondary-fixed': '#ffdbd1',
        'secondary-fixed-dim': '#ffb59f',
        'on-secondary': '#ffffff',
        'on-secondary-container': '#783925',
        'on-secondary-fixed': '#3a0a00',
        'on-secondary-fixed-variant': '#733422',

        // Tertiary: warm gold — used for confidence scores, AI accuracy
        tertiary: '#695f1d',
        'tertiary-container': '#b9ac62',
        'tertiary-fixed': '#f2e494',
        'tertiary-fixed-dim': '#d5c77a',
        'on-tertiary': '#ffffff',
        'on-tertiary-container': '#484000',
        'on-tertiary-fixed': '#201c00',
        'on-tertiary-fixed-variant': '#504705',

        // Surface: soft lavender whites — the main background family
        surface: '#fdf7ff',
        'surface-bright': '#fdf7ff',
        'surface-dim': '#dfd7e9',
        'surface-variant': '#e7dff1',
        'surface-container': '#f3ebfd',
        'surface-container-low': '#f8f1ff',
        'surface-container-high': '#ede5f7',
        'surface-container-highest': '#e7dff1',
        'surface-container-lowest': '#ffffff',
        'surface-tint': '#635789',
        'on-surface': '#1d1a26',
        'on-surface-variant': '#48454e',
        'inverse-surface': '#322e3c',
        'inverse-on-surface': '#f6eeff',

        // Outline: border and divider colors
        outline: '#79757f',
        'outline-variant': '#cac4cf',

        // Error: red family — used for severe DR grades, error states
        error: '#ba1a1a',
        'error-container': '#ffdad6',
        'on-error': '#ffffff',
        'on-error-container': '#93000a',

        // Custom brand colors
        background: '#fdf7ff',
        'on-background': '#1d1a26',
        'navy-950': '#0a0a14', // dark sections (validation, footer)
        'navy-900': '#131326',
        cream: '#fdfaf6',     // nav background
        blush: '#fff5f2',
      },

      // ─── Border Radius ───────────────────────────────────────────────────
      borderRadius: {
        DEFAULT: '0.25rem',
        lg: '0.5rem',
        xl: '0.75rem',
        '2xl': '1rem',
        '3xl': '1.5rem',
        full: '9999px',
      },

      // ─── Spacing tokens (used in className spacing references) ───────────
      spacing: {
        'card-gap': '2rem',         // 32px — gap between grid cards
        'section-margin': '4rem',   // 64px — vertical section spacing
        'gutter': '1.5rem',         // 24px — inner column gaps
        'container-padding': '2.5rem', // 40px — main content left padding
      },

      // ─── Font Families ────────────────────────────────────────────────────
      fontFamily: {
        // Manrope: the primary UI font for all headings and body copy
        sans: ['Manrope', 'system-ui', 'sans-serif'],
        // Space Grotesk: monospaced-feel font for data labels and caps
        mono: ['Space Grotesk', 'monospace'],
        // Fraunces: editorial serif used on the landing page hero only
        serif: ['Fraunces', 'Georgia', 'serif'],
      },

      // ─── Font Size Scale ─────────────────────────────────────────────────
      fontSize: {
        'label-caps': ['12px', { lineHeight: '1', letterSpacing: '0.08em', fontWeight: '500' }],
        'data-mono': ['14px', { lineHeight: '1.4', fontWeight: '400' }],
        'body-md': ['16px', { lineHeight: '1.6', fontWeight: '400' }],
        'body-lg': ['18px', { lineHeight: '1.6', fontWeight: '400' }],
        h2: ['24px', { lineHeight: '1.3', fontWeight: '600' }],
        h1: ['32px', { lineHeight: '1.2', fontWeight: '600' }],
        display: ['48px', { lineHeight: '1.1', letterSpacing: '-0.02em', fontWeight: '700' }],
      },
    },
  },
  plugins: [],
}
