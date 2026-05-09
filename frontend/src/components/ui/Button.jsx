/*
 * Button — reusable button with design system variants.
 *
 * Variants:
 *   primary   — solid violet, main CTA (Start Analysis, Login, etc.)
 *   secondary — outlined terracotta border, secondary actions
 *   ghost     — transparent with hover, tertiary actions
 *   danger    — solid error red, Emergency Support
 *
 * Size:
 *   sm  — small pill (table actions)
 *   md  — default (forms, most buttons)
 *   lg  — large pill (hero CTAs)
 */

import Icon from './Icon.jsx'

const VARIANTS = {
  primary:   'bg-primary text-on-primary hover:opacity-90 shadow-md',
  secondary: 'border border-secondary text-secondary hover:bg-secondary/5',
  ghost:     'text-on-surface-variant hover:bg-white/20',
  danger:    'bg-error text-on-error hover:opacity-90',
  light:     'bg-white/50 border border-white/80 text-on-surface hover:bg-white/80',
  container: 'bg-primary-container text-on-primary-container hover:brightness-105 shadow-md',
}

const SIZES = {
  sm: 'px-4 py-2 text-xs font-bold tracking-wide rounded-full',
  md: 'px-6 py-3 text-sm font-semibold rounded-full',
  lg: 'px-8 py-4 text-base font-semibold rounded-full',
}

export default function Button({
  variant = 'primary',
  size = 'md',
  icon,          // optional Icon name to show after the label
  iconLeft,      // optional Icon name to show before the label
  loading = false,
  disabled = false,
  className = '',
  children,
  ...props
}) {
  return (
    <button
      disabled={disabled || loading}
      className={`
        inline-flex items-center justify-center gap-2 transition-all duration-200
        active:scale-[0.97] disabled:opacity-50 disabled:cursor-not-allowed
        ${VARIANTS[variant]} ${SIZES[size]} ${className}
      `}
      {...props}
    >
      {loading && (
        // Simple CSS spinner shown when loading=true
        <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
      )}
      {iconLeft && !loading && <Icon name={iconLeft} className="text-[18px]" />}
      {children}
      {icon && <Icon name={icon} className="text-[18px]" />}
    </button>
  )
}
