/*
 * GlassCard — the frosted-glass panel used throughout all portal pages.
 *
 * Wraps children in a div with the glass morphism effect.
 * All card-like elements in the design use this pattern.
 *
 * Usage:
 *   <GlassCard className="p-8 rounded-3xl">...</GlassCard>
 *   <GlassCard hover className="p-6">...</GlassCard>  ← adds inner-glow on hover
 */

export default function GlassCard({ children, hover = true, className = '', ...props }) {
  return (
    <div
      className={`
        glass bg-white/40
        ${hover ? 'glass-hover transition-shadow duration-300' : ''}
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  )
}
