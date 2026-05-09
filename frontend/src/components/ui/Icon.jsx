/*
 * Icon — thin wrapper around Material Symbols Outlined.
 *
 * Usage:
 *   <Icon name="dashboard" />
 *   <Icon name="verified" filled />
 *   <Icon name="arrow_forward" className="text-primary" />
 *
 * The `filled` prop sets FILL=1 for solid icons (active nav states, badges).
 */

export default function Icon({ name, filled = false, className = '' }) {
  return (
    <span
      className={`material-symbols-outlined select-none leading-none ${filled ? 'icon-filled' : ''} ${className}`}
    >
      {name}
    </span>
  )
}
