/*
 * Toggle — animated on/off switch matching the design.
 *
 * Usage:
 *   const [on, setOn] = useState(true)
 *   <Toggle checked={on} onChange={setOn} />
 *
 * Visual: pill track, white dot that slides left (off) or right (on).
 * When on: track is bg-primary-container, dot is right.
 * When off: track is bg-slate-200, dot is left.
 */

export default function Toggle({ checked = false, onChange, disabled = false }) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange?.(!checked)}
      className={`
        relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full
        transition-colors duration-200 ease-in-out
        focus:outline-none focus-visible:ring-2 focus-visible:ring-primary
        disabled:opacity-50 disabled:cursor-not-allowed
        ${checked ? 'bg-primary-container' : 'bg-slate-200'}
      `}
    >
      {/* Sliding dot */}
      <span
        className={`
          pointer-events-none inline-block h-4 w-4 transform rounded-full
          bg-white shadow-sm transition-transform duration-200 ease-in-out
          mt-1
          ${checked ? 'translate-x-6' : 'translate-x-1'}
        `}
      />
    </button>
  )
}
