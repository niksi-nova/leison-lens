/*
 * SeverityBadge — colored indicator for DR grade severity.
 *
 * Displayed in the Patient History table and Dashboard activity feed.
 * Shows a vertical color bar + uppercase label.
 *
 * Usage:
 *   <SeverityBadge grade={2} />   → "MODERATE" with terracotta bar
 *   <SeverityBadge grade={3} />   → "SEVERE" with error-red bar
 */

import { SEVERITY_BADGE } from '../../mock/mockPatients.js'

export default function SeverityBadge({ grade }) {
  const config = SEVERITY_BADGE[grade] ?? SEVERITY_BADGE[0]

  return (
    <div className="flex items-center gap-2">
      {/* Vertical color stripe — the design uses a 2px wide, 32px tall bar */}
      <div className={`w-0.5 h-8 rounded-full ${config.bg.replace('/20', '')}`} />
      <span className={`text-xs font-bold uppercase tracking-wide ${config.text}`}>
        {config.label}
      </span>
    </div>
  )
}
