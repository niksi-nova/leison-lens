/*
 * AnalyticsPage — route: "/analytics"
 *
 * Longitudinal performance dashboard for the AI diagnostic system.
 * Currently a placeholder — charts and deeper metrics are Phase 7.
 *
 * Planned sections (Phase 7):
 *   • Model accuracy trend (line chart — precision/recall/F1 by week)
 *   • Grade distribution donut (grades 0-4 across all scans)
 *   • Lesion detection heatmap (MA/HE/EX/SE rates over time)
 *   • Clinician override rate (cases where doctor overrode AI grade)
 *
 * HOW TO CONNECT:
 *   GET /api/analytics?range=30d → returns chart datasets
 */

import PortalLayout from '../components/layout/PortalLayout.jsx'
import GlassCard from '../components/ui/GlassCard.jsx'
import Icon from '../components/ui/Icon.jsx'

// Placeholder stat cards shown while full chart views are built
const PLACEHOLDER_STATS = [
  { label: 'Model Accuracy',    value: '94.7%', icon: 'insights',       color: 'text-primary' },
  { label: 'Scans This Month',  value: '1,284',  icon: 'bar_chart',      color: 'text-secondary' },
  { label: 'Avg. Confidence',   value: '98.4%', icon: 'analytics',      color: 'text-tertiary' },
  { label: 'Override Rate',     value: '2.1%',  icon: 'manage_history', color: 'text-error' },
]

export default function AnalyticsPage() {
  return (
    <PortalLayout activePage="analytics">
      {/* ─── Header ───────────────────────────────────────────────────── */}
      <header className="mb-10">
        <h1 className="text-h1 font-bold text-on-surface">Analytics</h1>
        <p className="text-on-surface-variant mt-1">
          Model performance metrics and longitudinal retinopathy screening trends.
        </p>
      </header>

      {/* ─── Stat row ─────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {PLACEHOLDER_STATS.map((stat) => (
          <GlassCard key={stat.label} className="p-6 rounded-2xl flex flex-col gap-3">
            <Icon name={stat.icon} className={`${stat.color} text-2xl`} />
            <div>
              <p className="font-mono text-[10px] text-on-surface-variant uppercase tracking-widest mb-1">
                {stat.label}
              </p>
              <p className={`font-bold text-3xl ${stat.color}`}>{stat.value}</p>
            </div>
          </GlassCard>
        ))}
      </div>

      {/* ─── Coming-soon placeholder ──────────────────────────────────── */}
      <GlassCard className="rounded-3xl p-10 md:p-16 flex flex-col items-center justify-center text-center min-h-64">
        <div className="w-16 h-16 rounded-2xl bg-primary-container/20 flex items-center justify-center mb-6">
          <Icon name="bar_chart" className="text-primary text-3xl" />
        </div>
        <h2 className="font-semibold text-on-surface text-xl mb-3">
          Detailed Charts & Heatmaps
        </h2>
        <p className="text-on-surface-variant text-sm max-w-md leading-relaxed">
          Full time-series charts, grade distribution breakdowns, and lesion frequency
          heatmaps will be added.
        </p>
      </GlassCard>
    </PortalLayout>
  )
}
