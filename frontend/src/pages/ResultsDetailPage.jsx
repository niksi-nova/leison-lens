/*
 * ResultsDetailPage — route: "/scan/:scanId/results"
 *
 * The most content-rich page — shows the full AI diagnostic output.
 *
 * Sections:
 *   1. Patient info header bar — name, ID, date, Export PDF + Finalize buttons
 *   2. Side-by-side scan images — original (grayscale) + AI heatmap overlay
 *   3. DR Severity Scale — 5-step track (No DR → Proliferative), active node
 *   4. Lesion Probabilities panel — MA/HE/EX/SE progress bars
 *   5. AI Diagnostic Insight card — text insight + "View Full Clinical Report" link
 *
 * HOW TO CONNECT:
 *   On mount → getScanResults(scanId) → GET /api/scan/:id/results
 *   This returns: { drGrade, confidence, originalUrl, heatmapUrl,
 *                   lesionProbs: {MA, HE, EX, SE}, aiInsight, patientName, ... }
 *   "Export PDF" → GET /api/scan/:id/export (PDF download)
 *   "Finalize Diagnosis" → PATCH /api/scan/:id/finalize → marks scan as reviewed
 *
 * The heatmapUrl is the Grad-CAM output from src/gradcam.py in your project.
 * When the Flask backend serves /api/scan/:id/results, include both image URLs.
 */

import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import PortalLayout from '../components/layout/PortalLayout.jsx'
import GlassCard from '../components/ui/GlassCard.jsx'
import Icon from '../components/ui/Icon.jsx'
import Button from '../components/ui/Button.jsx'
import { getScanResults } from '../api/index.js'
import { DR_GRADE_LABELS } from '../data/mockPatients.js'

// DR severity scale steps — matches APTOS 2019 grade labels
const DR_STEPS = [
  { grade: 0, label: 'No DR' },
  { grade: 1, label: 'Mild' },
  { grade: 2, label: 'Moderate' },
  { grade: 3, label: 'Severe' },
  { grade: 4, label: 'Proliferative' },
]

// Lesion type metadata — full name + color for the progress bar
const LESION_META = {
  MA: { full: 'MA (Microaneurysms)', barColor: 'bg-primary-container' },
  HE: { full: 'HE (Hemorrhages)',    barColor: 'bg-secondary-container' },
  EX: { full: 'EX (Exudates)',       barColor: 'bg-outline-variant' },
  SE: { full: 'SE (Soft Exudates)',  barColor: 'bg-outline-variant' },
}

export default function ResultsDetailPage() {
  const { scanId }        = useParams()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getScanResults(scanId).then((data) => {
      setResult(data)
      setLoading(false)
    })
  }, [scanId])

  // While loading, show skeleton placeholders
  if (loading) {
    return (
      <PortalLayout activePage="scan">
        <div className="space-y-6 animate-pulse">
          <div className="h-20 bg-white/30 rounded-2xl" />
          <div className="h-96 bg-white/30 rounded-3xl" />
          <div className="h-32 bg-white/30 rounded-3xl" />
        </div>
      </PortalLayout>
    )
  }

  const drConfig = DR_GRADE_LABELS[result.drGrade]

  return (
    <PortalLayout activePage="scan">
      {/* ─── 1. Patient info header ────────────────────────────────────── */}
      <GlassCard className="rounded-2xl p-5 md:p-6 mb-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        {/* Patient details */}
        <div className="flex flex-wrap items-center gap-6 md:gap-8">
          <div>
            <p className="font-mono text-[10px] text-on-surface-variant uppercase tracking-widest mb-1">
              Patient Name
            </p>
            <h2 className="font-semibold text-primary">{result.patientName}</h2>
          </div>
          {/* Vertical divider */}
          <div className="hidden sm:block w-px h-10 bg-white/40" />
          <div>
            <p className="font-mono text-[10px] text-on-surface-variant uppercase tracking-widest mb-1">
              ID Number
            </p>
            <p className="font-mono text-sm text-on-surface">{result.patientId}</p>
          </div>
          <div className="hidden sm:block w-px h-10 bg-white/40" />
          <div>
            <p className="font-mono text-[10px] text-on-surface-variant uppercase tracking-widest mb-1">
              Last Scan
            </p>
            <p className="font-mono text-sm text-on-surface">{result.scanDate}</p>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex gap-3 flex-shrink-0">
          <Button variant="secondary" size="sm">
            Export PDF
          </Button>
          <Button variant="primary" size="sm">
            Finalize Diagnosis
          </Button>
        </div>
      </GlassCard>

      {/* ─── 2. Side-by-side scan images ──────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Original grayscale scan */}
        <div className="glass bg-white/30 rounded-3xl overflow-hidden relative group h-64 md:h-96">
          <img
            src={result.originalUrl}
            alt="Original retinal scan"
            className="w-full h-full object-cover grayscale opacity-90 group-hover:scale-105 transition-transform duration-700"
          />
          <div className="absolute top-4 left-4 bg-black/30 backdrop-blur-sm px-3 py-1.5 rounded-full border border-white/20">
            <span className="font-mono text-[10px] text-white uppercase tracking-widest">
              Original Scan
            </span>
          </div>
        </div>

        {/* AI Heatmap overlay (Grad-CAM output) */}
        <div className="glass bg-white/30 rounded-3xl overflow-hidden relative group h-64 md:h-96">
          {/* Warm violet/orange gradient blend on top of heatmap */}
          <div className="absolute inset-0 bg-gradient-to-tr from-violet-500/20 to-orange-500/20 mix-blend-overlay z-10" />
          <img
            src={result.heatmapUrl}
            alt="AI Grad-CAM heatmap overlay"
            className="w-full h-full object-cover opacity-80 group-hover:scale-105 transition-transform duration-700"
          />
          <div className="absolute top-4 left-4 bg-primary/40 backdrop-blur-sm px-3 py-1.5 rounded-full border border-white/20 z-20">
            <span className="font-mono text-[10px] text-white uppercase tracking-widest">
              AI Heatmap Overlay
            </span>
          </div>
          {/* Color legend dots */}
          <div className="absolute bottom-4 right-4 flex gap-2 z-20">
            <div className="w-2 h-2 rounded-full bg-orange-400" />
            <div className="w-2 h-2 rounded-full bg-violet-400" />
          </div>
          {/* Active border */}
          <div className="absolute inset-0 border-2 border-primary-container/40 rounded-3xl pointer-events-none" />
        </div>
      </div>

      {/* ─── 3. DR Severity Scale ─────────────────────────────────────── */}
      <GlassCard className="rounded-3xl p-6 md:p-8 mb-6">
        <h3 className="font-mono text-[11px] text-on-surface-variant/60 uppercase tracking-[0.2em] mb-8 text-center">
          Diabetic Retinopathy Severity Classification
        </h3>

        {/* 5-step track */}
        <div className="relative px-6 md:px-12">
          {/* Track line behind the dots */}
          <div className="absolute h-0.5 top-[6px] left-12 right-12 bg-outline-variant/40 rounded-full" />

          <div className="relative flex justify-between">
            {DR_STEPS.map((step) => {
              const isActive = step.grade === result.drGrade
              return (
                <div key={step.grade} className="flex flex-col items-center gap-3 relative z-10">
                  {/* Dot indicator */}
                  <div
                    className={`rounded-full ring-4 ring-white transition-all ${
                      isActive
                        ? 'w-6 h-6 bg-primary-container -mt-1.5 shadow-xl shadow-primary-container/40'
                        : 'w-3 h-3 bg-outline-variant/50'
                    }`}
                  />
                  {/* Label */}
                  {isActive ? (
                    <div className="flex flex-col items-center gap-1">
                      <div className="bg-primary-container px-3 py-0.5 rounded-full">
                        <span className="font-bold text-[11px] text-on-primary-container">{step.label}</span>
                      </div>
                    </div>
                  ) : (
                    <span className="font-semibold text-[11px] text-on-surface-variant/50">
                      {step.label}
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </GlassCard>

      {/* ─── 4 + 5. Lesion Probabilities + AI Insight (2 columns) ─────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Lesion probability bars (2/3 width) */}
        <GlassCard className="col-span-1 md:col-span-2 rounded-3xl p-6 md:p-8">
          <div className="flex justify-between items-end mb-8">
            <h3 className="font-mono text-[11px] text-on-surface-variant/60 uppercase tracking-[0.2em]">
              Lesion Probabilities
            </h3>
            <p className="font-bold text-xs text-primary">
              Confidence: {result.confidence}%
            </p>
          </div>

          <div className="space-y-6">
            {Object.entries(result.lesionProbs).map(([key, pct]) => {
              const meta = LESION_META[key]
              return (
                <div key={key}>
                  <div className="flex justify-between mb-2">
                    <span className="text-xs font-bold text-on-surface">{meta.full}</span>
                    <span className="font-mono text-xs">
                      {String(pct).padStart(2, '0')}%
                    </span>
                  </div>
                  <div className="h-1 w-full bg-outline-variant/20 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${meta.barColor} progress-animate rounded-full`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </GlassCard>

        {/* AI Diagnostic Insight (1/3 width) */}
        <GlassCard className="col-span-1 rounded-3xl p-6 md:p-8 flex flex-col bg-primary/5 border border-primary/20">
          {/* Brain / neurology icon */}
          <div className="w-12 h-12 bg-white rounded-2xl flex items-center justify-center mb-6 shadow-sm">
            <Icon name="neurology" className="text-primary text-2xl" />
          </div>

          <h3 className="text-lg font-bold text-primary mb-4">AI Diagnostic Insight</h3>

          <p className="text-sm text-on-surface-variant leading-relaxed flex-1">
            {result.aiInsight}
          </p>

          {/* Link to full report */}
          <div className="mt-6 pt-6 border-t border-primary/10">
            <a
              href="#"
              className="font-mono text-xs font-bold text-primary flex items-center gap-2 group uppercase tracking-widest"
            >
              View Full Clinical Report
              <Icon
                name="arrow_forward"
                className="text-sm group-hover:translate-x-1 transition-transform"
              />
            </a>
          </div>
        </GlassCard>
      </div>
    </PortalLayout>
  )
}
