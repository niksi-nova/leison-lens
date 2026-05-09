/*
 * LandingPage — the public marketing page (route: "/")
 *
 * Sections (top → bottom):
 *   1. LandingNav (fixed top bar)
 *   2. Hero — headline, stats, retinal scan image in dark rounded frame
 *   3. Workflow — "Precision in Three Steps" (3 staggered glass cards)
 *   4. Validation — dark navy section with peer-review stats + eye image
 *   5. CTA — gradient purple/terracotta strip with demo request buttons
 *   6. LandingFooter — dark navy footer
 *
 * All buttons on this page link to /login or /signup via React Router.
 * The retinal images are from the Stitch design assets (Google hosted).
 * In production replace with your own CDN images.
 */

import { Link } from 'react-router-dom'
import LandingNav from '../components/layout/LandingNav.jsx'
import LandingFooter from '../components/layout/LandingFooter.jsx'
import Icon from '../components/ui/Icon.jsx'

// Retinal eye image — hero section dark card
const HERO_EYE_IMG = 'https://lh3.googleusercontent.com/aida-public/AB6AXuArmH2hUy0nVBGa1iMxm9elCM6jE1ue5APWyHqX0pz2Oimuf3B5ECKrGh7Z5GEz_L6s-b73xl61TMPDLCG6sMWQsiwpO39cv6UPiDMQFcH7ypFM1jK-tnCFAvWORLCEqnSlIH7SV41oPVvxCiMcE160DTOHXCKn96g0THY7fHXrPyxUVuwgW8Uci8-Wc5s2bthFuDqYMWq4egSURLVYD_Vry2ZppnRLrnhKLCrtv6WDemzq7hqDrypXtwQtUEti3Dy2Oh_QKPsK8I1w'

// Validation section eye scan image
const VALIDATION_IMG = 'https://lh3.googleusercontent.com/aida/ADBb0ugD61CD1UF6MAC8a_awPLkj2UrfyAzSGQ6XfV3I8iHmyOPYBBYTeTgfhC9iQ8VHl_rpVFVZSPxIZtTRDh71easK_e3XY1GiC-ALO62NjIk6q_HiUII7g5B9L_yQRAc8rEk6lkMtIjsrE60DqbGifz0P9WGlBK9eaab16G5sU6e6N2O86bFrzB7b_yyZamIaearOBBLk0fnX4jq3kYRZ_TfTy_-Tu30OU0Fn0M9jEOP6IxNy5rmSNCUu70XizohejZqpyEBV1_QHQJg'

// Hero background abstract art
const HERO_BG_IMG = 'https://lh3.googleusercontent.com/aida-public/AB6AXuBPc2y5jWEHSk85Bggi7o20yH1zuTePKt25HJnS_XPT-5brob_Ib2KcDUsPLQz3ye3sGDl6hCvrD-KjLDwjlPj2UmazIQ-2uPZXSxQhCzfDNq2tAwqF18INO7nZmhJ4H17_uYo74Um_Op4UyRLyQKr0LspLAEKfO8X9y2B4m1JJwQD_2JPquO0lK4NgfveCDyoj1wDnYtCrVeBlAKhz2fU9yehqx9-ZvDJcTPCcHZYGx4_FzmxazNsA9Kle07zlw_iiIZylR9wvPf5P'

// ─── Workflow step data ───────────────────────────────────────────────────
const WORKFLOW_STEPS = [
  {
    num: '01',
    title: 'Data Input',
    desc: 'Securely upload high-resolution OCT or Fundus imagery directly via our encrypted cloud gateway.',
    bg: 'bg-primary',
    icon: 'cloud_upload',
    offset: '',
  },
  {
    num: '02',
    title: 'Neural Analysis',
    desc: 'Our dual-stage AI architecture segments pathologies with 98.4% sensitivity in under 300ms.',
    bg: 'bg-secondary',
    icon: 'psychology',
    offset: 'md:translate-y-16',
  },
  {
    num: '03',
    title: 'Diagnostic Result',
    desc: 'Download a comprehensive PDF report with heatmaps, measurements, and clinician-verified labels.',
    bg: 'bg-tertiary',
    icon: 'fact_check',
    offset: 'md:translate-y-32',
  },
]

export default function LandingPage() {
  return (
    <div className="bg-surface overflow-x-hidden">
      <LandingNav />

      <main className="pt-16">
        {/* ══════════════════════════════════════════════════════════════
            SECTION 1 — Hero
            Split layout: text left (55%), retinal scan frame right (45%)
            ══════════════════════════════════════════════════════════════ */}
        <section
          id="science"
          className="relative min-h-screen flex items-center hero-gradient px-6 md:px-10 py-20"
        >
          {/* Background abstract art overlay */}
          <div className="absolute inset-0 z-0 opacity-40 pointer-events-none">
            <img
              src={HERO_BG_IMG}
              alt=""
              className="w-full h-full object-cover mix-blend-overlay blur-3xl scale-110"
            />
          </div>
          {/* Subtle grainy texture */}
          <div className="absolute inset-0 grainy" />

          <div className="relative z-10 max-w-7xl mx-auto w-full flex flex-col lg:flex-row items-center gap-12 lg:gap-16">
            {/* ─── Left: Text content ──────────────────────────────── */}
            <div className="w-full lg:w-[55%] space-y-8">
              <span className="inline-block font-mono text-xs text-primary uppercase tracking-widest">
                Clinical-Grade AI
              </span>

              <h1 className="font-serif text-5xl md:text-6xl lg:text-7xl leading-tight text-on-surface">
                See What Others Miss.
              </h1>

              <p className="text-body-lg text-on-surface-variant max-w-xl">
                Leison Lens utilizes neural architecture inspired by biological vision to provide
                sub-millimeter precision in diagnostic retinal scans, empowering clinicians with
                immediate, actionable insights.
              </p>

              {/* CTA Buttons */}
              <div className="flex flex-wrap gap-4 pt-2">
                <Link
                  to="/scan/new"
                  className="bg-primary text-on-primary px-8 py-4 rounded-full font-semibold flex items-center gap-2 hover:opacity-90 transition-all active:scale-95"
                >
                  Start Analysis
                  <Icon name="arrow_forward" className="text-[18px]" />
                </Link>
                <a
                  href="#workflow"
                  className="border border-secondary text-secondary px-8 py-4 rounded-full font-semibold hover:bg-secondary/5 transition-all"
                >
                  View Methodology
                </a>
              </div>

              {/* Stat pills */}
              <div className="flex gap-10 pt-4">
                {[
                  { value: '98.4%', label: 'Sensitivity' },
                  { value: '0.3s',  label: 'Processing' },
                  { value: 'CE',    label: 'Certified Class IIa' },
                ].map((stat) => (
                  <div key={stat.label} className="space-y-1">
                    <div className="font-mono text-xl font-bold text-primary">{stat.value}</div>
                    <div className="font-mono text-xs text-on-surface-variant/60 uppercase tracking-widest">
                      {stat.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* ─── Right: Retinal scan in dark rounded frame ────────── */}
            <div className="w-full lg:w-[45%] relative">
              {/* Dark rounded frame */}
              <div className="bg-navy-950 rounded-[40px] p-6 md:p-8 aspect-square relative overflow-hidden shadow-2xl shadow-primary/20 max-w-sm md:max-w-none mx-auto">
                <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent" />
                {/* Outer ring with scanning arc */}
                <div className="relative z-10 w-full h-full border-2 border-primary/20 rounded-full flex items-center justify-center p-4 md:p-6">
                  <div className="absolute inset-0 border-[3px] border-primary/30 rounded-full scale-95 opacity-50" />
                  {/* Rotating scan arc */}
                  <div className="absolute inset-0 rounded-full border-t-2 border-primary shadow-[0_0_15px_rgba(99,87,137,0.5)] rotate-45" />
                  {/* Eye image inside circle */}
                  <div className="w-full h-full rounded-full overflow-hidden bg-black relative">
                    <img
                      src={HERO_EYE_IMG}
                      alt="Retinal scan"
                      className="w-full h-full object-cover opacity-80"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-navy-950/80 via-transparent to-transparent" />
                  </div>
                </div>
              </div>
              {/* Ambient glow blob */}
              <div className="absolute -z-10 -bottom-10 -right-10 w-64 h-64 bg-primary/20 rounded-full blur-3xl" />
            </div>
          </div>

          {/* Ambient glow blobs */}
          <div className="absolute bottom-10 left-10 w-64 h-64 bg-primary/10 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute top-20 right-10 w-96 h-96 bg-secondary/10 rounded-full blur-3xl pointer-events-none" />
        </section>

        {/* ══════════════════════════════════════════════════════════════
            SECTION 2 — Workflow ("Precision in Three Steps")
            3 staggered glass cards on a light background
            ══════════════════════════════════════════════════════════════ */}
        <section id="workflow" className="py-24 md:py-32 px-6 md:px-10 bg-surface">
          <div className="max-w-7xl mx-auto">
            {/* Section label */}
            <div className="mb-16 md:mb-20">
              <span className="font-mono text-xs text-primary uppercase tracking-widest block mb-4">
                Workflow
              </span>
              <h2 className="font-serif text-3xl md:text-4xl text-on-surface">
                Precision in Three Steps
              </h2>
            </div>

            {/* 3 staggered cards — each shifts down on md+ */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12 items-start pb-32">
              {WORKFLOW_STEPS.map((step) => (
                <div
                  key={step.num}
                  className={`glass bg-white/40 p-8 rounded-3xl shadow-sm glass-hover transition-all ${step.offset}`}
                >
                  {/* Step number badge */}
                  <div
                    className={`w-12 h-12 rounded-xl ${step.bg} text-white flex items-center justify-center mb-6 font-mono font-bold`}
                  >
                    {step.num}
                  </div>
                  <h3 className="font-serif text-2xl mb-4 text-on-surface">{step.title}</h3>
                  <p className="text-on-surface-variant text-sm leading-relaxed">{step.desc}</p>

                  {/* Visual accent per step */}
                  {step.num === '01' && (
                    <div className="mt-8 border-2 border-dashed border-primary/30 rounded-xl py-8 flex items-center justify-center bg-primary/5">
                      <Icon name={step.icon} className="text-primary text-3xl" />
                    </div>
                  )}
                  {step.num === '02' && (
                    <div className="mt-8 h-2 w-full bg-surface-container-high rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-primary to-secondary w-2/3 progress-animate" />
                    </div>
                  )}
                  {step.num === '03' && (
                    <div className="mt-8 flex gap-2">
                      <div className="h-1 flex-1 bg-tertiary/20 rounded" />
                      <div className="h-1 flex-1 bg-tertiary/20 rounded" />
                      <div className="h-1 flex-1 bg-tertiary rounded" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ══════════════════════════════════════════════════════════════
            SECTION 3 — Validation (dark navy background)
            Left: text + stats. Right: eye photo with CE badge overlay.
            ══════════════════════════════════════════════════════════════ */}
        <section id="security" className="bg-navy-950 text-cream py-24 md:py-32">
          <div className="px-6 md:px-10 max-w-7xl mx-auto flex flex-col lg:flex-row items-center gap-16 lg:gap-20">
            {/* Left: copy */}
            <div className="w-full lg:w-1/2 space-y-8">
              <span className="font-mono text-xs text-secondary uppercase tracking-widest">
                Trust &amp; Validation
              </span>
              <h2 className="font-serif text-4xl md:text-5xl leading-tight">
                Validated by Peer-Reviewed Data
              </h2>
              <p className="text-slate-400 text-body-lg leading-relaxed">
                In a multi-center clinical study involving over 15,000 unique patient eyes, Leison
                Lens demonstrated performance parity with fellowship-trained retina specialists
                across 12 distinct ocular pathologies.
              </p>
              {/* Big stat numbers */}
              <div className="grid grid-cols-2 gap-8 pt-4">
                <div className="space-y-2">
                  <div className="text-4xl md:text-5xl font-bold text-primary-container font-mono">
                    98.4%
                  </div>
                  <div className="font-mono text-xs uppercase tracking-widest text-slate-500">
                    AUC Metric
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="text-4xl md:text-5xl font-bold text-primary-container font-mono">
                    0.3s
                  </div>
                  <div className="font-mono text-xs uppercase tracking-widest text-slate-500">
                    Latency Target
                  </div>
                </div>
              </div>
              <a
                href="#"
                className="inline-flex items-center gap-3 text-secondary-container font-semibold hover:underline group"
              >
                Read Clinical Whitepaper
                <Icon name="open_in_new" className="group-hover:translate-x-1 transition-transform" />
              </a>
            </div>

            {/* Right: image with CE badge */}
            <div className="w-full lg:w-1/2 relative">
              <div className="relative rounded-2xl overflow-hidden shadow-2xl">
                <img
                  src={VALIDATION_IMG}
                  alt="Clinical eye scan visualization"
                  className="w-full aspect-[4/3] object-cover opacity-90"
                />
                {/* Floating CE badge */}
                <div className="absolute top-4 right-4 md:top-6 md:right-6 backdrop-blur-md bg-white/10 border border-white/20 px-3 py-2 md:px-4 rounded-lg flex items-center gap-3">
                  <div className="w-7 h-7 rounded-full bg-secondary-container flex items-center justify-center">
                    <Icon name="verified" filled className="text-on-secondary-container text-sm" />
                  </div>
                  <span className="font-mono text-[10px] tracking-wider text-white">
                    CE Certified Class IIa
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ══════════════════════════════════════════════════════════════
            SECTION 4 — CTA strip
            Gradient background, "Ready to transform your practice?"
            ══════════════════════════════════════════════════════════════ */}
        <section className="relative overflow-hidden" id="about">
          {/* Gradient overlay */}
          <div className="absolute inset-0 bg-gradient-to-r from-primary via-secondary to-primary-container opacity-90" />
          {/* Dot grid texture overlay */}
          <div className="absolute inset-0 opacity-20 dot-grid" />

          <div className="relative z-10 px-6 md:px-10 py-20 md:py-24 max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-10 text-center md:text-left">
            <h2 className="font-serif text-3xl md:text-5xl text-white max-w-md">
              Ready to transform your practice?
            </h2>
            <div className="flex flex-col sm:flex-row gap-4 flex-shrink-0">
              <Link
                to="/signup"
                className="bg-white text-primary px-8 md:px-10 py-4 md:py-5 rounded-full font-bold hover:bg-cream transition-all shadow-xl"
              >
                Request a Demo
              </Link>
              <a
                href="mailto:hello@leisonlens.com"
                className="border-2 border-white/40 text-white px-8 md:px-10 py-4 md:py-5 rounded-full font-bold hover:bg-white/10 transition-all"
              >
                Contact Sales
              </a>
            </div>
          </div>
        </section>
      </main>

      <LandingFooter />
    </div>
  )
}
