/*
 * LandingNav — top navigation bar for the public landing page only.
 *
 * Fixed at the top, glass morphism on cream background.
 * Contains: brand name (serif), Science/Security/About links, Clinician Portal CTA.
 *
 * On mobile: nav links collapse, only brand + CTA button remain.
 */

import { Link } from 'react-router-dom'

export default function LandingNav() {
  return (
    <nav className="fixed top-0 w-full z-50 backdrop-blur-xl bg-cream/80 border-b border-white/50">
      <div className="flex justify-between items-center px-6 md:px-10 py-4 max-w-7xl mx-auto">
        {/* Brand — italic serif matches the design */}
        <div className="text-xl md:text-2xl font-serif italic text-navy-950">
          Leison Lens
        </div>

        {/* Desktop nav links */}
        <div className="hidden md:flex gap-8 items-center">
          <a href="#science"   className="text-primary font-semibold border-b-2 border-primary/60 pb-0.5 text-sm">Science</a>
          <a href="#security"  className="text-on-surface-variant hover:text-primary transition-colors text-sm">Security</a>
          <a href="#about"     className="text-on-surface-variant hover:text-primary transition-colors text-sm">About</a>
        </div>

        {/* Clinician Portal CTA → goes to login */}
        <Link
          to="/login"
          className="bg-primary text-on-primary px-5 py-2.5 rounded-full text-sm font-semibold hover:opacity-90 transition-all active:scale-95"
        >
          Clinician Portal
        </Link>
      </div>
    </nav>
  )
}
