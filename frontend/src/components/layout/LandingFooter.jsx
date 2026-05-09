/*
 * LandingFooter — dark navy footer used at the bottom of the landing page.
 * Matches the navy-950 background from the design.
 */

export default function LandingFooter() {
  return (
    <footer className="bg-navy-950 border-t border-navy-900 w-full py-12">
      <div className="flex flex-col md:flex-row justify-between items-center px-10 max-w-7xl mx-auto gap-6">
        {/* Brand + copyright */}
        <div className="flex flex-col gap-1 items-center md:items-start">
          <span className="font-serif italic text-cream text-lg">Leison Lens</span>
          <p className="text-slate-400 text-sm">© 2024 Leison Lens. All rights reserved.</p>
        </div>

        {/* Legal links */}
        <div className="flex gap-8">
          {['Privacy Policy', 'Terms of Service'].map((link) => (
            <a
              key={link}
              href="#"
              className="text-slate-400 hover:text-cream transition-colors text-sm"
            >
              {link}
            </a>
          ))}
        </div>
      </div>
    </footer>
  )
}
