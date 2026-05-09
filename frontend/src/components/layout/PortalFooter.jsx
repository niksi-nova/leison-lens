/*
 * PortalFooter — shared footer used at the bottom of all portal pages.
 * Sits inside PortalLayout, below the main content.
 */

export default function PortalFooter() {
  return (
    <footer className="w-full flex flex-col md:flex-row justify-between items-center px-10 py-8 border-t border-white/20 bg-white/10 backdrop-blur-sm">
      <p className="font-mono text-xs uppercase tracking-widest text-on-surface-variant/60">
        © 2024 Leison Lens. High-Tech Sanctuary for Clinicians.
      </p>
      <div className="flex gap-6 mt-4 md:mt-0">
        {['Privacy Protocol', 'Terms of Clinical Use', 'Security Standards'].map((link) => (
          <a
            key={link}
            href="#"
            className="font-mono text-xs uppercase tracking-widest text-on-surface-variant/50 hover:text-primary transition-colors"
          >
            {link}
          </a>
        ))}
      </div>
    </footer>
  )
}
