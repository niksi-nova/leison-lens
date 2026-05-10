/*
 * PortalFooter — shared footer used at the bottom of all portal pages.
 * Sits inside PortalLayout, below the main content.
 */

export default function PortalFooter() {
  return (
    <footer className="w-full flex flex-col md:flex-row justify-between items-center px-10 py-8 border-t border-white/20 bg-white/10 backdrop-blur-sm">
      <p className="font-mono text-xs uppercase tracking-widest text-on-surface-variant/60">
        © 2026 Leison Lens. 
      </p>
      
    </footer>
  )
}
