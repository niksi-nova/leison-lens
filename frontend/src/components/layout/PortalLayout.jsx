/*
 * PortalLayout — wrapper for all authenticated portal pages.
 *
 * Renders the SideNav on the left and the page content on the right.
 * All portal pages (Dashboard, Scan, History, Results, Settings) use this.
 *
 * Props:
 *   activePage — passed through to SideNav to highlight the correct item
 *   children   — the page content
 *   className  — optional extra classes on the main content area
 */

import SideNav from './SideNav.jsx'
import PortalFooter from './PortalFooter.jsx'

export default function PortalLayout({ activePage, children, className = '' }) {
  return (
    // The brand gradient background fills the entire viewport
    <div className="flex min-h-screen brand-gradient">
      {/* Fixed sidebar — 256px wide on md+ */}
      <SideNav activePage={activePage} />

      {/* Main content area — offset by sidebar width on md+ */}
      <main className={`flex-1 md:ml-64 flex flex-col min-h-screen ${className}`}>
        <div className="flex-1 p-6 md:p-10">
          {children}
        </div>
        <PortalFooter />
      </main>
    </div>
  )
}
