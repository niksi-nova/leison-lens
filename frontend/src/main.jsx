/*
 * main.jsx — React entry point
 *
 * Mounts the app into #root (defined in index.html).
 *
 * Provider order (outer → inner):
 *   BrowserRouter  — gives all components access to React Router hooks
 *   AuthProvider   — exposes user/token/login/logout via useAuth()
 *   App            — renders the route tree
 *
 * StrictMode is intentionally kept on for development to surface
 * double-invocation bugs; it has zero effect in production builds.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext.jsx'
import App from './App.jsx'
import './index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <App />
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>
)
