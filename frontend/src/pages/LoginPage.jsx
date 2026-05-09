/*
 * LoginPage — route: "/login"
 *
 * Layout: full-screen split (60% artistic retinal bg / 40% form panel).
 * On mobile the left panel is hidden, only the form fills the screen.
 *
 * HOW TO CONNECT:
 *   On submit → calls useAuth().login(email, password)
 *   → auth/index.js login() → POST /api/auth/login (when backend is ready)
 *   On success → navigates to /dashboard
 *   On failure → shows inline error message
 */

import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import Icon from '../components/ui/Icon.jsx'
import Button from '../components/ui/Button.jsx'
import { useAuth } from '../context/AuthContext.jsx'

// Artistic retinal background for the left 60% panel
const BG_IMG = 'https://lh3.googleusercontent.com/aida-public/AB6AXuDXf_BmH7MZZTjYv6VspZqR4JSF0bKOOW37dZfnZVtJM1lFNNt3gzRVrV7OUCkNVF96892PhI7K-FLXhvDBD3XxBXaSOers_OC0r9MmRIu8ZsAvLzalkMdUrbvVoyPgHo9FvTUZbw1xLRK1itwp9QNqDS7CSJo5PqFLX4RJAWDYAkg7Lx8k7_tJdRSKJhEinzTMPvGtWMSICzEOxGG9vkBoGLvFqdUl6QPS84F2QY4zmFap8bQ9QV3Cd8rvcvbTNTnTUTZoCP34-7Yq'

export default function LoginPage() {
  const navigate = useNavigate()
  const { login } = useAuth()

  // Form state
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [remember, setRemember] = useState(false)
  const [showPwd, setShowPwd]   = useState(false)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await login(email, password)
      navigate('/dashboard')
    } catch (err) {
      setError('Invalid credentials. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="flex h-screen w-full overflow-hidden">
      {/* ─── Left 60%: Artistic retinal background ───────────────────── */}
      <section className="hidden md:flex md:w-3/5 h-full relative overflow-hidden bg-slate-900">
        {/* Blurred retinal art fills the panel */}
        <div className="absolute inset-0 z-0">
          <img
            src={BG_IMG}
            alt=""
            className="w-full h-full object-cover retinal-blur opacity-60"
          />
        </div>
        {/* Dark gradient overlay for text legibility */}
        <div className="absolute inset-0 bg-gradient-to-r from-slate-900/40 to-transparent" />

        {/* Branding copy in the bottom-left */}
        <div className="relative z-10 p-12 lg:p-16 flex flex-col justify-between h-full w-full">
          {/* Top: brand logo */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary-container rounded-lg flex items-center justify-center">
              <Icon name="visibility" className="text-white text-2xl" />
            </div>
            <span className="text-xl font-bold text-white tracking-wide">Leison Lens</span>
          </div>

          {/* Bottom: tagline */}
          <div className="max-w-md">
            <h2 className="font-serif text-4xl md:text-5xl text-white mb-6 leading-tight">
              Precision in every pixel.
            </h2>
            <p className="text-white/80 text-body-lg">
              Our advanced retinal analysis platform provides clinicians with unparalleled
              diagnostic clarity through AI-driven insights.
            </p>
          </div>

          <span className="font-mono text-xs text-white/40 uppercase tracking-widest">
            Trusted by Leading Clinics Globally
          </span>
        </div>
      </section>

      {/* ─── Right 40%: Login form ───────────────────────────────────── */}
      <section className="w-full md:w-2/5 h-full flex flex-col items-center justify-center relative bg-gradient-to-br from-[#f0edf8] to-[#fde8e4] px-6 md:px-10">
        {/* Mobile logo (hidden on desktop) */}
        <div className="md:hidden absolute top-8 left-8 flex items-center gap-2">
          <div className="w-8 h-8 bg-primary rounded flex items-center justify-center">
            <Icon name="visibility" className="text-white text-lg" />
          </div>
          <span className="font-bold text-on-surface">Leison Lens</span>
        </div>

        {/* Glass form card */}
        <div className="glass bg-white/40 w-full max-w-md p-8 md:p-10 rounded-2xl">
          <div className="mb-8">
            <h1 className="text-h1 font-bold text-on-surface mb-2">Welcome Back</h1>
            <p className="text-on-surface-variant">
              Access your clinical dashboard and patient records.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Email field */}
            <div className="space-y-1.5">
              <label className="font-mono text-xs text-on-surface-variant uppercase tracking-widest pl-1">
                Clinician Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="dr.smith@hospital.org"
                required
                className="w-full bg-white/50 border border-outline-variant focus:border-primary focus:ring-0 rounded-xl px-4 py-3 text-on-surface placeholder:text-outline/60 transition-colors"
              />
            </div>

            {/* Password field */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center pl-1">
                <label className="font-mono text-xs text-on-surface-variant uppercase tracking-widest">
                  Password
                </label>
                <a href="#" className="font-mono text-[10px] text-on-secondary-container hover:underline uppercase tracking-wider">
                  Forgot Password?
                </a>
              </div>
              <div className="relative">
                <input
                  type={showPwd ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••••••"
                  required
                  className="w-full bg-white/50 border border-outline-variant focus:border-primary focus:ring-0 rounded-xl px-4 py-3 pr-12 text-on-surface transition-colors"
                />
                <button
                  type="button"
                  onClick={() => setShowPwd(!showPwd)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-outline hover:text-primary"
                >
                  <Icon name={showPwd ? 'visibility_off' : 'visibility'} />
                </button>
              </div>
            </div>

            {/* Remember me */}
            <div className="flex items-center gap-3 py-1">
              <input
                type="checkbox"
                id="remember"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
                className="w-5 h-5 rounded border-outline-variant text-primary focus:ring-0 cursor-pointer"
              />
              <label htmlFor="remember" className="text-on-surface-variant cursor-pointer select-none text-sm">
                Remember this session
              </label>
            </div>

            {/* Error message */}
            {error && (
              <p className="text-error text-sm bg-error-container px-4 py-3 rounded-xl">{error}</p>
            )}

            {/* Submit */}
            <Button
              type="submit"
              variant="container"
              loading={loading}
              className="w-full justify-center py-4 rounded-xl text-base"
            >
              Log In
              {!loading && <Icon name="arrow_forward" className="text-[18px]" />}
            </Button>
          </form>

          {/* Footer links */}
          <div className="mt-8 pt-6 border-t border-white/30 flex flex-col items-center gap-3">
            <p className="text-on-surface-variant text-sm">Need clinical access?</p>
            <Link
              to="/signup"
              className="font-mono text-[11px] uppercase tracking-wider text-on-secondary-container bg-white/40 px-6 py-2 rounded-full border border-white/60 hover:bg-white/60 transition-all"
            >
              Request an Account
            </Link>
          </div>
        </div>

        {/* Small footer */}
        <footer className="absolute bottom-6 w-full px-8 flex flex-col sm:flex-row justify-between items-center text-on-surface-variant/50 gap-2">
          <span className="font-mono text-[10px] uppercase tracking-widest text-center">
            © 2024 Leison Lens. High-Tech Sanctuary for Clinicians.
          </span>
          <div className="flex gap-4">
            <a href="#" className="font-mono text-[10px] uppercase tracking-wider hover:text-primary">Security</a>
            <a href="#" className="font-mono text-[10px] uppercase tracking-wider hover:text-primary">Privacy</a>
          </div>
        </footer>
      </section>
    </main>
  )
}
