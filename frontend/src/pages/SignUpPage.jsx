/*
 * SignUpPage — route: "/signup"
 *
 * Layout: same split as Login (60% retinal art / 40% form).
 * Form fields: Full Name, Clinical Institution, Professional ID, Work Email,
 *              Security Credentials (password), Privacy Protocol toggle.
 *
 * HOW TO CONNECT:
 *   On submit → calls api/index.js register(formData)
 *   → POST /api/auth/register
 *   On success → navigate to /login with a success toast
 */

import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import Icon from '../components/ui/Icon.jsx'
import Button from '../components/ui/Button.jsx'
import { register } from '../api/index.js'

// Retinal art for the left panel — same as login for brand consistency
const BG_IMG = 'https://lh3.googleusercontent.com/aida-public/AB6AXuCLj1Ie5qNQyJKHpdAtQrVKCQ0qIV-5NGUSl6Qtkzr-u6fRZhvA-zVHtya-6qk8GjdUXoSeSKGXY2k_AMBrNNH1sdmW_R82ETS4yXnSvFcXwhyqsX-nbuLF20EmIp5FgUaZNMc5CHoEtu7J2myUfWc2P4-WkIpz6v05N6jx7Is4opriZ_bX5LjGx56MnONgC_9OeZ5BoPJSVrbxVD83aNl28Gt3t2NzqxZr7EUVA1hyaUT8g0tAmJhOKM5zWbKdoTll91UPZdu9JSKV'

export default function SignUpPage() {
  const navigate = useNavigate()

  const [form, setForm] = useState({
    fullName: '',
    institution: '',
    professionalId: '',
    email: '',
    password: '',
    acceptedTerms: false,
  })
  const [showPwd, setShowPwd] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  // Generic field updater — works for all text inputs
  function setField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!form.acceptedTerms) {
      setError('Please accept the Privacy Protocol and Terms of Clinical Use.')
      return
    }
    setError(null)
    setLoading(true)
    try {
      await register(form)
      navigate('/login')
    } catch (err) {
      setError('Registration failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="flex min-h-screen overflow-hidden">
      {/* ─── Left: Retinal art panel ─────────────────────────────────── */}
      <div className="hidden lg:flex lg:w-1/2 relative bg-primary-fixed overflow-hidden">
        {/* Background retinal art */}
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url('${BG_IMG}')`, filter: 'contrast(1.1) brightness(1.05)' }}
        />
        {/* Subtle right-side fade to not clash with form */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent to-surface/20" />

        {/* Bottom-left brand copy */}
        <div className="absolute bottom-16 left-12 lg:left-16 z-10 max-w-md">
          <h1 className="font-serif text-5xl text-on-primary-fixed tracking-tight leading-tight">
            Leison Lens
          </h1>
          <p className="text-body-lg text-on-primary-fixed-variant mt-4 opacity-80">
            Advancing diabetic retinopathy screening through clinical precision and empathetic design.
          </p>
        </div>

        {/* Decorative glow */}
        <div className="absolute top-16 right-16 border border-white/30 rounded-full w-64 h-64 blur-3xl opacity-30 bg-primary-container" />
      </div>

      {/* ─── Right: Sign-up form ─────────────────────────────────────── */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-6 md:p-10 bg-gradient-to-br from-[#f0edf8] via-surface to-[#fde8e4] overflow-y-auto">
        {/* Glass card */}
        <div className="glass bg-white/40 w-full max-w-lg p-8 md:p-10 rounded-[2rem] shadow-2xl relative my-8">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-3">
              <Icon name="visibility" filled className="text-primary text-3xl" />
              <span className="font-mono text-xs text-primary tracking-[0.2em] uppercase">
                Clinical Portal
              </span>
            </div>
            <h2 className="text-h1 font-bold text-on-surface">Create your account</h2>
            <p className="text-on-surface-variant mt-2">
              Join our network of healthcare professionals.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Full Name */}
            <div className="space-y-1.5">
              <label className="font-mono text-xs text-on-surface-variant uppercase tracking-widest pl-1">
                Full Name
              </label>
              <input
                type="text"
                value={form.fullName}
                onChange={(e) => setField('fullName', e.target.value)}
                placeholder="Dr. Julian Vestergaard"
                required
                className="w-full bg-white/40 border border-outline-variant focus:border-primary focus:ring-0 rounded-xl px-4 py-3.5 text-on-surface placeholder:text-outline/50 transition-colors"
              />
            </div>

            {/* Clinical Institution */}
            <div className="space-y-1.5">
              <label className="font-mono text-xs text-on-surface-variant uppercase tracking-widest pl-1">
                Clinical Institution
              </label>
              <input
                type="text"
                value={form.institution}
                onChange={(e) => setField('institution', e.target.value)}
                placeholder="Nordic Eye Institute"
                required
                className="w-full bg-white/40 border border-outline-variant focus:border-primary focus:ring-0 rounded-xl px-4 py-3.5 text-on-surface placeholder:text-outline/50 transition-colors"
              />
            </div>

            {/* Professional ID + Work Email — 2 columns */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="font-mono text-xs text-on-surface-variant uppercase tracking-widest pl-1">
                  Professional ID
                </label>
                <input
                  type="text"
                  value={form.professionalId}
                  onChange={(e) => setField('professionalId', e.target.value)}
                  placeholder="MD-8829-X"
                  required
                  className="w-full bg-white/40 border border-outline-variant focus:border-primary focus:ring-0 rounded-xl px-4 py-3.5 text-on-surface placeholder:text-outline/50 transition-colors"
                />
              </div>
              <div className="space-y-1.5">
                <label className="font-mono text-xs text-on-surface-variant uppercase tracking-widest pl-1">
                  Work Email
                </label>
                <input
                  type="email"
                  value={form.email}
                  onChange={(e) => setField('email', e.target.value)}
                  placeholder="clinical@leison.org"
                  required
                  className="w-full bg-white/40 border border-outline-variant focus:border-primary focus:ring-0 rounded-xl px-4 py-3.5 text-on-surface placeholder:text-outline/50 transition-colors"
                />
              </div>
            </div>

            {/* Security Credentials (password) */}
            <div className="space-y-1.5">
              <label className="font-mono text-xs text-on-surface-variant uppercase tracking-widest pl-1">
                Security Credentials
              </label>
              <div className="relative">
                <input
                  type={showPwd ? 'text' : 'password'}
                  value={form.password}
                  onChange={(e) => setField('password', e.target.value)}
                  placeholder="••••••••••••"
                  required
                  minLength={8}
                  className="w-full bg-white/40 border border-outline-variant focus:border-primary focus:ring-0 rounded-xl px-4 py-3.5 pr-12 text-on-surface transition-colors"
                />
                <button
                  type="button"
                  onClick={() => setShowPwd(!showPwd)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-outline hover:text-primary"
                >
                  <Icon name={showPwd ? 'visibility_off' : 'visibility'} className="text-sm" />
                </button>
              </div>
            </div>

            {/* Privacy Protocol toggle */}
            <div className="flex items-start gap-3 py-2">
              {/* Custom animated toggle matching design */}
              <button
                type="button"
                role="switch"
                aria-checked={form.acceptedTerms}
                onClick={() => setField('acceptedTerms', !form.acceptedTerms)}
                className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full transition-colors duration-200 mt-0.5 ${
                  form.acceptedTerms ? 'bg-primary-container' : 'bg-outline-variant/30'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 mt-1 transform rounded-full bg-white shadow-sm transition-transform duration-200 ${
                    form.acceptedTerms ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
              <label className="text-sm text-on-surface-variant">
                I accept the{' '}
                <a href="#" className="text-primary hover:underline">Privacy Protocol</a>{' '}
                and{' '}
                <a href="#" className="text-primary hover:underline">Terms of Clinical Use</a>.
              </label>
            </div>

            {/* Error */}
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
              Register Account
            </Button>
          </form>

          {/* Sign in link */}
          <div className="mt-6 text-center">
            <p className="text-on-surface-variant text-sm">
              Already registered?{' '}
              <Link to="/login" className="text-primary font-semibold hover:underline ml-1">
                Sign in to Portal
              </Link>
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}
