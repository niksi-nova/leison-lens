/*
 * SettingsPage — route: "/settings"
 *
 * Clinician account management and portal configuration.
 *
 * Sections:
 *   1. Clinician Profile — avatar, name, license ID, email (editable form)
 *   2. Notifications    — 3 toggles (Critical Findings, Weekly Analytics, Audit Logs)
 *   3. Institution      — hospital name, department, timezone select
 *   4. HIPAA & Privacy  — encryption status badge, Rotate Keys + Access Log buttons, 2FA
 *
 * Footer row: Discard Changes + Save Settings buttons.
 *
 * HOW TO CONNECT:
 *   On mount → load current settings from AuthContext (user) or GET /api/settings
 *   "Save Settings" → saveSettings(payload) → PATCH /api/settings
 *   "Rotate Keys"  → POST /api/settings/rotate-keys
 *   "Access Log"   → GET  /api/settings/access-log (download or navigate)
 */

import { useState } from 'react'
import PortalLayout from '../components/layout/PortalLayout.jsx'
import GlassCard from '../components/ui/GlassCard.jsx'
import Toggle from '../components/ui/Toggle.jsx'
import Button from '../components/ui/Button.jsx'
import Icon from '../components/ui/Icon.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import { saveSettings } from '../api/index.js'

// Timezone options for the institution section
const TIMEZONES = [
  'UTC',
  'Europe/London',
  'Europe/Copenhagen',
  'Europe/Berlin',
  'America/New_York',
  'America/Chicago',
  'America/Los_Angeles',
  'Asia/Dubai',
  'Asia/Kolkata',
  'Asia/Singapore',
  'Australia/Sydney',
]

export default function SettingsPage() {
  const { user } = useAuth()

  // ── Profile state ──────────────────────────────────────────────────────────
  const [profile, setProfile] = useState({
    name:       user?.name       ?? 'Dr. A. Vestergaard',
    licenseId:  user?.licenseId  ?? 'MD-8829-X',
    email:      user?.email      ?? 'a.vestergaard@leison.org',
    specialty:  user?.specialty  ?? 'Ophthalmology',
  })

  // ── Notification toggles ───────────────────────────────────────────────────
  const [notifications, setNotifications] = useState({
    criticalFindings: true,
    weeklyAnalytics:  true,
    auditLogs:        false,
  })

  // ── Institution state ──────────────────────────────────────────────────────
  const [institution, setInstitution] = useState({
    hospital:   'Nordic Eye Institute',
    department: 'Retinal Diagnostics',
    timezone:   'Europe/Copenhagen',
  })

  // ── Privacy / 2FA ─────────────────────────────────────────────────────────
  const [twoFactorEnabled, setTwoFactorEnabled] = useState(false)

  // ── UI state ──────────────────────────────────────────────────────────────
  const [saving, setSaving] = useState(false)
  const [saved,  setSaved]  = useState(false)
  const [error,  setError]  = useState(null)

  // Generic field updater for nested state objects
  function updateProfile(key, value) {
    setProfile((prev) => ({ ...prev, [key]: value }))
    setSaved(false)
  }

  function updateInstitution(key, value) {
    setInstitution((prev) => ({ ...prev, [key]: value }))
    setSaved(false)
  }

  function toggleNotification(key) {
    setNotifications((prev) => ({ ...prev, [key]: !prev[key] }))
    setSaved(false)
  }

  async function handleSave() {
    setError(null)
    setSaving(true)
    try {
      await saveSettings({ profile, notifications, institution, twoFactorEnabled })
      setSaved(true)
    } catch {
      setError('Failed to save settings. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  function handleDiscard() {
    // Reset to values from AuthContext / original defaults
    setProfile({
      name:      user?.name      ?? 'Dr. A. Vestergaard',
      licenseId: user?.licenseId ?? 'MD-8829-X',
      email:     user?.email     ?? 'a.vestergaard@leison.org',
      specialty: user?.specialty ?? 'Ophthalmology',
    })
    setNotifications({ criticalFindings: true, weeklyAnalytics: true, auditLogs: false })
    setInstitution({ hospital: 'Nordic Eye Institute', department: 'Retinal Diagnostics', timezone: 'Europe/Copenhagen' })
    setTwoFactorEnabled(false)
    setSaved(false)
    setError(null)
  }

  // Shared label style — consistent across all form fields
  const fieldLabel = 'font-mono text-[10px] text-on-surface-variant uppercase tracking-widest mb-1.5 block pl-1'
  const fieldInput = 'w-full bg-white/40 border border-outline-variant focus:border-primary focus:ring-0 rounded-xl px-4 py-3 text-on-surface text-sm placeholder:text-outline/50 transition-colors'

  return (
    <PortalLayout activePage="settings">
      {/* ─── Page Header ──────────────────────────────────────────────────── */}
      <header className="mb-10">
        <h1 className="text-h1 font-bold text-on-surface">Settings</h1>
        <p className="text-on-surface-variant mt-1">
          Manage your clinician profile 
           {/* notification preferences, and data-privacy controls. */}
        </p>
      </header>

      <div className="space-y-6">
        {/* ─── 1. Clinician Profile ───────────────────────────────────────── */}
        <GlassCard className="rounded-3xl p-6 md:p-8">
          {/* Section heading */}
          <div className="flex items-center gap-3 mb-8">
            <div className="w-9 h-9 rounded-xl bg-primary-container/20 flex items-center justify-center">
              <Icon name="person" className="text-primary text-xl" />
            </div>
            <h2 className="font-semibold text-on-surface text-lg">Clinician Profile</h2>
          </div>

          <div className="flex flex-col sm:flex-row gap-8 items-start">
            {/* Avatar upload area */}
            <div className="flex flex-col items-center gap-3 flex-shrink-0">
              <div className="w-24 h-24 rounded-full bg-primary-container/20 flex items-center justify-center border-2 border-primary/20 relative overflow-hidden">
                <span className="font-bold text-3xl text-primary">
                  {profile.name.split(' ').map((n) => n[0]).join('').slice(0, 2)}
                </span>
              </div>
              <button className="font-mono text-[10px] text-primary uppercase tracking-widest hover:underline">
                {/* Change Photo */}
              </button>
            </div>

            {/* Form fields — 2-column grid on sm+ */}
            <div className="flex-1 grid grid-cols-1 sm:grid-cols-2 gap-5 w-full">
              {/* Full Name */}
              <div>
                <label className={fieldLabel}>Full Name</label>
                <input
                  type="text"
                  value={profile.name}
                  onChange={(e) => updateProfile('name', e.target.value)}
                  className={fieldInput}
                />
              </div>

              {/* License ID */}
              <div>
                <label className={fieldLabel}>License ID</label>
                <input
                  type="text"
                  value={profile.licenseId}
                  onChange={(e) => updateProfile('licenseId', e.target.value)}
                  className={fieldInput}
                />
              </div>

              {/* Work Email */}
              <div>
                <label className={fieldLabel}>Work Email</label>
                <input
                  type="email"
                  value={profile.email}
                  onChange={(e) => updateProfile('email', e.target.value)}
                  className={fieldInput}
                />
              </div>

              {/* Specialty */}
              <div>
                <label className={fieldLabel}>Specialty</label>
                <input
                  type="text"
                  value={profile.specialty}
                  onChange={(e) => updateProfile('specialty', e.target.value)}
                  className={fieldInput}
                />
              </div>
            </div>
          </div>
        </GlassCard>

        {/* ─── 2. Notifications ───────────────────────────────────────────── */}
        {/* <GlassCard className="rounded-3xl p-6 md:p-8"> */}
          {/* Section heading */}
          {/* <div className="flex items-center gap-3 mb-8">
            <div className="w-9 h-9 rounded-xl bg-secondary-container/20 flex items-center justify-center">
              <Icon name="notifications" className="text-secondary text-xl" />
            </div>
            <h2 className="font-semibold text-on-surface text-lg">Notifications</h2>
          </div> */}

          {/* <div className="space-y-6"> */}
            {/* Critical Findings */}
            {/* <div className="flex items-center justify-between gap-4 py-3 border-b border-white/20">
              <div>
                <p className="font-semibold text-on-surface text-sm">Critical Findings Alert</p>
                <p className="text-xs text-on-surface-variant mt-0.5">
                  Receive immediate notification when a scan is graded Severe or Proliferative DR.
                </p>
              </div>
              <Toggle
                checked={notifications.criticalFindings}
                onChange={() => toggleNotification('criticalFindings')}
              />
            </div> */}

            {/* Weekly Analytics */}
            {/* <div className="flex items-center justify-between gap-4 py-3 border-b border-white/20">
              <div>
                <p className="font-semibold text-on-surface text-sm">Weekly Analytics Report</p>
                <p className="text-xs text-on-surface-variant mt-0.5">
                  Summary of scan volumes, grade distribution, and model performance sent every Monday.
                </p>
              </div>
              <Toggle
                checked={notifications.weeklyAnalytics}
                onChange={() => toggleNotification('weeklyAnalytics')}
              />
            </div> */}

            {/* Security Audit Logs */}
            {/* <div className="flex items-center justify-between gap-4 py-3">
              <div>
                <p className="font-semibold text-on-surface text-sm">Security Audit Log Digest</p>
                <p className="text-xs text-on-surface-variant mt-0.5">
                  Monthly digest of login events, key rotations, and data access records.
                </p>
              </div>
              <Toggle
                checked={notifications.auditLogs}
                onChange={() => toggleNotification('auditLogs')}
              />
            </div>
          </div>
        </GlassCard> */}

        {/* ─── 3. Institution Details ─────────────────────────────────────── */}
        {/* <GlassCard className="rounded-3xl p-6 md:p-8"> */}
          {/* Section heading */}
          {/* <div className="flex items-center gap-3 mb-8">
            <div className="w-9 h-9 rounded-xl bg-tertiary-container/20 flex items-center justify-center">
              <Icon name="local_hospital" className="text-tertiary text-xl" />
            </div>
            <h2 className="font-semibold text-on-surface text-lg">Institution Details</h2>
          </div> */}

          {/* <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5"> */}
            {/* Hospital name */}
            {/* <div>
              <label className={fieldLabel}>Hospital / Clinic Name</label>
              <input
                type="text"
                value={institution.hospital}
                onChange={(e) => updateInstitution('hospital', e.target.value)}
                className={fieldInput}
              />
            </div> */}

            {/* Department */}
            {/* <div>
              <label className={fieldLabel}>Department</label>
              <input
                type="text"
                value={institution.department}
                onChange={(e) => updateInstitution('department', e.target.value)}
                className={fieldInput}
              />
            </div> */}

            {/* Timezone */}
            {/* <div>
              <label className={fieldLabel}>Timezone</label>
              <select
                value={institution.timezone}
                onChange={(e) => updateInstitution('timezone', e.target.value)}
                className={`${fieldInput} cursor-pointer`}
              >
                {TIMEZONES.map((tz) => (
                  <option key={tz} value={tz}>{tz}</option>
                ))}
              </select>
            </div>
          </div>
        </GlassCard> */}

        {/* ─── 4. HIPAA & Privacy ─────────────────────────────────────────── */}
        {/* <GlassCard className="rounded-3xl p-6 md:p-8"> */}
          {/* Section heading */}
          {/* <div className="flex items-center gap-3 mb-8">
            <div className="w-9 h-9 rounded-xl bg-error-container/20 flex items-center justify-center">
              <Icon name="security" className="text-error text-xl" />
            </div>
            <h2 className="font-semibold text-on-surface text-lg">HIPAA &amp; Privacy</h2>
          </div>

          <div className="space-y-6"> */}
            {/* Encryption status */}
            {/* <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 bg-white/20 rounded-2xl border border-white/30">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-primary-container/20 flex items-center justify-center">
                  <Icon name="lock" className="text-primary text-base" />
                </div>
                <div>
                  <p className="font-semibold text-on-surface text-sm">Data Encryption</p>
                  <p className="text-xs text-on-surface-variant">AES-256 at rest · TLS 1.3 in transit</p>
                </div>
              </div> */}
              {/* Active badge */}
              {/* <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-primary-container/20 text-primary rounded-full text-xs font-bold self-start sm:self-auto">
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                Active
              </span>
            </div> */}

            {/* Key rotation + access log */}
            {/* <div className="flex flex-col sm:flex-row gap-4">
              <Button variant="secondary" size="sm" icon="refresh" className="flex-1 justify-center">
                Rotate Encryption Keys
              </Button>
              <Button variant="ghost" size="sm" icon="description" className="flex-1 justify-center">
                Download Access Log
              </Button>
            </div> */}

            {/* 2FA toggle */}
            {/* <div className="flex items-center justify-between gap-4 py-3 border-t border-white/20 mt-2">
              <div>
                <p className="font-semibold text-on-surface text-sm">Two-Factor Authentication</p>
                <p className="text-xs text-on-surface-variant mt-0.5">
                  Require OTP on each login to protect patient data access.
                </p>
              </div>
              <Toggle
                checked={twoFactorEnabled}
                onChange={() => { setTwoFactorEnabled((v) => !v); setSaved(false) }}
              />
            </div>
          </div>
        </GlassCard> */}

        {/* ─── Action Footer ──────────────────────────────────────────────── */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 py-2">
          {/* Status feedback */}
          <div>
            {saved && (
              <p className="text-sm text-primary flex items-center gap-2">
                <Icon name="check_circle" className="text-base" />
                Settings saved successfully.
              </p>
            )}
            {error && (
              <p className="text-sm text-error flex items-center gap-2">
                <Icon name="error" className="text-base" />
                {error}
              </p>
            )}
          </div>

          <div className="flex gap-3 w-full sm:w-auto">
            <Button
              variant="ghost"
              size="md"
              onClick={handleDiscard}
              disabled={saving}
              className="flex-1 sm:flex-none justify-center"
            >
              Discard Changes
            </Button>
            <Button
              variant="primary"
              size="md"
              loading={saving}
              onClick={handleSave}
              className="flex-1 sm:flex-none justify-center px-8"
            >
              Save Settings
            </Button>
          </div>
        </div>
      </div>
    </PortalLayout>
  )
}
