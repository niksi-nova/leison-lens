/*
 * API stubs — the single place where all backend calls live.
 *
 * HOW TO CONNECT TO YOUR FLASK BACKEND:
 * Each function currently returns mock data. When you build the backend,
 * replace the body with a real fetch() call. The vite.config.js proxy
 * already forwards /api/* to localhost:5000, so no CORS issues.
 *
 * Example replacement for login():
 *   export async function login(email, password) {
 *     const res = await fetch('/api/auth/login', {
 *       method: 'POST',
 *       headers: { 'Content-Type': 'application/json' },
 *       body: JSON.stringify({ email, password }),
 *     })
 *     if (!res.ok) throw new Error('Login failed')
 *     return res.json() // { token, user }
 *   }
 */

import { MOCK_PATIENTS, MOCK_SCAN_RESULT } from '../data/mockPatients.js'
import { MOCK_STATS, MOCK_ACTIVITY, MOCK_CLINICIAN } from '../data/mockDashboard.js'

// Simulates network latency so UI loading states are visible during dev
const delay = (ms = 400) => new Promise((r) => setTimeout(r, ms))

// ─── Auth ────────────────────────────────────────────────────────────────

/**
 * Login — POST /api/auth/login
 * @returns {{ token: string, user: object }}
 */
export async function login(email, password) {
  await delay()
  // Mock: accept any credentials
  return { token: 'mock-jwt-token', user: MOCK_CLINICIAN }
}

/**
 * Register — POST /api/auth/register
 * @returns {{ message: string }}
 */
export async function register(formData) {
  await delay(600)
  return { message: 'Account created successfully' }
}

// ─── Dashboard ───────────────────────────────────────────────────────────

/**
 * Dashboard stats — GET /api/dashboard/stats
 * @returns {{ stats: array, activity: array, clinician: object }}
 */
export async function getDashboard() {
  await delay()
  return { stats: MOCK_STATS, activity: MOCK_ACTIVITY, clinician: MOCK_CLINICIAN }
}

// ─── Patients ────────────────────────────────────────────────────────────

/**
 * Patient list — GET /api/patients?page=1&search=&filter=
 * @returns {{ patients: array, total: number }}
 */
export async function getPatients({ search = '', page = 1 } = {}) {
  await delay()
  const filtered = MOCK_PATIENTS.filter(
    (p) =>
      p.name.toLowerCase().includes(search.toLowerCase()) ||
      p.patientId.toLowerCase().includes(search.toLowerCase())
  )
  return { patients: filtered, total: 1284 }
}

// ─── Scans ───────────────────────────────────────────────────────────────

/**
 * Upload scan image — POST /api/scan/upload (multipart/form-data)
 * @param {File} file
 * @returns {{ scanId: string, previewUrl: string }}
 */
export async function uploadScan(file) {
  await delay(1200)
  // In production: upload file, get back a scan ID and preview URL
  return { scanId: 'scan-mock-001', previewUrl: URL.createObjectURL(file) }
}

/**
 * Start analysis — POST /api/scan/:scanId/analyze
 * @returns {{ jobId: string }}
 */
export async function startAnalysis(scanId) {
  await delay(2000)
  // In production: triggers the EfficientNet-B4 model inference
  return { jobId: 'job-mock-001', resultId: 'result-mock-001' }
}

/**
 * Get scan results — GET /api/scan/:id/results
 * @returns scan result object with drGrade, lesionProbs, heatmapUrl, aiInsight
 */
export async function getScanResults(scanId) {
  await delay()
  return MOCK_SCAN_RESULT
}

// ─── Settings ────────────────────────────────────────────────────────────

/**
 * Save settings — PUT /api/settings
 * @returns {{ message: string }}
 */
export async function saveSettings(settingsData) {
  await delay(500)
  return { message: 'Settings saved successfully' }
}
