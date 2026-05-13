export const DR_GRADE_LABELS = {
  0: { label: 'No DR', severity: 'None' },
  1: { label: 'Mild NPDR', severity: 'Early' },
  2: { label: 'Moderate NPDR', severity: 'Moderate' },
  3: { label: 'Severe NPDR', severity: 'Advanced' },
  4: { label: 'Proliferative DR', severity: 'Critical' },
}

export const MOCK_PATIENTS = [
  { id: 1, name: 'Priya Sharma', patientId: 'PT-000001', scanDate: '12 May 2026', drGrade: 2, confidence: 94.2, previewUrl: 'https://placehold.co/200x200/png?text=Scan' },
  { id: 2, name: 'Arun Kumar', patientId: 'PT-000002', scanDate: '12 May 2026', drGrade: 0, confidence: 98.7, previewUrl: 'https://placehold.co/200x200/png?text=Scan' },
  { id: 3, name: 'Lakshmi Devi', patientId: 'PT-000003', scanDate: '11 May 2026', drGrade: 4, confidence: 96.1, previewUrl: 'https://placehold.co/200x200/png?text=Scan' },
  { id: 4, name: 'Ravi Patel', patientId: 'PT-000004', scanDate: '11 May 2026', drGrade: 1, confidence: 92.8, previewUrl: 'https://placehold.co/200x200/png?text=Scan' },
  { id: 5, name: 'Sunita Reddy', patientId: 'PT-000005', scanDate: '10 May 2026', drGrade: 3, confidence: 95.5, previewUrl: 'https://placehold.co/200x200/png?text=Scan' },
]

export const MOCK_SCAN_RESULT = {
  drGrade: 2,
  confidence: 94.2,
  patientName: 'Priya Sharma',
  patientId: 'PT-000001',
  scanDate: '12 May 2026',
  originalUrl: '/assets/fundus-scan.png',
  heatmapUrl: '/assets/gradcam-heatmap.png',
  lesionProbs: { MA: 72, HE: 45, EX: 33, SE: 18 },
  aiInsight: 'Moderate NPDR detected. Microaneurysms and dot-blot hemorrhages present in the superior temporal quadrant. Referral to ophthalmologist within 6 months recommended.',
}

export const SEVERITY_BADGE = {
  0: { bg: 'bg-success/20', text: 'text-success', label: 'No DR' },
  1: { bg: 'bg-info/20', text: 'text-info', label: 'Mild' },
  2: { bg: 'bg-warning/20', text: 'text-warning', label: 'Moderate' },
  3: { bg: 'bg-error/20', text: 'text-error', label: 'Severe' },
  4: { bg: 'bg-error/20', text: 'text-error', label: 'Proliferative' },
}
