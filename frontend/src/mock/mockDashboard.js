export const MOCK_STATS = [
  { id: 'total', label: 'Total Scans', value: '1,284', color: 'text-primary', accentColor: 'bg-primary', subtext: '+12% from last month' },
  { id: 'early', label: 'Early Detections', value: '847', color: 'text-tertiary', accentColor: 'bg-tertiary', subtext: '66% of total scans' },
  { id: 'accuracy', label: 'AI Accuracy', value: '98.4%', color: 'text-secondary', accentColor: 'bg-secondary', subtext: '+0.3% this week' },
  { id: 'backlog', label: 'Backlog', value: '23', color: 'text-error', accentColor: 'bg-error', subtext: 'Needs review' },
]

export const MOCK_ACTIVITY = [
  { id: 1, initials: 'PS', name: 'Priya Sharma', patientId: 'PT-000001', scanDate: '12 May 2026', drDotColor: 'bg-warning', drGradeColor: 'text-warning', drGradeLabel: 'Moderate NPDR', confidence: '94.2%' },
  { id: 2, initials: 'AK', name: 'Arun Kumar', patientId: 'PT-000002', scanDate: '12 May 2026', drDotColor: 'bg-success', drGradeColor: 'text-success', drGradeLabel: 'No DR', confidence: '98.7%' },
  { id: 3, initials: 'LD', name: 'Lakshmi Devi', patientId: 'PT-000003', scanDate: '11 May 2026', drDotColor: 'bg-error', drGradeColor: 'text-error', drGradeLabel: 'Proliferative DR', confidence: '96.1%' },
  { id: 4, initials: 'RP', name: 'Ravi Patel', patientId: 'PT-000004', scanDate: '11 May 2026', drDotColor: 'bg-info', drGradeColor: 'text-info', drGradeLabel: 'Mild NPDR', confidence: '92.8%' },
  { id: 5, initials: 'SR', name: 'Sunita Reddy', patientId: 'PT-000005', scanDate: '10 May 2026', drDotColor: 'bg-warning', drGradeColor: 'text-warning', drGradeLabel: 'Severe NPDR', confidence: '95.5%' },
]

export const MOCK_CLINICIAN = {
  name: 'Dr. Ananya Gupta',
}
