// Field definitions mirror the validation rules in backend/schemas.py.

interface NumberField {
  key: string
  label: string
  kind: 'number'
  min: number
  max: number
  step: number
  unit?: string
}

interface SelectField {
  key: string
  label: string
  kind: 'select'
  options: { value: number; label: string }[]
}

export type Field = NumberField | SelectField

export const FIELDS: Field[] = [
  { key: 'age', label: 'Age', kind: 'number', min: 18, max: 100, step: 1, unit: 'years' },
  { key: 'sex', label: 'Sex', kind: 'select', options: [
    { value: 1, label: 'Male' },
    { value: 0, label: 'Female' },
  ] },
  { key: 'cp', label: 'Chest Pain Type', kind: 'select', options: [
    { value: 1, label: 'Typical angina' },
    { value: 2, label: 'Atypical angina' },
    { value: 3, label: 'Non-anginal pain' },
    { value: 4, label: 'Asymptomatic' },
  ] },
  { key: 'trestbps', label: 'Resting BP', kind: 'number', min: 80, max: 220, step: 1, unit: 'mmHg' },
  { key: 'chol', label: 'Cholesterol', kind: 'number', min: 100, max: 600, step: 1, unit: 'mg/dl' },
  { key: 'fbs', label: 'Fasting Blood Sugar', kind: 'select', options: [
    { value: 0, label: '120 mg/dl or below' },
    { value: 1, label: 'Above 120 mg/dl' },
  ] },
  { key: 'restecg', label: 'Resting ECG', kind: 'select', options: [
    { value: 0, label: 'Normal' },
    { value: 1, label: 'ST-T abnormality' },
    { value: 2, label: 'LV hypertrophy' },
  ] },
  { key: 'thalach', label: 'Max Heart Rate', kind: 'number', min: 60, max: 220, step: 1, unit: 'bpm' },
  { key: 'exang', label: 'Exercise Angina', kind: 'select', options: [
    { value: 0, label: 'No' },
    { value: 1, label: 'Yes' },
  ] },
  { key: 'oldpeak', label: 'ST Depression', kind: 'number', min: 0, max: 7, step: 0.1, unit: 'mm' },
  { key: 'slope', label: 'ST Slope', kind: 'select', options: [
    { value: 1, label: 'Upsloping' },
    { value: 2, label: 'Flat' },
    { value: 3, label: 'Downsloping' },
  ] },
  { key: 'ca', label: 'Vessels Colored', kind: 'select', options: [
    { value: 0, label: '0' },
    { value: 1, label: '1' },
    { value: 2, label: '2' },
    { value: 3, label: '3' },
  ] },
  { key: 'thal', label: 'Thal', kind: 'select', options: [
    { value: 3, label: 'Normal' },
    { value: 6, label: 'Fixed defect' },
    { value: 7, label: 'Reversible defect' },
  ] },
]

export const FEATURE_LABELS: Record<string, string> =
  Object.fromEntries(FIELDS.map(f => [f.key, f.label]))

// Form values are kept as strings so a number box can be cleared while typing.
export const DEFAULT_PATIENT: Record<string, string> = {
  age: '52', sex: '1', cp: '2', trestbps: '130', chol: '245',
  fbs: '0', restecg: '0', thalach: '160', exang: '0',
  oldpeak: '1.4', slope: '2', ca: '0', thal: '3'
}

export function validatePatient(values: Record<string, string>) {
  const patient: Record<string, number> = {}
  const errors: Record<string, string> = {}

  for (const field of FIELDS) {
    const raw = (values[field.key] ?? '').trim()
    const num = Number(raw)
    if (raw === '' || !Number.isFinite(num)) {
      errors[field.key] = 'Enter a number'
    } else if (field.kind === 'number' && (num < field.min || num > field.max)) {
      errors[field.key] = `Must be between ${field.min} and ${field.max}`
    } else {
      patient[field.key] = num
    }
  }

  return { patient, errors }
}
