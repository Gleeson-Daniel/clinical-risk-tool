import type { FormEvent } from 'react'
import { FIELDS } from '../fields'
import { cardStyle, colors, eyebrowStyle } from '../theme'

interface Props {
  values: Record<string, string>
  errors: Record<string, string>
  loading: boolean
  onChange: (key: string, value: string) => void
  onSubmit: () => void
}

export default function PatientForm({ values, errors, loading, onChange, onSubmit }: Props) {
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    onSubmit()
  }

  return (
    <form onSubmit={handleSubmit} noValidate style={cardStyle}>
      <div style={{ ...eyebrowStyle, fontSize: 12, marginBottom: 16 }}>
        PATIENT PARAMETERS
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        {FIELDS.map(field => {
          const error = errors[field.key]
          const inputStyle = {
            width: '100%', background: colors.bg,
            border: `1px solid ${error ? colors.accent : colors.border}`,
            borderRadius: 5, padding: '6px 10px',
            color: colors.text, fontSize: 13
          }
          return (
            <div key={field.key}>
              <label htmlFor={field.key} style={{ fontSize: 11, color: colors.muted,
                display: 'block', marginBottom: 4 }}>
                {field.label}
                {field.kind === 'number' && field.unit ? ` (${field.unit})` : ''}
              </label>
              {field.kind === 'select' ? (
                <select
                  id={field.key}
                  value={values[field.key]}
                  onChange={e => onChange(field.key, e.target.value)}
                  style={inputStyle}
                >
                  {field.options.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              ) : (
                <input
                  id={field.key}
                  type="number"
                  value={values[field.key]}
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  aria-invalid={error ? true : undefined}
                  onChange={e => onChange(field.key, e.target.value)}
                  style={inputStyle}
                />
              )}
              {error && (
                <div style={{ fontSize: 11, color: colors.accent, marginTop: 3 }}>
                  {error}
                </div>
              )}
            </div>
          )
        })}
      </div>
      <button
        type="submit"
        disabled={loading}
        style={{
          marginTop: 16, width: '100%', padding: '12px 0',
          background: colors.accent, border: 'none', borderRadius: 7,
          color: '#ffffff', fontWeight: 600, fontSize: 14,
          cursor: loading ? 'wait' : 'pointer'
        }}>
        {loading ? 'Analyzing...' : 'Predict Risk'}
      </button>
    </form>
  )
}
