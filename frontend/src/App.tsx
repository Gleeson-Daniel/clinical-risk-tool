import { useState } from 'react'
import { ApiError, predictRisk } from './api'
import type { PredictionResult } from './api'
import PatientForm from './components/PatientForm'
import RiskResults from './components/RiskResults'
import { DEFAULT_PATIENT, validatePatient } from './fields'
import { colors, eyebrowStyle } from './theme'

export default function App() {
  const [values, setValues] = useState<Record<string, string>>(DEFAULT_PATIENT)
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({})
  const [apiErrors, setApiErrors] = useState<string[]>([])
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)

  const handleChange = (key: string, value: string) => {
    setValues(v => ({ ...v, [key]: value }))
    setFieldErrors(errs => {
      const rest = { ...errs }
      delete rest[key]
      return rest
    })
  }

  const handleSubmit = async () => {
    const { patient, errors } = validatePatient(values)
    setFieldErrors(errors)
    setApiErrors([])
    if (Object.keys(errors).length > 0) {
      setResult(null)
      return
    }

    setLoading(true)
    try {
      setResult(await predictRisk(patient))
    } catch (e) {
      setResult(null)
      setApiErrors(e instanceof ApiError ? e.messages : ['Something went wrong.'])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ background: colors.bg, minHeight: '100vh',
      color: colors.text, padding: 28,
      maxWidth: 900, margin: '0 auto' }}>

      <div style={{ marginBottom: 20 }}>
        <div style={{ ...eyebrowStyle, marginBottom: 4 }}>CLINICAL DECISION SUPPORT</div>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: colors.heading }}>
          Cardiac Risk Stratification
        </h1>
      </div>

      <div role="note" style={{ background: colors.noticeBg,
        border: `1px solid ${colors.noticeBorder}`, borderRadius: 8,
        padding: '10px 14px', marginBottom: 20,
        fontSize: 12, color: colors.noticeText }}>
        <strong>Not for clinical use.</strong> This is a demonstration model
        trained on 297 patients from the 1988 Cleveland heart disease dataset.
        It has not been clinically validated and must not be used to make
        decisions about real patients.
      </div>

      <div style={{ display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 20 }}>

        <PatientForm
          values={values}
          errors={fieldErrors}
          loading={loading}
          onChange={handleChange}
          onSubmit={handleSubmit}
        />

        {/* Result Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {apiErrors.length > 0 && (
            <div role="alert" style={{ background: colors.accentBg,
              border: `1px solid ${colors.accentBorder}`, borderRadius: 8,
              padding: '10px 14px', fontSize: 13, color: colors.accent }}>
              <strong>Could not get a prediction</strong>
              <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
                {apiErrors.map(msg => <li key={msg}>{msg}</li>)}
              </ul>
            </div>
          )}
          <RiskResults result={result} />
        </div>
      </div>
    </div>
  )
}
