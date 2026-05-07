import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Cell
} from 'recharts'

const API = 'http://localhost:8001'

interface Result {
  risk_score: number
  risk_label: string
  shap_values: Record<string, number>
  model_auc: number
}

const FEATURE_LABELS: Record<string, string> = {
  age: 'Age', sex: 'Sex', cp: 'Chest Pain Type',
  trestbps: 'Resting BP', chol: 'Cholesterol', fbs: 'Fasting Blood Sugar',
  restecg: 'Resting ECG', thalach: 'Max Heart Rate', exang: 'Exercise Angina',
  oldpeak: 'ST Depression', slope: 'ST Slope', ca: 'Vessels Colored', thal: 'Thal'
}

const DEFAULT_PATIENT = {
  age: 52, sex: 1, cp: 2, trestbps: 130, chol: 245,
  fbs: 0, restecg: 0, thalach: 160, exang: 0,
  oldpeak: 1.4, slope: 2, ca: 0, thal: 3
}

export default function App() {
  const [patient, setPatient] = useState<Record<string, number>>(DEFAULT_PATIENT)
  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patient)
      })
      setResult(await res.json())
    } catch (e) {
      alert('API error — is the backend running on port 8001?')
    } finally {
      setLoading(false)
    }
  }

  const riskColor = result
    ? result.risk_label === 'Low' ? '#34d399'
    : result.risk_label === 'Moderate' ? '#fbbf24' : '#fb7185'
    : '#6b7391'

  const shapData = result
    ? Object.entries(result.shap_values)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 8)
        .map(([k, v]) => ({ name: FEATURE_LABELS[k] ?? k, value: v }))
    : []

  return (
    <div style={{ background: '#0a0b0f', minHeight: '100vh',
      color: '#d8dce8', padding: 28, fontFamily: 'sans-serif',
      maxWidth: 900, margin: '0 auto' }}>

      <div style={{ marginBottom: 28 }}>
        <div style={{ fontSize: 11, fontFamily: 'monospace',
          color: '#6b7391', marginBottom: 4 }}>CLINICAL DECISION SUPPORT</div>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#fff' }}>
          Cardiac Risk Stratification
        </h1>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>

        {/* Input Form */}
        <div style={{ background: '#111318', border: '1px solid #1e2230',
          borderRadius: 10, padding: 20 }}>
          <div style={{ fontSize: 12, fontFamily: 'monospace',
            color: '#6b7391', marginBottom: 16 }}>PATIENT PARAMETERS</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            {Object.entries(patient).map(([key, val]) => (
              <div key={key}>
                <label style={{ fontSize: 11, color: '#6b7391',
                  display: 'block', marginBottom: 4 }}>
                  {FEATURE_LABELS[key] ?? key}
                </label>
                <input
                  type="number"
                  value={val}
                  step="0.1"
                  onChange={e => setPatient(p =>
                    ({ ...p, [key]: parseFloat(e.target.value) || 0 })
                  )}
                  style={{
                    width: '100%', background: '#0a0b0f',
                    border: '1px solid #1e2230', borderRadius: 5,
                    padding: '6px 10px', color: '#d8dce8', fontSize: 13
                  }}
                />
              </div>
            ))}
          </div>
          <button
            onClick={handleSubmit}
            disabled={loading}
            style={{
              marginTop: 16, width: '100%', padding: '12px 0',
              background: '#34d399', border: 'none', borderRadius: 7,
              color: '#0a0b0f', fontWeight: 600, fontSize: 14,
              cursor: loading ? 'wait' : 'pointer'
            }}>
            {loading ? 'Analyzing...' : 'Predict Risk'}
          </button>
        </div>

        {/* Result Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {result ? (
            <>
              {/* Risk Score */}
              <div style={{ background: '#111318', border: `1px solid #1e2230`,
                borderTop: `3px solid ${riskColor}`, borderRadius: 10, padding: 20 }}>
                <div style={{ fontSize: 11, fontFamily: 'monospace',
                  color: '#6b7391', marginBottom: 8 }}>RISK ASSESSMENT</div>
                <div style={{ fontSize: 48, fontWeight: 700, color: riskColor }}>
                  {(result.risk_score * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: 18, color: riskColor, marginTop: 4 }}>
                  {result.risk_label} Risk
                </div>
                <div style={{ fontSize: 11, color: '#6b7391', marginTop: 8 }}>
                  Model AUC: {result.model_auc}
                </div>
              </div>

              {/* SHAP Chart */}
              <div style={{ background: '#111318', border: '1px solid #1e2230',
                borderRadius: 10, padding: 20, flex: 1 }}>
                <div style={{ fontSize: 11, fontFamily: 'monospace',
                  color: '#6b7391', marginBottom: 12 }}>
                  TOP RISK FACTORS (SHAP)
                </div>
                <div style={{ fontSize: 11, color: '#6b7391', marginBottom: 12 }}>
                  Positive = increases risk · Negative = reduces risk
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={shapData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e2230" />
                    <XAxis type="number" tick={{ fill: '#6b7391', fontSize: 10 }} />
                    <YAxis type="category" dataKey="name" width={120}
                      tick={{ fill: '#d8dce8', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ background: '#111318',
                        border: '1px solid #1e2230', borderRadius: 6 }}
                      formatter={(v) => [typeof v === 'number' ? v.toFixed(4) : String(v ?? ''), 'SHAP value']}
                    />
                    <ReferenceLine x={0} stroke="#2e3347" />
                    <Bar dataKey="value">
                      {shapData.map((entry, i) => (
                        <Cell key={i}
                          fill={entry.value > 0 ? '#fb7185' : '#34d399'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <div style={{ background: '#111318', border: '1px solid #1e2230',
              borderRadius: 10, padding: 40, display: 'flex',
              alignItems: 'center', justifyContent: 'center',
              color: '#2e3347', fontSize: 14, flex: 1 }}>
              Enter patient parameters and click Predict
            </div>
          )}
        </div>
      </div>
    </div>
  )
}