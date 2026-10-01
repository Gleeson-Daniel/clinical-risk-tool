import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Cell
} from 'recharts'
import type { PredictionResult } from '../api'
import { FEATURE_LABELS } from '../fields'
import { cardStyle, colors, eyebrowStyle } from '../theme'

const RISK_COLORS: Record<string, string> = {
  Low: colors.low, Moderate: colors.moderate, High: colors.high
}

export default function RiskResults({ result }: { result: PredictionResult | null }) {
  if (!result) {
    return (
      <div style={{ ...cardStyle, padding: 40, display: 'flex',
        alignItems: 'center', justifyContent: 'center',
        color: colors.faint, fontSize: 14, flex: 1 }}>
        Enter patient parameters and click Predict
      </div>
    )
  }

  const riskColor = RISK_COLORS[result.risk_label] ?? colors.muted

  const shapData = Object.entries(result.shap_values)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 8)
    .map(([k, v]) => ({ name: FEATURE_LABELS[k] ?? k, value: v }))

  return (
    <>
      {/* Risk Score */}
      <div style={{ ...cardStyle, borderTop: `3px solid ${riskColor}` }}>
        <div style={{ ...eyebrowStyle, marginBottom: 8 }}>RISK ASSESSMENT</div>
        <div style={{ fontSize: 48, fontWeight: 700, color: riskColor }}>
          {(result.risk_score * 100).toFixed(1)}%
        </div>
        <div style={{ fontSize: 18, color: riskColor, marginTop: 4 }}>
          {result.risk_label} Risk
        </div>
        <div style={{ fontSize: 11, color: colors.muted, marginTop: 8 }}>
          Model AUC: {result.model_auc.toFixed(2)} ± {result.model_auc_std.toFixed(2)} (5-fold
          cross-validation)
        </div>
      </div>

      {/* SHAP Chart */}
      <div style={{ ...cardStyle, flex: 1 }}>
        <div style={{ ...eyebrowStyle, marginBottom: 12 }}>
          TOP RISK FACTORS (SHAP)
        </div>
        <div style={{ fontSize: 11, color: colors.muted, marginBottom: 12 }}>
          Positive = increases risk · Negative = reduces risk. Values are in
          log-odds, not percentage points.
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={shapData} layout="vertical" margin={{ bottom: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={colors.border} />
            <XAxis type="number" tick={{ fill: colors.muted, fontSize: 10 }}
              label={{ value: 'Contribution to log-odds of disease',
                position: 'insideBottom', offset: -10,
                fill: colors.muted, fontSize: 10 }} />
            <YAxis type="category" dataKey="name" width={120}
              tick={{ fill: colors.text, fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: colors.card,
                border: `1px solid ${colors.border}`, borderRadius: 6 }}
              formatter={(v) => [typeof v === 'number' ? v.toFixed(4) : String(v ?? ''), 'Log-odds']}
            />
            <ReferenceLine x={0} stroke={colors.faint} />
            <Bar dataKey="value">
              {shapData.map((entry, i) => (
                <Cell key={i}
                  fill={entry.value > 0 ? colors.high : colors.low} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </>
  )
}
