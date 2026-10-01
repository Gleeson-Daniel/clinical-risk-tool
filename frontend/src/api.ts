import { FEATURE_LABELS } from './fields'

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8001'

export interface PredictionResult {
  risk_score: number
  risk_label: string
  shap_values: Record<string, number>
  model_auc: number
  model_auc_std: number
}

interface ValidationIssue {
  loc?: (string | number)[]
  msg?: string
}

// Turns a FastAPI error body into one readable line per problem.
function errorMessages(status: number, body: unknown): string[] {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (Array.isArray(detail)) {
    return (detail as ValidationIssue[]).map(issue => {
      const key = String(issue.loc?.at(-1) ?? '')
      return `${FEATURE_LABELS[key] ?? key}: ${issue.msg ?? 'invalid value'}`
    })
  }
  if (typeof detail === 'string') return [detail]
  return [`The API returned an error (HTTP ${status}).`]
}

export class ApiError extends Error {
  messages: string[]

  constructor(messages: string[]) {
    super(messages.join('; '))
    this.messages = messages
  }
}

export async function predictRisk(
  patient: Record<string, number>
): Promise<PredictionResult> {
  let res: Response
  try {
    res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patient)
    })
  } catch {
    throw new ApiError([`Could not reach the API at ${API}. Is the backend running?`])
  }

  const body: unknown = await res.json().catch(() => null)
  if (!res.ok) throw new ApiError(errorMessages(res.status, body))
  return body as PredictionResult
}
