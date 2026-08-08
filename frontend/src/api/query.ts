// api/query.ts
import type { QueryRequest, QueryResponse } from '../types'
import { API_BASE_URL } from './config'

export async function postQuery(request: QueryRequest): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error(`/query 요청 실패: ${response.status}`)
  }

  return response.json()
}
