// api/query.ts
import type { QueryRequest, QueryResponse } from '@/types'
import { mockResponseSequences } from '@/api/mockResponses'
import { API_BASE_URL } from '@/api/config'

const mockCallCounts = new Map<string, number>()

export async function postQuery(request: QueryRequest): Promise<QueryResponse> {
  return mockPostQuery(request)

  // const response = await fetch(`${API_BASE_URL}/query`, {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify(request),
  // })

  // if (!response.ok) {
  //   throw new Error(`/query 요청 실패: ${response.status}`)
  // }

  // return response.json()
}

function mockPostQuery(request: QueryRequest): Promise<QueryResponse> {
  return new Promise((resolve) => {
    setTimeout(() => {
      const caseId = Number(request.thread_id.split('-')[1])
      const sequence = mockResponseSequences[caseId]
      const callCount = mockCallCounts.get(request.thread_id) ?? 0
      mockCallCounts.set(request.thread_id, callCount + 1)
      resolve(sequence[callCount])
    }, 300)
  })
}
