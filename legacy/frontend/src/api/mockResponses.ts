// api/mockResponses.ts
import type { QueryResponse } from '@/types'

// 케이스 id별 응답 시퀀스. 실제 test.py 검증 결과(케이스 1~10)를 그대로 반영함.
export const mockResponseSequences: Record<number, QueryResponse[]> = {
  1: [{ is_interrupted: false, is_finished: true, date: null }],
  2: [
    { is_interrupted: false, is_finished: false, date: null },
    { is_interrupted: true, is_finished: false, date: '2026-08-08' },
    { is_interrupted: false, is_finished: true, date: null },
  ],
  3: [
    { is_interrupted: false, is_finished: false, date: null },
    { is_interrupted: true, is_finished: false, date: '2026-08-08' },
    { is_interrupted: false, is_finished: true, date: null },
  ],
  4: [
    { is_interrupted: false, is_finished: false, date: null },
    { is_interrupted: true, is_finished: false, date: null },
    { is_interrupted: false, is_finished: true, date: null },
  ],
  5: [
    { is_interrupted: false, is_finished: false, date: null },
    { is_interrupted: true, is_finished: false, date: null },
    { is_interrupted: false, is_finished: true, date: null },
  ],
  6: [
    { is_interrupted: false, is_finished: false, date: null },
    { is_interrupted: true, is_finished: false, date: '2026-08-08' },
    { is_interrupted: false, is_finished: true, date: null },
  ],
  7: [
    { is_interrupted: false, is_finished: false, date: null },
    { is_interrupted: true, is_finished: false, date: '2026-08-08' },
    { is_interrupted: false, is_finished: true, date: null },
  ],
  8: [
    { is_interrupted: false, is_finished: false, date: null },
    { is_interrupted: false, is_finished: true, date: null },
  ],
  9: [
    { is_interrupted: false, is_finished: false, date: null },
    { is_interrupted: false, is_finished: true, date: null },
  ],
  10: [
    { is_interrupted: false, is_finished: false, date: null },
    { is_interrupted: false, is_finished: true, date: null },
  ],
}
