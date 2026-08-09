// data/caseScripts.ts
import type { CaseScript } from '@/types'

export const caseScripts: CaseScript[] = [
  {
    id: 1,
    title: '일정 질문 없음 - 즉시 END',
    steps: [{ role: 'questioner', message: '내일 산책 할거야' }],
  },
  {
    id: 2,
    title: '일정 질문 O, 긍정 응답, 날짜 O, 저장 승인',
    steps: [
      { role: 'questioner', message: '내일 산책 할래?' },
      { role: 'responder', message: '응 좋아' },
      { role: 'responder', confirm: true },
    ],
  },
  {
    id: 3,
    title: '일정 질문 O, 긍정 응답, 날짜 O, 저장 거부',
    steps: [
      { role: 'questioner', message: '내일 산책 할래?' },
      { role: 'responder', message: '응 좋아' },
      { role: 'responder', confirm: false },
    ],
  },
  {
    id: 4,
    title: '일정 질문 O, 긍정 응답, 날짜 X, 저장 승인',
    steps: [
      { role: 'questioner', message: '산책 할래?' },
      { role: 'responder', message: '응 좋아' },
      { role: 'responder', confirm: true },
    ],
  },
  {
    id: 5,
    title: '일정 질문 O, 긍정 응답, 날짜 X, 저장 거부',
    steps: [
      { role: 'questioner', message: '산책 할래?' },
      { role: 'responder', message: '응 좋아' },
      { role: 'responder', confirm: false },
    ],
  },
  {
    id: 6,
    title: '일정 질문 O, 애매한 응답(긍정 추론), 날짜 O, 저장 승인',
    steps: [
      { role: 'questioner', message: '내일 산책 할래?' },
      { role: 'responder', message: '비 안 오면 가자' },
      { role: 'responder', confirm: true },
    ],
  },
  {
    id: 7,
    title: '일정 질문 O, 애매한 응답(긍정 추론), 날짜 O, 저장 거부',
    steps: [
      { role: 'questioner', message: '내일 산책 할래?' },
      { role: 'responder', message: '비 안 오면 가자' },
      { role: 'responder', confirm: false },
    ],
  },
  {
    id: 8,
    title: '일정 질문 O, 부정 응답',
    steps: [
      { role: 'questioner', message: '내일 산책 할래?' },
      { role: 'responder', message: '아니' },
    ],
  },
  {
    id: 9,
    title: '일정 질문 O, 애매한 응답(부정 추론)',
    steps: [
      { role: 'questioner', message: '내일 산책 할래?' },
      { role: 'responder', message: '꼭 가야 돼?' },
    ],
  },
  {
    id: 10,
    title: '일정 질문 O, 애매한 응답(추론 불가)',
    steps: [
      { role: 'questioner', message: '내일 산책 할래?' },
      { role: 'responder', message: '배고파' },
    ],
  },
]
