// pages/CaseChatPage.tsx
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import { caseScripts } from '@/data/caseScripts'
import { postQuery } from '@/api/query'
import type { ChatBubbleData } from '@/types'

export function CaseChatPage() {
  const { id } = useParams()
  const caseScript = caseScripts.find((c) => c.id === Number(id))

  const [threadId, setThreadId] = useState('')
  const [stepIndex, setStepIndex] = useState(0)
  const [bubbles, setBubbles] = useState<ChatBubbleData[]>([])
  const [isInterrupted, setIsInterrupted] = useState(false)
  const [interruptDate, setInterruptDate] = useState<string | null>(null)
  const [isFinished, setIsFinished] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (id) {
      setThreadId(`case-${id}-${crypto.randomUUID()}`)
    }
  }, [id])

  if (!caseScript) {
    return <div>존재하지 않는 케이스입니다.</div>
  }

  const currentStep = caseScript.steps[stepIndex]

  async function handleNext() {
    if (!currentStep || isLoading) return
    setIsLoading(true)

    if ('message' in currentStep) {
      setBubbles((prev) => [
        ...prev,
        { role: currentStep.role, text: currentStep.message },
      ])
    }

    try {
      const response = await postQuery(
        'message' in currentStep
          ? { thread_id: threadId, message: currentStep.message }
          : { thread_id: threadId, confirm: currentStep.confirm },
      )

      if (response.is_interrupted) {
        setIsInterrupted(true)
        setInterruptDate(response.date)
      } else if (response.is_finished) {
        setIsFinished(true)
      }
      setStepIndex((prev) => prev + 1)
    } finally {
      setIsLoading(false)
    }
  }

  async function handleConfirm() {
    if (!currentStep || !('confirm' in currentStep)) return
    setIsLoading(true)
    try {
      const response = await postQuery({
        thread_id: threadId,
        confirm: currentStep.confirm,
      })
      setIsInterrupted(false)
      setInterruptDate(null)
      if (response.is_finished) {
        setIsFinished(true)
      }
      setStepIndex((prev) => prev + 1)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-screen p-6">
      <h1 className="text-lg font-semibold mb-4">
        케이스 {caseScript.id} - {caseScript.title}
      </h1>

      <div className="flex-1 overflow-y-auto space-y-2">
        {bubbles.map((bubble, i) => (
          <div
            key={i}
            className={`flex ${
              bubble.role === 'questioner' ? 'justify-start' : 'justify-end'
            }`}
          >
            <div className="rounded-lg bg-muted px-4 py-2 max-w-xs">
              {bubble.text}
            </div>
          </div>
        ))}
      </div>

      <Button
        onClick={handleNext}
        disabled={isLoading || isFinished || isInterrupted}
      >
        {isFinished ? '완료' : '다음'}
      </Button>

      <Dialog open={isInterrupted}>
        <DialogContent showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>저장 확인</DialogTitle>
          </DialogHeader>
          <p>{interruptDate}에 저장하시겠습니까?</p>
          <DialogFooter>
            <Button
              onClick={handleConfirm}
              disabled={
                isLoading ||
                !currentStep ||
                !('confirm' in currentStep) ||
                currentStep.confirm !== true
              }
            >
              승인
            </Button>
            <Button
              onClick={handleConfirm}
              disabled={
                isLoading ||
                !currentStep ||
                !('confirm' in currentStep) ||
                currentStep.confirm !== false
              }
              variant="outline"
            >
              거부
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
