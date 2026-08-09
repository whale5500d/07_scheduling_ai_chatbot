export type ChatStep =
  | { role: 'questioner'; message: string }
  | { role: 'responder'; message: string }
  | { role: 'responder'; confirm: boolean }

export type CaseScript = {
  id: number
  title: string
  steps: ChatStep[]
}

export type ChatBubbleData = {
  role: 'questioner' | 'responder'
  text: string
}

export type QueryRequest = {
  thread_id: string
  message?: string
  confirm?: boolean
}

export type QueryResponse = {
  is_interrupted: boolean
  is_finished: boolean
  date: string | null
}
