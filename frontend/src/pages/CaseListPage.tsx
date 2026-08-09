// pages/CaseListPage.tsx
import { useNavigate } from 'react-router-dom'
import { Card, CardHeader, CardTitle, CardFooter } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { caseScripts } from '@/data/caseScripts'

export function CaseListPage() {
  const navigate = useNavigate()

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-6">
      {caseScripts.map((caseScript) => (
        <Card key={caseScript.id}>
          <CardHeader>
            <CardTitle>
              케이스 {caseScript.id} - {caseScript.title}
            </CardTitle>
          </CardHeader>
          <CardFooter>
            <Button onClick={() => navigate(`/case/${caseScript.id}`)}>
              시작
            </Button>
          </CardFooter>
        </Card>
      ))}
    </div>
  )
}
