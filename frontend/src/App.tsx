import './App.css'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { CaseListPage } from './pages/CaseListPage'
import { CaseChatPage } from './pages/CaseChatPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<CaseListPage />} />
        <Route path="/case/:id" element={<CaseChatPage />} />
      </Routes>
    </BrowserRouter>
  )
}
