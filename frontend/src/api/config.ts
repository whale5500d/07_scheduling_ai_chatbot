// api/config.ts

// 현재: EC2 공인 IP + 서버 포트로 직접 호출 (최소 의존성 원칙).
// 확장: 실제 배포/도메인 연결 단계에서 Nginx 리버스 프록시로 전환 검토.
// 전환 시 클라이언트 컨테이너가 /api 요청을 서버 컨테이너로 프록시하고,
// 브라우저는 클라이언트 포트 하나만 호출하도록 구성 예정.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL
export const USE_MOCK_DATA = import.meta.env.VITE_USE_MOCK_DATA === 'true'
