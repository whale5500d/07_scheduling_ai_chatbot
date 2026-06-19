from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from schemas.schemas import GenerateRequest, GenerateResponse
from generate import generate as generate_text
from generate import _load_model_and_tokenizer

# ============================================================
# TODO: 학습 필요 - FastAPI Lifespan 패턴
# 
# - 기존 @app.on_event("startup") 방식보다 권장되는 현대적 패턴
# - asynccontextmanager를 사용하여 앱 시작/종료 시 리소스 관리
# - 특히 ML 모델, DB 커넥션 등 무거운 리소스를 한 번만 로딩할 때 유용
# - 참고: FastAPI 공식 문서 "Lifespan" 섹션
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 모델과 토크나이저 로딩
    print("[FastAPI] 모델과 토크나이저를 로딩합니다...")
    _load_model_and_tokenizer()
    print("[FastAPI] 모델 로딩 완료")
    yield
    # 서버 종료 시 (필요한 cleanup이 있으면 여기에 작성)

app = FastAPI(
    title="한국어 챗봇 API",
    description="Transformer + BPE 기반 한국어 챗봇 (구조 연결 완료 단계)",
    version="0.2.0",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {
        "message": "Transformer + BPE 기반 챗봇 API가 동작 중입니다.",
        "stage": "real_model_connected"
    }

@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(request: GenerateRequest):
    if not request.prompt or len(request.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="prompt는 비어있을 수 없습니다.")
    
    try:
        # generate 함수 호출 (max_new_tokens로 전달)
        result = generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_length,   # ← 여기 주의
            stop_sequences=None
        )
        return GenerateResponse(
            generated_text=result,
            prompt=request.prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"생성 중 오류가 발생했습니다: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "stage": "real_model_connected",
        "message": "TransformerLanguageModel + BPE Tokenizer가 연결된 상태입니다."
    }