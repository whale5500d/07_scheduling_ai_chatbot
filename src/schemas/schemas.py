from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str                    # 사용자가 보내는 문장
    max_length: int = 50           # 최대 생성 길이 (기본값)

class GenerateResponse(BaseModel):
    generated_text: str            # 생성된 전체 문장
    prompt: str                    # 사용자가 보낸 원래 문장