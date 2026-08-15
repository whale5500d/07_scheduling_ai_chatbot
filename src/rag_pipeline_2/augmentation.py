from langchain_core.documents import Document

from rag_pipeline_2.schemas import ChatMessage

SYSTEM_PROMPT_TEMPLATE = "아래 참고자료를 바탕으로 질문에 답하시오.\n\n{context}"


def build_augmented_messages(documents: list[Document], user_query: str) -> list[ChatMessage]:
    # 1. 컨텍스트 구성: 검색된 문서를 "[참고자료 N] 제목 + 본문" 형식으로 나열
    context_blocks = []
    for index, document in enumerate(documents, start=1):
        title = document.metadata["title"]
        context_blocks.append(f"[참고자료 {index}] {title}\n{document.page_content}")
    context_text = "\n\n".join(context_blocks)

    # 2. 메시지 구성: 컨텍스트를 담은 system 메시지 + 질문을 담은 user 메시지
    system_message = ChatMessage(role="system", content=SYSTEM_PROMPT_TEMPLATE.format(context=context_text))
    user_message = ChatMessage(role="user", content=user_query)

    return [system_message, user_message]