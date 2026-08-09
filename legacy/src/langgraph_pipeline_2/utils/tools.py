# tools.py
from langchain_core.tools import tool
from langgraph_pipeline_2.utils.indexing import build_vector_store

_store = build_vector_store()

@tool
def search_case_examples(query: str) -> str:
    """애매한 응답 판정을 위해 과거 유사 판정 사례를 검색합니다.
    응답이 긍정/부정 키워드로 명확히 분류되지 않을 때 사용합니다."""
    results = _store.similarity_search(query)

    if len(results) > 0:
        return "\n\n".join(doc.page_content for doc in results)
    return "관련 사례를 찾을 수 없습니다 (스텁)"

def get_tools():
    """Tools를 list로 구성하기"""
    return [search_case_examples]