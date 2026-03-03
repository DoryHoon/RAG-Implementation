from rag_engine import RAGManager
import streamlit as st

# 1. 엔진 가동
@st.cache_resource
def get_rag():
    # RAGManager 클래스의 인스턴스를 생성하여 반환
    return RAGManager()

rag = get_rag()
chain = rag.get_chain()

st.set_page_config(page_title="삼성전자 실적 도우미", layout="wide")
st.title("🤖 삼성전자 실적 RAG 챗봇")

# 2. 채팅 메시지 세션 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 사용자 입력 및 답변 생성
if prompt := st.chat_input("삼성전자 실적에 대해 궁금한 점을 물어보세요!"):
    # 사용자 질문 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 엔진을 통해 답변 생성
        response = chain.invoke(prompt)
        st.markdown(response)
        # 대화 기록 저장
        st.session_state.messages.append({"role": "assistant", "content": response})