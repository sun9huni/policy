import streamlit as st
import time
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. 페이지 기본 설정 및 CSS ---
st.set_page_config(
    page_title="정책 큐레이터",
    page_icon="🤖",
    layout="wide"
)

# [개선] 세련된 UI를 위한 커스텀 CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'IBM Plex Sans KR', sans-serif; }
.stApp { background-color: #F0F2F6; }
[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }
.stButton > button {
    border: 1px solid #E0E0E0; border-radius: 10px; color: #31333F;
    background-color: #FFFFFF; transition: all 0.2s ease-in-out;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stButton > button:hover {
    border-color: #0068C9; color: #0068C9;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
div[data-testid="stSidebar"] .stButton > button {
    background-color: #0068C9; color: white; border: none;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #0055A3; color: white;
}
</style>
""", unsafe_allow_html=True)

# --- 2. 백엔드 로직: RAG 파이프라인 설정 (오류 방지 강화) ---

# 데이터베이스 경로 설정
DB_PATH = "./vector_db"
DATA_PATH = "./data"

def check_api_key():
    """API 키 유무를 확인하고, 없으면 명확한 에러 메시지를 표시 후 앱을 중지합니다."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
    except (KeyError, FileNotFoundError):
        st.error("오류: Google API 키가 설정되지 않았습니다.")
        st.info("Streamlit Cloud 배포 시, 앱 설정의 'Secrets'에 GOOGLE_API_KEY를 추가해야 합니다.")
        st.stop() # [개선] API 키가 없으면 앱 실행을 즉시 중단하여 혼란 방지

def create_and_load_db(embedding_function):
    """데이터베이스를 생성하고 로드하는 함수."""
    st.sidebar.info("새로운 데이터베이스를 구축하고 있습니다. 잠시만 기다려주세요...")
    
    # [개선] 데이터 폴더 및 파일 유무 명시적 확인
    if not os.path.exists(DATA_PATH) or not any(f.endswith('.pdf') for f in os.listdir(DATA_PATH)):
        st.error(f"오류: '{DATA_PATH}' 폴더를 찾을 수 없거나 폴더 내에 PDF 파일이 없습니다.")
        st.info("배포 시, GitHub 리포지토리에 'data' 폴더와 그 안에 PDF 파일을 포함해야 합니다.")
        st.stop()

    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents=texts, embedding=embedding_function, persist_directory=DB_PATH)
    st.sidebar.success("데이터베이스 구축 완료!")
    return vectorstore

@st.cache_resource
def get_rag_pipeline():
    """
    RAG 파이프라인 전체를 설정하고 반환하는 함수.
    오류 발생 가능성이 있는 부분들을 명확히 분리하고 확인합니다.
    """
    # 1. API 키 확인 (가장 먼저)
    check_api_key()

    # 2. 임베딩 모델 초기화
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 3. 벡터 데이터베이스 로드 또는 생성
    if os.path.exists(DB_PATH):
        st.sidebar.info("기존 데이터베이스를 로드했습니다.")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        vectorstore = create_and_load_db(embeddings)

    retriever = vectorstore.as_retriever()

    # 4. LLM 및 프롬프트 설정
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    template = """당신은 대한민국 정부 정책 전문가입니다. 사용자의 질문에 대해 아래의 '문서 내용'을 바탕으로, 명확하고 친절하게 답변해주세요.
    답변은 항상 한국어로 작성해야 합니다. 문서 내용에 없는 정보는 답변에 포함하지 마세요.
    [문서 내용]
    {context}
    [사용자 질문]
    {question}
    [답변]
    """
    prompt_template = PromptTemplate.from_template(template)

    # 5. RAG 체인 구성 (간소화된 표현)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# --- 3. 애플리케이션 실행 ---

# RAG 파이프라인 로드
try:
    rag_chain, retriever = get_rag_pipeline()
except Exception as e:
    st.error(f"RAG 파이프라인을 설정하는 중 심각한 오류가 발생했습니다: {e}")
    st.stop()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# 좌측 사이드바 UI
with st.sidebar:
    st.header("🎯 나의 맞춤 조건 설정")
    st.markdown("AI가 더 정확한 정책을 추천하도록 정보를 입력해주세요.")
    age = st.number_input("나이(만)", min_value=18, max_value=100, value=st.session_state.profile.get("age", 25))
    interests = st.multiselect(
        "주요 관심 분야",
        ['주거 지원', '일자리/창업', '금융/자산 형성', '생활/복지'],
        default=st.session_state.profile.get("interests", [])
    )
    if st.button("✅ 조건 저장 및 반영", type="primary", use_container_width=True):
        st.session_state.profile = { "age": age, "interests": interests }
        st.success("맞춤 조건이 저장되었습니다!")
        time.sleep(1)
        st.rerun()

# 메인 화면 UI
st.title("🤖 정책 큐레이터")
st.caption("AI 기반 맞춤형 정책 탐색기")

# 추천 질문
recommended_questions_db = {
    "주거 지원": ["청년 월세 지원 자격 알려줘", "신혼부부 전세 대출 조건"],
    "일자리/창업": ["창업 지원금 종류 알려줘", "내일채움공제 신청 방법"],
}
st.markdown("##### 무엇을 도와드릴까요?")
profile_interests = st.session_state.get("profile", {}).get("interests", [])
questions_to_show = recommended_questions_db.get(profile_interests[0], ["청년 월세 지원 자격 알려줘", "내일채움공제 신청 방법"]) if profile_interests else ["청년 월세 지원 자격 알려줘", "내일채움공제 신청 방법"]
cols = st.columns(len(questions_to_show))
for i, question in enumerate(questions_to_show):
    if cols[i].button(question, use_container_width=True, key=f"rec_q_{i}"):
        st.session_state.selected_question = question

# 채팅 인터페이스 로직

# 동적 온보딩 메시지
if not st.session_state.messages:
    profile = st.session_state.get("profile", {})
    welcome_message = f"안녕하세요! {profile['age']}세, '{profile['interests'][0]}' 분야에 관심이 있으시군요." if profile.get("age") and profile.get("interests") else "안녕하세요! 어떤 정책이 궁금하신가요?"
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# 이전 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 근거 자료 확인하기"):
                for source in message["sources"]:
                    st.info(f"출처: {source.metadata.get('source', 'N/A')} (페이지: {source.metadata.get('page', 'N/A')})")
                    st.write(source.page_content)

# 사용자 입력 처리
prompt = st.chat_input("궁금한 정책에 대해 질문해보세요.")
if st.session_state.selected_question:
    prompt = st.session_state.selected_question
    st.session_state.selected_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중입니다..."):
            try:
                response = rag_chain.invoke(prompt)
                sources = retriever.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant", "content": response, "sources": sources
                })
            except Exception as e:
                error_message = f"답변 생성 중 오류가 발생했습니다: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    st.rerun()
