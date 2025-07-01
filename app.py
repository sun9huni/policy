import streamlit as st
import time
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# --- 1. 페이지 기본 설정 및 CSS ---
st.set_page_config(
    page_title="정책 큐레이터",
    page_icon="🤖",
    layout="wide"
)

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

# --- 2. 백엔드 로직: RAG 파이프라인 설정 ---

DATA_PATH = "./data"

def check_api_key():
    """API 키 유무를 확인하고, 없으면 명확한 에러 메시지를 표시 후 앱을 중지합니다."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
    except (KeyError, FileNotFoundError):
        st.error("오류: Google API 키가 설정되지 않았습니다.")
        st.info("Streamlit Cloud 배포 시, 앱 설정의 'Secrets'에 GOOGLE_API_KEY를 추가해야 합니다.")
        st.stop()

@st.cache_resource
def get_rag_components():
    """
    RAG 파이프라인의 핵심 구성 요소들을 설정하고 반환합니다.
    앱 시작 시 한 번만 실행되며, 결과는 캐시에 저장됩니다.
    """
    check_api_key()

    st.sidebar.info("문서를 로드하고 있습니다...")
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
    st.sidebar.info("데이터베이스를 구축하고 있습니다...")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Qdrant.from_documents(
        texts, embeddings, location=":memory:", collection_name="policy_documents",
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 10}) # [개선] 더 많은 후보군 확보를 위해 k값 증가
    st.sidebar.success("데이터베이스 구축 완료!")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    prompt_template = PromptTemplate.from_template(
        """당신은 대한민국 정부 정책 전문가입니다. 사용자의 질문에 대해 아래의 '문서 내용'을 바탕으로, 명확하고 친절하게 답변해주세요.
        답변은 항상 한국어로 작성해야 합니다. 문서 내용에 없는 정보는 답변에 포함하지 마세요.
        [문서 내용]
        {context}
        [사용자 질문]
        {question}
        [답변]
        """
    )
    
    # [개선] Re-ranker 모델 로드 추가 (안정적인 모델로 변경)
    st.sidebar.info("Re-ranker 모델을 로드하고 있습니다...")
    reranker_model = CrossEncoder('bongsoo/kpf-cross-encoder-v1')
    st.sidebar.success("모든 컴포넌트 로드 완료!")
    
    return retriever, llm, prompt_template, reranker_model

# --- 3. 애플리케이션 실행 ---
try:
    retriever, llm, prompt_template, reranker_model = get_rag_components()
except Exception as e:
    st.error(f"RAG 구성 요소를 설정하는 중 심각한 오류가 발생했습니다: {e}")
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
        with st.spinner("질문을 분석하고 관련 정보를 찾는 중입니다..."):
            try:
                # --- 1. 쿼리 확장(Query Expansion) 단계 ---
                expansion_prompt = PromptTemplate.from_template(
                    """당신은 한국 정부 정책 관련 검색어 생성 전문가입니다.
                    사용자의 질문을 보고, 관련성이 높은 검색어를 2개 생성해주세요.
                    공식 명칭, 동의어, 약어 등을 포함해야 합니다.
                    결과는 쉼표로 구분된 하나의 문자열로만 응답해주세요.
                    질문: {question}"""
                )
                query_expansion_chain = expansion_prompt | llm | StrOutputParser()
                expanded_queries_str = query_expansion_chain.invoke({"question": prompt})
                # 원본 질문과 확장된 질문을 모두 사용
                expanded_queries = [prompt] + [q.strip() for q in expanded_queries_str.split(',') if q.strip()]

                # --- 2. 확장된 검색(Expanded Retrieval) 단계 ---
                all_retrieved_docs = []
                for q in expanded_queries:
                    all_retrieved_docs.extend(retriever.invoke(q))
                
                # 중복된 문서 제거
                unique_docs = list({doc.page_content: doc for doc in all_retrieved_docs}.values())
                
                # --- 3. [개선] Re-ranking 단계 ---
                final_docs = []
                if unique_docs:
                    st.spinner("찾은 정보의 정확도를 높이는 중입니다...")
                    pairs = [[prompt, doc.page_content] for doc in unique_docs]
                    scores = reranker_model.predict(pairs)
                    
                    doc_scores = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)
                    # 재정렬된 문서 중 상위 3개만 선택
                    final_docs = [doc for score, doc in doc_scores[:3]]

                # --- 4. 최종 답변 생성 단계 ---
                if final_docs:
                    context = "\n\n".join(doc.page_content for doc in final_docs)
                    final_prompt = prompt_template.format(context=context, question=prompt)
                    response = llm.invoke(final_prompt).content
                else:
                    response = "죄송합니다. 관련 문서를 찾을 수 없어 답변을 생성할 수 없습니다. 더 많은 문서를 추가하면 답변 품질이 향상됩니다."
                
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant", "content": response, "sources": final_docs
                })
            except Exception as e:
                error_message = f"답변 생성 중 오류가 발생했습니다: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    st.rerun()
