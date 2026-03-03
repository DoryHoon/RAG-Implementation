import yaml
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 로드: API 및 폴더 경로
load_dotenv()

class RAGManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            self.settings = config['rag_settings']

        # config에서 설정값 불러오기
        self.file_paths = [self.settings['file_path']] # 리스트 형태
        self.model_name = self.settings['model_name']
        self.chunk_size = self.settings['chunk_size']
        self.chunk_overlap = self.settings['chunk_overlap']

        self.vectorstore = None
        self.retriever = None
        
        # 엔진 초기화 실행
        self._setup_engine()

    def _setup_engine(self):
        """문서 로드부터 리트리버 설정까지 한 번에 처리"""
        # 1. 문서 로드
        docs = []
        for path in self.file_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        # 2. 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        splits = text_splitter.split_documents(docs)

        # 3. 임베딩 및 벡터 스토어 생성
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings
        )
        
        # 4. 리트리버 설정
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    def get_chain(self):
        """질문에 답하는 최종 RAG 체인 반환"""
        llm = ChatOpenAI(model=self.model_name, temperature=0)
        
        # 프롬프트 구성
        template = """You are an expert AI assistant specializing in Samsung Electronics earnings reports.
        Answer the question based on the provided context.
        If the question is not related to "Samsung Electronics Earnings Call", 
        provide an answer at your base capability, but mention that it is not directly from the uploaded documents.
        Always include specific numbers and financial figures when available.

        #Context:
        {context}

        #Question:
        {question}

        #Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        # 체인 생성
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain