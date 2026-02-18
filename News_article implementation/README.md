# LangChain & HuggingFace를 활용한 RAG 챗봇

---

## 1. LangChain이란?

LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션 개발을 단순화하기 위해 만들어진 오픈소스 프레임워크입니다. 직접 API를 호출하고 프롬프트, 검색기, 출력 파서 등을 수동으로 연결하는 대신, LangChain은 문서 로더, 텍스트 분할기, 벡터 저장소, 프롬프트 템플릿, 체인 등 표준화된 컴포넌트들을 제공하여 이를 손쉽게 조합해 엔드-투-엔드 파이프라인을 구성할 수 있게 해줍니다.

LangChain의 핵심 강점은 **컴포저빌리티(composability)** 입니다. 각 컴포넌트가 일관된 인터페이스를 따르기 때문에, 임베딩 모델이나 벡터 저장소, LLM을 교체하더라도 나머지 코드를 수정할 필요가 없습니다. 이러한 특성 덕분에 LangChain은 RAG 시스템, 에이전트, 챗봇 등 다양한 LLM 기반 애플리케이션을 구축하는 데 표준적으로 활용되는 프레임워크입니다.

---

## 2. HuggingFace란?

HuggingFace는 머신러닝 모델을 중심으로 한 플랫폼이자 오픈소스 생태계입니다. LLM, 임베딩 모델, 이미지 모델 등 수만 개의 사전학습된 모델을 호스팅하며, 누구나 다운로드하거나 API를 통해 접근할 수 있습니다.

이 프로젝트에서 HuggingFace는 두 가지 역할을 담당합니다:
- **임베딩:** `all-MiniLM-L6-v2` 문장 변환(sentence-transformer) 모델이 텍스트를 의미 기반 검색을 위한 수치 벡터로 변환합니다.
- **LLM 추론:** `google/gemma-2-9b-it` 모델을 HuggingFace Inference Endpoints를 통해 서빙하여, 로컬 GPU 없이도 강력한 오픈소스 LLM을 사용할 수 있습니다.

---

## 3. RAG 챗봇이란?

### 3-1. RAG가 왜 필요한가?

일반적인 LLM은 특정 시점까지의 고정된 데이터셋으로 학습됩니다. 이로 인해 두 가지 근본적인 문제가 발생합니다:

- **지식의 한계:** 학습 데이터 이후에 발생한 사건, 기사, 문서에 대해 모델은 알 수 없습니다.
- **환각(Hallucination):** 모델이 모르는 내용을 질문받으면, 그럴듯하지만 잘못된 답변을 만들어내는 경향이 있습니다.

**RAG(Retrieval-Augmented Generation, 검색 증강 생성)** 는 질문이 들어올 때마다 외부의 최신 지식 소스에서 관련 정보를 검색해 모델에게 제공함으로써 이 두 가지 문제를 해결합니다. 모델은 학습 중에 기억한 내용에만 의존하는 대신, 실제 출처 문서를 참고하여 답변하도록 지시받습니다. 그 결과 답변의 정확성, 근거, 검증 가능성이 크게 향상됩니다.

### 3-2. RAG 챗봇을 만들기 위해 LangChain과 HuggingFace가 왜 필요한가?

RAG 파이프라인을 처음부터 직접 구현하려면 여러 복잡한 단계가 필요합니다: 문서 수집 및 파싱, 텍스트 청킹, 임베딩 생성, 벡터 저장 및 검색, 프롬프트 구성, LLM 호출 등. 프레임워크 없이 이를 구현하면 방대한 양의 반복 코드와 연결 코드가 필요합니다.

**LangChain**은 이 모든 컴포넌트를 기본으로 제공하며, 무엇보다 이들을 자유롭게 조합할 수 있습니다. 검색 → 프롬프트 → 생성으로 이어지는 전체 파이프라인을 단 몇 줄의 코드로 정의할 수 있습니다.

**HuggingFace**는 가장 핵심적인 두 단계, 즉 임베딩(텍스트를 검색 가능한 벡터로 변환)과 생성(최종 답변 생성)을 담당하는 모델을 제공합니다. HuggingFace가 API를 통해 오픈소스 모델을 호스팅하기 때문에, Gemma와 같은 고성능 모델을 별도의 추론 인프라 없이도 사용할 수 있습니다.

정리하면, LangChain이 **오케스트레이션(파이프라인 구성)** 을 담당하고, HuggingFace가 **모델 백본** 을 제공합니다.

### 3-3. RAG 챗봇의 전체 구조 및 흐름

RAG 파이프라인은 두 단계로 구성됩니다: 지식 베이스를 준비하는 **인덱싱 단계** (최초 1회 실행)와 사용자 질문에 답하는 **쿼리 단계** (매 질문마다 실행)입니다.

```
===============================================
  인덱싱 단계  (지식 베이스 준비)
===============================================

  [소스 문서 / 웹 페이지]
              |
              v
      [문서 로더 (Document Loader)]
              |
              v
      [텍스트 분할기 (Text Splitter)]  -->  텍스트 청크
              |
              v
    [임베딩 모델 (Embedding Model)]  -->  벡터 표현
              |
              v
       [벡터 저장소 (Vector Store)]  -->  인덱싱 및 저장


===============================================
  쿼리 단계  (사용자 질문에 답변)
===============================================

  [사용자 질문]
        |
        +------------------------------+
        v                              v
  [임베딩 모델]                  (원본 질문 유지)
        |
        v
  [벡터 저장소]  -->  상위 k개의 관련 청크 검색
        |
        v
  [프롬프트 템플릿]
    "아래 문맥만을 사용하여 질문에 답하세요.
     문맥: {검색된 청크}
     질문: {사용자 질문}"
        |
        v
       [LLM]
        |
        v
  [최종 답변]
```

---

## 4. 코드 구현 방법

### Step 1 — 문서 로드

LangChain의 `WebBaseLoader`를 사용하여 네이버 뉴스 기사를 웹에서 불러옵니다. BeautifulSoup을 활용해 기사 본문과 제목에 해당하는 HTML `<div>` 요소만 파싱하고, 내비게이션이나 광고 등 불필요한 내용은 제거합니다.

```python
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
)
docs = loader.load()
```

### Step 2 — 청크 분할

문맥이 경계에서 손실되지 않도록 텍스트를 겹치는 청크로 분할합니다. 청크 크기는 1000자, 겹침(overlap)은 50자로 설정합니다.

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
```

### Step 3 — 임베딩 및 인덱싱

HuggingFace의 `all-MiniLM-L6-v2` 모델로 각 청크를 임베딩합니다. 이 모델은 가볍지만 의미 유사도 검색에 효과적인 문장 변환 모델입니다. 생성된 벡터는 FAISS 인메모리 벡터 저장소에 저장되고, 이를 기반으로 검색기(retriever)를 생성합니다.

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()
```

### Step 4 — LLM 설정

`google/gemma-2-9b-it` 모델을 HuggingFace Inference Endpoints를 통해 불러옵니다. API 토큰은 보안을 위해 `.env` 파일에서 읽어오며, LangChain의 채팅 모델 인터페이스에 맞게 `ChatHuggingFace`로 래핑합니다.

```python
llm_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=hf_token
)
chat_llm = ChatHuggingFace(llm=llm_endpoint)
```

### Step 5 — 프롬프트 정의

검색된 문맥에서만 답변하도록 모델을 엄격하게 제한하고, 답변은 반드시 한국어로 작성하도록 프롬프트를 설계합니다. 이를 통해 환각을 최소화합니다.

```python
template = """당신은 질문-답변을 도와주는 AI 어시스턴트입니다.
아래의 제공된 문맥(Context)을 활용해서만 질문에 답하세요.
답을 모른다면 모른다고 말하고, 직접적인 답이 문맥에 없다면 문맥을 바탕으로 추론하지 마세요.
답변은 반드시 한국어로 작성하세요.

#Context:
{context}

#Question:
{question}

#Answer:"""

prompt = ChatPromptTemplate.from_template(template)
```

### Step 6 — RAG 체인 구성 및 실행

전체 파이프라인을 LangChain LCEL 체인으로 구성합니다. 검색기가 자동으로 관련 청크를 가져오고, 이를 질문과 함께 프롬프트에 주입한 뒤 LLM을 거쳐 최종적으로 문자열로 파싱됩니다.

```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat_llm
    | StrOutputParser()
)

question = "부영그룹은 출산 직원에게 얼마의 지원을 제공하나요?"
response = rag_chain.invoke(question)
# → "부영그룹은 출산 직원에게 1억원을 지원합니다."
```
