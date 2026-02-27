# Multi-Query RAG — 삼성전자 4Q25 실적 분석

단일 쿼리 RAG의 벡터 유사도 한계를 극복하기 위해 Multi-Query 기법을 적용한 실험 프로젝트. 삼성전자 2025년 4분기 실적 발표 자료(PDF 2종)를 소스 문서로 사용했으며, 전 과정을 HuggingFace 오픈소스 모델로 구성했습니다.

---

## 구현 방식

### 기술 스택

| 구성 요소 | 사용 모델 / 라이브러리 |
|---|---|
| 문서 로더 | `PyPDFLoader` (LangChain) |
| 텍스트 분할 | `RecursiveCharacterTextSplitter` |
| 임베딩 | `BAAI/bge-m3` (HuggingFace, GPU) |
| 벡터 스토어 | `FAISS` |
| LLM | `google/gemma-2-9b-it` (HuggingFace Inference API) |
| 파이프라인 | LangChain LCEL |
| 관측 | LangSmith |

### 소스 문서

- `삼성_2025Q4_conference_eng_presentation.pdf` — 실적 발표 슬라이드 (15페이지)
- `삼성_2025Q4_script_eng_AudioScript.pdf` — 실적 발표 컨퍼런스콜 스크립트 (34페이지)
- 총 49페이지 → `chunk_size=1000`, `chunk_overlap=120` 기준 96개 청크로 분할

### 파이프라인 흐름

```
사용자 질문 (1개)
      ↓
[Step 1] Multi-Query 생성
  QUERY_PROMPT | gemma-2-9b-it | StrOutputParser | split("\n")
  → 동일한 질문을 5가지 다른 관점으로 재작성
      ↓
[Step 2] 병렬 검색
  retriever.map()
  → 5개의 질문 각각에 대해 FAISS 검색 (쿼리당 k=4)
  → 최대 20개 청크 반환
      ↓
[Step 3] 중복 제거 (Unique Union)
  get_unique_union()
  → dumps()로 Document를 string 변환 후 set()으로 중복 제거
  → loads()로 Document 객체 복원
  → 실제 검색 결과: unique 문서 18개
      ↓
[Step 4] 최종 답변 생성
  RAG_PROMPT | gemma-2-9b-it | StrOutputParser
  → unique 문서를 context로 사용해 최종 답변 생성
```

### 버그 수정 이력

초기 구현에서 `QUERY_PROMPT | retriever.map()`으로 연결 시 `TypeError: object of type 'ChatPromptValue' has no len()` 오류 발생. `QUERY_PROMPT` 단독으로는 `ChatPromptValue` 객체를 출력하는데, `retriever.map()`은 string 리스트를 입력으로 기대하기 때문. LLM과 파서를 포함한 `query_chain`을 먼저 거치도록 수정하여 해결.

```python
# 수정 전 (오류 발생)
retrieval_chain = QUERY_PROMPT | retriever.map() | get_unique_union

# 수정 후
retrieval_chain = query_chain | retriever.map() | get_unique_union
```

---

## 테스트 결과 및 문제점

### 실행 결과 (실제 출력)

```
질문: 삼성전자 2025 4분기 실적 발표에 대해서 알려줘

매출: 93.8조원 (전년 동기 대비 11% 증가)       ← 수치는 맞으나 YoY 오류
영업이익: 20.1조원 (전년 동기 대비 33% 증가)    ← 수치는 맞으나 YoY 오류
DS 매출: 44조원 (전년 동기 대비 33% 증가)       ← YoY 오류
환율 효과: 1.6조원 추가 영업이익                ← 정확
```

### 정확도 평가

| 항목 | 모델 답변 | 실제 수치 | 판정 |
|---|---|---|---|
| 4Q25 매출 | 93.8조 | 93.8조 | ✅ |
| 4Q25 영업이익 | 20.1조 | 20.1조 | ✅ |
| 매출 YoY | +11% | +24% (4Q 기준) | ❌ |
| 영업이익 YoY | +33% | +209% (6.5조→20.1조) | ❌ |
| 환율 효과 | 1.6조 | 1.6조 | ✅ |

핵심 수치 자체는 정확하게 가져왔으나, 전년 동기 대비 증가율(YoY %)을 지속적으로 틀리는 패턴이 확인됐다.

---

## 원인 분석

### 원인 1. PDF 파싱 과정에서 테이블 구조 붕괴

프레젠테이션 PDF에 포함된 재무 테이블은 `PyPDFLoader`를 통과하면서 구조가 무너진다. 실제 청크를 확인한 결과 아래와 같이 테이블 제목(섹션 헤더) 없이 숫자만 나열된 형태로 저장되는 것을 확인했다.

```
# 실제 청크 내용 (page 6)
KRW trillion 4Q24 3Q25 4Q25 QoQ YoY 2024 2025 YoY
Total 6.5 12.2 20.1 7.9↑ 13.6↑ 32.7 43.6 10.9↑
DX 2.3 3.5 1.3 ...
```

이 청크만으로는 `20.1`이 영업이익인지 매출인지, `11%`가 분기 YoY인지 연간 YoY인지 구분할 수 없다. 테이블 제목인 "Operating Profit"이 이전 청크에 잘려나갔기 때문이다. LLM은 라벨 없이 숫자만 받아서 임의로 매핑한다.

### 원인 2. HuggingFace 소형 모델의 한계

`gemma-2-9b-it`는 9B 파라미터 규모의 오픈소스 모델로, GPT-4o 계열 대형 모델 대비 수치 추론 및 표 해석 능력에 한계가 있다. 특히 불완전한 컨텍스트를 받았을 때 "정보가 부족하다"고 답하지 않고 그럴듯한 값을 생성하는 경향(hallucination)이 강하다. 임베딩 모델인 `BAAI/bge-m3` 자체의 검색 성능은 양호한 수준으로, 관련 청크는 정확히 가져왔다. 문제는 검색이 아닌 해석 단계에서 발생했다.

---

## 다음 프로젝트 계획

위 두 가지 원인을 해소하기 위해 다음 버전에서 아래와 같이 개선할 예정이다.

- **LLM 교체:** `gemma-2-9b-it` → `gpt-4o-mini` — 수치 추론 및 불확실 상황에서의 응답 품질 개선
- **임베딩 교체:** `BAAI/bge-m3` → `text-embedding-3-small` — GPU 의존성 제거 및 API 기반 운영
- **소스 문서 조정:** 프레젠테이션 PDF 제거, 오디오 스크립트만 사용 — 자연어 문장 형태로 숫자와 라벨이 같은 문장 안에 존재하여 파싱 오류 최소화. 또는 새로운 Document Loader 활용
- **벡터 스토어:** `FAISS` → `Chroma`

관련 구현은 `Multi-Query_RAG_OpenAI.ipynb` 참고.
