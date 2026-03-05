# 🍽️ 식당 리뷰 챗봇 - RAG 엔진 코드 설명

네이버 검색 API와 LLM을 결합한 RAG(Retrieval-Augmented Generation) 기반 식당 추천 챗봇입니다.

---

## 📁 핵심 파일 구조

```
rag_engine.py   # 네이버 API 호출 + LLM 체인 관리
app.py          # Streamlit UI
config.yaml     # 모델 및 검색 설정
```

---

## 🔌 NaverSearchClient

네이버 검색 API 호출을 담당하는 클라이언트 클래스입니다.

### API 엔드포인트

| 목적 | 엔드포인트 |
|------|-----------|
| 식당 위치 및 명단 | `https://openapi.naver.com/v1/search/local` |
| 식당 리뷰 및 평판 | `https://openapi.naver.com/v1/search/blog` |

### 인증 방식

```python
self.headers = {
    "X-Naver-Client-Id": self.client_id,
    "X-Naver-Client-Secret": self.client_secret,
}
```

`requests.get()` 호출 시 `headers`를 함께 전달하면, 네이버 서버가 본문을 읽기 전에 헤더를 먼저 확인하여 요청의 유효성을 검사합니다. API 키는 반드시 `.env` 파일에 설정해야 합니다.

```
NAVER_CLIENT_ID=your_client_id
NAVER_CLIENT_SECRET=your_client_secret
```

---

### `search_local()` - 식당 기본 정보 검색

```python
def search_local(self, query: str, display: int = 5) -> list[dict]:
    params = {"query": query, "display": display, "sort": "comment"}
```

- `sort="comment"`: 리뷰(댓글)가 많은 순으로 정렬하여 검증된 맛집을 상단에 표시합니다.
- `display`: 가져올 결과 개수 (기본값 5개)

**반환 데이터 필드**

| 필드명 | 설명 | 예시 |
|--------|------|------|
| `title` | 식당 이름 | 맛있는 김치찌개 본점 |
| `category` | 업종 분류 | 음식점 > 한식 > 찌개,전골 |
| `address` | 지번 주소 | 서울특별시 강남구 역삼동... |
| `roadAddress` | 도로명 주소 | 서울특별시 강남구 테헤란로... |
| `telephone` | 전화번호 | 02-123-4567 |
| `mapx`, `mapy` | 지도 좌표 (KATECH) | 312345, 543210 |

---

### `search_blog()` - 식당 리뷰 검색

```python
def search_blog(self, query: str, display: int = 5) -> list[dict]:
    params = {"query": f"{query} 리뷰 맛집", "display": display, "sort": "sim"}
```

- 검색어 뒤에 자동으로 `" 리뷰 맛집"`을 붙여 더 정확한 리뷰를 탐색합니다.
- `sort="sim"`: 연관도(정확도) 높은 순으로 정렬합니다.
- `timeout=5`: 5초 안에 응답이 없으면 요청을 포기합니다.
- `raise_for_status()`: 네트워크 에러(404, 401 등) 발생 시 즉시 예외를 발생시킵니다.

---

## 🧠 RAGManager

### `__init__()` - 설정값 자동 로드

`config.yaml`에서 모델 이름, 검색 결과 개수 등을 불러옵니다. 코드 수정 없이 파일만 바꿔도 설정이 바로 반영됩니다.

> ⚠️ **주의**: 원본 코드의 `self.settings.get("naver_search, {}")` 는 오타입니다. `self.settings.get("naver_search", {})` 로 수정해야 정상 동작합니다.

---

### `_history_search_query()` - 문맥 기반 검색어 재작성

대화 흐름을 반영해 실제 검색에 사용할 키워드를 재구성합니다.

**왜 필요한가?**

```
사용자: "강남역 맛집 알려줘"   →  검색어: "강남역 맛집"      ✅
AI:     "A식당, B식당이 있습니다."
사용자: "거기 메뉴는 뭐야?"   →  검색어: "거기 메뉴"        ❌ (결과 없음)
                               →  재작성: "A식당 메뉴"       ✅
```

- 최근 **6개 메시지**만 참조하여 비용(토큰)과 혼란을 최소화합니다.
- `HumanMessage` / `AIMessage` 타입을 판별해 대화 기록을 깔끔한 텍스트로 변환합니다.

---

### `_fetch_context()` - 네이버 데이터 수집 및 전처리

LLM에 넘길 컨텍스트 문자열과 UI용 원본 데이터를 함께 반환합니다.

**데이터의 이중 구조**

| 반환값 | 용도 |
|--------|------|
| `context_str` | LLM에 전달하는 읽기용 텍스트 요약 |
| `raw_data` | 식당 주소, 링크 등 UI에 직접 사용하는 원본 데이터 |

**주요 전처리 로직**

```python
# HTML 태그 제거: <b>맛집</b> → 맛집
name = item.get("title", "").replace("<b>", "").replace("</b>", "")

# 주소 Fallback: 도로명 주소 우선, 없으면 지번 주소
address = item.get("roadAddress") or item.get("address", "")
```

**에러 핸들링**: `try-except`로 네이버 서버 장애 시에도 앱이 종료되지 않으며, 오류 내용을 컨텍스트에 포함시켜 LLM이 "정보를 가져올 수 없습니다"라고 안내하게 합니다.

**Hallucination 방지**: 검색 결과가 없을 경우 `"검색 결과가 없습니다"`를 컨텍스트에 포함시켜 LLM이 정보를 지어내지 않도록 유도합니다.

---

### `get_chain()` - RAG 체인 구성

#### 시스템 프롬프트 설계

LLM에게 전문가 페르소나를 부여하고 답변 형식을 강제합니다.

```
1. 추천 식당명
   a. 식당의 주소
   b. 식당의 영업 시간
   c. 식당의 메뉴
   d. 식당의 가격
   e. 분위기
```

#### MessagesPlaceholder - 대화 흐름 유지

```python
MessagesPlaceholder(variable_name="chat_history")
```

LLM은 기본적으로 이전 대화를 기억하지 못합니다. 이 한 줄이 지금까지의 대화 기록을 프롬프트 중간에 삽입하여 문맥을 유지시켜 줍니다.

#### 실행 흐름 (3단계)

```
1. 쿼리 재작성   →  "거기 어때?" → "강남역 A식당 어때?"
2. 데이터 수집   →  네이버 API 호출 → {context} 구성
3. 답변 생성     →  LLM에 context + chat_history + question 전달
```

#### 반환 구조

```python
return {
    "answer":      # 사용자에게 보여줄 답변 텍스트
    "restaurants": # 식당명 + 주소 리스트 (UI용)
    "debug": {
        "search_query": # 실제 사용된 검색어
        "context":      # LLM에 전달된 컨텍스트 전문
    }
}
```

---

## 📊 전체 데이터 흐름 요약

| 구분 | 정보 성격 | 데이터 예시 |
|------|----------|------------|
| Local API | 객관적 팩트 | 식당 이름, 위치, 전화번호, 카테고리 |
| Blog API | 주관적 평판 | 실제 맛, 친절도, 웨이팅 시간, 분위기 |

```
사용자 질문
    ↓
_history_search_query()  # 문맥 반영 검색어 재작성
    ↓
_fetch_context()         # 네이버 Local + Blog API 호출
    ↓
get_chain()              # LLM에 컨텍스트 + 히스토리 전달
    ↓
구조화된 답변 반환
```
