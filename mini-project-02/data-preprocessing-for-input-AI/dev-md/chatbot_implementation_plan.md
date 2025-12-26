# 챗봇 구축 전체 프로세스 보고서

> **버전**: 1.0
> **목표**: 텍스트 임베딩 기반 QA 챗봇 시스템 구축
> **Frontend**: Next.js
> **Backend**: Python Django + Django REST Framework

---

## 1. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1: 데이터 전처리                              │
│                                                                               │
│   Raw Text → Cleaning → Tokenizing → Embedding → Vector DB 저장              │
│                                                                               │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 2: 챗봇 모델 구축                             │
│                                                                               │
│   Query → Embedding → Vector Search → Context Retrieval → LLM Response       │
│                                                                               │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 3: 백엔드 (Django)                           │
│                                                                               │
│   REST API: /api/chat, /api/history, /api/feedback                           │
│   WebSocket: 실시간 채팅                                                      │
│                                                                               │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 4: 프론트엔드 (Next.js)                       │
│                                                                               │
│   Chat UI, Message History, Streaming Response                               │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. PHASE 1: 데이터 전처리 파이프라인

### 2.1 구현 단계

| 단계 | 작업 | 도구/라이브러리 |
|------|------|----------------|
| **정제** | HTML 제거, URL/이메일 처리, 공백 정규화 | `re`, `html` |
| **토크나이징** | BPE/WordPiece 서브워드 분리 | `transformers.AutoTokenizer` |
| **인코딩** | BERT/RoBERTa 기반 벡터 변환 | `transformers.AutoModel` |
| **풀링** | Mean Pooling으로 문장 벡터화 | PyTorch |
| **저장** | Vector DB 저장 | FAISS, ChromaDB, Pinecone |

### 2.2 디렉토리 구조

```
data-preprocessing/
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py         # TextCleaner
│   │   └── tokenizer.py       # TextTokenizer
│   ├── encoder/
│   │   ├── __init__.py
│   │   ├── base.py            # TextEncoder ABC
│   │   ├── bert.py            # BERTEncoder
│   │   └── sentence_transformer.py
│   ├── pooling/
│   │   ├── __init__.py
│   │   └── pooler.py          # TextPooler
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # TextDataset
│   │   └── exporter.py        # TextDatasetExporter
│   ├── pipeline.py            # TextEmbeddingPipeline
│   ├── quality.py             # TextQualityAssessor
│   └── utils.py
├── data/
│   ├── raw/                   # 원본 데이터
│   ├── processed/             # 정제된 데이터
│   └── embeddings/            # 임베딩 벡터 (.h5, .npy)
├── scripts/
│   ├── preprocess.py
│   ├── generate_embeddings.py
│   └── evaluate.py
├── configs/
│   ├── base.yaml
│   └── encoder/
│       ├── bert.yaml
│       └── roberta.yaml
├── requirements.txt
└── README.md
```

### 2.3 핵심 클래스 요약

| 클래스 | 파일 | 역할 |
|--------|------|------|
| `TextCleaner` | `preprocessing/cleaner.py` | HTML, URL, 특수문자 정제 |
| `TextTokenizer` | `preprocessing/tokenizer.py` | 토큰화 및 패딩 |
| `TextPreprocessor` | `preprocessing/` | 정제 + 토큰화 통합 |
| `BERTEncoder` | `encoder/bert.py` | BERT 기반 인코딩 |
| `TextPooler` | `pooling/pooler.py` | CLS/Mean/Max Pooling |
| `OutputProjection` | `projection/output.py` | 차원 투영 |
| `TextEmbeddingPipeline` | `pipeline.py` | E2E 파이프라인 |
| `TextQualityAssessor` | `quality.py` | 품질 필터링 |

---

## 3. PHASE 2: 챗봇 모델 구축 (RAG 기반)

### 3.1 RAG 시스템 구조

```
User Query
    │
    ▼
┌──────────────────┐
│  Query Embedding │ ← 동일한 텍스트 인코더 사용
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Vector Search   │ ← FAISS/ChromaDB에서 유사 문서 검색
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Context Builder │ ← Top-K 문서를 프롬프트에 삽입
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM Response    │ ← OpenAI GPT / Claude API
└──────────────────┘
```

### 3.2 핵심 컴포넌트

| 컴포넌트 | 역할 | 권장 도구 |
|----------|------|----------|
| **Embedding Model** | 쿼리/문서 벡터화 | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | 벡터 저장/검색 | FAISS (로컬), ChromaDB, Pinecone |
| **LLM** | 응답 생성 | OpenAI GPT-4, Claude API |
| **Orchestrator** | RAG 파이프라인 통합 | LangChain, LlamaIndex |

### 3.3 RAG 엔진 디렉토리 구조

```
chatbot-engine/
├── src/
│   ├── embedder/
│   │   ├── __init__.py
│   │   └── text_embedder.py   # 쿼리 임베딩
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── vector_store.py    # FAISS/ChromaDB 래퍼
│   │   └── retriever.py       # Top-K 검색
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── llm_client.py      # OpenAI/Claude API
│   │   └── prompt_builder.py  # 프롬프트 템플릿
│   └── rag_pipeline.py        # RAG 통합 파이프라인
├── configs/
│   ├── llm.yaml
│   └── retriever.yaml
└── requirements.txt
```

---

## 4. PHASE 3: Django 백엔드

### 4.1 프로젝트 구조

```
chatbot-backend/
├── config/
│   ├── __init__.py
│   ├── settings/
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py                # WebSocket 지원
├── apps/
│   ├── chat/
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── models.py          # Conversation, Message
│   │   ├── views.py           # ChatViewSet
│   │   ├── serializers.py
│   │   ├── urls.py
│   │   ├── consumers.py       # WebSocket Consumer
│   │   └── routing.py         # WebSocket URL
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embedder.py        # 텍스트 임베딩
│   │   ├── retriever.py       # 벡터 검색
│   │   ├── generator.py       # LLM 응답 생성
│   │   └── pipeline.py        # RAG 파이프라인
│   └── users/
│       ├── __init__.py
│       ├── models.py
│       ├── views.py
│       └── serializers.py
├── static/
├── media/
├── requirements/
│   ├── base.txt
│   ├── development.txt
│   └── production.txt
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── manage.py
└── README.md
```

### 4.2 데이터 모델

```python
# apps/chat/models.py

class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Message(models.Model):
    class Role(models.TextChoices):
        USER = 'user', 'User'
        ASSISTANT = 'assistant', 'Assistant'
        SYSTEM = 'system', 'System'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    role = models.CharField(max_length=10, choices=Role.choices)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    # RAG 관련 메타데이터
    retrieved_docs = models.JSONField(null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)

class Feedback(models.Model):
    message = models.OneToOneField(Message, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
```

### 4.3 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/chat/` | 새 메시지 전송 및 응답 |
| GET | `/api/v1/chat/conversations/` | 대화 목록 조회 |
| GET | `/api/v1/chat/conversations/{id}/` | 특정 대화 상세 |
| GET | `/api/v1/chat/conversations/{id}/messages/` | 대화 메시지 기록 |
| POST | `/api/v1/chat/messages/{id}/feedback/` | 메시지 피드백 |
| DELETE | `/api/v1/chat/conversations/{id}/` | 대화 삭제 |
| WS | `/ws/chat/{conversation_id}/` | 실시간 스트리밍 채팅 |

### 4.4 기술 스택

| 영역 | 도구 | 버전 |
|------|------|------|
| Framework | Django | 5.x |
| REST API | Django REST Framework | 3.14+ |
| WebSocket | Django Channels | 4.x |
| Message Broker | Redis | 7.x |
| Database | PostgreSQL | 15+ |
| Vector DB | ChromaDB / FAISS | 최신 |
| Task Queue | Celery | 5.x |
| Caching | Redis | 7.x |

---

## 5. PHASE 4: Next.js 프론트엔드

### 5.1 프로젝트 구조

```
chatbot-frontend/
├── app/
│   ├── page.tsx               # 메인 채팅 페이지
│   ├── layout.tsx
│   ├── globals.css
│   ├── chat/
│   │   └── [id]/
│   │       └── page.tsx       # 특정 대화 페이지
│   └── api/                   # API Routes (필요시)
├── components/
│   ├── Chat/
│   │   ├── ChatWindow.tsx     # 채팅 윈도우 컨테이너
│   │   ├── MessageList.tsx    # 메시지 목록
│   │   ├── MessageBubble.tsx  # 개별 메시지 버블
│   │   ├── InputBox.tsx       # 입력창
│   │   ├── TypingIndicator.tsx # 타이핑 인디케이터
│   │   └── StreamingText.tsx  # 스트리밍 텍스트 표시
│   ├── Sidebar/
│   │   ├── Sidebar.tsx        # 사이드바 컨테이너
│   │   └── ConversationList.tsx # 대화 목록
│   ├── Layout/
│   │   ├── Header.tsx
│   │   └── Footer.tsx
│   └── UI/
│       ├── Button.tsx
│       ├── Input.tsx
│       └── Modal.tsx
├── hooks/
│   ├── useChat.ts             # 채팅 상태 관리
│   ├── useWebSocket.ts        # WebSocket 연결
│   ├── useConversations.ts    # 대화 목록 관리
│   └── useStreamingResponse.ts # 스트리밍 응답 처리
├── lib/
│   ├── api.ts                 # API 클라이언트 (axios/fetch)
│   ├── websocket.ts           # WebSocket 클라이언트
│   └── utils.ts               # 유틸리티 함수
├── stores/
│   └── chatStore.ts           # Zustand 스토어
├── types/
│   ├── chat.ts                # 채팅 관련 타입
│   └── api.ts                 # API 응답 타입
├── styles/
│   └── chat.module.css
├── public/
├── next.config.js
├── tailwind.config.js
├── tsconfig.json
├── package.json
└── README.md
```

### 5.2 주요 기능

| 기능 | 설명 | 구현 방식 |
|------|------|----------|
| **실시간 채팅** | 스트리밍 응답 표시 | WebSocket + SSE |
| **마크다운 렌더링** | 코드 블록, 링크 지원 | react-markdown |
| **대화 기록** | 세션별 히스토리 | API + LocalStorage |
| **새 대화** | 새 대화 시작 | POST /conversations |
| **대화 전환** | 이전 대화 불러오기 | GET /conversations/{id} |
| **피드백** | 응답 평가 | POST /feedback |
| **반응형 UI** | 모바일/데스크톱 | Tailwind CSS |
| **다크 모드** | 테마 전환 | Tailwind + CSS 변수 |

### 5.3 기술 스택

| 영역 | 도구 | 버전 |
|------|------|------|
| Framework | Next.js (App Router) | 14+ |
| Language | TypeScript | 5.x |
| Styling | Tailwind CSS | 3.x |
| State Management | Zustand | 4.x |
| Data Fetching | TanStack Query | 5.x |
| WebSocket | socket.io-client | 4.x |
| Markdown | react-markdown | 최신 |
| Syntax Highlight | react-syntax-highlighter | 최신 |
| Icons | Lucide React | 최신 |

---

## 6. 구현 순서 로드맵

### Step 1: 데이터 전처리 파이프라인 구축
- [ ] TextCleaner 구현
- [ ] TextTokenizer 구현
- [ ] BERTEncoder + Mean Pooling 구현
- [ ] TextEmbeddingPipeline 통합
- [ ] 임베딩 생성 스크립트 작성
- [ ] Vector DB (ChromaDB) 저장

### Step 2: RAG 챗봇 엔진 구축
- [ ] Vector Store 연동 (ChromaDB/FAISS)
- [ ] Retriever 구현 (Top-K 검색)
- [ ] LLM Client 구현 (OpenAI/Claude)
- [ ] Prompt Builder 구현
- [ ] RAG Pipeline 통합

### Step 3: Django 백엔드 개발
- [ ] 프로젝트 초기화
- [ ] 데이터 모델 정의 (Conversation, Message)
- [ ] REST API 구현 (DRF ViewSet)
- [ ] RAG 엔진 통합
- [ ] WebSocket 설정 (Django Channels)
- [ ] Redis 연동
- [ ] 인증/인가 (JWT)

### Step 4: Next.js 프론트엔드 개발
- [ ] 프로젝트 초기화
- [ ] 레이아웃 구성
- [ ] 채팅 UI 컴포넌트 구현
- [ ] API 클라이언트 구현
- [ ] WebSocket 연동
- [ ] 스트리밍 응답 처리
- [ ] 반응형 디자인

### Step 5: 통합 및 배포
- [ ] Docker Compose 설정
- [ ] CI/CD 파이프라인 (GitHub Actions)
- [ ] Render 배포 (Backend)
- [ ] Vercel 배포 (Frontend)
- [ ] 도메인 연결

---

## 7. 데이터셋 권장

| 용도 | 데이터셋 | 크기 | 링크 |
|------|---------|------|------|
| **QA 챗봇** | SQuAD 2.0 | 150K QA | https://rajpurkar.github.io/SQuAD-explorer |
| **한국어 QA** | KorQuAD 2.0 | 100K+ QA | https://korquad.github.io |
| **대화형** | AI Hub 한국어 대화 | 100만+ 대화 | https://aihub.or.kr |
| **감성 분류** | NSMC | 200K 리뷰 | https://github.com/e9t/nsmc |

---

## 8. 배포 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                          Vercel                                  │
│                    (Next.js Frontend)                           │
│                  chatbot.example.com                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Render                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Django API     │  │  Redis          │  │  PostgreSQL     │ │
│  │  (Web Service)  │  │  (Cache/WS)     │  │  (Database)     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    ChromaDB / FAISS                         ││
│  │                    (Vector Store)                           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      External APIs                               │
│            OpenAI GPT / Claude API (LLM)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 환경 변수

### Backend (.env)
```env
# Django
SECRET_KEY=your-secret-key
DEBUG=False
ALLOWED_HOSTS=api.example.com

# Database
DATABASE_URL=postgres://user:pass@host:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379/0

# LLM API
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx

# Vector DB
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=https://api.example.com
NEXT_PUBLIC_WS_URL=wss://api.example.com/ws
```

---

> **문서 작성**: Claude
> **최종 수정**: 2025년 12월 26일
