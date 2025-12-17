# 성적 및 출결 관리 시스템

학생들의 출결 및 성적을 효율적으로 관리하는 웹 기반 시스템입니다.

## 주요 기능

- **사용자 관리**: Admin, Teacher, Staff, Student 역할 기반 인증
- **학사 기본**: 학생 등록, 과정/반 관리, 수강 배정
- **출결 관리**: 회차별 출결 입력, 집계, 변경 이력
- **성적 처리**: 평가항목 설정, 점수 입력, 자동 산출
- **리포트**: 출결부, 성적표 PDF/Excel 출력

## 기술 스택

### Backend
- FastAPI 0.104.1
- PostgreSQL 15
- SQLAlchemy 2.0
- JWT + RBAC 인증

### Frontend
- React 18 + TypeScript
- Vite
- TailwindCSS
- React Query

## 시작하기

### 사전 요구사항
- Python 3.11+
- Node.js 20+
- PostgreSQL 15+
- Docker & Docker Compose (선택)

### Docker로 실행 (권장)

```bash
# 프로젝트 디렉토리로 이동
cd grade_management_system

# Docker Compose로 실행
docker-compose up -d

# 브라우저에서 접속
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/docs
```

### 개별 실행

#### Backend
```bash
cd backend

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일 수정

# 데이터베이스 마이그레이션
alembic upgrade head

# 서버 실행
uvicorn app.main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

## 기본 계정

| 역할 | 이메일 | 비밀번호 |
|------|--------|----------|
| Admin | admin@example.com | admin123 |

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 프로젝트 구조

```
grade_management_system/
├── backend/
│   ├── app/
│   │   ├── models/      # SQLAlchemy 모델
│   │   ├── schemas/     # Pydantic 스키마
│   │   ├── routers/     # API 라우터
│   │   ├── services/    # 비즈니스 로직
│   │   ├── utils/       # 유틸리티
│   │   └── middleware/  # 미들웨어
│   ├── alembic/         # DB 마이그레이션
│   └── tests/           # 테스트
├── frontend/
│   ├── src/
│   │   ├── components/  # React 컴포넌트
│   │   ├── pages/       # 페이지
│   │   ├── services/    # API 서비스
│   │   ├── store/       # 상태 관리
│   │   └── types/       # TypeScript 타입
│   └── public/
└── docker-compose.yml
```

## 라이선스

MIT License
