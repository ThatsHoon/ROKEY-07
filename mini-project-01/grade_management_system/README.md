# 성적 및 출결 관리 시스템

학생들의 출결 및 성적을 효율적으로 관리하는 웹 기반 시스템입니다.

## 배포 URL

- **서버 실행(1-2소요)**: https://grade-management-frontend.onrender.com
- **접속 링크**: https://grade-management-frontend-bkoh.onrender.com
- **Backend API**: https://grade-management-api-q3q6.onrender.com
- **API 문서**: https://grade-management-api-q3q6.onrender.com/docs

> 참고: Render 무료 플랜은 15분 비활성 후 서비스가 슬립 상태로 전환됩니다. 첫 접속 시 1-2분 정도 로딩 시간이 필요합니다.

## 주요 기능

- **사용자 관리**: Admin, Teacher, Staff, Student 역할 기반 인증
- **학사 기본**: 학생 등록, 과정/반 관리, 수강 배정
- **출결 관리**: 회차별 출결 입력, 집계, 변경 이력
- **성적 처리**: 평가항목 설정, 점수 입력, 자동 산출
- **리포트**: 출결부, 성적표 PDF/Excel 출력

## 기술 스택

### Backend
- FastAPI 0.104.1
- SQLite (개발/배포) / PostgreSQL (프로덕션 옵션)
- SQLAlchemy 2.0
- JWT + RBAC 인증
- Pydantic v2

### Frontend
- React 18 + TypeScript
- Vite
- TailwindCSS
- React Query (TanStack Query)
- Zustand (상태관리)

### 배포
- Render (Backend: Web Service, Frontend: Static Site)

## 시작하기

### 사전 요구사항
- Python 3.11+
- Node.js 20+
- Git

### 로컬 실행

#### Backend
```bash
cd backend

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend

# 의존성 설치
npm install

# 환경변수 설정 (로컬 개발용)
# .env 파일 생성
echo "VITE_API_URL=http://localhost:8000/api/v1" > .env

# 개발 서버 실행
npm run dev
```

### Render 배포

프로젝트에 `render.yaml` Blueprint 파일이 포함되어 있습니다.

1. Render Dashboard에서 "New +" → "Blueprint" 선택
2. GitHub 저장소 연결
3. 자동으로 Backend와 Frontend 서비스 생성

## 기본 계정

| 역할 | 이메일 | 비밀번호 |
|------|--------|----------|
| Admin | admin@example.com | admin123 |

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs (로컬)
- ReDoc: http://localhost:8000/redoc (로컬)
- 배포: https://grade-management-api-q3q6.onrender.com/docs

## 프로젝트 구조

```
grade_management_system/
├── backend/
│   ├── app/
│   │   ├── models/      # SQLAlchemy 모델
│   │   ├── schemas/     # Pydantic 스키마
│   │   ├── routers/     # API 라우터
│   │   └── utils/       # 유틸리티 (인증, 보안)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/  # React 컴포넌트
│   │   ├── pages/       # 페이지
│   │   ├── services/    # API 서비스
│   │   ├── store/       # Zustand 상태 관리
│   │   └── types/       # TypeScript 타입
│   ├── package.json
│   └── vite.config.ts
├── dev_md/              # 개발 문서
├── render.yaml          # Render 배포 설정
└── README.md
```

## 개발 현황

- [x] Backend API 구현 완료
- [x] Frontend UI 구현 완료
- [x] 사용자 인증/권한 시스템
- [x] 학생/과정/반 관리
- [x] 출결 관리 시스템
- [x] 성적 처리 시스템
- [x] PDF/Excel 리포트 출력
- [x] Render 배포 완료

## 라이선스

MIT License
