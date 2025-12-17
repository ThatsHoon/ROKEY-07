# 성적 및 출결 관리 시스템 설계서

## 1. 시스템 개요

### 1.1 프로젝트 목적
학생들의 출결 및 성적을 효율적으로 관리하는 웹 기반 시스템 구축

### 1.2 핵심 MVP 기능
| 기능 | 설명 |
|------|------|
| 사용자 관리 | Admin, Teacher, Staff, Student 역할 기반 인증 |
| 학사 기본 | 학생 등록, 과정/반 관리, 수강 배정 |
| 출결 관리 | 회차별 출결 입력, 집계, 변경 이력 |
| 성적 처리 | 평가항목 설정, 점수 입력, 자동 산출 |
| 기본 리포트 | 출결부, 성적표 PDF/Excel 출력 |

---

## 2. 기술 스택

### 2.1 Backend
```
Framework: FastAPI 0.104.1
- 고성능 비동기 API
- 자동 OpenAPI 문서화
- Pydantic 기반 타입 검증

ASGI Server: Uvicorn 0.24.0
ORM: SQLAlchemy 2.0.23
Migration: Alembic 1.12.1
```

### 2.2 Database
```
RDBMS: PostgreSQL 15+
- 트랜잭션 안정성
- 감사 로그 지원
- JSON 타입 지원
```

### 2.3 Authentication
```
JWT (JSON Web Token)
- Access Token: 30분
- Refresh Token: 7일
RBAC (Role-Based Access Control)
```

### 2.4 Frontend
```
Framework: React 18 + TypeScript
State: Zustand / React Query
UI: Tailwind CSS + shadcn/ui
Build: Vite
```

### 2.5 추가 라이브러리
```python
# 인증/보안
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# 리포트 생성
reportlab==4.0.7          # PDF 생성
openpyxl==3.1.2           # Excel 생성
jinja2==3.1.2             # 템플릿 엔진

# 유틸리티
python-multipart==0.0.6   # 파일 업로드
python-dateutil==2.8.2    # 날짜 처리
```

---

## 3. 시스템 아키텍처

### 3.1 전체 구조
```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Web App    │  │  Mobile     │  │  Admin      │         │
│  │  (React)    │  │  (PWA)      │  │  Dashboard  │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           │ HTTPS
┌──────────────────────────┼──────────────────────────────────┐
│                    API Gateway                              │
│  ┌───────────────────────┴───────────────────────┐         │
│  │              FastAPI Application               │         │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐         │         │
│  │  │  Auth   │ │  CORS   │ │  Rate   │         │         │
│  │  │Middleware│ │Middleware│ │ Limiter │         │         │
│  │  └─────────┘ └─────────┘ └─────────┘         │         │
│  └───────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                   Service Layer                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│  │    User    │ │   Course   │ │ Attendance │ │  Grade   │ │
│  │  Service   │ │  Service   │ │  Service   │ │ Service  │ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
│  ┌────────────┐ ┌────────────┐                             │
│  │   Report   │ │   Audit    │                             │
│  │  Service   │ │  Service   │                             │
│  └────────────┘ └────────────┘                             │
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                    Data Layer                               │
│  ┌───────────────────────┴───────────────────────┐         │
│  │              PostgreSQL Database               │         │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐         │         │
│  │  │  Users  │ │ Courses │ │ Grades  │         │         │
│  │  └─────────┘ └─────────┘ └─────────┘         │         │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐         │         │
│  │  │Attendance│ │  Audit  │ │ Reports │         │         │
│  │  └─────────┘ └─────────┘ └─────────┘         │         │
│  └───────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 디렉토리 구조
```
grade_management_system/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI 앱 진입점
│   │   ├── config.py               # 환경설정
│   │   ├── database.py             # DB 연결
│   │   │
│   │   ├── models/                 # SQLAlchemy 모델
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── course.py
│   │   │   ├── student.py
│   │   │   ├── attendance.py
│   │   │   ├── grade.py
│   │   │   └── audit_log.py
│   │   │
│   │   ├── schemas/                # Pydantic 스키마
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── course.py
│   │   │   ├── attendance.py
│   │   │   └── grade.py
│   │   │
│   │   ├── routers/                # API 라우터
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── users.py
│   │   │   ├── courses.py
│   │   │   ├── students.py
│   │   │   ├── attendance.py
│   │   │   ├── grades.py
│   │   │   └── reports.py
│   │   │
│   │   ├── services/               # 비즈니스 로직
│   │   │   ├── __init__.py
│   │   │   ├── auth_service.py
│   │   │   ├── user_service.py
│   │   │   ├── attendance_service.py
│   │   │   ├── grade_service.py
│   │   │   └── report_service.py
│   │   │
│   │   ├── utils/                  # 유틸리티
│   │   │   ├── __init__.py
│   │   │   ├── security.py         # JWT, 비밀번호 해싱
│   │   │   ├── pdf_generator.py
│   │   │   └── excel_generator.py
│   │   │
│   │   └── middleware/             # 미들웨어
│   │       ├── __init__.py
│   │       └── audit.py
│   │
│   ├── alembic/                    # DB 마이그레이션
│   │   ├── versions/
│   │   └── env.py
│   │
│   ├── tests/                      # 테스트
│   │   ├── __init__.py
│   │   ├── test_auth.py
│   │   ├── test_attendance.py
│   │   └── test_grades.py
│   │
│   ├── alembic.ini
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/
│   │   │   ├── attendance/
│   │   │   ├── grades/
│   │   │   └── reports/
│   │   │
│   │   ├── pages/
│   │   │   ├── Login.tsx
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Students.tsx
│   │   │   ├── Attendance.tsx
│   │   │   ├── Grades.tsx
│   │   │   └── Reports.tsx
│   │   │
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── store/
│   │   ├── types/
│   │   └── utils/
│   │
│   ├── package.json
│   ├── vite.config.ts
│   └── Dockerfile
│
├── docker-compose.yml
├── docker-compose.prod.yml
└── README.md
```

---

## 4. 데이터베이스 설계

### 4.1 ERD (Entity Relationship Diagram)
```
┌─────────────────┐       ┌─────────────────┐
│      users      │       │     roles       │
├─────────────────┤       ├─────────────────┤
│ id (PK)         │──┐    │ id (PK)         │
│ email           │  │    │ name            │
│ password_hash   │  │    │ permissions     │
│ name            │  └───>│                 │
│ role_id (FK)    │       └─────────────────┘
│ is_active       │
│ created_at      │
└────────┬────────┘
         │
         │ 1:1 (Teacher/Staff)
         ▼
┌─────────────────┐       ┌─────────────────┐
│    students     │       │    courses      │
├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │
│ user_id (FK)    │       │ name            │
│ student_number  │       │ code            │
│ class_id (FK)   │──┐    │ description     │
│ enrolled_at     │  │    │ start_date      │
│ status          │  │    │ end_date        │
└────────┬────────┘  │    │ teacher_id (FK) │
         │           │    └────────┬────────┘
         │           │             │
         │           │             │
         ▼           ▼             ▼
┌─────────────────────────────────────────────┐
│               classes (반)                   │
├─────────────────────────────────────────────┤
│ id (PK)                                     │
│ course_id (FK)                              │
│ name                                        │
│ capacity                                    │
│ schedule                                    │
└─────────────────────────────────────────────┘
         │
         │
         ▼
┌─────────────────┐       ┌─────────────────┐
│   attendance    │       │ attendance_logs │
├─────────────────┤       ├─────────────────┤
│ id (PK)         │──────>│ id (PK)         │
│ student_id (FK) │       │ attendance_id   │
│ class_id (FK)   │       │ changed_by      │
│ session_no      │       │ old_status      │
│ date            │       │ new_status      │
│ status          │       │ reason          │
│ check_in_time   │       │ changed_at      │
│ check_out_time  │       └─────────────────┘
│ notes           │
└─────────────────┘

┌─────────────────┐       ┌─────────────────┐
│  evaluations    │       │     grades      │
├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │
│ course_id (FK)  │──────>│ evaluation_id   │
│ name            │       │ student_id (FK) │
│ type            │       │ score           │
│ weight (%)      │       │ max_score       │
│ max_score       │       │ graded_by (FK)  │
│ due_date        │       │ graded_at       │
└─────────────────┘       │ comments        │
                          └─────────────────┘

┌─────────────────────────────────────────────┐
│              audit_logs                      │
├─────────────────────────────────────────────┤
│ id (PK)                                     │
│ user_id (FK)                                │
│ action                                      │
│ table_name                                  │
│ record_id                                   │
│ old_values (JSONB)                          │
│ new_values (JSONB)                          │
│ ip_address                                  │
│ created_at                                  │
└─────────────────────────────────────────────┘
```

### 4.2 주요 테이블 상세

#### users (사용자)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | UUID | Primary Key |
| email | VARCHAR(255) | 이메일 (UNIQUE) |
| password_hash | VARCHAR(255) | 암호화된 비밀번호 |
| name | VARCHAR(100) | 이름 |
| role_id | INTEGER | 역할 FK |
| is_active | BOOLEAN | 활성 상태 |
| created_at | TIMESTAMP | 생성일시 |
| updated_at | TIMESTAMP | 수정일시 |

#### attendance (출결)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | UUID | Primary Key |
| student_id | UUID | 학생 FK |
| class_id | UUID | 반 FK |
| session_no | INTEGER | 회차 번호 |
| date | DATE | 날짜 |
| status | ENUM | 출석/지각/조퇴/결석/공결 |
| check_in_time | TIME | 입실 시간 |
| check_out_time | TIME | 퇴실 시간 |
| notes | TEXT | 비고 |

#### grades (성적)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | UUID | Primary Key |
| evaluation_id | UUID | 평가항목 FK |
| student_id | UUID | 학생 FK |
| score | DECIMAL(5,2) | 취득점수 |
| max_score | DECIMAL(5,2) | 만점 |
| graded_by | UUID | 채점자 FK |
| graded_at | TIMESTAMP | 채점일시 |
| comments | TEXT | 코멘트 |

---

## 5. API 설계

### 5.1 인증 API
```
POST   /api/v1/auth/login           # 로그인
POST   /api/v1/auth/logout          # 로그아웃
POST   /api/v1/auth/refresh         # 토큰 갱신
GET    /api/v1/auth/me              # 현재 사용자 정보
PUT    /api/v1/auth/password        # 비밀번호 변경
```

### 5.2 사용자 관리 API
```
GET    /api/v1/users                # 사용자 목록
POST   /api/v1/users                # 사용자 생성
GET    /api/v1/users/{id}           # 사용자 상세
PUT    /api/v1/users/{id}           # 사용자 수정
DELETE /api/v1/users/{id}           # 사용자 삭제
```

### 5.3 학생 관리 API
```
GET    /api/v1/students             # 학생 목록
POST   /api/v1/students             # 학생 등록
GET    /api/v1/students/{id}        # 학생 상세
PUT    /api/v1/students/{id}        # 학생 수정
DELETE /api/v1/students/{id}        # 학생 삭제
POST   /api/v1/students/{id}/enroll # 수강 신청
```

### 5.4 출결 관리 API
```
GET    /api/v1/attendance                      # 출결 목록
POST   /api/v1/attendance                      # 출결 입력
PUT    /api/v1/attendance/{id}                 # 출결 수정
GET    /api/v1/attendance/class/{class_id}     # 반별 출결 현황
GET    /api/v1/attendance/student/{student_id} # 학생별 출결 현황
GET    /api/v1/attendance/summary              # 출결 집계
GET    /api/v1/attendance/{id}/logs            # 출결 변경 이력
```

### 5.5 성적 관리 API
```
GET    /api/v1/evaluations                     # 평가항목 목록
POST   /api/v1/evaluations                     # 평가항목 생성
PUT    /api/v1/evaluations/{id}                # 평가항목 수정
DELETE /api/v1/evaluations/{id}                # 평가항목 삭제

GET    /api/v1/grades                          # 성적 목록
POST   /api/v1/grades                          # 성적 입력
POST   /api/v1/grades/bulk                     # 성적 일괄 입력
PUT    /api/v1/grades/{id}                     # 성적 수정
GET    /api/v1/grades/student/{student_id}     # 학생별 성적
GET    /api/v1/grades/calculate/{student_id}   # 성적 자동 산출
```

### 5.6 리포트 API
```
GET    /api/v1/reports/attendance/pdf          # 출결부 PDF
GET    /api/v1/reports/attendance/excel        # 출결부 Excel
GET    /api/v1/reports/grades/pdf              # 성적표 PDF
GET    /api/v1/reports/grades/excel            # 성적표 Excel
GET    /api/v1/reports/transcript/{student_id} # 개인 성적증명서
```

---

## 6. 역할 기반 접근 제어 (RBAC)

### 6.1 역할 정의
| 역할 | 설명 |
|------|------|
| **Admin** | 시스템 전체 관리, 모든 권한 |
| **Teacher** | 담당 과정/반의 출결/성적 관리 |
| **Staff** | 학사 정보 조회, 리포트 생성 |
| **Student** | 본인 출결/성적 조회 |

### 6.2 권한 매트릭스
| 기능 | Admin | Teacher | Staff | Student |
|------|:-----:|:-------:|:-----:|:-------:|
| 사용자 관리 | ✅ | ❌ | ❌ | ❌ |
| 과정/반 생성 | ✅ | ❌ | ❌ | ❌ |
| 과정/반 조회 | ✅ | ✅ | ✅ | ✅ |
| 학생 등록 | ✅ | ❌ | ✅ | ❌ |
| 출결 입력 | ✅ | ✅ | ❌ | ❌ |
| 출결 조회 (전체) | ✅ | ✅ | ✅ | ❌ |
| 출결 조회 (본인) | ✅ | ✅ | ✅ | ✅ |
| 성적 입력 | ✅ | ✅ | ❌ | ❌ |
| 성적 조회 (전체) | ✅ | ✅ | ✅ | ❌ |
| 성적 조회 (본인) | ✅ | ✅ | ✅ | ✅ |
| 리포트 생성 | ✅ | ✅ | ✅ | ❌ |
| 감사 로그 조회 | ✅ | ❌ | ❌ | ❌ |

---

## 7. 핵심 기능 상세

### 7.1 출결 관리 시스템
```python
# 출결 상태 정의
class AttendanceStatus(Enum):
    PRESENT = "출석"      # 정상 출석
    LATE = "지각"         # 지각 (10분 이내)
    EARLY_LEAVE = "조퇴"  # 조퇴
    ABSENT = "결석"       # 결석
    EXCUSED = "공결"      # 공결 (사유 필요)

# 출결 집계 로직
def calculate_attendance_summary(student_id, class_id):
    """
    학생별 출결 집계
    - 출석률 = (출석 + 지각*0.5) / 전체 회차 * 100
    - 지각 3회 = 결석 1회 환산
    """
    pass
```

### 7.2 성적 처리 시스템
```python
# 평가 유형
class EvaluationType(Enum):
    MIDTERM = "중간고사"
    FINAL = "기말고사"
    QUIZ = "퀴즈"
    ASSIGNMENT = "과제"
    PROJECT = "프로젝트"
    ATTENDANCE = "출결"

# 성적 산출 로직
def calculate_final_grade(student_id, course_id):
    """
    최종 성적 산출
    - 각 평가항목별 가중치 적용
    - 총점 = Σ(점수/만점 * 가중치)
    - 등급 산출: A(90+), B(80+), C(70+), D(60+), F(<60)
    """
    evaluations = get_evaluations(course_id)
    total = 0
    for eval in evaluations:
        grade = get_grade(student_id, eval.id)
        weighted = (grade.score / eval.max_score) * eval.weight
        total += weighted
    return total, get_letter_grade(total)
```

### 7.3 변경 이력 (Audit Log)
```python
# 감사 로그 자동 기록
@audit_log
def update_attendance(attendance_id, new_status, reason):
    """
    출결 변경 시 자동으로 audit_log 테이블에 기록
    - 변경 전/후 값
    - 변경 사유
    - 변경자 정보
    - 변경 시각
    """
    pass
```

---

## 8. 보안 설계

### 8.1 인증 흐름
```
┌─────────┐    1. Login Request     ┌─────────┐
│ Client  │ ──────────────────────> │  Server │
│         │    (email, password)    │         │
│         │                         │         │
│         │ <────────────────────── │         │
│         │    2. JWT Tokens        │         │
│         │    (access, refresh)    │         │
│         │                         │         │
│         │    3. API Request       │         │
│         │ ──────────────────────> │         │
│         │    Authorization:       │         │
│         │    Bearer {access_token}│         │
│         │                         │         │
│         │ <────────────────────── │         │
│         │    4. Response          │         │
└─────────┘                         └─────────┘
```

### 8.2 보안 체크리스트
- [x] 비밀번호 bcrypt 해싱
- [x] JWT 토큰 만료 설정
- [x] HTTPS 강제 적용
- [x] CORS 정책 설정
- [x] SQL Injection 방지 (ORM 사용)
- [x] XSS 방지 (프론트엔드 이스케이프)
- [x] Rate Limiting 적용
- [x] 민감 정보 환경변수 관리

---

## 9. 리포트 출력

### 9.1 출결부 (PDF/Excel)
```
┌─────────────────────────────────────────────────────────────┐
│                      출 결 부                               │
│                                                             │
│  과정: Python 심화과정        반: A반                       │
│  기간: 2024.01.01 ~ 2024.03.31                              │
│                                                             │
├─────┬──────────┬────┬────┬────┬────┬────┬────┬─────────────┤
│ No. │   이름   │ 1회│ 2회│ 3회│ ... │출석│결석│  출석률     │
├─────┼──────────┼────┼────┼────┼────┼────┼────┼─────────────┤
│  1  │ 홍길동   │ ○ │ ○ │ △ │ ... │ 18 │  2 │   90.0%     │
│  2  │ 김철수   │ ○ │ × │ ○ │ ... │ 15 │  5 │   75.0%     │
└─────┴──────────┴────┴────┴────┴────┴────┴────┴─────────────┘
                                        ○:출석 △:지각 ×:결석
```

### 9.2 성적표 (PDF/Excel)
```
┌─────────────────────────────────────────────────────────────┐
│                      성 적 표                               │
│                                                             │
│  학번: 2024001        이름: 홍길동                          │
│  과정: Python 심화과정                                      │
│                                                             │
├─────────────────┬────────┬────────┬────────┬───────────────┤
│     평가항목    │  만점  │  점수  │ 가중치 │    환산점     │
├─────────────────┼────────┼────────┼────────┼───────────────┤
│ 중간고사        │  100   │   85   │  30%   │    25.5       │
│ 기말고사        │  100   │   90   │  30%   │    27.0       │
│ 과제            │  100   │   95   │  20%   │    19.0       │
│ 출결            │  100   │   90   │  20%   │    18.0       │
├─────────────────┼────────┼────────┼────────┼───────────────┤
│      총점       │        │        │        │    89.5       │
│      등급       │        │        │        │      B+       │
└─────────────────┴────────┴────────┴────────┴───────────────┘
```

---

## 10. 배포 환경

### 10.1 Docker Compose 구성
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: grade_db
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  backend:
    build: ./backend
    environment:
      DATABASE_URL: postgresql://admin:${DB_PASSWORD}@postgres:5432/grade_db
      JWT_SECRET: ${JWT_SECRET}
    ports:
      - "8000:8000"
    depends_on:
      - postgres

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - frontend
      - backend

volumes:
  postgres_data:
```

### 10.2 환경 변수
```env
# .env
DATABASE_URL=postgresql://admin:password@localhost:5432/grade_db
JWT_SECRET=your-super-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# PostgreSQL
POSTGRES_DB=grade_db
POSTGRES_USER=admin
POSTGRES_PASSWORD=secure-password

# Frontend
VITE_API_URL=http://localhost:8000/api/v1
```

---

## 11. 개발 로드맵

### Phase 1: 기반 구축
- [ ] 프로젝트 초기 환경 설정
- [ ] PostgreSQL 스키마 설계 및 마이그레이션
- [ ] FastAPI 기본 구조 구현
- [ ] JWT 인증 시스템 구현

### Phase 2: 핵심 기능
- [ ] 사용자/권한 관리 API
- [ ] 학생/과정/반 관리 API
- [ ] 출결 관리 시스템 (입력/집계/변경이력)
- [ ] 성적 처리 시스템 (평가/점수/산출)

### Phase 3: 리포트 & 프론트엔드
- [ ] PDF/Excel 리포트 생성
- [ ] React 프론트엔드 구축
- [ ] 대시보드 구현

### Phase 4: 배포 & 고도화
- [ ] Docker 컨테이너화
- [ ] CI/CD 파이프라인 구축
- [ ] 성능 최적화

---

## 12. 결론

본 시스템은 FastAPI + PostgreSQL + React 기술 스택을 기반으로 한 완전한 성적 및 출결 관리 시스템입니다.

**핵심 특징:**
- 역할 기반 접근 제어 (RBAC)로 안전한 권한 관리
- 모든 데이터 변경에 대한 감사 로그 기록
- 자동화된 성적 산출 및 출결 집계
- PDF/Excel 형식의 리포트 출력 지원
- 확장 가능한 마이크로서비스 친화적 설계

**기대 효과:**
- 수작업 출결/성적 관리의 자동화
- 데이터 무결성 및 투명성 확보
- 실시간 현황 파악 및 리포트 생성
- 관리자/교사/학생 각 역할별 최적화된 인터페이스 제공
