# Todo List

## 완료된 작업

- [x] GitHub 리포지토리 클론 및 프로젝트 초기 환경 설정
- [x] 시스템 설계 보고서 분석 및 MVP 범위 확정
- [x] Python FastAPI 기반 백엔드 환경 구축
- [x] SQLite/PostgreSQL 스키마 설계 및 모델 생성
- [x] 사용자 인증/권한(RBAC) 시스템 구현
- [x] 학생/과정/반 관리 API 구현
- [x] 출결 관리 시스템 구현 (입력/집계/감사로그)
- [x] 성적 처리 시스템 구현 (평가/점수/산출)
- [x] 리포트 생성 및 PDF/Excel 출력 기능 구현
- [x] React 기반 웹 프론트엔드 구축
- [x] Render 클라우드 배포 완료

## 배포 정보

- **Frontend**: https://grade-management-frontend-bkoh.onrender.com
- **Backend API**: https://grade-management-api-q3q6.onrender.com
- **API 문서**: https://grade-management-api-q3q6.onrender.com/docs

## 핵심 MVP 기능 (모두 완료)

1. **사용자 관리**: Admin, Teacher, Staff, Student 역할 기반 인증 ✅
2. **학사 기본**: 학생 등록, 과정/반 관리, 수강 배정 ✅
3. **출결 관리**: 회차별 출결 입력, 집계, 변경 이력 ✅
4. **성적 처리**: 평가항목 설정, 점수 입력, 자동 산출 ✅
5. **기본 리포트**: 출결부, 성적표 PDF/Excel 출력 ✅

## 기술 스택

- **Backend**: FastAPI (고성능 API, 타입 안정성, 확장성)
- **Database**: SQLite (개발/배포) / PostgreSQL (프로덕션 옵션)
- **Authentication**: JWT + RBAC
- **Frontend**: React + TypeScript + Vite + TailwindCSS
- **State Management**: Zustand + React Query
- **Deployment**: Render (Free Tier)

## 향후 개선사항 (Optional)

- [ ] PostgreSQL로 데이터베이스 마이그레이션
- [x] 대시보드 통계 기능 강화
- [ ] 이메일 알림 기능
- [ ] 모바일 반응형 UI 개선
- [ ] 테스트 코드 작성
- [ ] CI/CD 파이프라인 구축

## 개발 일지

### 2025-12-22
**버그 수정 및 배포 테스트 완료**

#### 버그 수정

1. **프론트엔드 - 반 생성 시 schedule 필드 오류** (`Courses.tsx:96-106`)
   - 문제: `schedule` 필드를 문자열로 전송하여 백엔드 422 오류 발생
   - 원인: 백엔드 스키마가 `Dict[str, Any]` 타입을 기대
   - 해결: JSON 객체 `{ description: "..." }` 또는 `null`로 전송하도록 수정

2. **백엔드 - SQLite dict 저장 불가 오류** (`courses.py`)
   - 문제: SQLite가 Python dict 타입을 직접 저장할 수 없어 500 오류 발생
   - 원인: `schedule` 필드를 직렬화 없이 저장 시도
   - 해결: `json.dumps()`로 저장, `json.loads()`로 조회 시 역직렬화

#### 배포 테스트 결과

| 기능 | 상태 |
|------|------|
| 과정 등록/조회 | ✅ 정상 |
| 반 생성 (일정 포함) | ✅ 정상 |
| 학생 등록 | ✅ 정상 |
| 학생 수강 배정 | ✅ 정상 |
| 출결 관리 UI | ✅ 정상 |
| 성적 관리/평가항목 추가 | ✅ 정상 |
| 리포트 페이지 | ✅ 정상 |

#### 커밋 내역
- `1101e93` - fix: Class creation schedule field - send as JSON object
- `56882b0` - fix: Backend Class schedule JSON serialization for SQLite

#### 참고사항
- Render Free Tier는 SQLite를 사용하여 배포 시마다 데이터가 초기화됨
- 영구 데이터 저장이 필요하면 PostgreSQL 등 외부 DB 연동 필요

---

### 2024-12-22
**프론트엔드 미구현 기능 전체 구현 완료**

#### Courses.tsx
- 과정 수정 모달 구현 (코드, 이름, 설명, 회차, 시작/종료일)
- 반(Class) 관리 기능 구현 (생성, 조회)
- 과정 상세 보기 모달 구현 (과정 정보 + 반 목록)

#### Students.tsx
- 학생 정보 수정 모달 구현 (학번, 상태 변경)
- 수강 배정(Enrollment) UI 구현 (과정 → 반 선택 후 학생 배정)

#### Attendance.tsx
- 출결 일괄 입력 모달 구현 (날짜, 회차 선택 후 학생별 출결 상태 일괄 입력)
- 출결 수정 모달 구현 (상태, 입/퇴실 시간, 비고)
- 필터 수정: 과정 → 반 2단계 계층 구조로 변경

#### Grades.tsx
- 평가 항목 추가 모달 구현 (유형, 가중치, 만점, 마감일, 설명)
- 성적 입력 모달 구현 (평가 카드 클릭 시)
- 성적 수정 모달 구현 (테이블에서 수정 버튼 클릭)

#### Reports.tsx
- 버그 수정: 출결 리포트가 course_id 대신 class_id를 사용하도록 수정
- 과정/반 2단계 선택 UI 구현 (출결부는 반 필수, 성적표는 과정만)

#### Dashboard.tsx
- 실제 API 데이터 기반 통계 연동 (오늘 출석률, 평균 성적)
- 빠른 작업 버튼 네비게이션 연결 (학생등록, 출결입력, 성적입력, 리포트)
- 최근 학생/과정 목록에 "전체 보기" 링크 추가

#### 기타
- TypeScript 빌드 오류 수정 (미사용 변수 제거, Map 타입 명시)

### 2024-12-17
- Render 배포 완료
- TypeScript 빌드 에러 수정 (미사용 import 제거, vite-env.d.ts 추가)
- Backend CORS 및 환경변수 설정 수정
- Frontend API URL 환경변수 설정
- UUID to String 변환 이슈 해결 (SQLite 호환성)
- Excel 리포트 한글 파일명 인코딩 수정
