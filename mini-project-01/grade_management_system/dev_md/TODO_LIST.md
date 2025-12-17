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

- **Frontend**: https://grade-management-frontend.onrender.com
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
- [ ] 대시보드 통계 기능 강화
- [ ] 이메일 알림 기능
- [ ] 모바일 반응형 UI 개선
- [ ] 테스트 코드 작성
- [ ] CI/CD 파이프라인 구축

## 개발 일지

### 2024-12-17
- Render 배포 완료
- TypeScript 빌드 에러 수정 (미사용 import 제거, vite-env.d.ts 추가)
- Backend CORS 및 환경변수 설정 수정
- Frontend API URL 환경변수 설정
- UUID to String 변환 이슈 해결 (SQLite 호환성)
- Excel 리포트 한글 파일명 인코딩 수정
