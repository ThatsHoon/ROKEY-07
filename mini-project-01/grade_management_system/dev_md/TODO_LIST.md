Todo List
☐ GitHub 리포지토리 클론 및 프로젝트 초기 환경 설정
☐ 시스템 설계 보고서 분석 및 MVP 범위 확정
☐ Python FastAPI/Django 기반 백엔드 환경 구축
☐ PostgreSQL 스키마 설계 및 마이그레이션 생성
☐ 사용자 인증/권한(RBAC) 시스템 구현
☐ 학생/과정/반 관리 API 구현
☐ 출결 관리 시스템 구현 (입력/집계/감사로그)
☐ 성적 처리 시스템 구현 (평가/점수/산출)
☐ 리포트 생성 및 PDF/Excel 출력 기능 구현
☐ React 기반 웹 프론트엔드 구축

  핵심 MVP 기능:

1. 사용자 관리: Admin, Teacher, Staff, Student 역할 기반 인증

2. 학사 기본: 학생 등록, 과정/반 관리, 수강 배정

3. 출결 관리: 회차별 출결 입력, 집계, 변경 이력

4. 성적 처리: 평가항목 설정, 점수 입력, 자동 산출

5. 기본 리포트: 출결부, 성적표 PDF/Excel 출력
   기술 스택 결정:
- Backend: FastAPI (고성능 API, 타입 안정성, 확장성)
- Database: PostgreSQL (트랜잭션 안정성, 감사 로그)
- Authentication: JWT + RBAC
- Frontend: React + TypeScript
