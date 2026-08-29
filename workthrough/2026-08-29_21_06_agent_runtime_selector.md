# 에이전트 런타임 선택기

## 개요
OpenCode 전용 설정을 OpenCode, Claude Code, Codex, Hermes 공통 런타임 설정으로 확장했다. 실행 파일, 모델, 프로젝트, 기존 세션을 자동으로 찾아 웹과 Electron 앱에서 선택하고 현재 대화 엔진에 즉시 반영할 수 있다.

## 주요 변경사항
- 개발한 것: 네 런타임 선택 UI, 상태 검사 API, 자동 탐지 카탈로그 API
- 개발한 것: CLI 실행 파일 자동 탐지와 모델·프로젝트·세션 선택기
- 개발한 것: OpenCode와 Hermes의 Direct CLI/oMLX 실행 모드
- 개발한 것: Claude Code, Codex, Hermes 연속 대화 세션 어댑터
- 개발한 것: Electron 네이티브 프로젝트 폴더 선택기
- 수정한 것: 연결 확인 시 저장하지 않은 런타임 선택과 입력값이 초기화되던 문제
- 수정한 것: Hermes safe-mode가 사용자 oMLX 공급자 설정을 무시해 401을 내던 문제
- 개선한 것: Claude 도구 비활성화, Codex 읽기 전용, Hermes 규칙·도구 비활성화
- 개선한 것: OpenCode의 agent 입력을 의미가 분명한 OpenCode profile 고급 설정으로 정리
- 개선한 것: 영문/중문 기본 설정과 기존 설정 업그레이드 호환성
- 개선한 것: 타임아웃 및 취소 시 CLI 자식 프로세스 정리
- 개선한 것: Electron 44 업그레이드, 렌더러 샌드박스 활성화, Node 통합 비활성화
- 개선한 것: 고유 앱 ID와 완전한 로컬 ad-hoc 번들 서명 적용

## 결과
- 백엔드 테스트 20개 통과, Ruff 검사 및 포맷 통과
- OpenCode, Claude Code, Codex, Hermes 실제 설치 상태 확인
- OpenCode 연결과 Claude Code, Codex, Hermes 실행 파일·버전 검사 통과
- oMLX 모델 16개, 프로젝트 12개, 각 런타임 기존 세션 탐지 확인
- OpenCode 기존 세션 선택과 프로젝트 연동 확인
- Hermes oMLX 실제 응답 `VTUBER_OK` 및 세션 ID 생성 확인
- 웹 프로덕션 빌드, Electron 빌드, macOS arm64 앱 패키징 성공
- 실제 설정 UI에서 네 런타임, Direct CLI/oMLX, 모델, 프로젝트, 세션 선택 검증
- Electron 앱 실행 프로세스 유지와 패키지 렌더러의 Chromium 샌드박스 적용 확인

## 다음 단계
- 기존 Live2D SDK TypeScript 오류를 별도 정리
- 외부 배포 전 Apple Developer ID 인증서로 서명하고 Apple 공증 완료
