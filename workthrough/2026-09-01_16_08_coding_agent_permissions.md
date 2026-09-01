# 코딩 에이전트 권한 및 런타임 통합

## 개요
OpenCode, Claude Code, Codex, Hermes를 네이티브 런타임으로 연결하고 프로젝트별 세션, 명령, 추론, 도구 실행과 권한 요청을 하나의 코딩 모드에서 처리하도록 완성했다.

## 주요 변경사항
- 개발한 것: 런타임별 세션/모델/명령 카탈로그와 수동·자동·계획·비활성 권한 모드
- 수정한 것: OpenCode SSE, Claude SDK, Codex app-server, Hermes ACP 이벤트와 승인 흐름
- 개선한 것: 실행 파일 자동 탐지, 세션 이름 변경, 프로젝트별 세션 필터링, ASR/TTS 절전 설정

## 결과
- 백엔드 전체 테스트 96개 통과
- Ruff, compileall, diff 검사 통과
- 네 런타임 실응답 및 Codex 권한 거절 흐름 확인

## 다음 단계
- 배포용 Apple Developer ID 서명과 공증 자동화
