# 런타임 세션 제목 개선

## 개요
OpenCode, Claude Code, Codex, Hermes 세션 이름 변경 경로를 검증했다. Codex는 데스크톱 카탈로그의 정리된 제목을 우선 표시하고, 이름 변경 시 두 로컬 저장소를 함께 갱신하도록 개선했다.

## 주요 변경사항
- Codex `local_thread_catalog.display_title`을 세션 목록 제목으로 반영
- 사용자 지정 `threads.name`은 자동 생성 제목보다 항상 우선하도록 유지
- Codex 이름 변경 시 `state_5.sqlite`와 `codex*.db`를 함께 갱신
- 네 런타임별 이름 변경 저장 테스트 보강

## 결과
- 세션 관련 백엔드 테스트 21개 통과
- Ruff 검사 통과
- 실제 Codex 로컬 카탈로그 제목 조회 확인

## 다음 단계
- 새 Codex 저장소 버전이 추가될 때 카탈로그 스키마 호환성을 회귀 테스트에 추가
