# 원본 세션 재개 계약 검증

## 개요
기존 세션 선택 시 별도 복사본을 만들지 않고 각 런타임의 원본 세션 ID를 그대로 재개하는지 검증했다. 활성 네이티브 어댑터 기준의 회귀 테스트를 추가했다.

## 주요 변경사항
- OpenCode가 기존 `/session/{id}`에 최신 입력만 추가하는지 검증
- Claude Code가 SDK `resume`에 선택한 원본 ID를 전달하는지 검증
- Codex가 `thread/start` 대신 `thread/resume`을 호출하는지 검증
- Hermes가 `new_session` 대신 ACP `load_session`을 호출하는지 검증

## 결과
- 관련 백엔드 테스트 81개 통과
- Ruff 및 diff 검사 통과

## 다음 단계
- 런타임 프로토콜 변경 시 같은 원본 ID 유지 계약을 우선 회귀 검증
