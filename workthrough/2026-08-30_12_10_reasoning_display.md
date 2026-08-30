# 에이전트 Reasoning 표시 옵션

## 개요
OpenCode, Claude Code, Codex, Hermes가 제공하는 네이티브 reasoning을 최종 답변과 분리해 채팅에 표시하는 옵션을 추가했다. 옵션을 끄면 기존 채팅 UI로 돌아가며 provider의 reasoning 설정과 세션 기록은 변경하지 않는다.

## 주요 변경사항
- 개발한 것: provider별 reasoning 이벤트 파싱과 WebSocket 전달
- 개발한 것: 항상 펼쳐진 reasoning 영역과 설정 스위치
- 수정한 것: Hermes의 강제 `reasoning none` 제거 및 네이티브 세션 DB 연동
- 수정한 것: 중단/종료 시 진행 상태와 `Thinking...` 자막 정리
- 개선한 것: reasoning을 VTuber 대화 메모리와 히스토리 미리보기에서 분리

## 결과
- 백엔드 관련 테스트 26개 통과
- Ruff, 프런트 TypeScript, ESLint, diff 검사 통과
- 웹 및 Electron production 빌드 성공
- Electron에서 OpenCode ON/OFF 실응답과 설정 재실행 지속성 확인

## 다음 단계
- Claude Code, Codex, Hermes 실제 계정 환경을 포함한 선택형 E2E 테스트 자동화
- provider CLI 출력 형식 변경을 감지하는 호환성 테스트 추가
