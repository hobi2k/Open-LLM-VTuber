# OpenCode 어댑터 연동

## 개요
Open-LLM-VTuber가 Ollama나 개별 API 키 대신 실행 중인 OpenCode 서버의 provider와 model을 사용할 수 있도록 전용 스트리밍 어댑터를 추가했다. 새 포크를 다시 설치하고 로컬 oMLX 모델로 실제 대화까지 검증했다.

## 주요 변경사항
- `opencode_llm` 제공자와 Pydantic 설정, 팩토리 연결 추가
- OpenCode SSE에서 추론을 제외한 답변 텍스트만 실시간 전달
- 대화 이력과 이미지 입력 전달, Basic Auth, 타임아웃 지원
- 기본 도구 권한 차단과 일회용 세션 자동 삭제 적용
- 프로젝트용 `vtuber` OpenCode 에이전트 및 사용 문서 추가
- HTTP/SSE 가짜 서버 기반 회귀 테스트 4개 추가

## 결과
- Ruff 저장소 전체 검사 통과
- 영문·중문 설정 템플릿 전체 검증 통과
- 실제 OpenCode + oMLX 모델 토큰 스트리밍 통과
- 브라우저 입력부터 Live2D 화면, TTS, 응답 표시까지 통합 검증 통과

## 다음 단계
- 필요하면 OpenCode와 VTuber 서버를 함께 시작하는 런처 추가
- OpenCode provider/model 목록을 UI에서 선택하는 설정 화면 추가
