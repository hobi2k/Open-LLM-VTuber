# CLI reasoning effort 연결

## 개요
Claude Code와 Codex CLI 설정에 reasoning effort를 추가했다. 새 대화와 기존 세션 재개 모두에서 각 CLI가 요구하는 인자로 정확히 전달되도록 구성했다.

## 주요 변경사항
- 런타임 설정 스키마와 기본 설정 템플릿에 `reasoning_effort` 추가
- Claude Code에 `--effort`, Codex에 `model_reasoning_effort` 설정 전달
- Codex 모델 캐시에서 지원 reasoning 단계 제공
- 설정 저장, 카탈로그, 새 세션 및 재개 세션 테스트 추가

## 결과
- 관련 단위 테스트 28개 통과
- Ruff 검사 및 변경 파일 포맷 검사 통과

## 다음 단계
- 향후 CLI가 새 effort 단계를 추가할 경우 카탈로그 감지 결과로 자동 노출되는지 회귀 확인
