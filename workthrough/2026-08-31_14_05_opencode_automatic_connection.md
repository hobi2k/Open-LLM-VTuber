# OpenCode 자동 연결과 ASR 설치 복구

## 개요
OpenCode Desktop 및 `omlx launch opencode` 환경에서 사용자가 포트를 입력하지 않아도 실행 중인 로컬 OpenCode 서버를 탐색하도록 개선했다. 접근 가능한 서버가 없으면 앱 전용 보조 서버를 자동으로 시작하고 앱 종료 시 함께 정리한다.

## 주요 변경사항
- OpenCode 리스너 자동 탐지와 관리형 보조 서버 수명 주기 추가
- 탐지된 URL을 모델, 프로젝트, 세션 조회에 즉시 반영
- 설정 화면에서 실제 연결 URL 표시
- 중단된 SenseVoice 다운로드와 압축 해제를 안전하게 복구

## 결과
- 백엔드 테스트 51개 통과
- Ruff, 프런트엔드 타입 검사, 린트, 웹 및 macOS 앱 빌드 통과
- 실제 앱에서 OpenCode 109개, Claude 20개, Codex 3,451개, Hermes 16개 세션 조회 확인
- 앱 종료 시 관리형 OpenCode 서버 정리 확인

## 다음 단계
- 다른 CLI가 공식 서버 탐색 API를 제공하면 프로세스 탐지를 해당 API로 보강
