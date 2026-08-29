# 에이전트 런타임 선택기

## 개요
OpenCode 전용 설정을 OpenCode, Claude Code, Codex, Hermes 공통 런타임 설정으로 확장했다. 웹과 Electron 앱에서 런타임을 선택하고 설정을 저장하면 현재 대화 엔진에 즉시 반영된다.

## 주요 변경사항
- 개발한 것: 네 런타임 선택 UI와 상태 검사 API
- 개발한 것: Claude Code, Codex, Hermes one-shot CLI 어댑터
- 수정한 것: 연결 확인 시 저장하지 않은 런타임 선택과 입력값이 초기화되던 문제
- 개선한 것: Claude 도구 비활성화, Codex 읽기 전용, Hermes 안전 모드 실행
- 개선한 것: 영문/중문 기본 설정과 기존 설정 업그레이드 호환성
- 개선한 것: 타임아웃 및 취소 시 CLI 자식 프로세스 정리
- 개선한 것: Electron 44 업그레이드, 렌더러 샌드박스 활성화, Node 통합 비활성화
- 개선한 것: 고유 앱 ID와 완전한 로컬 ad-hoc 번들 서명 적용

## 결과
- 백엔드 테스트 13개 통과, Ruff 검사 및 포맷 통과
- OpenCode, Claude Code, Codex, Hermes 실제 설치 상태 확인
- 네 런타임 설정 API 선택 및 저장 전환 통과
- Claude Code, Codex, Hermes 실제 응답 `OK` 확인
- 웹 프로덕션 빌드, Electron 빌드, macOS arm64 앱 패키징 성공
- 실제 설정 UI에서 네 런타임 선택, 연결 확인, 취소 시 원복 검증
- 프로덕션 npm 의존성 취약점 0건, macOS 엄격 코드 서명 검증 통과
- 패키지 렌더러의 Chromium 샌드박스 적용 확인

## 다음 단계
- 기존 Live2D SDK TypeScript 오류를 별도 정리
- 외부 배포 전 Apple Developer ID 인증서로 서명하고 Apple 공증 완료
