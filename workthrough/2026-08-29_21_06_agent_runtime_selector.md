# 에이전트 런타임 선택기

## 개요
OpenCode 전용 설정을 OpenCode, Claude Code, Codex, Hermes 공통 런타임 설정으로 확장했다. 실행 파일, 모델, 프로젝트, 기존 세션을 자동으로 찾아 웹과 Electron 앱에서 선택하고 현재 대화 엔진에 즉시 반영할 수 있다.

## 주요 변경사항
- 개발한 것: 네 런타임 선택 UI, 상태 검사 API, 자동 탐지 카탈로그 API
- 개발한 것: CLI 실행 파일 자동 탐지와 모델·프로젝트·세션 선택기
- 개발한 것: OpenCode와 Hermes의 Direct CLI/oMLX 실행 모드
- 개발한 것: Claude Code, Codex, Hermes 연속 대화 세션 어댑터
- 개발한 것: Electron 네이티브 프로젝트 폴더 선택기
- 개발한 것: 영어·중국어와 동일한 범위를 갖는 일본어 UI 및 언어 선택
- 개발한 것: 실행 파일·프로바이더·모델·프로젝트·세션을 목록에서 고르거나 직접 입력할 수 있는 편집형 선택기
- 수정한 것: 연결 확인 시 저장하지 않은 런타임 선택과 입력값이 초기화되던 문제
- 수정한 것: OpenCode 실행 파일이 서버 연결 상태와 별개로 자동 탐지·직접 경로 오류를 표시하도록 분리
- 수정한 것: VAD의 ONNX Runtime 자산 경로가 `/assets/libs/`로 잘못 해석되고 `.mjs` 로더가 누락되던 문제
- 수정한 것: Live2D 설정 도착 전에 `/undefined/undefined.model3.json`을 요청하던 초기화 순서 문제
- 수정한 것: WebGL2가 없을 때 WebGL1으로 폴백하고 차단 팝업 대신 안전하게 초기화를 중단하도록 개선
- 수정한 것: Electron preload가 샌드박스에서 `@electron-toolkit/preload`을 찾지 못하던 패키징 문제
- 수정한 것: Electron 종료 시 Live2D 렌더 루프가 해제된 객체를 다시 사용하던 경쟁 조건
- 수정한 것: 정상 중단된 OpenCode 요청을 일반 응답 실패로 기록하고 오류 문구를 내보내던 문제
- 수정한 것: Hermes safe-mode가 사용자 oMLX 공급자 설정을 무시해 401을 내던 문제
- 개선한 것: Claude 도구 비활성화, Codex 읽기 전용, Hermes 규칙·도구 비활성화
- 개선한 것: OpenCode의 agent 입력을 의미가 분명한 OpenCode profile 고급 설정으로 정리
- 개선한 것: 영문/중문 기본 설정과 기존 설정 업그레이드 호환성
- 개선한 것: 타임아웃 및 취소 시 CLI 자식 프로세스 정리
- 개선한 것: Electron 44 업그레이드, 렌더러 샌드박스 활성화, Node 통합 비활성화
- 개선한 것: 고유 앱 ID와 완전한 로컬 ad-hoc 번들 서명 적용
- 개선한 것: OriginKit의 조밀한 도구형 구성을 참고해 설정 패널, 탭, 상태 표시, 세그먼트 컨트롤의 대비와 정보 밀도 정리

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
- 일본어·영어·중국어 번역 키 일치, 변경 파일 ESLint 및 백엔드 집중 테스트 14개 통과
- 네 CLI 자동 탐지, OpenCode 잘못된 직접 경로 오류, 사용자 지정 프로바이더·모델·세션·실행 파일 입력 확인
- 일본어 브라우저 UI에서 OpenCode 기존 세션 선택, 네 런타임 전환, 취소 시 설정 미저장을 실제 조작으로 확인
- 최신 웹 번들 및 macOS arm64 Electron 앱 재빌드 성공
- 웹에서 Silero 모델, ONNX `.mjs`·WASM, AudioWorklet 요청 200 및 실제 마이크 활성화 확인
- Electron 패키지 내부의 ONNX `.mjs` 4개·WASM 4개 포함과 오디오 프로세스 실행 확인
- 프런트엔드 Node/Web TypeScript 검사 0 오류, 변경 파일 ESLint 및 웹 빌드 통과
- 웹 초기 로드 5회에서 잘못된 Live2D 요청·VAD 오류·WebGL 팝업 없음 확인
- Electron preload, VAD, Live2D 실제 시작과 앱 종료 후 Uncaught 오류 없음 확인
- OpenCode 정상 중단 회귀를 포함한 런타임 백엔드 테스트 22개 통과

## 다음 단계
- 외부 배포 전 Apple Developer ID 인증서로 서명하고 Apple 공증 완료
