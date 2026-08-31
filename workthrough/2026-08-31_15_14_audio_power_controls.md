# ASR/TTS 전원 및 메모리 해제

## 개요
ASR과 TTS를 설정 화면에서 독립적으로 켜고 끌 수 있게 구현했다. OFF 상태에서는 백엔드 모델, 브라우저 VAD, 마이크 스트림, 대기 중인 음성 재생 자원을 해제하며 재시작 후에도 상태가 유지된다.

## 주요 변경사항
- ASR/TTS 설정에 `enabled` 상태와 실제 엔진 `loaded` 상태 추가
- 연결된 세션이 하나의 음성 엔진을 공유하도록 재초기화 경로 정리
- OFF 시 엔진 참조, Python GC, CUDA/MPS 캐시 정리
- ASR OFF 시 마이크/VAD 차단, TTS OFF 시 텍스트 응답만 전달
- 영어, 일본어, 중국어 설정 UI와 저장 완료 대기 상태 추가

## 결과
- 백엔드 전체 테스트 60개 통과
- Ruff, ESLint, TypeScript 타입 검사 통과
- 웹 빌드와 Electron arm64 앱 빌드 성공
- 실제 TTS 생성 및 생성 음원의 Sherpa ASR 인식 성공
- ASR/TTS OFF 시 서버 RSS 약 1,259,424 KiB에서 651,776 KiB로 감소
- OFF 상태 재시작 시 RSS 약 131 MiB, ON 복원 및 Live2D 렌더링 확인

## 다음 단계
- 진행 중인 음성 인식이나 합성 작업을 전원 OFF 즉시 취소하는 작업 레지스트리 추가 검토
