# OpenCode 설정 UI와 캐릭터 프리셋 복구

## 개요
OpenCode 어댑터 설정을 웹 UI에서 확인하고 저장할 수 있게 연결했다. 누락된 Shizuku Live2D 모델 등록을 복구하고 실제 설정 저장, 캐릭터 전환, 로컬 모델 응답까지 검증했다.

## 주요 변경사항
- Agent 탭에 OpenCode 서버, 프로바이더, 모델, 에이전트, 작업 폴더와 실행 옵션 추가
- OpenCode 연결 상태 조회와 설정 저장 API 추가
- 열린 세션에 변경된 Agent 설정을 즉시 반영하고 `conf.yaml`에 저장
- `shizuku-local` 모델 등록과 모든 캐릭터 프리셋의 모델 경로 회귀 테스트 추가
- 사용자 포크의 프런트엔드 소스 및 웹 빌드 브랜치 연결

## 결과
- Python 단위 테스트 8개 통과
- React 프로덕션 웹 빌드 성공
- OpenCode 설정 UI에서 연결 상태와 저장 동작 확인
- `en_nuke_debator` 선택 시 Shizuku 모델, 모션, 텍스처 로드 확인
- OpenCode와 oMLX를 통한 실제 캐릭터 응답 완료 확인

## 다음 단계
- 로컬 `Qwen3.8-27B-oQ4e-mtp`의 약 3분대 첫 응답 시간을 줄이기 위한 oMLX 프로필 조정
- 업스트림 Live2D SDK 타입 오류를 정리해 전체 프런트 타입 검사를 통과시키기
