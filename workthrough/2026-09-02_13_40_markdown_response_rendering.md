# AI 응답 Markdown 렌더링 배포

## 개요
프런트엔드의 AI 응답 Markdown 렌더링 배포본을 백엔드 저장소와 최종 macOS 앱에 반영했다. 웹 서브모듈과 앱·DMG 산출물을 같은 빌드 기준으로 맞췄다.

## 주요 변경사항
- 수정한 것: `frontend` 서브모듈을 Markdown 응답 배포 커밋으로 갱신
- 배포한 것: Apple Silicon 앱과 arm64/x64 DMG 재생성 및 release 폴더 교체
- 검증한 것: 앱 ad-hoc 서명, 아키텍처, DMG 무결성, 웹 Pages 배포

## 결과
- 채팅·추론·펫모드 Markdown 표시 반영
- 로컬 최종 앱과 원격 저장소 동기화

## 다음 단계
- 정식 외부 배포 시 Apple Developer ID 서명 및 notarization 적용
