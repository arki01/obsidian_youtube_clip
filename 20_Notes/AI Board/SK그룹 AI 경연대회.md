---
Created: "2025-09-07 18:12"
tags:
---

# 개요


# 내용

## 구현 방향성

### Phase A: 통합 뼈대 완성(현 로직 유지 중심)
- A1: TargetManager v0(가까운 미청정 룸) + Room Locator + Clean Trigger
- A2: GlobalPlanning v0(직선 fallback) + Follower(PP) + Safety Stop(현행)
- A3: DoD: “모든 룸을 한 번씩 방문해 CLEAN 시도 후 도킹 복귀”가 평균 2/3회 성공

### Phase B: 경로 안정화
- B1: A* + 벽 인플레이션(1~2셀) 도입, 웨이포인트 기반 PP 추종
- B2: Command Clamp(가감속 제한) & Stall Recovery(짧은 회전/후진)
- B3: DoD: 무충돌 3/3, 문/가구 코너에서 정체 없음

###  Phase C: 동적 회피 & 스케줄링
- C1: DWA-lite(20개 내외 샘플)로 인접 동적장애물 회피
- C2: Clean Scheduler v1(점수 기반: 오염–거리–리스크)
- C3: DoD: 동적 장애물 시 완전정지 빈도↓, 총 소요시간↓, 청정 완료율 3/3

# 출처


# 관련 노트



