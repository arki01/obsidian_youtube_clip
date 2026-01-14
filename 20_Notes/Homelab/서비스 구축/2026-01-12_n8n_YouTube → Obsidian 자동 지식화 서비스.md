---
Created: "2026-01-14 16:28"
tags:
  - AI/개발
---

## 0. 개요/목적
프로젝트명 : 

목적
- 유튜브 시청을 넘어, 최소 입력으로 자동 생성되는 고품질 지식 노트를 장기 보존 가능한 구조로 축적한다. 
- 콘텐츠 소비를 지식 축적으로 전환하는 개인 자동화 파이프라인이 목적이다.

핵심 원칙
- 입력 최소화(링크만) · 결과물은 리서치 노트 수준
- Transcript만 활용(추정/허구 금지)
- 저장/동기화는 충돌 없이 안정적

---

## 1. 최종 흐름
1) Android에서 Telegram으로 유튜브 링크 공유  
2) n8n Trigger로 링크 수신  
3) Transcript로 자막/스크립트 수집  
4) GPT로 제목 및 내용 요약  
5) Obsidian 템플릿 생성  
6) Ubuntu의 Obsidian 폴더에 Markdown 파일 저장  
7) Syncthing으로 PC/Mobile 동기화

---

## 2. 저장/동기화 구조
- Single Source of Truth: /home/zoai/Obsidian
- 폴더 구조 핵심
```
Obsidian/
└─ 10_Clippings/
   └─ Youtube/
      └─ _Inbox   ← n8n 자동 저장 위치
```
- 동기화: Syncthing (HJ-VM-UBUNTU 기준 → HJ-PC-MAIN, HJ-VM-WIN, HJ-MO-EDGE)
- Git: 실시간 아님, 스냅샷/백업용(후순위)

---

## 3. 입력 UX
- Telegram 봇 1:1 채팅만 사용
- 링크만 전송하면 즉시 실행(모바일/PC 동일 UX)

---

## 4. 요약 품질 전략
- System/User Prompt 분리
- 규칙: 세부 항목 4개 이상, 각 4~6문장, 단순 요약 금지(논리 재구성)
- 예시 개수 제거로 anchor effect 방지
- 결과물은 “강의/인터뷰 리서치 노트” 수준

---

## 5. n8n 구현 메모
- Execute Command의 invalid syntax 원인: Expression 파서 충돌
- 해결: 명령어에서 Expression 제거, 역할 분리
  - n8n: 콘텐츠 생성(파일 노드)
  - Execute Command: 디렉토리 생성/파일 이동만 담당

---

## 6. 현재 상태 평가
- 실사용 가능: ✅
- 자동화 안정성: ✅
- 요약 품질: ✅
- 다기기 동기화: ✅
- 구조적 확장성: 높음
- Git 연계: ⏳(주기적 스냅샷 예정)

---

## 7. 다음 작업(Backlog)
- n8n 파일 생성 로직 최종 안정화
- Inbox → 정리 폴더 자동 분류 룰
- 태그/주제 자동 추출
- Git 스냅샷 커밋/히스토리 전략 수립

---

## 8. 한 문장 요약
“유튜브가 자동으로 내 지식 자산이 되는 시스템”을 Obsidian+Syncthing 기준으로 완성했다.