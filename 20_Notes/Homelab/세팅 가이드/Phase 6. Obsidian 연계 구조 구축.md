---
Created: 2025-12-14 15:52
tags:
  - AI/개발
---

# 개요


Phase 6의 목표는 **Obsidian을 ‘PC 하나에 종속된 노트 앱’이 아니라  
서버 중심의 지식 생산 시스템으로 확장**하는 것이다.

이 Phase에서의 핵심 전략은 다음과 같다.

- **지식 생산의 중심은 개인 PC**
    
- **지식 저장·자동화·AI 처리는 서버**
    
- **회사 PC는 지식을 “직접 쓰는 곳”이 아니라  
    결과를 활용하는 “접근 창(Client)” 역할만 수행**
    

즉,

- 개인 PC + 서버 = **지식 생산 시스템**
    
- 회사 PC = **지식 활용 인터페이스**
# 내용

## 6.1 전체 역할 분리 전략 (가장 중요)

### 6.1.1 개인 PC (Primary Authoring Environment)

- Obsidian 주 사용 환경
    
- 모든 노트 작성/편집은 개인 PC에서 수행
    
- Obsidian 플러그인 적극 활용
    
    - Templater
        
    - Dataview
        
    - Buttons
        
    - (향후) Custom Command
        
- AI 요청의 “트리거” 역할 수행
    

👉 **사람이 직접 생각하고 쓰는 공간**

---

### 6.1.2 서버 (Knowledge Core)

- Vault의 실제 저장 위치
    
- Git 저장소
    
- n8n 자동화 엔진
    
- AI API 연계(OpenAI 등)
    
- 문서 가공/요약/태깅/기획 생성 담당
    

👉 **지식의 두뇌 + 자동 손**

---

### 6.1.3 회사 PC (Client / Viewer)

- Obsidian 설치 필수 아님
    
- Vault 직접 편집 ❌
    
- 역할:
    
    - 서버에 접속
        
    - 자동화 결과 확인
        
    - AI 생성 문서 열람
        
    - 회사 문서로 복사/활용
        

👉 **지식을 생산하지 않고, 활용만 하는 창**

---

## 6.2 권장 전체 아키텍처

```
[개인 PC]
Obsidian (작성/편집)
 └─ Sync (Syncthing or Git)
      ↓
[Ubuntu Server]
Vault (SSOT)
 + Git
 + n8n
 + AI 처리
      ↑
[회사 PC]
웹 / SSH / VS Code
(결과 열람·활용)
```

---

## 6.3 개인 PC ↔ 서버 Vault 연계

### 6.3.1 Vault 기준 위치 (서버)

```text
/home/hyungjun/homelab/data/vault
```

- 서버가 **Single Source of Truth**
    
- 개인 PC Vault는 이 경로와 동기화된 복제본
    

---

### 6.3.2 동기화 방식 (권장)

#### 1순위: Syncthing

- 개인 PC ↔ 서버 간 실시간 동기화
    
- Git보다 충돌 관리에 유리
    
- 회사 PC는 동기화 대상에서 제외
    

```
개인 PC ↔ Syncthing ↔ 서버
회사 PC ❌
```

#### Git의 역할 (보조)

- 서버에서만 사용
    
- 변경 이력 관리
    
- n8n 자동화 트리거 용도
    

---

## 6.4 Obsidian 내부 “AI 트리거” 설계

### 6.4.1 기본 개념

- Obsidian 자체에서 AI를 직접 실행하지 않음
    
- Obsidian은:
    
    - “요청 신호”만 생성
        
- 서버(n8n)가:
    
    - 실제 AI 처리 수행
        
    - 결과를 Vault에 기록
        

---

### 6.4.2 예시 트리거 방식

- 특정 태그 추가
    
    - `#ai/summary`
        
    - `#ai/tagging`
        
- 특정 폴더 이동
    
    - `00_Inbox/`
        
- 버튼 클릭
    
    - “요약 생성”
        
    - “기획 초안 생성”
        

---

## 6.5 n8n 기반 자동화 시나리오

### 6.5.1 문서 요약 자동화

- 트리거:
    
    - `00_Inbox` 신규 문서
        
- 처리:
    
    - Markdown 파싱
        
    - AI 요약
        
- 결과:
    
    - 동일 문서 하단에 `## 요약` 자동 삽입
        

---

### 6.5.2 태그 자동 생성

- 입력:
    
    - 문서 본문
        
- 처리:
    
    - 키워드 추출
        
    - 사전 정의 태그 체계 매핑
        
- 결과:
    
    - Frontmatter에 `tags:` 추가
        

---

### 6.5.3 기획 초안 생성

- 입력:
    
    - 여러 노트 선택
        
- 처리:
    
    - n8n에서 병합
        
    - 구조화 프롬프트 적용
        
- 결과:
    
    - `Draft_*.md` 신규 생성
        

---

## 6.6 회사 PC 사용 방식 (중요 정책)

### 6.6.1 회사 PC에서 하지 않는 것

- Vault 직접 편집 ❌
    
- Obsidian 필수 설치 ❌
    
- Syncthing/Git 동기화 ❌
    

---

### 6.6.2 회사 PC에서 하는 것

- 서버 접속
    
    - 브라우저 (n8n UI)
        
    - VS Code Remote-SSH
        
    - SSH
        
- AI 자동화 결과 열람
    
- Markdown 문서 복사 → 회사 문서 활용
    

👉 **회사 보안 정책과 충돌 최소화**

---

## 6.7 확장 여지 (Phase 9 이후)

- 개인 AI Agent
    
- RAG 기반 Vault 검색
    
- 회사 PoC용 지식 자동 정리
    
- 프로젝트 단위 Agent 분리
# 요약

- Obsidian 주 사용자는 **개인 PC**
    
- 서버는:
    
    - Vault 저장소
        
    - 자동화 허브
        
    - AI 처리 담당
        
- 회사 PC는:
    
    - 지식 생산 ❌
        
    - 결과 활용 ✅
        
- Obsidian은 AI를 “호출”만 하고
    
- 실제 AI 처리는 **n8n + 서버**에서 수행
    
- 개인용이면서도 **회사 보안·설명에 모두 안전한 구조**

# 출처


# 관련 노트





    
