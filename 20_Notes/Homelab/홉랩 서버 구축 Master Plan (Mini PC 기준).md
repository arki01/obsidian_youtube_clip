---
Created: 2025-12-13 23:43
tags:
  - AI/개발
---

# 개요

### 0.1 구축 목적
- Mini PC 1대를 **개인 홈랩 서버**로 사용
    
- 회사/로컬 PC에서 원격 접속하여:
    
    - 개발·테스트 환경 운영
        
    - 자동화(n8n)
        
    - 개인 API 서버
        
    - 지식 관리(Obsidian 자동화)
        
- 서비스 실행은 **Docker 기반**
    
- VM은 **서버 역할**, 모바일/외부는 **클라이언트**
    
### 0.2 기본 아키텍처
```
Mini PC
 └─ Proxmox VE (Host, Linux)
     ├─ VM: Ubuntu Server (Main)
     │    ├─ Docker
     │    ├─ n8n
     │    ├─ AI 테스트 환경
     │    ├─ 개인 API 서버
     │    └─ Obsidian 자동화 (Git, Sync)
     └─ VM: Windows (Optional)
          ├─ GUI 필요 작업
          ├─ 회사 전용 툴 테스트
          └─ 필요 시에만 기동
```



# [[Phase 1. 사전 준비 단계 (Windows 유지 상태)]]
### 1.1 데이터 백업

- Windows 11 내 모든 데이터 백업 완료
    
    - Documents / Desktop / Downloads
        
    - Obsidian Vault
        
    - 개인 설정 파일
        
- 백업 위치
    
    - 외장 SSD 또는 NAS 또는 클라우드 중 1개 이상
        

### 1.2 하드웨어 및 네트워크 확인

- SSD 용량 확인 (실사용 기준 1TB)
    
- 유선 LAN 포트 사용 확인
    
- 공유기 Admin 접근 여부 확인
    
    - 현재 접근 불가 → 고정 IP 전략은 추후 진행
        

### 1.3 BIOS 설정

- Virtualization (Intel VT-x): 활성화
    
- Secure Boot: 비활성화
    
- Fast Boot: 비활성화
    
- 부팅 우선순위: USB 우선
    

---

# [[Phase 2. Windows 제거 및 Proxmox VE 설치]]

### 2.1 Windows 유지 여부 최종 결정

- Windows 11은 Host OS로 사용하지 않음
    
- Windows는 **필요 시 VM으로만 사용**
    
- Proxmox 단일 Host 구조로 확정
    

### 2.2 설치 미디어 준비

- Proxmox VE ISO 다운로드
    
- Rufus로 설치 USB 생성
    
    - GPT / UEFI
        
    - 파일시스템: FAT32
        

### 2.3 Windows 포맷 전 최종 체크

- 데이터 백업 완료 재확인
    
- Windows 라이선스 키 필요 여부 확인
    
- Proxmox 설치 시 **디스크 전체 초기화** 인지
    

---

# [[Phase 3. Proxmox 초기 정비 및 Ubuntu Server VM 생성]]

### 3.1 Proxmox 설치

- USB 부팅 후 Proxmox VE 설치
    
- 디스크 설정
    
    - 파일시스템: EXT4 (단일 디스크 기준)
        
- 네트워크 설정
    
    - DHCP 사용
        
    - 공유기 접근 가능 시 고정 IP 또는 DHCP 예약 예정
        

### 3.2 Proxmox 초기 접속 확인

- Web UI 접속
    
    `https://192.168.75.203:8006`
    
- root 로그인 확인
    
- 노드(pve) 정상 인식 확인
    

### 3.3 Proxmox 기본 세팅

- Non-subscription repository 설정
    
- 시스템 업데이트
    
- Timezone 확인 (Host 기준)
    

---

# [[Phase 4. Ubuntu Server VM 구축(Main Server) & VS Code 연결]]

### 4.1 Ubuntu Server VM 생성

- 권장 리소스 (N100 기준)
    
    - CPU: 2 Core
        
    - RAM: 8GB
        
    - Disk: 150GB (local-lvm)
        
    - Network: Bridge (vmbr0, DHCP)
        

### 4.2 Ubuntu 기본 설정

- OpenSSH 설치
    
- 사용자 계정 생성
    
- 시스템 업데이트
    
- Time Zone 설정
    
    - Asia/Seoul
        
- NTP 동기화 정상 확인
    

### 4.3 원격 접속 테스트

- 로컬 / 회사 PC → SSH → Ubuntu VM
    
- SSH 키 기반 인증 적용
    
- 비밀번호 로그인 차단
    
- 내부 IP 기준 접속 정상 확인
    

### 4.4 Proxmox 연동 및 기본 보안

- QEMU Guest Agent 설치
    
    - Proxmox Summary에서 VM IP 표시 확인
        
- UFW 활성화
    
    - incoming deny
        
    - outgoing allow
        
    - SSH(22/tcp)만 허용
        
- 고정 IP
    
    - 공유기 Admin 접근 불가로 **DHCP 유지**
        
    - 추후 설정 예정
    

---

# [[Phase 5. Docker 기반 서비스 환경 구축]]

### 5.1 기본 도구 설치

- Docker Engine
    
- Docker Compose
    
- Portainer
    

### 5.2 1차 핵심 서비스

- Portainer (컨테이너 관리)
    
- n8n (자동화)
    
- Git 기반 저장소 연동 (GitHub)
    
- 파일 동기화 도구
    
    - Syncthing 또는 Nextcloud
        

---

# Phase 6. Obsidian 연계 구조 구축

### 6.1 기본 개념

- Obsidian은 회사 PC(Windows)에서 사용
    
- 서버는:
    
    - Vault 저장
        
    - 버전 관리
        
    - 자동화 및 AI 처리 담당
        

### 6.2 권장 구조

`회사 PC (Obsidian)  └─ Git / Sync       ↓ Ubuntu Server  └─ Vault + 자동화 + AI 처리`

### 6.3 서버 자동화 방향

- 문서 요약
    
- 태깅 자동 생성
    
- 기획 초안 생성
    
- AI Agent 연계
    

---

# Phase 7. 원격 접속 및 보안 구성

### 7.1 원격 접속 전략

- 포트포워딩 최소화
    
- 회사 네트워크 제약 고려
    

### 7.2 권장 솔루션

- Tailscale (1순위)
    
- WireGuard (대안)
    

### 7.3 접근 흐름
회사 PC
 └─ VPN (Tailscale)
      ↓
Ubuntu VM
      ↓
Docker 서비스


---

# Phase 8. Windows VM 구축 (선택 사항)

### 8.1 Windows VM 사용 목적

- GUI 필수 툴
    
- 회사 전용 프로그램 테스트
    
- 브라우저 격리 환경
    

### 8.2 운영 원칙

- 기본 상태: OFF
    
- 필요 시에만 ON
    
- 스냅샷 적극 활용
    
- 리소스 최소 할당
    

---

# Phase 9. 운영 및 확장 단계

### 9.1 백업 전략

- Proxmox VM Snapshot
    
- 중요 데이터 외부 백업
    

### 9.2 모니터링

- 리소스 사용량
    
- 디스크 여유 공간
    
- 서비스 상태
    

### 9.3 향후 확장

- 추가 VM / LXC
    
- AI 서비스 고도화
    
- 회사 PoC 환경 미러링
    

# 요약

- Windows는 **출발점**, Linux는 **목적지**
    
- Proxmox는 서버의 뼈대
    
- Ubuntu VM은 실제 작업 공간
    
- Obsidian은 지식 인터페이스
    
- 이 구조는 개인용이면서도 **회사에 설명 가능한 서버 아키텍처**

# 출처


# 관련 노트




