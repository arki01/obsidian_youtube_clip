---
Created: 2025-12-14 09:33
tags:
  - AI/개발
---

# 개요

Phase 3는 Proxmox 설치가 완료된 이후,  **실제 사용을 위한 기본 정비**와  **메인 작업 서버가 될 Ubuntu Server VM을 생성하는 단계**이다.

이 단계의 목표는 다음과 같다.

- Proxmox 호스트를 안정적인 상태로 정비
    
- VM 생성에 필요한 기본 리소스 구조 확립
    
- Ubuntu Server VM을 생성하고 네트워크 연결까지 확인
    
# 내용

## 3.1 Proxmox 초기 정비 (Host 레벨)

### 3.1.1 Proxmox 웹 UI 접속

- 브라우저에서:
    
    https://192.168.75.203:8006
    
- 계정:
    
    - Username: root
        
    - Realm: PAM
        
    - Password: skan097942
        

---

### 3.1.2 저장소(Repository) 설정

#### Enterprise 저장소 비활성화

- 경로:
    
    `Datacenter → Node(pve) → Updates → Repositories`
    
- `enterprise` 저장소 비활성화
    

#### Community 저장소 활성화

- `No-Subscription` 저장소 활성화
    
- 개인 홈랩 환경에서 표준적인 선택
    

---

### 3.1.3 시스템 업데이트

- 경로:
    
    `Node(pve) → Updates`
    
- 패키지 업데이트 실행
    
- 커널 업데이트가 포함된 경우 재부팅 수행
    

---

### 3.1.4 시간 및 NTP 확인

- 경로:
    
    `Node(pve) → System → Time`
    
- Timezone: Asia/Seoul
    
- NTP 동기화 정상 여부 확인
    

---

## 3.2 스토리지 구조 점검

### 3.2.1 기본 스토리지 확인

- 경로:
    
    `Datacenter → Storage`
    
- 기본적으로 다음 스토리지가 존재해야 한다.
    
    - `local`
        
    - `local-lvm` (또는 ZFS pool)
        

---

### 3.2.2 ISO 이미지 업로드 위치

- Ubuntu Server ISO는 `local` 스토리지에 업로드
    
- Content Type:
    
    - ISO Image 체크 여부 확인
        

---

## 3.3 Ubuntu Server ISO 준비

### 3.3.1 ISO 다운로드

- Ubuntu Server LTS 버전 권장
    
    - 22.04 LTS 또는 24.04 LTS
        

### 3.3.2 ISO 업로드

- 경로:
    
    `Node(pve) → local → ISO Images → Upload`
    

---

## 3.4 Ubuntu Server VM 생성

### 3.4.1 VM 생성 시작

- Proxmox 웹 UI 우측 상단:
    
    `Create VM`
    

---

### 3.4.2 General

- VM ID: 자동 또는 수동 지정
    
- Name: `ubuntu-main` (예시)
    

---

### 3.4.3 OS

- ISO Image:
    
    - 업로드한 Ubuntu Server ISO 선택
        
- Guest OS:
    
    - Linux
        
    - Version: 6.x - 2.6 Kernel
        

---

### 3.4.4 System

- BIOS: UEFI
    
- Machine: q35
    
- SCSI Controller: VirtIO SCSI
    
- QEMU Agent: 체크 (중요)
    

---

### 3.4.5 Disks

- Bus/Device: SCSI
    
- Storage: local-lvm
    
- Disk Size: 100~200GB 권장
    
- Cache: Default
    

---

### 3.4.6 CPU

- Cores: 2~3
    
- Type: host
    

---

### 3.4.7 Memory

- RAM: 8GB
    
- Ballooning: 체크 해제 권장 (초기 안정성)
    

---

### 3.4.8 Network

- Bridge: vmbr0
    
- Model: VirtIO (paravirtualized)
    
- Firewall: 필요 시 나중에 설정
    

---

### 3.4.9 Confirm

- 설정 확인 후 Create
    

---

## 3.5 Ubuntu Server 설치

### 3.5.1 VM 시작

- 생성된 VM 선택
    
- Start 클릭
    
- Console 열기
    

---

### 3.5.2 Ubuntu 설치 주요 선택

- Language: English (권장)
    
- Keyboard: 실제 키보드 레이아웃에 맞게
    
- Network: DHCP (기본)
    
- Storage: Use entire disk
    
- Profile: Minimal Install
    
- OpenSSH Server: 체크
    
- 사용자 계정 생성
    

---

### 3.5.3 설치 완료

- 설치 완료 후 재부팅
    
- ISO 제거 또는 부팅 순서 조정
    

---

## 3.6 Ubuntu VM 초기 점검

### 3.6.1 SSH 접속 확인

- 회사 PC 또는 로컬 PC에서:
    
    `ssh <user>@<Ubuntu_VM_IP>`
    

### 3.6.2 네트워크 확인

`ip a`

---

## 3.7 Phase 3 완료 기준

- Proxmox Host 업데이트 완료
    
- Ubuntu Server VM 정상 부팅
    
- vmbr0를 통해 외부 네트워크 접속 가능
    
- SSH 접속 성공
    
# 요약

Phase 3는 Proxmox를 “쓸 수 있는 서버” 상태로 만드는 단계이다.  
이 단계가 끝나면:

- Proxmox는 안정적인 하이퍼바이저로 준비 완료
    
- Ubuntu Server VM이 메인 작업 서버로 준비 완료
    
- 이후 단계(Docker, n8n, AI 환경)로 자연스럽게 연결된다

이어서 [[Phase 4. Ubuntu Server VM 구축(Main Server) & VS Code 연결]] 단계로 넘어간다.

# 출처


# 관련 노트

