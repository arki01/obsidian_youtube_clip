---
Created: 2025-12-13 23:50
tags:
  - AI/개발
---

# 개요

본 단계는 현재 **Windows 11이 설치된 Mini PC**를  
향후 **Proxmox 기반 홈랩 서버**로 전환하기 전,  
되돌릴 수 없는 작업(포맷, OS 교체)에 앞서 **환경·데이터·하드웨어를 정리하는 단계**이다.

이 단계의 목표는 다음과 같다.

- 데이터 유실 방지
    
- 하드웨어 및 네트워크 제약 요소 사전 제거
    
- 서버 전환에 대한 명확한 의사결정
# 내용

### 1. 현재 상태 정의 및 기록

- Mini PC 기본 정보 기록
	- 장치 이름	Hyungjuns-MiniPC
	- 프로세서	Intel(R) N100   800 MHz
	- 설치된 RAM	16.0GB(15.8GB 사용 가능)

- 해당 정보는 Obsidian에 기준점(Baseline)으로 저장
    
- 향후 재설치, 장애 대응, 회사 아키텍처 설명 시 활용
    

---

### 2. 데이터 백업

- 백업 대상
    
    - Documents / Desktop / Downloads
        
    - 개인 프로젝트 폴더
        
    - Obsidian Vault
        
    - API Key, 환경설정, 스크립트
        
- Obsidian Vault는 **폴더 전체 단위로 백업**
    
- 백업 위치는 최소 1곳 이상
    
    - 외장 SSD, NAS, 클라우드 중 선택
        
- 백업 후 다른 PC에서 정상 열림 여부 검증
    

---

### 3. 디스크 및 저장공간 점검

- SSD 총 용량, 사용량, 여유 공간 확인
    
- 홈랩 기준 예상 사용량 검토
    
    - Proxmox Host
        
    - Ubuntu Server VM
        
    - Windows VM(선택)
        
- SSD 용량이 부족한 경우:
    
    - Windows VM 생략 또는 외장 스토리지 고려
        

---

### 4. 네트워크 환경 점검

- 유선 LAN 사용 가능 여부 확인
    
- 공유기 관리자 접근 가능 여부 확인
	- ID : admin
	- PW : 567620_admin
	    
- Proxmox Host에 고정 IP 또는 DHCP 예약 가능 여부 확인
    
- 현재 IP 및 네트워크 정보 기록
	- 링크-로컬 IPv6 주소 . . . . : fe80::ab45:daaa:3ab1:6133%7
	- IPv4 주소 . . . . . . . . . : 192.168.75.203
	- 서브넷 마스크 . . . . . . . : 255.255.255.0
	- 기본 게이트웨이 . . . . . . : 192.168.75.1
	    

---

### 5. BIOS 설정 확인

- BIOS 진입 키 확인
    
- 필수 설정
    
    - Intel VT-x (Virtualization) 활성화
        
    - Secure Boot 비활성화
        
    - Fast Boot 비활성화
        
    - UEFI 모드 유지
        
- 설정 저장 후 Windows 정상 부팅 확인
    

---

### 6. Proxmox 설치 준비

- 다른 PC에서 Proxmox VE ISO 다운로드
    
- Rufus를 이용한 설치 USB 생성 (https://m.blog.naver.com/17beans/221333143809)
    
    - GPT / UEFI 기반
        
- USB를 Mini PC에 연결한 상태 유지
    

---

### 7. Windows 제거 전 최종 점검

- Windows는 향후 Host OS로 사용하지 않음
    
- Windows 환경은 필요 시 VM으로 대체
    
- 디스크 전체 초기화에 대한 인지 확인
    
- 다음 항목이 모두 충족되어야 다음 단계 진행
    
    - 데이터 백업 완료
        
    - BIOS 가상화 설정 완료
        
    - 설치 USB 준비 완료
        
    - 네트워크 환경 확인 완료

# 요약

- 본 단계는 **홈랩 구축을 위한 마지막 안전 구간**이다.
    
- 이 단계가 끝나면 Mini PC는:
    
    - 개인용 PC → 서버 장비로 전환될 준비가 완료된다.
        
- 핵심 체크 포인트는 다음과 같다.
    
    - 데이터는 모두 백업되었는가
        
    - 가상화 및 네트워크 제약은 제거되었는가
        
    - Windows를 더 이상 메인 OS로 사용하지 않아도 되는가
        

위 조건이 모두 충족되면  
다음 단계인 [[Phase 2. Windows 제거 및 Proxmox VE 설치]]로 진행한다.

# 출처


# 관련 노트


