---
Created: 2025-12-14 11:28
tags:
  - AI/개발
---

# 개요

Phase 4의 목표는 **Ubuntu VM(lab-main)을 “항상 켜져 있는 메인 서버”로 완성**시키는 것이다.  
설치 직후 상태는 “OS만 설치된 상태”라서, 원격접속/업데이트/에이전트/보안/스토리지/도커 기반 운영까지 한 번에 정리해두는 게 핵심이다.

# 4.1 설치 완료 직후 체크

## 4.1.1 ISO 분리(부팅 루프 방지)

설치가 끝나고 `Reboot now`를 누른 뒤, VM이 다시 설치 화면으로 돌아오면 ISO가 여전히 연결된 상태다.

- Proxmox에서 해당 VM 선택
    
- `Hardware → CD/DVD Drive`
    
    - `Do not use any media`로 변경하거나
        
    - ISO 연결 해제
        
- VM 재부팅
    

## 4.1.2 VM 콘솔 로그인 확인

- 설치 시 만든 계정으로 로그인
    
- `hostname` 확인
    
    ```
    hostname
    ```
    
    기대값: `lab-main` (네가 설정한 값)
    

## 4.1.3 IP 확인

```
ip a
```

또는

```
hostname -I
```

이 IP가 이후 SSH 접속 주소가 된다.

---

# 4.2 기본 업데이트 및 필수 패키지

## 4.2.1 패키지 업데이트

```
sudo apt update
sudo apt -y upgrade
sudo apt -y autoremove
```

## 4.2.2 기본 유틸 설치(운영 편의)

```
sudo apt -y install curl wget git vim htop unzip ca-certificates gnupg lsb-release
```

---

# 4.3 SSH 원격 접속 안정화

## 4.3.1 SSH 서비스 상태 확인

```
sudo systemctl status ssh
```

Active(running)면 정상.

## 4.3.2 로컬 PC에서 접속 테스트

윈도우 PowerShell에서:

```
ssh hyungjun@192.168.75.81
```

## 4.3.3 SSH 키 기반 로그인 전환(권장)

로컬 PC에서 키 생성:

```
ssh-keygen
```

키 복사:

```
ssh-copy-id hyungjun@<VM_IP>
```

윈도우에서 `ssh-copy-id`가 없으면 아래 대체:

```
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh hyungjun@<VM_IP> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

## 4.3.4 비밀번호 로그인 차단(키 설정 후)

서버에서:

```
sudo nano /etc/ssh/sshd_config
```

다음 값 확인/수정:

```
PasswordAuthentication no
PermitRootLogin no
```

반영:

```
sudo systemctl restart ssh
```

---

# 4.4 Proxmox Qemu Guest Agent 설치(강력 권장)

## 4.4.1 Ubuntu에서 설치

```
sudo apt -y install qemu-guest-agent
sudo systemctl enable --now qemu-guest-agent
```

## 4.4.2 Proxmox에서 확인

- VM Summary에서 IP가 표시되거나
    
- Shutdown이 안정적으로 동작하면 정상
    

---

# 4.5 고정 IP 전략(지금은 선택)

## 4.5.1 권장 방식: 공유기 DHCP 예약

- VM의 MAC 주소를 기준으로
    
- 공유기에서 IP를 고정(예약)하는 방식
    
- Ubuntu 설정 건드릴 필요가 없음
    

Proxmox에서 MAC 확인:

- VM → Hardware → Network Device
    

## 4.5.2 Ubuntu 내부 고정 IP는 언제?

- 공유기 접근이 불가능하거나
    
- 서버망에서 정책상 필요할 때만
    

---

# 4.6 시간 동기화(NTP) 확인

```
timedatectl
```

확인 포인트:

- `Time zone: Asia/Seoul`
    
- `System clock synchronized: yes`
    

---

# 4.7 방화벽(UFW) 기본 정책(권장)

## 4.7.1 UFW 활성화

```
sudo apt -y install ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw enable
sudo ufw status
```



# 4.8 VS Code 기반 원격 작업 환경 구축 (권장)

## 4.8.1 목적 및 위치 정의

이 단계의 목적은 **Ubuntu VM을 “터미널로만 접속하는 서버”가 아니라  
VS Code를 통해 효율적으로 관리·운영 가능한 작업 환경으로 전환**하는 것이다.

- Phase 5부터는:
    
    - docker-compose.yml 작성
        
    - Git 기반 설정 관리
        
    - 자동화 스크립트 수정
        
    - 로그 확인 및 디버깅
        
- 따라서 **브라우저 SSH / 단순 터미널은 생산성에 한계**가 있음
    

이 단계는:

- 서버에는 **GUI를 설치하지 않고**
    
- 로컬 PC(회사/개인)를 **클라이언트**
    
- Ubuntu VM은 **Headless Server**로 유지하는 구조를 완성한다.
    

---

## 4.8.2 로컬 PC(VS Code) 준비

### 4.8.2.1 VS Code 설치

- [https://code.visualstudio.com](https://code.visualstudio.com/)
    
- 회사 PC / 개인 PC 모두 설치 가능
    
- 설치 시 기본 옵션 그대로 진행
    

### 4.8.2.2 필수 Extension 설치

VS Code Extension 탭에서 아래 항목 설치:

- **Remote - SSH** (Microsoft)
    
- **Remote Explorer** (선택)
    
- **YAML** (docker-compose 작성용)
    
- **Docker** (Phase 5 이후 사용)
    
- **GitLens** (선택, Git 이력 가시화)
    

---

## 4.8.3 VS Code → Ubuntu VM 원격 연결

### 4.8.3.1 SSH 연결 정보 등록

VS Code에서:

- `Ctrl + Shift + P`
    
- `Remote-SSH: Add New SSH Host`
    
- 아래 형식으로 입력
    

```
ssh hyungjun@192.168.75.81
```

- SSH config 파일 위치 선택 (기본값 권장)
    

### 4.8.3.2 원격 접속

- `Ctrl + Shift + P`
    
- `Remote-SSH: Connect to Host`
    
- 방금 등록한 `lab-main` 선택
    

정상일 경우:

- VS Code 하단 좌측에 `SSH: lab-main` 표시
    
- 최초 접속 시 VS Code Server가 자동 설치됨
    

---

## 4.8.4 서버 측 상태 확인 (중요)

VS Code 원격 접속 후, **VS Code 터미널**에서 아래 확인:

```bash
whoami
hostname
pwd
```

기대 결과:

- 사용자: `hyungjun`
    
- 호스트명: `lab-main`
    
- 홈 디렉토리에서 시작
    

---

## 4.8.5 홈랩 작업 디렉토리 준비

Phase 5부터 사용할 표준 작업 공간을 미리 생성한다.

```bash
mkdir -p ~/homelab
cd ~/homelab
```

VS Code 좌측 Explorer에서:

- `Open Folder`
    
- `/home/hyungjun/homelab` 선택
    

이 폴더가 이후:

- Docker 설정
    
- Git 저장소
    
- 자동화 스크립트
    
- Obsidian 연계 작업  
    의 **기준 작업 디렉토리**가 된다.
    

---

## 4.8.6 VS Code 터미널 기반 작업 확인

VS Code 내 Terminal(`Ctrl + ``)에서:

```bash
ls
ip a
sudo apt update
```

확인 포인트:

- 로컬 PC가 아닌 **서버에서 명령이 실행됨**
    
- SSH 세션과 VS Code 터미널이 동일 환경임을 확인
    

---

## 4.8.7 운영 관점 권장 사항

- 서버에는:
    
    - 데스크톱 환경
        
    - GUI 도구
        
    - IDE 직접 설치 ❌
        
- 모든 작업은:
    
    - VS Code Remote-SSH
        
    - Docker 컨테이너
        
    - Git 기반 설정으로 수행
        

이 구조는:

- 개인 홈랩
    
- 회사 PoC
    
- 클라우드 VM  
    어디로 옮겨가도 동일하게 재사용 가능하다.
    
# 요약

Phase 4 종료 시점에서 Ubuntu VM(lab-main)은:

- SSH 키 기반 원격 접속 완료
    
- 기본 보안(UFW) 적용
    
- Proxmox 연동(Qemu Agent) 완료
    
- **VS Code Remote-SSH 기반 작업 환경 구축 완료**
    

이제 서버는:

- “접속 테스트용 VM”이 아니라
    
- **실제 개발·운영이 가능한 메인 서버** 상태가 된다.
    

다음 단계인 [[Phase 5. Docker 기반 서비스 환경 구축]] 에서는  
VS Code를 통해 바로 다음 작업을 진행한다.

- docker-compose 작성
    
- 컨테이너 실행/중지
    
- 자동화(n8n) 환경 구성
    
- Git 기반 설정 관리

# 출처


# 관련 노트


