---
Created: 2025-12-14 12:52
tags:
  - AI/개발
---

# 개요

- Ubuntu Server VM(Main Server)을 **실제 서비스 실행 환경**으로 전환
    
- 모든 서비스는 **Docker 컨테이너 기반**으로 운영
    
- 초기 목표
    
    - 컨테이너 실행 표준 확립
        
    - 자동화(n8n) 기반 확보
        
    - 파일 동기화 및 Git 연동으로 Phase 6(Obsidian 자동화) 연결
        
- 운영 원칙
    
    - 설정은 코드(docker-compose)로 관리
        
    - 데이터는 컨테이너와 분리하여 영속 저장
        
    - 외부 노출 최소화 (내부망 기준 검증)
        
# 내용

## 5.1 기본 도구 설치 (Docker / Docker Compose / Portainer)

### 5.1.1 사전 준비

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg git
```

### 5.1.2 Docker Engine 설치

```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
| sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

```bash
sudo systemctl enable --now docker
docker --version
docker compose version
```

### 5.1.3 Docker 권한 설정

```bash
sudo usermod -aG docker $USER
newgrp docker
docker ps
```

### 5.1.4 홈랩 표준 디렉토리 구조

```bash
mkdir -p ~/homelab/{stack,data,backup}
mkdir -p ~/homelab/stack/{portainer,n8n,syncthing}
```

### 5.1.5 Portainer 설치 (컨테이너 관리 UI)

```bash
docker volume create portainer_data

docker run -d \
  --name portainer \
  --restart=always \
  -p 9000:9000 \
  -p 9443:9443 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce:latest
```

- 접속
    
    - `http://<ubuntu_vm_ip>:9000`
        
    - 또는 `https://<ubuntu_vm_ip>:9443`
        
- 최초 접속 시 admin 계정 생성 후 Local Docker 환경 선택
    

---

## 5.2 자동화 플랫폼 n8n 구축

### 5.2.1 n8n 디렉토리 준비

```bash
mkdir -p ~/homelab/stack/n8n
cd ~/homelab/stack/n8n
```

### 5.2.2 docker-compose.yml

```yaml
services:
  n8n:
    image: n8nio/n8n:latest
    container_name: n8n
    restart: always
    ports:
      - "5678:5678"
    environment:
      - TZ=Asia/Seoul
      - N8N_HOST=localhost
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - N8N_SECURE_COOKIE=false
    volumes:
      - ../../data/n8n:/home/node/.n8n
```

```bash
docker compose up -d
docker logs -f n8n
```

- 접속: `http://<ubuntu_vm_ip>:5678`
    
- n8n 설정/워크플로우는 `~/homelab/data/n8n`에 영구 저장
    

---

## 5.3 Git 기반 형상관리 연동 (GitHub 기준)

### 5.3.1 서버 SSH 키 생성

```bash
ssh-keygen -t ed25519 -C "homelab-ubuntu" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

- GitHub → Settings → SSH Keys에 등록
    

### 5.3.2 Git 기본 설정

```bash
git config --global user.name "Hyungjun"
git config --global user.email "<email@example.com>"
```

### 5.3.3 홈랩 구성 레포 관리 예시

```bash
cd ~/homelab
git init
git add .
git commit -m "init homelab docker stack"
```

- 데이터 폴더는 `.gitignore`로 제외 권장
    

```bash
echo "data/" >> .gitignore
```

---

## 5.4 파일 동기화 도구 구축 (Syncthing)

### 5.4.1 Syncthing 디렉토리

```bash
mkdir -p ~/homelab/stack/syncthing
cd ~/homelab/stack/syncthing
```

### 5.4.2 docker-compose.yml

```yaml
services:
  syncthing:
    image: syncthing/syncthing:latest
    container_name: syncthing
    restart: always
    environment:
      - TZ=Asia/Seoul
      - PUID=1000
      - PGID=1000
    volumes:
      - ../../data/syncthing:/var/syncthing
      - ../../data/vault:/vault
    ports:
      - "8384:8384"
      - "22000:22000"
      - "21027:21027/udp"
```

```bash
docker compose up -d
```

- 접속: `http://<ubuntu_vm_ip>:8384`
    
- `/vault` 경로는 Phase 6에서 Obsidian Vault와 직접 연계 예정



# 요약

이번 단계에서는 아래와 같은 사항을 진행했다.

- Docker / Docker Compose 설치로 **컨테이너 기반 운영 환경 확립**
    
- 홈랩 표준 디렉토리 구조(`homelab/stack`, `homelab/data`) 고정
    
- Portainer로 컨테이너 상태/로그/재시작 관리 가능
    
- n8n을 통해 자동화 플랫폼 기반 확보
    
- Git SSH 연동으로 설정·구성의 코드 관리 준비 완료
    
- Syncthing으로 Vault 및 파일 동기화 기반 구축
    
- Phase 6에서 Obsidian 자동화 및 AI 처리로 자연스럽게 확장 가능

다음 단계로는 Phase 6 (Obsidian 연계 구조)를

- Git vs Syncthing 역할 분리
    
- 회사 PC / 서버 / 자동화 흐름
    
- “사람이 쓰는 Obsidian ↔ 서버가 처리하는 Obsidian” 구조  
    이 관점에서 정리해주는 게 가장 자연스러워 보여.

# 출처


# 관련 노트



