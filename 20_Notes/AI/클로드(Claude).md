---
Created: 2025-05-28 10:57
tags:
  - AI/LLM
---

# 개요

**Claude**는 OpenAI 전 직원들이 설립한 **Anthropic**이 개발한 대형 언어 모델(LLM)로, **안전성, 윤리성, 정밀한 문서 해석 능력**에 초점을 둔 차별화된 AI 모델이다.

- 개발사: Anthropic
- 출시 시점: Claude 1 (2023), Claude 2 (2023.7), Claude 3 (2024.3)
- 핵심 철학: Constitutional AI (헌법 기반 AI)
- 주요 경쟁 모델: GPT-4 (OpenAI), Gemini (Google), Command R (Cohere)

# 내용

## Claude의 철학: Constitutional AI

Anthropic은 AI에게 인간 피드백(RLHF) 대신, **“AI의 헌법”이라 부를 수 있는 규칙 집합**을 기반으로 행동 원칙을 학습시킴.

### 핵심 가치:
- **안전성(Safety)**: 유해 발언, 조작적 응답 방지
- **정직함(Honesty)**: 모르는 건 모른다고 말함
- **도움 됨(Helpfulness)**: 명확하고 유익한 응답 생성

---

## Claude 3 시리즈 소개

| 모델 | 특징 | 용도 |
|------|------|------|
| **Claude 3 Haiku** | 가장 빠르고 저비용, 실시간 응답 최적화 | 챗봇, 단순 대화 |
| **Claude 3 Sonnet** | 균형 잡힌 성능과 속도 | 일반적인 업무 도우미 |
| **Claude 3 Opus** | 최고 성능, GPT-4와 동급 또는 초과 | 복잡한 작업, 멀티 문서 분석 |

### Claude 3 Opus 성능 예시:
- 벤치마크 기준 GPT-4보다 더 나은 추론, 분석, 문해력 능력
- 2024년 기준 최고의 문서 기반 AI 모델 중 하나

---

## Claude의 기술적 강점

### 1. 초장문의 문서 처리 능력
- 최대 200,000 tokens (약 150,000단어 이상)
- 논문, 계약서, 백서 등 전체 문서 분석 가능

### 2. 보수적이고 신중한 언어 스타일
- 정중하고 예측 가능한 표현 방식
- 기업/법률/교육 분야에 적합

### 3. AWS Bedrock 기반 API 제공
- 직접 Anthropic API 또는 AWS 환경에서 Claude 모델 호출 가능
- 보안성·컴플라이언스 요구 높은 기업에 적합

---

## Claude와 AI Agent

### 현재 Claude는 직접적인 "에이전트 실행 기능"은 미지원
- 툴 실행, 브라우징, 외부 시스템 자동 연동 불가
- 그러나 **분석 Agent**나 **문서 비서 Agent**로는 매우 적합

### Claude 기반 Agent 사용 예시
- 법률 문서 요약 Agent
- 학술 자료 비교 Agent
- 회의록 다중 문서 통합 요약 Agent
- 기업 보고서 정제 및 리포트 작성 도우미

---

## Claude vs GPT-4 (Agent 관점 간단 비교)

| 항목 | Claude | GPT-4 (ChatGPT) |
|------|--------|-----------------|
| 철학 | 헌법 기반 AI | 인간 피드백 기반 강화학습 |
| 도구 실행 | ❌ 없음 | ✅ Plugins, Actions 등 가능 |
| 문서 처리 | ✅ 200K tokens | 최대 128K tokens |
| 자동화 구성 | 외부 시스템 필요 | Assistant API, Zapier 연동 용이 |
| 스타일 | 정중, 보수적, 신중함 | 창의적, 다양한 스타일 대응 |

---

## Claude가 유리한 사용 사례

- 긴 문서 중심의 전문 분석
- 보안/윤리 기준이 중요한 산업 (법률, 의료, 정부)
- 멀티 문서 통합 요약
- 정밀한 언어 분석 기반의 리서치/컨설팅 보조

---

## 향후 발전 방향 (예상)

- Claude 4 개발 중 (2025~)
- 브라우징 및 툴 호출 기능 도입 가능성 낮음
- 대신, 더 깊은 추론 및 실시간 지식처리 능력에 집중할 가능성


# 출처


# 관련 노트


# 참고 링크

- Anthropic 공식 사이트: [https://www.anthropic.com](https://www.anthropic.com)
- Claude 사용 사이트: [https://claude.ai](https://claude.ai)
- AWS Bedrock: [https://aws.amazon.com/bedrock](https://aws.amazon.com/bedrock)
