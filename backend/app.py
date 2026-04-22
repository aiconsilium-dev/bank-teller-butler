"""
행원 집사 — 금융기관 현장직원 AI 법률 어시스턴트
FastAPI 백엔드
"""
import os
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

app = FastAPI(title="행원 집사", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

SYSTEM_PROMPT = """당신은 행원 집사 — 금융기관 현장직원(행원)을 위한 AI 법률 어시스턴트입니다.

은행, 저축은행, 새마을금고, 카드사 등 금융기관 행원이 고객 응대 및 업무 처리 중 마주치는 법률·규제 질문에 즉시 답변합니다.

## 근거 법령 (답변 시 조문 명시 필수)
- 은행법 / 은행업감독규정 / 은행업감독업무시행세칙
- 저축은행법 / 새마을금고법
- 여신전문금융업법
- 금융소비자보호법(금소법)
- 채권의 공정한 추심에 관한 법률(채권추심법)
- 민법 (소멸시효, 채권양도 등)
- 이자제한법 / 대부업법

## 답변 원칙
1. **실무 우선**: 행원이 지금 바로 적용할 수 있는 절차와 체크리스트 중심
2. **법령 명시**: 근거 조문 반드시 포함 (예: 금소법 제17조 제1항)
3. **리스크 경고**: 행원 개인 책임 또는 기관 제재 가능성 있는 경우 ⚠️로 명확히 표시
4. **고객 설명용**: 고객에게 설명할 수 있는 언어로 추가 안내
5. **한계 인정**: 법무팀/준법감시인 확인이 필요한 사안은 그렇게 안내 (과도한 자신감 금지)
6. 답변 언어: 한국어

## 응답 수준 판단 기준
- **L1 즉답**: 표준화된 절차 질문, 조문 확인 수준 — 5분 이내 처리 가능
- **L2 검토필요**: 사실관계 판단 필요, 개별 사안 복잡도 있음 — 준법감시인 또는 법무팀 검토 권고
- **L3 전문가 필수**: 소송 관련, 임직원 비위, 약관 분쟁 등 — 반드시 전문 법률 검토 필요

## 업무 분류
- BK-001 담보·등기: 근저당 설정/말소/이전, 담보가등기, 경매 신청
- BK-002 여신심사·실행: 대출 심사 기준, 실행 절차, 대출 한도, 보증
- BK-003 채권관리·추심: 연체 관리, 채권양도, 내부채권, NPL 매각
- BK-004 소멸시효: 시효 기산점 계산, 시효중단 방법, 시효 갱신
- BK-005 금소법·설명의무: 설명의무 이행, 부당권유 금지, 광고 규제, 적합성 원칙
- BK-006 금리인하요구권: 신청 요건, 검토 절차, 거절 사유, 통지 기한
- BK-007 기한이익상실: 상실 사유, 통보 방법, 절차, 효과
- BK-008 청약철회·해지: 청약철회권 행사 기간, 해지 절차, 조기상환수수료
- BK-009 금감원 민원대응: 민원 유형 분류, 처리 기한, 내부 보고 절차
- BK-010 불법추심 규제: 채권추심법 금지행위, 위반 시 제재, 위탁 관리
- BK-011 내부통제·준법: 내부통제 기준, 이상거래 보고, 준법감시 업무
- BK-012 임직원 횡령·배임: 인지 시 보고 의무, 수사기관 신고, 피해 보전 조치

## 답변 형식 (마크다운 사용)
```
### 답변 요약
[핵심 내용 1-2줄]

### 근거 법령
- 조문 명시

### 실무 처리 절차
1. 단계별 체크리스트

### 주의사항
⚠️ 리스크 항목

### 고객 설명 포인트
(해당 시) 고객에게 이렇게 설명하세요
```
"""

BK_KEYWORDS = {
    "BK-001": ["근저당", "담보", "등기", "경매", "말소", "설정", "이전", "가등기", "담보권", "저당권", "근저당권"],
    "BK-002": ["여신", "대출", "심사", "실행", "한도", "보증", "신용평가", "담보대출", "신용대출", "금리"],
    "BK-003": ["채권", "연체", "추심", "채권양도", "NPL", "부실채권", "내부채권", "독촉", "추심원"],
    "BK-004": ["소멸시효", "시효", "시효중단", "시효갱신", "채무승인", "압류", "가압류", "소제기"],
    "BK-005": ["금소법", "금융소비자", "설명의무", "부당권유", "적합성", "광고", "불공정", "판매규제"],
    "BK-006": ["금리인하요구권", "금리인하", "금리 인하", "인하요구", "금리조정"],
    "BK-007": ["기한이익", "기한이익상실", "기이상실", "상실사유", "기이"],
    "BK-008": ["청약철회", "철회권", "해지", "조기상환", "중도상환", "수수료", "청약취소"],
    "BK-009": ["금감원", "민원", "민원처리", "금융민원", "분쟁조정", "이의신청"],
    "BK-010": ["불법추심", "추심금지", "채권추심법", "추심행위", "야간추심", "공갈"],
    "BK-011": ["내부통제", "준법감시", "이상거래", "보고의무", "컴플라이언스", "내부신고"],
    "BK-012": ["횡령", "배임", "임직원", "비위", "수사", "형사고발", "피해보전"],
}

LAW_KEYWORDS = {
    "금소법": ["금소법", "금융소비자", "설명의무", "부당권유", "청약철회", "위법계약"],
    "채권추심법": ["채권추심", "추심", "불법추심", "추심원"],
    "은행법": ["은행법", "금융지주", "인가", "허가", "내부통제"],
    "민법": ["소멸시효", "채권양도", "질권", "저당권", "보증"],
    "여신전문금융업법": ["카드", "할부", "리스", "여신전문"],
}


def detect_bk(text: str) -> Optional[str]:
    scores = {}
    for bk, keywords in BK_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[bk] = score
    return max(scores, key=scores.get) if scores else None


def detect_response_level(text: str) -> str:
    l3_triggers = ["횡령", "배임", "소송", "약관 불공정", "형사", "수사기관", "고발"]
    l2_triggers = ["판단", "복잡", "개별", "사실관계", "검토 필요", "법무팀"]
    if any(t in text for t in l3_triggers):
        return "L3 전문가 필수"
    if any(t in text for t in l2_triggers):
        return "L2 검토 권고"
    return "L1 즉답"


def detect_law_refs(text: str) -> Optional[str]:
    refs = []
    for law, keywords in LAW_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            refs.append(law)
    return " · ".join(refs) if refs else None


def load_kb_context(bk_filter: Optional[str] = None) -> str:
    """데이터 폴더에서 법령 인덱스 로드 (없으면 빈 문자열 반환)"""
    kb_file = DATA_DIR / "law_index.json"
    if not kb_file.exists():
        return ""
    try:
        with open(kb_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if bk_filter and bk_filter in data:
            items = data[bk_filter][:5]
        else:
            items = []
            for v in data.values():
                items.extend(v[:2])
            items = items[:10]

        if not items:
            return ""

        lines = ["--- 관련 법령 및 가이드라인 ---"]
        for item in items:
            lines.append(f"[{item.get('source', '')}] {item.get('text', '')[:200]}")
        return "\n".join(lines) + "\n---"
    except Exception:
        return ""


class ChatRequest(BaseModel):
    message: str
    history: list = []
    bk_filter: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    detected_bk: Optional[str]
    response_level: str
    law_refs: Optional[str]


@app.get("/")
async def root():
    index_file = PROJECT_DIR / "frontend" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"status": "행원 집사 API", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not client.api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    detected_bk = req.bk_filter or detect_bk(req.message)
    kb_context = load_kb_context(detected_bk)

    messages = []
    for h in req.history[-10:]:
        messages.append({"role": h["role"], "content": h["content"]})

    user_message = req.message
    if kb_context:
        user_message = f"{req.message}\n\n{kb_context}"

    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    reply = response.content[0].text
    level = detect_response_level(reply)
    law_refs = detect_law_refs(reply)

    return ChatResponse(
        reply=reply,
        detected_bk=detected_bk,
        response_level=level,
        law_refs=law_refs,
    )


@app.get("/health")
async def health():
    kb_ready = (DATA_DIR / "law_index.json").exists()
    return {
        "status": "ok",
        "kb_ready": kb_ready,
        "model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
    }


@app.get("/categories")
async def get_categories():
    return {
        "categories": [
            {"code": code, "name": kws[0]}
            for code, kws in BK_KEYWORDS.items()
        ]
    }


frontend_dir = PROJECT_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8100)))
