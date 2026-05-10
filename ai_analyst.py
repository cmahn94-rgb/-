"""
ai_analyst.py — Gemini API 기반 AI 시장 해석 모듈
==================================================
이 파일이 하는 일:
  1) get_ai_market_commentary()
     → 현재 시장 레짐 + 상위 신호 종목 요약을 Gemini에게 전달
     → 짧고 실용적인 시장 해석 코멘트 생성
  2) get_ai_signal_reasons_batch()
     → 여러 종목의 매수 신호 이유를 한 번에 Gemini에게 요청
  3) translate_to_korean_one_line_batch()
     → 영어 뉴스들을 한 번의 호출로 한국어로 번역

[업그레이드 v3 변경사항]
  ① 지수 백오프(Exponential Backoff) 재시도 적용
     - 첫 실패 → 1초 대기 → 재시도
     - 두 번째 실패 → 2초 대기 → 재시도
     - 세 번째 실패 → 4초 대기 → 재시도
     - 모두 실패 시 빈 문자열 반환 (리포트 전송에는 영향 없음)
     - 503(서버 과부하), 429(요청 초과) 오류에 특히 효과적

  ② 배치 요청 수 제한 (BATCH_SIZE = 5   # 10→5 변경: 429 오류 방지 (신호 10개 초과 시 반복 실패 해결))
     - AI 신호 이유, 번역 등 배치 호출을 10개씩 끊어서 처리
     - 한 번에 100개 요청이 쏟아지던 것을 10개 × N회로 분산
     - 이렇게 하면 429(Too Many Requests) 오류가 크게 줄어든다

[필요 환경변수]
  GEMINI_API_KEY — GitHub Actions Secrets에 등록 필요
"""

import os
import json
import time
import requests
from dotenv import load_dotenv


_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_MODEL          = "gemini-2.5-flash"

# ─────────────────────────────────────────
# 배치 최대 크기: 한 번의 API 호출에 담을 최대 항목 수
# 10개 초과 시 10개씩 끊어서 순차 호출 → 429 오류 방지
# ─────────────────────────────────────────
BATCH_SIZE = 10

# ─────────────────────────────────────────
# 지수 백오프 재시도 설정
# MAX_RETRIES = 3 → 최대 3번 재시도 (총 4번 시도)
# BASE_DELAY  = 1 → 첫 대기 시간 1초, 이후 2배씩 증가 (1→2→4초)
# ─────────────────────────────────────────
MAX_RETRIES = 4
BASE_DELAY  = 2   # 초 단위 (1→2로 강화: 2→4→8→16초 대기)

# 번역 결과 캐시: 같은 문장은 API 재호출 없이 재사용
_TRANSLATION_CACHE: dict[str, str] = {}


def _call_gemini(system_prompt: str, user_content: str, max_output_tokens: int = 300) -> str:
    """
    Gemini API를 호출하고 텍스트 응답을 반환한다.

    [중학생 설명]
    Gemini라는 AI에게 질문을 보내고 답변을 받는 함수다.
    서버가 바쁘거나(503) 요청이 너무 많으면(429) 실패할 수 있는데,
    이때 바로 포기하지 않고 잠깐 기다렸다가 다시 시도한다.
    기다리는 시간은 매번 2배씩 늘어난다 (1초 → 2초 → 4초).
    이 방식을 '지수 백오프(Exponential Backoff)'라고 부른다.

    API 키 없음 또는 모든 재시도 실패 시 → 빈 문자열 반환 (안전 폴백)
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ""   # API 키 없으면 AI 분석 생략 (리포트는 정상 발송됨)

    headers = {
        "x-goog-api-key": api_key,
        "content-type":   "application/json",
    }
    url = f"{_GEMINI_API_URL}/{_MODEL}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {"maxOutputTokens": max_output_tokens},
    }

    # ── 지수 백오프 재시도 루프 ──────────────────────────────
    # 시도 횟수: 0(첫 시도), 1(1초 뒤), 2(2초 뒤), 3(4초 뒤)
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=25)

            # 429(요청 초과) 또는 503(서버 과부하)이면 재시도 대상
            if resp.status_code in (429, 503):
                if attempt < MAX_RETRIES:
                    대기_시간 = BASE_DELAY * (2 ** attempt)  # 1 → 2 → 4초
                    print(
                        f"⏳ Gemini API {resp.status_code} 오류 "
                        f"(시도 {attempt + 1}/{MAX_RETRIES + 1}) "
                        f"→ {대기_시간}초 후 재시도..."
                    )
                    time.sleep(대기_시간)
                    continue   # 다음 attempt로

                # 마지막 시도도 실패하면 빈 문자열 반환
                print(f"⚠️ AI 분석 오류 (최대 재시도 초과): {resp.status_code}")
                return ""

            # 그 외 오류(400, 401 등)는 재시도 없이 즉시 반환
            resp.raise_for_status()

            data       = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return ""
            content = candidates[0].get("content") or {}
            parts   = content.get("parts") or []
            if not parts:
                return ""
            text = parts[0].get("text")
            return text.strip() if isinstance(text, str) else ""

        except requests.exceptions.Timeout:
            # 타임아웃도 재시도 대상
            if attempt < MAX_RETRIES:
                대기_시간 = BASE_DELAY * (2 ** attempt)
                print(f"⏳ Gemini API 타임아웃 (시도 {attempt + 1}) → {대기_시간}초 후 재시도...")
                time.sleep(대기_시간)
                continue
            print("⚠️ AI 분석 오류 (타임아웃 반복): 분석 생략")
            return ""

        except Exception as e:
            # 예상치 못한 오류는 재시도 없이 즉시 반환
            print(f"⚠️ AI 분석 오류 (무시하고 계속): {e}")
            return ""

    return ""   # 모든 재시도 소진


# ─────────────────────────────────────────
# AI 시장 코멘트 생성
# ─────────────────────────────────────────

def get_ai_market_commentary(
    하락_레짐: bool,
    크립토_하락_레짐: bool,
    경고_목록: list,
    신호_종목_요약: list,
) -> str:
    """
    시장 레짐 상태와 신호 종목 목록을 바탕으로
    실용적인 한국어 시장 코멘트를 생성한다.

    [중학생 설명]
    "지금 시장 상황이 이러이러하고, 이런 종목들이 신호가 떴어"
    라고 AI에게 알려주면, AI가 2~3줄로 요약 코멘트를 만들어준다.

    반환값: 2~3줄 문자열 (실패 시 빈 문자열)
    """
    if not 신호_종목_요약 and not 하락_레짐:
        return ""   # 신호도 없고 하락장도 아니면 AI 호출 생략

    system = (
        "당신은 퀀트 헤지펀드의 수석 애널리스트입니다. "
        "한국어로 간결하고 실용적인 시장 코멘트를 2~3줄로 작성합니다. "
        "투자 권유가 아닌 데이터 기반 관찰 의견만 제시합니다. "
        "이모지는 사용하지 않습니다."
    )

    regime_desc = []
    if 하락_레짐:
        regime_desc.append(f"대세 하락장 ({', '.join(경고_목록)})")
    if 크립토_하락_레짐:
        regime_desc.append("크립토 하락 레짐")
    if not regime_desc:
        regime_desc.append("중립")

    종목_desc = ""
    if 신호_종목_요약:
        items = [f"{s['name']}(점수:{s['점수']}, RSI:{s['rsi']:.1f})" for s in 신호_종목_요약[:5]]
        종목_desc = f"매수 신호 종목: {', '.join(items)}"

    user_msg = (
        f"현재 시장 레짐: {', '.join(regime_desc)}\n"
        f"{종목_desc}\n\n"
        "위 데이터를 바탕으로 오늘 시장의 핵심 관찰 포인트를 2~3줄로 요약해 주세요."
    )

    result = _call_gemini(system, user_msg, max_output_tokens=250)
    if result:
        return f"\n*AI 시장 해석*\n{result}\n"
    return ""


# ─────────────────────────────────────────
# 번역 (단건)
# ─────────────────────────────────────────

def translate_to_korean_one_line(text: str) -> str:
    """
    영어(또는 비한국어) 문장을 한국어 한 줄로 짧게 번역한다.
    API 키가 없으면 빈 문자열 반환(=번역 생략)
    """
    if not text or not isinstance(text, str):
        return ""

    key = text.strip()
    if not key:
        return ""
    if key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[key]

    system = (
        "당신은 금융 뉴스 번역가입니다. "
        "입력 문장을 한국어로 자연스럽게 번역하되, 1줄(짧게)로 요약 번역합니다. "
        "불필요한 수식어는 줄이고 핵심만 남깁니다. "
        "이모지는 사용하지 않습니다."
    )
    user_msg = f"영문 뉴스 요약:\n{text}\n\n한국어 한 줄 번역:"
    result = _call_gemini(system, user_msg, max_output_tokens=120)
    _TRANSLATION_CACHE[key] = result   # 빈 문자열도 캐시해서 폭주 방지
    return result


# ─────────────────────────────────────────
# 번역 배치 (10개씩 끊어서 처리)
# ─────────────────────────────────────────

def translate_to_korean_one_line_batch(texts: list[str]) -> list[str]:
    """
    여러 문장을 10개씩 나눠서 Gemini에게 한국어 번역을 요청한다.

    [중학생 설명]
    예전엔 100개 문장을 한꺼번에 보냈다가 "너무 많다"며 거절당했다.
    이제는 10개씩 끊어서 보내고, 중간에 짧게 쉬어가면서(0.5초 대기)
    서버에 과부하가 걸리지 않게 한다.
    마치 편의점에서 한 번에 100개 계산하지 않고 10개씩 나눠서 계산하는 것.

    반환값: 입력과 동일한 길이의 번역 결과 리스트
    """
    if not texts:
        return []

    normalized = [(t or "").strip() for t in texts]
    out = [""] * len(normalized)

    # 캐시 히트 먼저 채우기 (캐시에 있는 건 API 호출 불필요)
    to_translate = []
    index_map    = []
    for i, t in enumerate(normalized):
        if not t:
            continue
        if t in _TRANSLATION_CACHE:
            out[i] = _TRANSLATION_CACHE[t]
        else:
            to_translate.append(t)
            index_map.append(i)

    if not to_translate:
        return out

    # ── 10개씩 배치로 나눠서 처리 ────────────────────────────
    system = (
        "당신은 금융 뉴스 번역가입니다. "
        "입력 문장들을 각각 한국어로 자연스럽게 번역하되, 각 항목은 1줄(짧게)로 요약 번역합니다. "
        "불필요한 수식어는 줄이고 핵심만 남깁니다. "
        "출력은 반드시 JSON 배열 형식으로만 반환합니다. "
        "배열 길이는 입력과 반드시 같아야 합니다. "
        "이모지는 사용하지 않습니다."
    )

    # 전체 결과를 담을 임시 리스트
    all_translated = [""] * len(to_translate)

    # BATCH_SIZE(10)개씩 슬라이싱
    for batch_start in range(0, len(to_translate), BATCH_SIZE):
        batch_texts = to_translate[batch_start: batch_start + BATCH_SIZE]

        user_msg = (
            "아래 JSON 배열의 각 문자열을 같은 순서로 한국어 한 줄 번역해서, JSON 배열로만 답하세요.\n"
            f"{json.dumps(batch_texts, ensure_ascii=False)}"
        )

        raw = _call_gemini(system, user_msg, max_output_tokens=400)

        parsed = _parse_json_list(raw, expected_len=len(batch_texts))

        for j, (t, ko) in enumerate(zip(batch_texts, parsed)):
            ko_text = ko.strip() if isinstance(ko, str) else ""
            _TRANSLATION_CACHE[t] = ko_text
            all_translated[batch_start + j] = ko_text

        # 배치 사이 3초 대기 → Gemini 무료 분당 15회 한도 대응
        if batch_start + BATCH_SIZE < len(to_translate):
            time.sleep(3.0)

    # 원래 인덱스 위치에 결과 채우기
    for t, orig_idx in zip(to_translate, index_map):
        out[orig_idx] = _TRANSLATION_CACHE.get(t, "")

    return out


# ─────────────────────────────────────────
# 매수 신호 이유 배치 생성 (10개씩 끊어서 처리)
# ─────────────────────────────────────────

def get_ai_signal_reasons_batch(items: list[dict]) -> dict[str, str]:
    """
    여러 종목의 '매수 신호 이유(한 문장)'를 10개씩 나눠서 Gemini에게 요청한다.

    [중학생 설명]
    "왜 이 종목이 매수 신호가 났어?"를 AI에게 물어보는 함수.
    예전엔 50개 종목을 한꺼번에 물어봐서 "너무 많다(429)"는 오류가 났다.
    이제는 10개씩 끊어서 묻고, 중간에 0.5초 쉰다.

    items 예시:
      [{"ticker":"AAPL","name":"애플","rsi":38.2,"score":4, ...}, ...]
    반환: {ticker: 이유 문자열}
    """
    if not items:
        return {}

    system = (
        "당신은 퀀트 트레이더입니다. "
        "각 종목에 대해 '왜 매수 신호가 나왔는지'를 한국어 한 문장(40자 이내)으로 작성합니다. "
        "출력은 반드시 JSON 객체로만 반환합니다. "
        "키는 ticker, 값은 이유 문자열입니다. "
        "이모지는 사용하지 않습니다."
    )

    result_map: dict[str, str] = {}

    # ── 10개씩 배치로 나눠서 처리 ────────────────────────────
    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start: batch_start + BATCH_SIZE]

        user_msg = (
            "아래 JSON 배열의 각 항목을 보고, ticker별 이유를 JSON 객체로만 답하세요.\n"
            f"{json.dumps(batch, ensure_ascii=False)}"
        )

        raw = _call_gemini(system, user_msg, max_output_tokens=500)
        parsed = _parse_json_dict(raw)

        for k, v in parsed.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip():
                result_map[k.strip()] = v.strip()

        # 배치 사이 3초 대기 → Gemini 무료 분당 15회 한도 대응
        if batch_start + BATCH_SIZE < len(items):
            time.sleep(3.0)

    return result_map


# ─────────────────────────────────────────
# 단건 신호 이유 (하위 호환용)
# ─────────────────────────────────────────

def get_ai_signal_reason(ticker: str, name: str, 지표: dict) -> str:
    """
    단일 종목의 기술지표를 분석해 매수 신호 이유를 한 문장으로 반환한다.
    (배치 함수가 없을 때를 위한 단건 호출용)
    """
    system = (
        "당신은 퀀트 트레이더입니다. "
        "기술지표 데이터를 보고 매수 신호 이유를 한 문장(40자 이내)으로 설명합니다. "
        "예: 'RSI 과매도 + MACD 반등으로 단기 반등 가능성'"
    )
    user_msg = (
        f"종목: {name} ({ticker})\n"
        f"RSI: {지표.get('rsi', 0):.1f}\n"
        f"점수: {지표.get('점수', 0)}/6\n"
        f"MA 정배열: {'예' if 지표.get('ma_정배열') else '아니오'}\n"
        f"MACD 음→양 전환: {'예' if 지표.get('macd_전환') else '아니오'}\n"
        f"거래량 증가: {'예' if 지표.get('거래량_증가') else '아니오'}\n"
        f"변동성 돌파: {'예' if 지표.get('변동성돌파') else '아니오'}\n\n"
        "매수 신호 이유를 한 문장으로 설명해 주세요."
    )
    return _call_gemini(system, user_msg, max_output_tokens=80)


# ─────────────────────────────────────────
# 내부 헬퍼: JSON 파싱 안전 처리
# ─────────────────────────────────────────

def _parse_json_list(raw: str, expected_len: int) -> list:
    """
    Gemini 응답 문자열에서 JSON 배열을 안전하게 파싱한다.

    [중학생 설명]
    AI 응답이 항상 깔끔한 JSON으로 오지 않을 수 있다.
    ```json ... ``` 같은 코드블록으로 감싸서 오거나,
    앞뒤에 설명 문장이 붙을 수도 있다.
    이 함수는 그런 경우에도 JSON 부분만 골라내서 파싱한다.
    파싱에 실패하면 빈 문자열 리스트를 반환한다.
    """
    if not raw:
        return [""] * expected_len

    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            start = raw.find("[")
            end   = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(raw[start:end + 1])
        except Exception:
            parsed = None

    if not isinstance(parsed, list) or len(parsed) != expected_len:
        return [""] * expected_len

    return parsed


def _parse_json_dict(raw: str) -> dict:
    """
    Gemini 응답 문자열에서 JSON 객체를 안전하게 파싱한다.
    파싱 실패 시 빈 딕셔너리 반환.
    """
    if not raw:
        return {}

    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            start = raw.find("{")
            end   = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(raw[start:end + 1])
        except Exception:
            parsed = None

    if not isinstance(parsed, dict):
        return {}

    return parsed
