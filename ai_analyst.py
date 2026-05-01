"""
ai_analyst.py — Gemini API 기반 AI 시장 해석 모듈
==================================================
이 파일이 하는 일:
  1) get_ai_market_commentary()
     → 현재 시장 레짐 + 상위 신호 종목 요약을 Gemini에게 전달
     → 짧고 실용적인 시장 해석 코멘트 생성
  2) get_ai_signal_reason()
     → 특정 종목의 기술지표 수치를 Gemini에게 주면
     → 왜 매수 신호가 나왔는지 한 문장 해석 반환

[필요 환경변수]
  GEMINI_API_KEY — GitHub Actions Secrets에 등록 필요
  GitHub Actions workflow의 env: 블록에 아래 추가:
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}

[비용 절감 설계]
  - flash 계열 모델 사용(빠르고 저렴)
  - 출력 길이를 짧게 제한
  - 매 실행 당 1~2회만 호출 (종목별 개별 호출 없음)
  - AI 분석 실패 시 빈 문자열 반환 → 전체 리포트에 영향 없음
"""

import os
import requests
from dotenv import load_dotenv


_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_MODEL          = "gemini-2.5-flash"

# AI 코멘트 기능(키가 없으면 AI만 생략하고 정상 진행)

_TRANSLATION_CACHE: dict[str, str] = {}


def _call_gemini(system_prompt: str, user_content: str, max_output_tokens: int = 300) -> str:
    """
    Gemini API를 호출하고 텍스트 응답을 반환한다.
    API 키 없음 또는 오류 시 빈 문자열 반환(안전 폴백).
    """
    # 로컬(.env)과 GitHub Actions(Secrets env) 둘 다 지원
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ""   # API 키 없으면 AI 분석 생략 (리포트 정상 발송)

    headers = {
        "x-goog-api-key": api_key,
        "content-type":  "application/json",
    }
    url = f"{_GEMINI_API_URL}/{_MODEL}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {"maxOutputTokens": max_output_tokens},
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # 응답 텍스트 경로: candidates[0].content.parts[0].text
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            return ""
        text = parts[0].get("text")
        return text.strip() if isinstance(text, str) else ""
    except Exception as e:
        print(f"⚠️ AI 분석 오류 (무시하고 계속): {e}")
        return ""


def get_ai_market_commentary(
    하락_레짐: bool,
    크립토_하락_레짐: bool,
    경고_목록: list,
    신호_종목_요약: list,   # [{"name": "삼성전자", "ticker": "005930.KS", "점수": 4, "rsi": 38.2}, ...]
) -> str:
    """
    시장 레짐 상태와 신호 종목 목록을 바탕으로
    실용적인 한국어 시장 코멘트를 생성한다.
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


def get_ai_signal_reason(ticker: str, name: str, 지표: dict) -> str:
    """
    단일 종목의 기술지표를 분석해 매수 신호 이유를 한 문장으로 반환한다.

    지표 dict 예시:
      {"rsi": 38.2, "점수": 4, "macd_전환": True, "거래량_증가": True,
       "변동성돌파": False, "ma_정배열": True}
    """
    system = (
        "당신은 퀀트 트레이더입니다. "
        "기술지표 데이터를 보고 매수 신호 이유를 한 문장(40자 이내)으로 설명합니다. "
        "예: 'RSI 과매도 + MACD 반등으로 단기 반등 가능성'"
    )

    user_msg = (
        f"종목: {name} ({ticker})\n"
        f"RSI: {지표.get('rsi', 'N/A'):.1f}\n"
        f"점수: {지표.get('점수', 0)}/6\n"
        f"MA 정배열: {'예' if 지표.get('ma_정배열') else '아니오'}\n"
        f"MACD 음→양 전환: {'예' if 지표.get('macd_전환') else '아니오'}\n"
        f"거래량 증가: {'예' if 지표.get('거래량_증가') else '아니오'}\n"
        f"변동성 돌파: {'예' if 지표.get('변동성돌파') else '아니오'}\n\n"
        "매수 신호 이유를 한 문장으로 설명해 주세요."
    )

    return _call_gemini(system, user_msg, max_output_tokens=80)


def translate_to_korean_one_line(text: str) -> str:
    """
    영어(또는 비한국어) 문장을 한국어 한 줄로 짧게 번역한다.
    - API 키가 없으면 빈 문자열 반환(=번역 생략)
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
    # 빈 문자열도 캐시해서 같은 입력에 대해 재시도 호출이 폭주하지 않게 한다.
    _TRANSLATION_CACHE[key] = result
    return result
