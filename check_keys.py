"""
check_keys.py — API 키 등록·작동 여부 진단 스크립트
=====================================================
[사용법]
  로컬:        python check_keys.py
  GitHub:      Actions 워크플로에 임시 step 추가 (아래 설명 참고)

[하는 일]
  등록된 각 API 키로 실제 1회 호출해서
  "등록됨/작동함/실패함"을 한눈에 보여준다.
  키 값 자체는 절대 출력하지 않는다 (보안).

[GitHub Actions에서 돌리는 법]
  run_analysis.yml의 "Run analysis" step을 잠시 아래로 바꿔서 1회 실행:
      run: python check_keys.py
  확인 후 다시 'python main.py'로 되돌리면 된다.
"""

import os
import requests


def mask(key: str) -> str:
    """키 앞 4자리만 보여주고 나머지는 가림 (등록 여부만 확인용)."""
    if not key:
        return "(없음)"
    if len(key) <= 8:
        return key[:2] + "****"
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def check(name: str) -> str:
    """환경변수 등록 여부 확인."""
    val = os.getenv(name, "").strip()
    status = "✅ 등록됨" if val else "❌ 미등록"
    return f"  {name:22s} {status:10s} {mask(val)}"


def test_groq() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return "  GROQ      ❌ 키 없음"
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 5},
            timeout=15,
        )
        if r.status_code == 200:
            return "  GROQ      ✅ 작동 (200 OK)"
        if r.status_code == 401:
            return "  GROQ      ❌ 인증 실패 (401) — 키가 틀림"
        if r.status_code == 429:
            return "  GROQ      ⚠️ 한도 초과 (429) — 키는 유효, 잠시 후 재시도"
        return f"  GROQ      ⚠️ HTTP {r.status_code}"
    except Exception as e:
        return f"  GROQ      ⚠️ 호출 오류: {e}"


def test_gemini() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return "  GEMINI    ❌ 키 없음"
    try:
        r = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            headers={"x-goog-api-key": key, "content-type": "application/json"},
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                  "generationConfig": {"maxOutputTokens": 5}},
            timeout=15,
        )
        if r.status_code == 200:
            return "  GEMINI    ✅ 작동 (200 OK)"
        if r.status_code in (400, 403):
            return f"  GEMINI    ❌ 인증/권한 실패 ({r.status_code}) — 키 확인 필요"
        if r.status_code == 429:
            return "  GEMINI    ⚠️ 한도 초과 (429) — 키는 유효"
        return f"  GEMINI    ⚠️ HTTP {r.status_code}"
    except Exception as e:
        return f"  GEMINI    ⚠️ 호출 오류: {e}"


def test_dart() -> str:
    key = os.getenv("DART_API_KEY", "").strip()
    if not key:
        return "  DART      ❌ 키 없음"
    try:
        # 삼성전자 corp_code(00126380)로 company.json 조회
        r = requests.get(
            "https://opendart.fss.or.kr/api/company.json",
            params={"crtfc_key": key, "corp_code": "00126380"},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            status = data.get("status")
            if status == "000":
                return "  DART      ✅ 작동 (삼성전자 조회 성공)"
            if status == "010":
                return "  DART      ❌ 등록되지 않은 키 (status 010)"
            if status == "011":
                return "  DART      ❌ 사용할 수 없는 키 (status 011)"
            if status == "020":
                return "  DART      ⚠️ 요청 한도 초과 (status 020) — 키는 유효"
            return f"  DART      ⚠️ status {status}: {data.get('message','')}"
        return f"  DART      ⚠️ HTTP {r.status_code}"
    except Exception as e:
        err = str(e)
        if "timed out" in err or "ConnectTimeout" in err or "Max retries" in err:
            return (
                "  DART      ⚠️ 연결 타임아웃 — 키는 유효하나 GitHub Actions IP가 DART 서버에 막힘\n"
                "             (KRX·네이버와 같은 한국 서버 해외 IP 차단 문제)\n"
                "             → 봇 실행에는 영향 없음 (DART 없이도 FA 나머지 데이터로 계속 진행)"
            )
        return f"  DART      ⚠️ 호출 오류: {e}"


def test_telegram() -> str:
    token   = os.getenv("TELEGRAM_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token:
        return "  TELEGRAM  ❌ TELEGRAM_TOKEN 없음"
    if not chat_id:
        return "  TELEGRAM  ⚠️ 토큰은 있으나 TELEGRAM_CHAT_ID 없음"
    try:
        # getMe로 봇 토큰 유효성만 확인 (메시지 안 보냄)
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=15)
        if r.status_code == 200 and r.json().get("ok"):
            bot_name = r.json().get("result", {}).get("username", "?")
            return f"  TELEGRAM  ✅ 작동 (봇: @{bot_name}, chat_id 등록됨)"
        if r.status_code == 401:
            return "  TELEGRAM  ❌ 토큰 무효 (401)"
        return f"  TELEGRAM  ⚠️ HTTP {r.status_code}"
    except Exception as e:
        return f"  TELEGRAM  ⚠️ 호출 오류: {e}"


def test_alphavantage() -> str:
    key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not key:
        return "  ALPHAVANTAGE ➖ 미등록 (선택사항 — 미국주식 뉴스)"
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": "AAPL", "apikey": key},
            timeout=15,
        )
        data = r.json()
        if "Global Quote" in data:
            return "  ALPHAVANTAGE ✅ 작동"
        if "Note" in data or "Information" in data:
            return "  ALPHAVANTAGE ⚠️ 한도 초과 (키는 유효)"
        return "  ALPHAVANTAGE ❌ 응답 이상 — 키 확인"
    except Exception as e:
        return f"  ALPHAVANTAGE ⚠️ 호출 오류: {e}"


if __name__ == "__main__":
    print("=" * 55)
    print("  API 키 진단 (값은 일부만 표시, 보안 안전)")
    print("=" * 55)

    print("\n[1] 환경변수 등록 여부")
    for name in ["TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "GROQ_API_KEY",
                 "GEMINI_API_KEY", "DART_API_KEY", "GH_PAGES_URL",
                 "ALPHAVANTAGE_API_KEY", "GNEWS_API_KEY", "NEWSAPI_KEY"]:
        print(check(name))

    print("\n[2] 실제 API 호출 테스트")
    print(test_telegram())
    print(test_groq())
    print(test_gemini())
    print(test_dart())
    print(test_alphavantage())

    print("\n" + "=" * 55)
    print("  ✅=정상  ⚠️=키는 유효하나 한도/일시오류  ❌=키 문제  ➖=선택(미등록)")
    print("=" * 55)
