"""
data_loader.py — 주가 데이터 및 뉴스 수집 (일괄 다운로드 + 캐싱)
==================================================================
[업비트 API]
  CRYPTO_KRW 종목(예: BTC/KRW)은 야후 파이낸스 대신 업비트 REST API 사용.
  업비트 티커 형식: "UPBIT:BTC" 또는 "BTC-KRW" → 내부에서 "KRW-BTC"로 변환.
  반환 DataFrame은 yfinance와 동일한 컬럼(Open/High/Low/Close/Volume) 구조.
"""

import time
import re
import json
import requests
import yfinance as yf
import pandas as pd

_cache = {}


def clear_cache():
    global _cache
    _cache = {}
    print("🗑️  데이터 캐시 초기화 완료")


# ─────────────────────────────────────────
# 업비트 REST API 연동
# ─────────────────────────────────────────

_UPBIT_PERIOD_DAYS = {
    "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
}


def _upbit_market_code(ticker: str) -> str:
    """내부 티커 → 업비트 마켓 코드 변환. 예) "UPBIT:BTC" → "KRW-BTC" """
    t = ticker.upper().strip()
    if t.startswith("UPBIT:"):
        return f"KRW-{t.split(':')[1]}"
    if t.endswith("-KRW"):
        return f"KRW-{t.replace('-KRW', '')}"
    if t.startswith("KRW-"):
        return t
    return f"KRW-{t}"


def _is_upbit_ticker(ticker: str) -> bool:
    t = ticker.upper()
    return t.startswith("UPBIT:") or t.endswith("-KRW") or t.startswith("KRW-")


def _fetch_upbit_ohlcv(market_code: str, count: int = 365) -> pd.DataFrame | None:
    """
    업비트 일봉 캔들 API로 OHLCV DataFrame을 가져온다.
    API: https://api.upbit.com/v1/candles/days
    반환 컬럼: Open / High / Low / Close / Volume (yfinance 동일 구조)
    """
    url = "https://api.upbit.com/v1/candles/days"
    headers = {"accept": "application/json"}
    all_rows = []
    remaining = count
    to_param = None

    while remaining > 0:
        fetch_count = min(remaining, 200)
        params = {"market": market_code, "count": fetch_count}
        if to_param:
            params["to"] = to_param

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"⚠️ 업비트 API 오류 ({market_code}): {e}")
            break

        if not data:
            break

        all_rows.extend(data)
        remaining -= len(data)
        to_param = data[-1]["candle_date_time_utc"]
        time.sleep(0.12)  # 업비트 Rate Limit 준수 (초당 10회)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["candle_date_time_utc"], utc=True)
    df = df.set_index("date").sort_index()
    df = df.rename(columns={
        "opening_price":           "Open",
        "high_price":              "High",
        "low_price":               "Low",
        "trade_price":             "Close",
        "candle_acc_trade_volume": "Volume",
    })[["Open", "High", "Low", "Close", "Volume"]]
    return df


def get_upbit_price(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    """
    업비트 API에서 KRW 기준 암호화폐 OHLCV를 가져온다.
    캐싱 지원 — 같은 ticker+period는 1회만 API 호출.

    ticker 예시: "UPBIT:BTC", "BTC-KRW", "KRW-BTC"
    """
    cache_key = f"{ticker}_{period}_upbit"
    if cache_key in _cache:
        # 캐시(cache): 이미 받은 데이터 재사용(다운로드 시간 절약)
        return _cache[cache_key]

    market_code = _upbit_market_code(ticker)
    days = _UPBIT_PERIOD_DAYS.get(period, 365)

    print(f"  📥 업비트 API: {market_code} ({period})...")
    df = _fetch_upbit_ohlcv(market_code, count=days)
    if df is not None and not df.empty:
        _cache[cache_key] = df
        print(f"  ✅ 업비트 {market_code}: {len(df)}일치 로드 완료 | 최근가: ₩{df['Close'].iloc[-1]:,.0f}")
    else:
        print(f"  ⚠️ 업비트 {market_code}: 데이터를 가져오지 못했습니다.")
    return df


# ─────────────────────────────────────────
# yfinance 일괄 다운로드
# ─────────────────────────────────────────

def bulk_download(tickers, period="1y", chunk_size=50):
    """
    여러 종목을 한 번에 묶어서 다운로드하고 캐시에 저장한다.
    CRYPTO_KRW 티커(업비트)는 자동으로 분리하여 별도 처리.
    """
    upbit_tickers = [t for t in tickers if _is_upbit_ticker(t)]
    yf_tickers    = [t for t in tickers if not _is_upbit_ticker(t)]

    # 업비트(KRW 마켓) 쪽은 yfinance로 못 받으니 업비트 API로 따로 받는다.
    for ticker in upbit_tickers:
        cache_key = f"{ticker}_{period}_upbit"
        if cache_key not in _cache:
            get_upbit_price(ticker, period)

    # yfinance는 티커를 너무 많이 한 번에 받으면 실패할 수 있어 chunk로 나눔
    for i in range(0, len(yf_tickers), chunk_size):
        묶음 = yf_tickers[i:i + chunk_size]
        ticker_str = " ".join(묶음)
        print(f"  📥 일괄 다운로드 ({i+1}~{min(i+chunk_size, len(yf_tickers))}/{len(yf_tickers)}종목, period={period})...")

        try:
            df_all = yf.download(
                ticker_str,
                period=period,
                progress=False,
                auto_adjust=True,
                group_by="ticker"
            )
            if df_all is None or df_all.empty:
                continue

            for ticker in 묶음:
                cache_key = f"{ticker}_{period}"
                try:
                    df_ticker = df_all if len(묶음) == 1 else df_all[ticker].dropna(how="all")
                    if df_ticker is not None and not df_ticker.empty:
                        _cache[cache_key] = df_ticker
                except Exception:
                    pass

        except Exception as e:
            print(f"⚠️ 일괄 다운로드 실패 → 개별 재시도: {e}")
            for ticker in 묶음:
                get_price_data(ticker, period)


def get_price_data(ticker, period="1y"):
    """
    캐시에서 주가 데이터를 반환한다.
    - CRYPTO_KRW 티커(업비트)는 업비트 API 사용
    - 그 외는 yfinance 사용
    """
    if _is_upbit_ticker(ticker):
        return get_upbit_price(ticker, period)

    cache_key = f"{ticker}_{period}"
    if cache_key in _cache:
        # 캐시에 있으면 바로 반환(빠름)
        return _cache[cache_key]

    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            print(f"⚠️ {ticker} 데이터가 비어 있습니다.")
            return None
        _cache[cache_key] = df
        return df
    except Exception as e:
        print(f"⚠️ {ticker} 데이터를 불러오지 못했습니다. (오류: {e})")
        return None


def get_news(ticker):
    """종목 뉴스를 가져온다. 업비트 티커는 빈 리스트 반환."""
    if _is_upbit_ticker(ticker):
        return []
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news if news else []
    except Exception:
        return []


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def _to_text(value) -> str:
    """
    yfinance 뉴스 필드는 문자열이 아닐 수도 있다(dict/list 등).
    정규식/요약 처리가 가능하도록 안전하게 문자열로 변환한다.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        # 뉴스 본문이 dict로 오는 경우(일부 공급자) 핵심 텍스트 후보를 먼저 뽑아본다.
        for k in ("title", "summary", "content", "description", "text", "body"):
            v = value.get(k)
            if isinstance(v, str) and v.strip():
                return v
        # dict 안에 문자열이 흩어져 있으면 이어붙여 요약 재료로 사용
        chunks = []
        for _, v in value.items():
            if isinstance(v, str) and v.strip():
                chunks.append(v.strip())
            if len(chunks) >= 3:
                break
        if chunks:
            return " ".join(chunks)
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    if isinstance(value, list):
        # 리스트는 문자열 요소만 일부 이어붙인다.
        chunks = []
        for v in value:
            if isinstance(v, str) and v.strip():
                chunks.append(v.strip())
            elif isinstance(v, dict):
                t = _to_text(v)
                if t.strip():
                    chunks.append(t.strip())
            if len(chunks) >= 3:
                break
        if chunks:
            return " ".join(chunks)
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _first_core_sentence(text: str) -> str:
    """
    긴 본문을 그대로 내보내지 않기 위해 '첫 핵심 문장'만 뽑는다.
    마침표/물음표/느낌표 기준으로 1문장을 우선 추출하고,
    그게 없으면 앞부분을 자른다.
    """
    t = _normalize_whitespace(_strip_html(_to_text(text)))
    if not t:
        return ""
    # 흔한 문장 구분자 기준으로 첫 문장 추출
    parts = re.split(r"(?<=[\.\!\?])\s+", t, maxsplit=1)
    first = parts[0] if parts else t
    # 한국어 문장(…다.)만 있는 케이스 보정
    if len(first) < 20 and "다." in t:
        first = t.split("다.", 1)[0].strip() + "다."
    return _normalize_whitespace(first)


def _one_or_two_sentences(text: str, min_first_len: int = 32) -> str:
    """
    첫 문장이 너무 짧으면(정보가 부족하면) 두 번째 문장까지 붙여준다.
    단, 최종 출력은 나중에 _truncate_one_line()로 길이 제한된다.
    """
    t = _normalize_whitespace(_strip_html(_to_text(text)))
    if not t:
        return ""
    sentences = re.split(r"(?<=[\.\!\?])\s+", t)
    if not sentences:
        return t
    first = _normalize_whitespace(sentences[0])
    if len(first) >= min_first_len or len(sentences) == 1:
        return first
    second = _normalize_whitespace(sentences[1])
    combined = _normalize_whitespace(f"{first} {second}".strip())
    return combined if combined else first


def _truncate_one_line(text: str, max_len: int = 140) -> str:
    t = _normalize_whitespace(text)
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"


def _classify_sentiment_kor(text: str) -> str:
    """
    키워드 점수 기반 감성 분류(가벼운 휴리스틱).
    반환: "호재" | "악재" | "중립"
    """
    t = (text or "").lower()
    pos_keywords = [
        "상향", "호실적", "흑자", "기대", "확대", "증가", "성장", "수주", "계약", "승인", "돌파",
        "투자", "채택", "파트너", "파트너십", "협력", "신제품", "출시", "인수", "자사주", "매입",
        "beat", "beats", "upgrade", "upgraded", "record", "surge", "rally",
        "raises guidance", "raise guidance", "guidance raise", "strong demand",
    ]
    neg_keywords = [
        "하향", "부진", "적자", "감소", "우려", "경고", "리콜", "소송", "규제", "조사", "충격",
        "중단", "연기", "해킹", "유출", "벌금", "제재", "하락", "급락", "리스크", "불확실",
        "miss", "misses", "downgrade", "downgraded", "plunge", "lawsuit", "recall",
        "cuts guidance", "cut guidance", "guidance cut", "weak demand",
    ]

    def _count_hits(keywords):
        return sum(1 for k in keywords if k in t)

    pos = _count_hits(pos_keywords)
    neg = _count_hits(neg_keywords)

    # 아주 강한 단어들은 가중치를 조금 더 준다
    if "급락" in t or "plunge" in t:
        neg += 1
    if "급등" in t or "surge" in t:
        pos += 1

    if pos >= neg + 1 and pos > 0:
        return "호재"
    if neg >= pos + 1 and neg > 0:
        return "악재"
    return "중립"


def _extract_news_text(item: dict) -> str:
    # 우선순위: title > summary > content > description
    text = (
        item.get("title")
        or item.get("summary")
        or item.get("content")
        or item.get("description")
    )
    return _to_text(text)


def get_news_summary(ticker, 현재가, 전일_종가):
    """
    뉴스 요약(사람이 읽기 쉬운 한 줄) 최대 3개와 당일 변동률을 반환한다.
    - title이 없더라도 summary/content/description이 있으면 활용한다.
    - HTML 제거, 공백 정리, 첫 문장 추출, 길이 제한(기본 150자)
    - 각 뉴스에 호재/악재/중립 라벨을 붙인다.
    """
    try:
        변동률 = ((현재가 - 전일_종가) / 전일_종가) * 100 if 전일_종가 else 0
        뉴스_raw = get_news(ticker)
        if not 뉴스_raw:
            return None, 변동률
        뉴스_목록 = []
        for item in 뉴스_raw:
            raw_text = _extract_news_text(item if isinstance(item, dict) else {})
            one = _truncate_one_line(_one_or_two_sentences(raw_text), max_len=140)
            if not one:
                continue
            sentiment = _classify_sentiment_kor(one)
            뉴스_목록.append({"text": one, "sentiment": sentiment})
            if len(뉴스_목록) >= 3:
                break
        return (뉴스_목록 if 뉴스_목록 else None), 변동률
    except Exception as e:
        print(f"뉴스를 불러오지 못했습니다: {e}")
        return None, 0
