"""
data_loader.py — 주가 데이터 및 뉴스 수집 (업그레이드 v3)
=============================================================
[뉴스 업그레이드 내용 — 중학생 설명]

■ 기존 방식의 문제점
  yfinance의 뉴스 API(Ticker.news)를 단독으로 사용했는데,
  Yahoo Finance가 2024년부터 API 구조를 자꾸 바꾸면서 뉴스가
  0개로 오거나 제목만 오고 내용이 없는 경우가 잦아졌다.
  또한 "오늘 왜 올랐나/내렸나"와 관련 없는 일반 기사가 섞여 들어왔다.

■ 새로운 방식: 3중 소스 폴백 체계
  소스 1 (yfinance)  → 안 되면 소스 2로
  소스 2 (Alpha Vantage NEWS_SENTIMENT API)
        → API 키 있으면 사용. 감성 점수(bullish/bearish)까지 제공
        → 무료 25회/일, 유료 시 무제한
        → 환경변수: ALPHAVANTAGE_API_KEY
  소스 3 (Gemini grounding 검색)
        → "왜 오늘 급등/급락했나"를 Gemini에게 직접 물어봄
        → 변동률 ±2% 이상인 날만 호출 (API 절약)
        → 환경변수: GEMINI_API_KEY (이미 있음)

■ 등락 관련 뉴스 필터링
  - 변동률 ±2% 이상인 날만 뉴스를 적극 수집 (평범한 날은 생략)
  - 이벤트 키워드(어닝/FDA/계약/소송 등) 감지 → 우선 표시
  - 감성 분류 강화: 단순 키워드 → 가중치 점수제로 변경
  - 뉴스 제목만이 아니라 summary까지 보고 분류

[API 키 설정 방법 — GitHub Secrets에 추가]
  ALPHAVANTAGE_API_KEY: https://www.alphavantage.co/support/#api-key
                        (무료 가입 후 즉시 발급, 25회/일)
  GEMINI_API_KEY: 이미 있음 (뉴스 검색에도 재사용)
"""

import os
import re
import json
import time
import requests
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_cache: dict = {}

# ── 이벤트 키워드: 주가 등락에 직접 영향을 주는 단어들 ──────────
# 이 단어가 뉴스에 있으면 "등락 관련 기사"로 우선 표시한다.
_EVENT_KEYWORDS_POS = [
    # 한국어
    "어닝 서프라이즈", "실적 상회", "흑자 전환", "깜짝 실적", "수주", "FDA 승인",
    "임상 성공", "계약 체결", "자사주 매입", "배당 증가", "목표주가 상향",
    "신규 상장", "합병", "인수", "파트너십", "매출 성장", "영업이익 증가",
    "가이던스 상향", "분기 최고", "연간 최고",
    # 영어
    "earnings beat", "beats estimates", "raises guidance", "fda approval",
    "clinical trial success", "contract win", "buyback", "dividend increase",
    "price target raised", "acquisition", "merger", "partnership",
    "record revenue", "record profit", "guidance raise", "strong demand",
    "upgrade", "upgraded",
]
_EVENT_KEYWORDS_NEG = [
    # 한국어
    "어닝 쇼크", "실적 부진", "적자 전환", "대규모 손실", "소송", "규제",
    "FDA 거부", "임상 실패", "리콜", "해킹", "유출", "벌금", "제재",
    "가이던스 하향", "목표주가 하향", "구조조정", "감원", "파산", "상장 폐지",
    "영업이익 감소", "매출 감소", "부도", "횡령", "배임",
    # 영어
    "earnings miss", "misses estimates", "cuts guidance", "fda rejection",
    "clinical trial failure", "recall", "lawsuit", "investigation", "fine",
    "sanctions", "layoffs", "bankruptcy", "guidance cut", "weak demand",
    "downgrade", "downgraded", "data breach", "cybersecurity incident",
]


def clear_cache():
    """
    메모리에 저장된 주가 데이터 캐시를 전부 삭제한다.

    [중학생 설명]
    같은 실행 안에서 데이터를 여러 번 쓸 때 속도를 위해 캐시에 저장해두는데,
    새 실행을 시작할 때는 오래된 데이터가 아닌 최신 데이터를 받아야 하므로
    시작 시점에 캐시를 비워준다.
    """
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
        time.sleep(0.12)

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
    업비트 API에서 원화(KRW) 기반 암호화폐 OHLCV 데이터를 가져온다.

    [중학생 설명]
    비트코인, 이더리움 등 업비트에서 거래되는 코인의 일봉 데이터를 받아온다.
    yfinance가 한국 원화 기준 코인 데이터를 제공하지 않기 때문에
    업비트 공개 REST API를 직접 호출해서 가져온다.

    ticker 예: 'BTC-KRW', 'UPBIT:ETH', 'KRW-SOL'
    반환값: OHLCV pandas DataFrame (Open/High/Low/Close/Volume)
    """
    cache_key = f"{ticker}_{period}_upbit"
    if cache_key in _cache:
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
    여러 종목의 주가 데이터를 한꺼번에 다운로드해서 캐시에 저장한다.

    [중학생 설명]
    100개 종목을 1개씩 받으면 100번 서버에 요청해야 해서 느리다.
    이 함수는 50개씩 묶어서 한 번에 요청하므로 훨씬 빠르다.
    받은 데이터는 _cache에 저장해두고, 이후 get_price_data()가 재사용한다.
    업비트 코인은 별도 API를 쓰므로 yf_tickers와 분리해서 처리한다.
    """
    upbit_tickers = [t for t in tickers if _is_upbit_ticker(t)]
    yf_tickers    = [t for t in tickers if not _is_upbit_ticker(t)]

    for ticker in upbit_tickers:
        cache_key = f"{ticker}_{period}_upbit"
        if cache_key not in _cache:
            get_upbit_price(ticker, period)

    for i in range(0, len(yf_tickers), chunk_size):
        묶음 = yf_tickers[i:i + chunk_size]
        ticker_str = " ".join(묶음)
        print(f"  📥 일괄 다운로드 ({i+1}~{min(i+chunk_size, len(yf_tickers))}/{len(yf_tickers)}종목, period={period})...")
        try:
            df_all = yf.download(
                ticker_str, period=period,
                progress=False, auto_adjust=True, group_by="ticker"
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
    종목 1개의 주가 데이터를 반환한다. 캐시에 있으면 캐시에서, 없으면 다운로드.

    [중학생 설명]
    bulk_download()가 먼저 실행되면 캐시에 데이터가 있어서 즉시 반환되고,
    캐시가 없으면 yfinance 또는 업비트 API에서 직접 받아온다.
    ticker 예: '005930.KS'(삼성전자), 'NVDA'(엔비디아), 'BTC-KRW'(비트코인)
    반환값: OHLCV pandas DataFrame 또는 None (데이터 없을 시)
    """
    if _is_upbit_ticker(ticker):
        return get_upbit_price(ticker, period)
    cache_key = f"{ticker}_{period}"
    if cache_key in _cache:
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


# ─────────────────────────────────────────
# 뉴스 소스 1: yfinance (기존)
# ─────────────────────────────────────────

def _get_news_yfinance(ticker: str) -> list[dict]:
    """
    yfinance Ticker.news로 뉴스를 가져온다.
    Yahoo Finance API가 자주 바뀌므로, 실패 시 빈 리스트 반환.
    """
    if _is_upbit_ticker(ticker):
        return []
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return []
        result = []
        for item in news[:5]:
            title   = _to_text(item.get("title") or item.get("headline") or "")
            summary = _to_text(item.get("summary") or item.get("description") or "")
            if not title and not summary:
                continue
            result.append({
                "title":   title,
                "summary": summary,
                "source":  "yfinance",
            })
        return result
    except Exception:
        return []


# ─────────────────────────────────────────
# 뉴스 소스 2: Alpha Vantage NEWS_SENTIMENT
# ─────────────────────────────────────────

def _get_news_alphavantage(ticker: str) -> list[dict]:
    """
    Alpha Vantage NEWS_SENTIMENT API로 뉴스와 감성 점수를 가져온다.

    [중학생 설명]
    Alpha Vantage는 금융 데이터 전문 회사로, 뉴스 API를 무료로 제공한다.
    단순히 제목만 주는 게 아니라 "이 뉴스가 주가에 호재인지 악재인지"를
    0~1 사이 점수로 알려준다 (0.5 이상 = 호재, 0.5 미만 = 악재).

    무료 25회/일 → 매수 신호 종목에만 호출하면 충분하다.
    API 키: https://www.alphavantage.co/support/#api-key (무료 가입)
    GitHub Secrets에 ALPHAVANTAGE_API_KEY로 등록 필요.

    ticker 형식 주의:
    - 한국주식 (005930.KS) → Alpha Vantage 지원 안 함 → 빈 리스트 반환
    - 미국주식 (AAPL, NVDA 등) → 지원
    - 크립토 (BTC-USD) → CRYPTO:BTC 형식으로 변환
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        # run_analysis.yml env 섹션에 ALPHAVANTAGE_API_KEY가 없거나
        # GitHub Secrets에 등록되지 않은 경우
        # → 뉴스 소스 2번(Alpha Vantage)을 건너뛰고 yfinance로 폴백
        return []

    # 한국 주식(.KS, .KQ)은 Alpha Vantage 티커 지원 안 함
    # → get_news() 함수에서 yfinance로 폴백됨
    if ticker.endswith((".KS", ".KQ")):
        return []

    # 크립토 티커 변환: BTC-USD → CRYPTO:BTC
    av_ticker = ticker
    if ticker.endswith("-USD") and len(ticker) <= 10:
        av_ticker = f"CRYPTO:{ticker.replace('-USD', '')}"

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function":                "NEWS_SENTIMENT",
            "tickers":                av_ticker,
            "apikey":                 api_key,
            "limit":                  10,   # 5→10: 필터링 후 충분한 여유분 확보
            "sort":                   "RELEVANCE",
            "relevance_score_threshold": "0.3",  # 관련도 0.3 미만 기사 제거
        }
        resp = requests.get(url, params=params, timeout=12)
        if resp.status_code != 200:
            return []

        data = resp.json()
        feed = data.get("feed", [])
        if not feed:
            return []

        result = []
        for item in feed[:5]:
            title   = item.get("title", "")
            summary = item.get("summary", "")
            if not title:
                continue

            # Alpha Vantage 감성 점수: 0~1 (0.5 기준, 높을수록 긍정)
            # ticker_sentiment 배열에서 이 종목의 점수만 추출
            # 해당 ticker의 sentiment가 없는 기사는 무관 기사일 가능성 높음 → 제외
            ticker_sentiments = item.get("ticker_sentiment", [])
            ticker_mentioned = any(
                ts.get("ticker", "").upper() == av_ticker.upper()
                for ts in ticker_sentiments
            )
            if not ticker_mentioned:
                continue  # 이 종목이 언급 안 된 기사는 건너뜀

            av_sentiment_label = "중립"
            for ts in ticker_sentiments:
                if ts.get("ticker", "").upper() == av_ticker.upper():
                    score = float(ts.get("ticker_sentiment_score", 0.5))
                    label = ts.get("ticker_sentiment_label", "Neutral")
                    if label in ("Bullish", "Somewhat-Bullish") or score >= 0.15:
                        av_sentiment_label = "호재"
                    elif label in ("Bearish", "Somewhat-Bearish") or score <= -0.15:
                        av_sentiment_label = "악재"
                    else:
                        av_sentiment_label = "중립"
                    break

            result.append({
                "title":     title,
                "summary":   summary[:200],
                "sentiment": av_sentiment_label,
                "source":    "alphavantage",
            })
        return result[:5]  # 최대 5개만 반환

    except Exception as e:
        print(f"⚠️ Alpha Vantage 뉴스 오류 ({ticker}): {e}")
        return []


# ─────────────────────────────────────────
# 뉴스 소스 3: Gemini Grounding 검색
# (변동률 ±2% 이상인 날, 급등/급락 이유 직접 검색)
# ─────────────────────────────────────────

def _get_news_gemini_grounding(ticker: str, name: str, 변동률: float) -> list[dict]:
    """
    Gemini의 Google Search Grounding 기능으로 급등/급락 이유를 검색한다.

    [중학생 설명]
    변동률이 ±2% 이상인 날에는 Gemini에게 직접 물어본다:
    "오늘 삼성전자가 왜 3% 올랐어?"
    Gemini는 Google 검색을 실시간으로 해서 답변을 만들어준다.
    이게 가장 "등락에 직접 연관된 뉴스"를 가져오는 방법이다.

    변동률 ±2% 미만인 날은 호출하지 않는다 (API 절약).
    """
    if abs(변동률) < 2.0:
        return []  # 소폭 등락은 Gemini 검색 불필요

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return []

    방향 = "급등" if 변동률 > 0 else "급락"
    query = (
        f"{name}({ticker}) 오늘 주가 {방향} 이유 "
        f"(변동률 {변동률:+.1f}%). "
        f"오늘 발표된 뉴스, 실적, 공시, 이벤트 중심으로 짧게 요약해줘. "
        f"한국어 2~3줄."
    )

    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent"
        )
        headers = {
            "x-goog-api-key": api_key,
            "content-type":   "application/json",
        }
        # grounding: Google Search 실시간 검색 활성화
        payload = {
            "contents": [{"role": "user", "parts": [{"text": query}]}],
            "tools": [{"google_search": {}}],  # Gemini Grounding 핵심 설정
            "generationConfig": {"maxOutputTokens": 300},
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code != 200:
            return []

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return []
        parts = candidates[0].get("content", {}).get("parts", [])
        text = " ".join(p.get("text", "") for p in parts if p.get("text"))
        text = text.strip()
        if not text:
            return []

        # 검색 출처(grounding sources) 추출
        grounding_meta = candidates[0].get("groundingMetadata", {})
        sources = grounding_meta.get("groundingChunks", [])
        source_titles = [
            s.get("web", {}).get("title", "")
            for s in sources[:2] if s.get("web")
        ]
        source_str = " | ".join(t for t in source_titles if t)

        결과_텍스트 = f"[{방향} 원인] {text}"
        if source_str:
            결과_텍스트 += f" (출처: {source_str})"

        # 감성은 변동률 방향으로 자동 결정
        sentiment = "호재" if 변동률 > 0 else "악재"

        return [{
            "title":     결과_텍스트[:250],
            "summary":   "",
            "sentiment": sentiment,
            "source":    "gemini_grounding",
        }]

    except Exception as e:
        print(f"⚠️ Gemini Grounding 뉴스 오류 ({ticker}): {e}")
        return []


# ─────────────────────────────────────────
# 뉴스 통합 수집 (3중 소스 폴백)
# ─────────────────────────────────────────



def get_today_open(ticker: str) -> float | None:
    """
    당일 시가를 1분봉으로 조회해서 갭을 확인한다.

    [중학생 설명]
    전날 종가 기준으로 신호가 났는데, 오늘 아침에 갭업(+3%)이면
    이미 늦은 진입이다. 이 함수는 장이 열린 직후 시가를 확인해서
    갭이 2% 이상이면 알림에 "갭 주의" 경고를 표시한다.

    반환값: 당일 시가 (float) 또는 None
    """
    if _is_upbit_ticker(ticker):
        return None  # 업비트는 24시간 거래라 갭 개념 없음
    try:
        df = yf.download(
            ticker, period="1d", interval="1m",
            progress=False, auto_adjust=True
        )
        if df is None or df.empty:
            return None
        return float(df["Open"].squeeze().iloc[0])
    except Exception:
        return None



def get_weekly_data(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """
    주간봉(Weekly) 데이터를 가져온다.

    [중학생 설명]
    일봉은 하루 단위 가격이고, 주간봉은 한 주 단위 가격이다.
    일봉이 단기 노이즈에 흔들릴 때, 주간봉은 진짜 큰 흐름을 보여준다.
    일봉 MACD는 상승이지만 주간봉 MACD가 하락이라면?
    → 단기 반등일 뿐이고 큰 흐름은 하락 → 진입 위험 신호

    반환값: 주간봉 OHLCV DataFrame 또는 None
    """
    if _is_upbit_ticker(ticker):
        return None  # yfinance interval 지원 안 함
    cache_key = f"{ticker}_{period}_weekly"
    if cache_key in _cache:
        return _cache[cache_key]
    try:
        df = yf.download(
            ticker, period=period, interval="1wk",
            progress=False, auto_adjust=True
        )
        if df is None or df.empty or len(df) < 26:
            return None
        _cache[cache_key] = df
        return df
    except Exception:
        return None

def get_news(ticker: str, name: str = "", 변동률: float = 0.0) -> list[dict]:
    """
    3중 소스 폴백으로 뉴스를 수집한다.

    [수집 우선순위]
    1) 변동률 ±2% 이상이면 → Gemini Grounding 검색 (가장 관련성 높음)
    2) Alpha Vantage API → 감성 점수 포함된 정제된 뉴스
    3) yfinance → 기존 방식 (폴백)

    결과는 이벤트 키워드가 있는 뉴스를 앞으로 정렬해서 반환한다.
    """
    if _is_upbit_ticker(ticker):
        return []

    raw_items: list[dict] = []

    # ── 소스 1: Gemini Grounding (급등/급락 원인 직접 검색) ──
    if abs(변동률) >= 2.0:
        gemini_news = _get_news_gemini_grounding(ticker, name, 변동률)
        raw_items.extend(gemini_news)

    # ── 소스 2: Alpha Vantage (감성 점수 포함) ──────────────
    # 미국 주식·크립토만 지원 (한국 주식은 yfinance 폴백)
    # ALPHAVANTAGE_API_KEY가 run_analysis.yml env에 있어야 동작
    if len(raw_items) < 3:
        av_news = _get_news_alphavantage(ticker)
        if av_news:
            print(f"  📰 Alpha Vantage 뉴스: {ticker} {len(av_news)}개")
        raw_items.extend(av_news)

    # ── 소스 3: yfinance (폴백) ─────────────────────────────
    if len(raw_items) < 2:
        yf_news = _get_news_yfinance(ticker)
        raw_items.extend(yf_news)

    if not raw_items:
        return []

    # ── 이벤트 키워드 스코어링 → 중요 뉴스 앞으로 ─────────
    def _event_score(item: dict) -> int:
        """
        이벤트 키워드가 많을수록 높은 점수.
        등락에 직접 영향을 주는 기사가 앞에 오도록 정렬한다.
        """
        text = (
            (item.get("title") or "") + " " +
            (item.get("summary") or "")
        ).lower()
        pos = sum(1 for k in _EVENT_KEYWORDS_POS if k in text)
        neg = sum(1 for k in _EVENT_KEYWORDS_NEG if k in text)
        # Gemini Grounding 결과는 가장 관련성이 높으므로 가산점
        source_bonus = 2 if item.get("source") == "gemini_grounding" else 0
        return pos + neg + source_bonus

    raw_items.sort(key=_event_score, reverse=True)
    return raw_items[:5]  # 최대 5개


# ─────────────────────────────────────────
# 감성 분류 (가중치 점수제 — 업그레이드)
# ─────────────────────────────────────────

def _classify_sentiment(item: dict, text: str) -> str:
    """
    뉴스 감성을 분류한다.

    [분류 우선순위]
    1) Alpha Vantage가 이미 감성을 계산해줬으면 그대로 사용
    2) Gemini Grounding 결과는 변동률 방향으로 이미 설정됨
    3) 그 외는 이벤트 키워드 가중치 점수로 자체 분류

    기존 단순 키워드 카운팅 → 가중치 점수제로 업그레이드:
    - 강력 이벤트 키워드(어닝 쇼크, FDA 승인 등)는 2점
    - 일반 키워드는 1점
    """
    # API가 이미 분류해준 경우 그대로 사용
    if item.get("sentiment") in ("호재", "악재", "중립"):
        return item["sentiment"]

    t = text.lower()

    # 강력 이벤트 (2점짜리)
    strong_pos = ["어닝 서프라이즈", "fda approval", "earnings beat", "raises guidance",
                  "record revenue", "깜짝 실적", "임상 성공"]
    strong_neg = ["어닝 쇼크", "fda rejection", "earnings miss", "cuts guidance",
                  "임상 실패", "파산", "bankruptcy", "대규모 손실"]

    pos_score = sum(2 for k in strong_pos if k in t)
    neg_score = sum(2 for k in strong_neg if k in t)

    pos_score += sum(1 for k in _EVENT_KEYWORDS_POS if k in t and k not in strong_pos)
    neg_score += sum(1 for k in _EVENT_KEYWORDS_NEG if k in t and k not in strong_neg)

    # 강도 부사 보정
    if "급락" in t or "plunge" in t or "crash" in t:
        neg_score += 2
    if "급등" in t or "surge" in t or "soar" in t:
        pos_score += 2

    if pos_score >= neg_score + 1 and pos_score > 0:
        return "호재"
    if neg_score >= pos_score + 1 and neg_score > 0:
        return "악재"
    return "중립"


# ─────────────────────────────────────────
# 뉴스 요약 최종 반환 (외부 인터페이스)
# ─────────────────────────────────────────

def get_news_summary(ticker: str, 현재가: float, 전일_종가: float, name: str = ""):
    """
    뉴스 요약(한 줄) 최대 3개와 당일 변동률을 반환한다.

    [업그레이드 내용]
    - 변동률을 먼저 계산해서 뉴스 수집 전략 결정 (±2% 기준)
    - 3중 소스 폴백 (Gemini Grounding > Alpha Vantage > yfinance)
    - 이벤트 키워드 기반 중요 뉴스 우선 정렬
    - 뉴스 출처 표시 (어디서 가져왔는지)
    - 감성 분류 강화 (가중치 점수제)

    반환값: (뉴스_목록, 변동률)
    뉴스_목록 형식: [{"text": "...", "sentiment": "호재/악재/중립"}, ...]
    """
    try:
        변동률 = ((현재가 - 전일_종가) / 전일_종가) * 100 if 전일_종가 else 0.0
    except Exception:
        변동률 = 0.0

    try:
        raw_items = get_news(ticker, name=name, 변동률=변동률)
        if not raw_items:
            return None, 변동률

        뉴스_목록 = []
        for item in raw_items:
            # 제목과 요약을 합쳐서 한 줄로 만들기
            title   = _to_text(item.get("title",   ""))
            summary = _to_text(item.get("summary", ""))
            source  = item.get("source", "yfinance")

            # Gemini Grounding은 title에 이미 충분한 내용이 있음 (이미 한국어)
            if source == "gemini_grounding":
                one_line = _truncate_one_line(title, max_len=200)
            else:
                # 제목이 짧으면 요약 일부 추가
                combined = title
                if len(title) < 40 and summary:
                    combined = f"{title} — {summary[:80]}"
                one_line = _truncate_one_line(
                    _normalize_whitespace(_strip_html(combined)),
                    max_len=160
                )

            if not one_line:
                continue

            sentiment = _classify_sentiment(item, one_line)
            # source 필드 보존: scheduler_job.py에서 번역 여부 판단에 사용
            # gemini_grounding = 이미 한국어, alphavantage/yfinance = 영어 → 번역 필요
            뉴스_목록.append({
                "text":      one_line,
                "sentiment": sentiment,
                "source":    source,
            })

            if len(뉴스_목록) >= 3:
                break

        return (뉴스_목록 if 뉴스_목록 else None), 변동률

    except Exception as e:
        print(f"뉴스를 불러오지 못했습니다: {e}")
        return None, 변동률


# ─────────────────────────────────────────
# 텍스트 처리 유틸리티 (기존 유지)
# ─────────────────────────────────────────

def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for k in ("title", "summary", "content", "description", "text", "body"):
            v = value.get(k)
            if isinstance(v, str) and v.strip():
                return v
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


def _truncate_one_line(text: str, max_len: int = 160) -> str:
    t = _normalize_whitespace(text)
    if len(t) <= max_len:
        return t
    return t[:max_len - 1].rstrip() + "…"
