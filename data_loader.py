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
        → 변동률 ±4.5% 이상인 날만 호출 (API 절약)
        → 환경변수: GEMINI_API_KEY (이미 있음)

■ 등락 관련 뉴스 필터링
  - 변동률 ±4.5% 이상인 날만 뉴스를 적극 수집 (평범한 날은 생략)
  - 이벤트 키워드(어닝/FDA/계약/소송 등) 감지 → 우선 표시
  - 감성 분류 강화: 단순 키워드 → 가중치 점수제로 변경
  - 뉴스 제목만이 아니라 summary까지 보고 분류

[VIX 폴백 — v5.9 추가]
  get_vix_from_fred(): yfinance ^VIX가 GitHub Actions 해외 IP에서 차단될 때
  FRED(미국 연준) 공개 API로 VIX를 가져온다. IP 차단 없음, 키 불필요.
  scheduler_job·market_phase가 공유한다 (DRY).

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
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# yfinance 1.4.1은 내부적으로 curl_cffi를 사용해 crumb을 자동 관리한다.
# 외부에서 별도 세션을 만들어 주입하면 crumb 없는 새 세션이 생겨 오히려 401이 증가하므로
# 외부 세션을 사용하지 않는다. _warmup_yfinance_crumb()으로 사전 crumb 획득.
_yf_session = None

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


def get_vix_from_fred() -> float | None:
    """
    FRED(미국 연방준비은행) 공개 API로 최신 VIX 종가를 가져온다.

    [중학생 설명]
    yfinance의 ^VIX는 GitHub Actions 해외 IP에서 자주 차단된다
    (로그에 "possibly delisted"로 찍힘 — 실제 폐지가 아니라 IP 차단).
    FRED는 미국 정부 공개 데이터라 IP 차단이 없고 API 키도 불필요해서
    ^VIX 폴백으로 안성맞춤이다.

    series_id "VIXCLS" = CBOE 변동성지수 일별 종가.
    최근 7일 중 가장 최신 유효값을 반환. 실패 시 None.

    scheduler_job(VIX 1회 주입)과 market_phase(국면 판단) 양쪽에서
    동일하게 쓰던 코드를 여기 하나로 모았다 (DRY).
    """
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id":         "VIXCLS",
                "api_key":           "anonymous",
                "file_type":         "json",
                "sort_order":        "desc",
                "limit":             5,
                "observation_start": (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),
            },
            timeout=10,
        )
        if r.status_code == 200:
            obs = [o for o in r.json().get("observations", [])
                   if o.get("value", ".") != "."]
            if obs:
                return float(obs[0]["value"])
    except Exception:
        pass
    return None


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

def _warmup_yfinance_crumb():
    """
    yfinance YfData 싱글톤의 crumb을 사전 획득한다.

    [중학생 설명]
    Yahoo Finance는 요청할 때마다 "crumb"이라는 인증 토큰이 필요하다.
    이 함수를 먼저 실행하면 토큰을 미리 받아두기 때문에
    이후 100개 종목을 병렬로 분석해도 "401 Invalid Crumb" 오류가 나지 않는다.

    crumb 획득 확인 후 실패 시 최대 3회 재시도.
    """
    import time as _time
    try:
        import yfinance as yf
        from yfinance.data import YfData

        for attempt in range(3):
            try:
                yfdata = YfData()
                if yfdata._crumb is not None:
                    # 이미 crumb 있음
                    print("  ✅ yfinance crumb 이미 보유 — 워밍업 건너뜀")
                    return

                # SPY + QQQ 순서로 다운로드해서 crumb 강제 획득
                # SPY가 실패해도 QQQ에서 crumb을 얻을 수 있음
                for sym in ["SPY", "QQQ"]:
                    try:
                        yf.download(sym, period="1d", progress=False, auto_adjust=True)
                    except Exception:
                        pass

                # crumb 획득 확인
                yfdata2 = YfData()
                if yfdata2._crumb is not None:
                    print(f"  ✅ yfinance crumb 사전 획득 완료 (시도 {attempt+1}회)")
                    return

            except Exception:
                pass

            if attempt < 2:
                _time.sleep(2)

        print("  ⚠️ crumb 워밍업 3회 모두 실패 — 계속 진행 (일부 401 발생 가능)")

    except Exception as e:
        print(f"  ⚠️ crumb 워밍업 오류 (무시): {e}")


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
# (변동률 ±4.5% 이상인 날, 급등/급락 이유 직접 검색)
# ─────────────────────────────────────────

def _get_news_gemini_grounding(ticker: str, name: str, 변동률: float) -> list[dict]:
    """
    Gemini의 Google Search Grounding 기능으로 급등/급락 이유를 검색한다.

    [중학생 설명]
    변동률이 ±4.5% 이상인 날에는 Gemini에게 직접 물어본다:
    "오늘 삼성전자가 왜 3% 올랐어?"
    Gemini는 Google 검색을 실시간으로 해서 답변을 만들어준다.
    이게 가장 "등락에 직접 연관된 뉴스"를 가져오는 방법이다.

    변동률 ±4.5% 미만인 날은 호출하지 않는다 (API 절약).
    """
    if abs(변동률) < 4.5:   # [v5.9] ±2% → ±4.5%: Gemini 호출 빈도 축소 (한도 절약)
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




def bulk_download_weekly(tickers, period="2y", chunk_size=50):
    """
    주간봉 데이터를 미리 일괄 다운로드해서 캐시에 저장한다.
    
    [이유]
    기존엔 calc_signals()가 종목마다 get_weekly_data()를 개별 호출해서
    103개 종목 × 2년치 주간봉 = 실행 시간이 늘어났다.
    이 함수를 run_analysis() 시작 시 한 번 호출하면
    이후 get_weekly_data()는 캐시에서 즉시 반환한다.
    """
    yf_tickers = [t for t in tickers
                  if not _is_upbit_ticker(t)]
    if not yf_tickers:
        return

    for i in range(0, len(yf_tickers), chunk_size):
        chunk = yf_tickers[i:i + chunk_size]
        chunk_str = " ".join(chunk)
        try:
            import yfinance as yf
            df_all = yf.download(
                chunk_str, period=period, interval="1wk",
                progress=False, auto_adjust=True, group_by="ticker",
            )
            for ticker in chunk:
                cache_key = f"{ticker}_{period}_weekly"
                if cache_key in _cache:
                    continue
                try:
                    if len(chunk) == 1:
                        df = df_all
                    else:
                        df = df_all[ticker] if ticker in df_all.columns.get_level_values(0) else None
                    if df is not None and not df.empty:
                        _cache[cache_key] = df
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ 주간봉 일괄 다운로드 오류: {e}")

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


# ─────────────────────────────────────────
# 뉴스 소스 4: GNews API (한국주식 포함 전종목)
# ─────────────────────────────────────────

def _get_news_gnews(ticker: str, name: str) -> list[dict]:
    """
    GNews API로 종목 관련 뉴스를 가져온다.

    [핵심 장점]
    - Google News 기반 → 한국어 뉴스 완벽 지원
    - 한국주식(삼성전자, SK하이닉스 등) 뉴스 가능 (AV는 한국주식 불가)
    - 100회/일 무료, API 키 발급 30초 (https://gnews.io)
    - GitHub Secrets: GNEWS_API_KEY

    쿼리 전략:
    - 한국주식: '{종목명} 주가' 한국어 검색
    - 미국주식: '{ticker} stock' 영어 검색
    """
    api_key = os.getenv("GNEWS_API_KEY", "")
    if not api_key:
        return []

    try:
        # 한국/미국 주식에 따라 검색어와 언어 설정
        if ticker.endswith((".KS", ".KQ")):
            query = f"{name} 주가 주식"
            lang  = "ko"
            country = "kr"
        else:
            query = f"{ticker} stock earnings"
            lang  = "en"
            country = "us"

        params = {
            "q":        query,
            "lang":     lang,
            "country":  country,
            "max":      5,
            "sortby":   "publishedAt",   # 최신순
            "token":    api_key,
        }
        resp = requests.get(
            "https://gnews.io/api/v4/search",
            params=params, timeout=10
        )
        if resp.status_code != 200:
            return []

        articles = resp.json().get("articles", [])
        if not articles:
            return []

        result = []
        for art in articles[:5]:
            title   = art.get("title",       "") or ""
            desc    = art.get("description", "") or ""
            if not title:
                continue
            result.append({
                "title":   title,
                "summary": desc[:150],
                "source":  "gnews",
            })
        return result

    except Exception as e:
        print(f"⚠️ GNews 뉴스 오류 ({ticker}): {e}")
        return []


# ─────────────────────────────────────────
# 뉴스 소스 5: NewsAPI.org (GNews 폴백)
# ─────────────────────────────────────────

def _get_news_newsapi(ticker: str, name: str) -> list[dict]:
    """
    NewsAPI.org로 종목 관련 뉴스를 가져온다.

    [특징]
    - 100회/일 무료 (GNews 한도 초과 시 폴백)
    - 한국어 검색 지원
    - API 키: https://newsapi.org (이메일 가입 즉시)
    - GitHub Secrets: NEWSAPI_KEY
    """
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        return []

    try:
        if ticker.endswith((".KS", ".KQ")):
            query = f"{name} 주식"
            lang  = "ko"
        else:
            query = f"{ticker} stock"
            lang  = "en"

        params = {
            "q":        query,
            "language": lang,
            "sortBy":   "publishedAt",
            "pageSize": 5,
            "apiKey":   api_key,
        }
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params=params, timeout=10
        )
        if resp.status_code != 200:
            return []

        articles = resp.json().get("articles", [])
        if not articles:
            return []

        result = []
        for art in articles[:5]:
            title = art.get("title",       "") or ""
            desc  = art.get("description", "") or ""
            if not title or title == "[Removed]":
                continue
            result.append({
                "title":   title,
                "summary": desc[:150],
                "source":  "newsapi",
            })
        return result

    except Exception as e:
        print(f"⚠️ NewsAPI 뉴스 오류 ({ticker}): {e}")
        return []

def _parse_rss_date(date_str: str):
    """RSS pubDate(RFC822)를 datetime으로 파싱. 실패 시 None."""
    if not date_str:
        return None
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str)
    except Exception:
        return None


def _clean_rss_title(title: str) -> str:
    """구글 뉴스 제목에서 ' - 출처' 꼬리를 제거."""
    title = (title or "").strip()
    if " - " in title:
        title = title.rsplit(" - ", 1)[0].strip()
    return title


# HTML 엔티티 → 일반 문자 매핑 (RSS description 정리용)
_HTML_ENTITIES = {
    "&quot;": '"', "&amp;": "&", "&lt;": "<",
    "&gt;": ">", "&#39;": "'", "&nbsp;": " ",
}


def _strip_html(text: str) -> str:
    """RSS description의 HTML 태그·엔티티를 제거하고 순수 텍스트만 남긴다."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)        # 모듈 상단 re 재사용
    for ent, ch in _HTML_ENTITIES.items():
        text = text.replace(ent, ch)
    return _normalize_whitespace(text)          # 공백 정리 헬퍼 재사용


# 구글 뉴스 RSS 요청 헤더 (브라우저로 위장 — 해외 IP 차단 완화)
_GOOGLE_RSS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

# 광고성·낚시성 제목 패턴 (품질 필터) — 이런 단어만 있고 알맹이 없는 기사 제외
_RSS_NOISE_PATTERNS = [
    "특징주", "이 종목", "오늘의 추천", "급등주 포착", "상한가 종목",
    "관심주", "추천주", "유망주", "테마주 정리", "장중 특징",
]


def _get_news_google_rss(ticker: str, name: str = "", 변동률: float = 0.0) -> list[dict]:
    """
    구글 뉴스 RSS로 뉴스를 수집한다. (한국주식 보강용 — v5.16 품질 개선)

    [중학생 설명]
    구글 뉴스 RSS는 키도 한도도 없는 공개 피드다. 한국 종목 뉴스가
    부족할 때 채워준다. v5.16에서 4가지를 개선해 품질을 높였다:
      1. 검색어 정교화 — 평상시 "종목명", 급등락 시 "종목명 실적 OR 수주 OR 계약"
         으로 '왜 움직였나'를 묻는 기사를 우선 수집
      2. 본문 요약(description) 활용 — 제목만 쓰던 걸 요약까지 파싱
      3. 발행일(pubDate) 필터 — 3일 초과 오래된 기사 제외, 최신 우선
      4. 품질 필터 — 광고성 제목·짧은 제목 제외, 한 언론사 최대 2개

    [주의 — 해외 IP 차단]
    구글이 GitHub Actions IP를 rate-limit할 수 있다. 검색어를 바꾸면
    해외 IP에서 결과가 달라질 수 있으므로 check_rss.py로 재검증 권장.
    실패 시 빈 리스트 → 폴백 체인에 영향 없음.

    한국 종목 전용 (.KS/.KQ). 미국은 Alpha Vantage가 커버.
    """
    import urllib.parse
    import xml.etree.ElementTree as _ET

    if not (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        return []
    if not name:
        name = ticker.replace(".KS", "").replace(".KQ", "")

    # ── 1. 검색어 정교화 ──────────────────────────────────
    # 급등락(±4.5% 이상)이면 "왜?"를 묻는 이벤트 지향 검색,
    # 평상시엔 종목명만으로 폭넓게 (단순 '주가' 기사 편중 방지)
    if abs(변동률) >= 4.5:
        검색어 = f"{name} 실적 OR 수주 OR 계약 OR 공시"
    else:
        검색어 = name
    q = urllib.parse.quote(검색어)
    url = (
        f"https://news.google.com/rss/search?q={q}"
        f"&hl=ko&gl=KR&ceid={urllib.parse.quote('KR:ko')}"
    )
    try:
        resp = requests.get(url, headers=_GOOGLE_RSS_HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        root = _ET.fromstring(resp.content)
    except Exception:
        return []

    # ── 3. 발행일 필터 기준 (3일 이내) ────────────────────
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=3)

    후보 = []
    for item in root.findall(".//item"):
        title = _clean_rss_title(item.findtext("title"))
        if not title:
            continue

        # ── 4. 품질 필터 ──────────────────────────────────
        # (a) 너무 짧은 제목 제외 (10자 미만 = 정보량 부족)
        if len(title) < 10:
            continue
        # (b) 광고성·낚시성 제목 제외
        if any(p in title for p in _RSS_NOISE_PATTERNS):
            continue

        # ── 3. 발행일 파싱 + 최신성 ───────────────────────
        pub_dt = _parse_rss_date(item.findtext("pubDate"))
        if pub_dt is not None and pub_dt < cutoff:
            continue  # 3일보다 오래된 기사 제외

        # ── 2. 본문 요약(description) 살리기 ──────────────
        summary = _strip_html(item.findtext("description"))
        # 요약이 제목과 거의 같으면 버림 (구글이 제목 반복하는 경우)
        if summary and summary[:20] == title[:20]:
            summary = ""

        src_el = item.find("source")
        publisher = (src_el.text or "").strip() if src_el is not None else "google_rss"

        후보.append({
            "title":     title,
            "summary":   summary[:200],   # 너무 길면 자름
            "source":    "google_rss",
            "publisher": publisher,
            "_pub":      pub_dt or now,    # 정렬용 (없으면 현재시각 취급)
        })

    if not 후보:
        return []

    # ── 3. 최신순 정렬 ────────────────────────────────────
    후보.sort(key=lambda x: x["_pub"], reverse=True)

    # ── 4. 한 언론사 최대 2개 (도배 방지) + 상위 5개 ──────
    출처_카운트 = {}
    result = []
    for c in 후보:
        pub = c["publisher"]
        출처_카운트[pub] = 출처_카운트.get(pub, 0) + 1
        if 출처_카운트[pub] > 2:
            continue
        c.pop("_pub", None)   # 내부 정렬용 필드 제거
        result.append(c)
        if len(result) >= 5:
            break

    return result


def get_news(ticker: str, name: str = "", 변동률: float = 0.0) -> list[dict]:
    """
    3중 소스 폴백으로 뉴스를 수집한다.

    [수집 우선순위 — 6중 폴백]
    1) Gemini Grounding → 변동률 ±4.5% 이상인 날, 급등/급락 이유 실시간 검색
    2) Alpha Vantage   → 미국주식 전용, 감성 점수 포함 (25회/일)
    3) 구글 뉴스 RSS    → 한국주식 1순위, 키·한도 없음 (v5.16 품질개선: 검색어 정교화·요약·최신성·품질필터)
    4) GNews           → RSS 미충족 종목 + 보강, Google News 기반 (100회/일)
    5) NewsAPI         → GNews 한도 초과 시 폴백 (100회/일)
    6) yfinance        → 최후 수단 (API 불안정)

    결과는 제목 중복 제거 후 이벤트 키워드 순으로 정렬해서 반환한다.
    """
    if _is_upbit_ticker(ticker):
        return []

    raw_items: list[dict] = []

    # ── 소스 1: Gemini Grounding (급등/급락 원인 직접 검색) ──
    # 변동률 ±4.5% 이상인 날만 Gemini Grounding 시도
    # ±1%는 일상적 노이즈 수준 — Gemini 호출 낭비 + 429 위험 증가
    if abs(변동률) >= 4.5:   # [v5.9] ±2% → ±4.5% (위 _get_news_gemini_grounding과 동일 기준 유지)
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

    # ── 소스 3: 구글 뉴스 RSS (한국주식 1순위, 키·한도 없음) ──
    # check_rss.py 진단 결과 해외 IP에서 한국·미국 모두 HTTP 200 확인됨(v5.11).
    # RSS는 무한 무료라 GNews(하루 100회)보다 먼저 써서 한도를 아낀다.
    # 한국 종목 전용 (미국은 Alpha Vantage가 이미 커버 → 함수 내부에서 빈 리스트).
    # ENABLE_GOOGLE_RSS=false 로 끌 수 있음 (기본 활성).
    if len(raw_items) < 3 and os.getenv("ENABLE_GOOGLE_RSS", "true").lower() == "true":
        rss_news = _get_news_google_rss(ticker, name, 변동률)
        if rss_news:
            print(f"  📰 구글RSS: {ticker} {len(rss_news)}개")
        raw_items.extend(rss_news)

    # ── 소스 4: GNews (RSS가 못 채운 종목 + 미국주식 한국어) ──
    # GNews는 하루 100회 한도라 RSS 다음 순서로 둬서 한도를 아낀다.
    if len(raw_items) < 3:
        gn_news = _get_news_gnews(ticker, name)
        if gn_news:
            print(f"  📰 GNews: {ticker} {len(gn_news)}개")
        raw_items.extend(gn_news)

    # ── 소스 5: NewsAPI (GNews 폴백) ────────────────────────
    if len(raw_items) < 3:
        na_news = _get_news_newsapi(ticker, name)
        if na_news:
            print(f"  📰 NewsAPI: {ticker} {len(na_news)}개")
        raw_items.extend(na_news)

    # ── 소스 6: yfinance (최후 수단) ────────────────────────
    # yfinance는 Yahoo Finance API 불안정으로 자주 실패.
    # GNews/NewsAPI로도 못 채웠을 때만 시도한다.
    if len(raw_items) < 2:
        yf_news = _get_news_yfinance(ticker)
        raw_items.extend(yf_news)

    if not raw_items:
        return []

    # ── 제목 중복 제거 (RSS·GNews가 같은 기사를 줄 수 있음) ──
    # 정규화(공백·기호 제거, 소문자)한 제목 앞 40자로 중복 판정
    _seen = set()
    _deduped = []
    for it in raw_items:
        _t = (it.get("title") or "").lower()
        _norm = "".join(c for c in _t if c.isalnum())[:40]
        if _norm and _norm in _seen:
            continue
        _seen.add(_norm)
        _deduped.append(it)
    raw_items = _deduped

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
    - 변동률을 먼저 계산해서 뉴스 수집 전략 결정 (±4.5% 기준)
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
