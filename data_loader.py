"""
data_loader.py — 주가 데이터 및 뉴스 수집 (일괄 다운로드 + 캐싱)
==================================================================
[속도 개선 핵심]
  기존: 종목 1개씩 개별 다운로드 → 100종목 = 300번 네트워크 요청
  개선: 종목 50개를 한 번에 묶어서 다운로드 → 100종목 = 6번 요청
        yfinance.download("AAPL NVDA TSLA ...", period="1y") 형태로 호출

  bulk_download(tickers, period) 함수가 핵심.
  run_analysis 시작 시 1회만 호출하면
  이후 get_price_data()는 모두 캐시에서 즉시 반환된다.
"""

import yfinance as yf
import pandas as pd

# 인메모리 캐시 (키: "티커_period" / 값: pandas DataFrame)
_cache = {}


def clear_cache():
    """캐시를 초기화한다. 매 분석 실행 시작 시 호출."""
    global _cache
    _cache = {}
    print("🗑️  데이터 캐시 초기화 완료")


def bulk_download(tickers, period="1y", chunk_size=50):
    """
    여러 종목을 한 번에 묶어서 다운로드하고 캐시에 저장한다.

    [왜 빠른가?]
    yfinance는 티커를 공백으로 구분해 한 번의 HTTP 요청으로
    여러 종목 데이터를 동시에 받을 수 있다.
    100종목 개별 요청 = 100번 왕복
    50개씩 묶음 요청  = 2번 왕복  → 약 50배 빠름

    chunk_size: 한 번에 요청할 최대 종목 수 (야후 서버 부하 방지를 위해 50 권장)
    """
    for i in range(0, len(tickers), chunk_size):
        묶음 = tickers[i:i + chunk_size]
        ticker_str = " ".join(묶음)
        print(f"  📥 일괄 다운로드 ({i+1}~{min(i+chunk_size, len(tickers))}/{len(tickers)}종목, period={period})...")

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
                    if len(묶음) == 1:
                        df_ticker = df_all
                    else:
                        df_ticker = df_all[ticker].dropna(how="all")

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
    캐시 히트 시 즉시 반환 (네트워크 요청 없음).
    캐시 미스 시 개별 다운로드 후 저장.
    """
    cache_key = f"{ticker}_{period}"

    if cache_key in _cache:
        return _cache[cache_key]

    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            print(f"⚠️ {ticker} 데이터가 비어 있습니다. 티커를 확인하세요.")
            return None
        _cache[cache_key] = df
        return df
    except Exception as e:
        print(f"⚠️ {ticker} 데이터를 불러오지 못했습니다. (오류: {e})")
        return None


def get_news(ticker):
    """종목 뉴스를 가져온다. 오류 시 빈 리스트 반환."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news if news else []
    except Exception:
        return []


def get_news_summary(ticker, 현재가, 전일_종가):
    """뉴스 제목 최대 3개와 당일 변동률을 반환한다."""
    try:
        변동률 = ((현재가 - 전일_종가) / 전일_종가) * 100 if 전일_종가 else 0
        뉴스_raw = get_news(ticker)
        if not 뉴스_raw:
            return None, 변동률
        뉴스_목록 = [item.get("title", "제목 없음") for item in 뉴스_raw[:3]]
        return 뉴스_목록, 변동률
    except Exception:
        print("뉴스를 불러오지 못했습니다. (yfinance 뉴스 API 일시 오류)")
        return None, 0
