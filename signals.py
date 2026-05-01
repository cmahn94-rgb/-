"""
signals.py — 마켓 레짐 필터 · 점수제 매수 신호 · 백테스트 · 포지션 사이징
==========================================================================
[점수제 매수 시스템 v2]
  기존 '필수 2개 AND 보조 2개 AND 변동성돌파' 방식 → 모두 AND라서 신호가 거의 안 나옴
  변경 → 6개 조건 각 1점, 합계 3점 이상이면 매수 신호 발생

  점수표:
    ① RSI < RSI_BUY          : 1점  (과매도)
    ② 현재가 > MA20 + 기울기↑ : 1점  (단기 상승 추세)
    ③ 거래량 > 평균 * 1.3     : 1점  (거래 증가, 기준 완화 1.5→1.3)
    ④ 현재가 <= 볼린저 중심선  : 1점  (눌림 구간)
    ⑤ MACD 히스토그램 음→양  : 1점  (모멘텀 반등)
    ⑥ 변동성 돌파              : 1점  (강한 상승 에너지)

  매수 신호: 총점 >= BUY_SCORE_THRESHOLD (settings.txt에서 설정, 기본 3)
  강력 매수: 총점 >= 5 → 리포트에 🔥 표시
"""

import numpy as np
import pandas as pd

from data_loader  import get_price_data
from indicators   import calc_rsi, calc_bollinger_bands, calc_macd, calc_atr


# ─────────────────────────────────────────
# 1. VIX 복합 마켓 레짐 필터
# ─────────────────────────────────────────

def get_market_regime():
    """
    대세 하락장인지 아닌지를 판단한다.
    S&P500, KOSPI, VIX, BTC 4가지 기준을 복합적으로 체크한다.
    반환값: (하락_레짐 bool, 크립토_하락_레짐 bool, 경고_목록 list)
    """
    경고_목록     = []
    하락_레짐     = False
    크립토_하락_레짐 = False

    try:
        sp500_df = get_price_data("^GSPC", period="1y")
        if sp500_df is not None and len(sp500_df) >= 200:
            현재가_sp500 = sp500_df["Close"].squeeze().iloc[-1]
            ma200_sp500  = sp500_df["Close"].squeeze().rolling(200).mean().iloc[-1]
            if 현재가_sp500 < ma200_sp500:
                하락_레짐 = True
                경고_목록.append("S&P500 MA200 하회")

        kospi_df = get_price_data("^KS11", period="1y")
        if kospi_df is not None and len(kospi_df) >= 200:
            현재가_kospi = kospi_df["Close"].squeeze().iloc[-1]
            ma200_kospi  = kospi_df["Close"].squeeze().rolling(200).mean().iloc[-1]
            if 현재가_kospi < ma200_kospi:
                하락_레짐 = True
                경고_목록.append("KOSPI MA200 하회")

        vix_df = get_price_data("^VIX", period="3mo")
        if vix_df is not None and len(vix_df) >= 20:
            현재_vix     = vix_df["Close"].squeeze().iloc[-1]
            평균_vix_20일 = vix_df["Close"].squeeze().rolling(20).mean().iloc[-1]
            if 현재_vix > 평균_vix_20일 * 1.2:
                하락_레짐 = True
                경고_목록.append("VIX 급등")

        btc_df = get_price_data("BTC-USD", period="1y")
        if btc_df is not None and len(btc_df) >= 200:
            현재가_btc = btc_df["Close"].squeeze().iloc[-1]
            ma200_btc  = btc_df["Close"].squeeze().rolling(200).mean().iloc[-1]
            if 현재가_btc < ma200_btc:
                크립토_하락_레짐 = True

    except Exception as e:
        print(f"⚠️ 마켓 레짐 분석 중 오류: {e}")

    return 하락_레짐, 크립토_하락_레짐, 경고_목록


# ─────────────────────────────────────────
# 2. 점수제 매수 신호 판단
# ─────────────────────────────────────────

def calc_signals(ticker, name, market, settings):
    """
    점수제 매수 신호 시스템.

    6개 조건 각 1점 → 합계 BUY_SCORE_THRESHOLD점 이상이면 매수 신호.
    settings.txt에서 BUY_SCORE_THRESHOLD 설정 가능 (기본 3).

    조건이 AND 체인 대신 점수 합산이므로 알람 빈도가 크게 증가한다.
    """
    df = get_price_data(ticker, period="1y")
    if df is None or len(df) < 60:
        return None

    try:
        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        # RSI: 숫자가 낮을수록 '많이 빠진 상태(과매도)'로 보는 지표
        rsi = calc_rsi(close, 14).iloc[-1]

        ma_window  = int(settings.get("MA_WINDOW", 20))
        ma20       = close.rolling(ma_window).mean()
        현재_ma20  = ma20.iloc[-1]
        ma20_기울기 = 현재_ma20 - ma20.iloc[-2]

        # calc_bollinger_bands 반환값: (중심선, 상단밴드, 하단밴드)
        bb_중심선, bb_상단, bb_하단 = calc_bollinger_bands(close)
        현재_bb_중심선 = bb_중심선.iloc[-1]
        현재_bb_하단  = bb_하단.iloc[-1]

        macd선, 시그널선, 히스토그램 = calc_macd(close)
        현재_히스토그램 = 히스토그램.iloc[-1]
        이전_히스토그램 = 히스토그램.iloc[-2]

        # 거래량 기준 완화: 1.5 → 1.3
        평균_거래량 = volume.rolling(20).mean().iloc[-1]
        오늘_거래량 = volume.iloc[-1]

        # 변동성 돌파: 오늘 시가 + (어제 변동폭 × 0.5) 위로 올라가면 1점
        오늘_시가  = df["Open"].squeeze().iloc[-1]
        전일_고가  = df["High"].squeeze().iloc[-2]
        전일_저가  = df["Low"].squeeze().iloc[-2]
        현재가     = close.iloc[-1]
        변동성_돌파_기준 = 오늘_시가 + (전일_고가 - 전일_저가) * 0.5

        rsi_buy = settings.get("RSI_BUY", 50)

        # ── 점수 계산 ───────────────────────────────────
        # 6개 조건: 맞으면 1점(AND가 아니라 점수 합으로 판단)
        조건1_rsi      = bool(rsi < rsi_buy)
        조건2_ma       = bool((현재가 > 현재_ma20) and (ma20_기울기 > 0))
        조건3_거래량   = bool(오늘_거래량 > 평균_거래량 * 1.3)   # 완화: 1.5→1.3
        조건4_볼린저   = bool(현재가 <= 현재_bb_중심선)
        조건5_macd     = bool(
            (현재_히스토그램 > 0) and
            (이전_히스토그램 <= 0) and
            (macd선.iloc[-1] > 시그널선.iloc[-1])
        )
        조건6_변동성돌파 = bool(현재가 > 변동성_돌파_기준)

        # 총 점수
        점수 = sum([조건1_rsi, 조건2_ma, 조건3_거래량,
                    조건4_볼린저, 조건5_macd, 조건6_변동성돌파])

        # 임계값=3이면 6개 중 3개 이상 맞을 때 매수 신호
        임계값 = int(settings.get("BUY_SCORE_THRESHOLD", 3))
        매수신호 = 점수 >= 임계값
        강력매수 = 점수 >= 5   # 5점 이상 = 강력 매수

        atr_변동폭 = calc_atr(df)

        return {
            "ticker":          ticker,
            "name":            name,
            "market":          market,
            "현재가":           현재가,
            "rsi":             rsi,
            "ma20":            현재_ma20,
            "bb_중심선":       현재_bb_중심선,
            "bb_하단":         현재_bb_하단,
            "atr":             atr_변동폭,
            "매수신호":         매수신호,
            "강력매수":         강력매수,
            "점수":             점수,
            "임계값":           임계값,
            "조건1_rsi":       조건1_rsi,
            "조건2_ma":        조건2_ma,
            "조건3_거래량":     조건3_거래량,
            "조건4_볼린저":     조건4_볼린저,
            "조건5_macd":      조건5_macd,
            "조건6_변동성돌파": 조건6_변동성돌파,
        }

    except Exception as e:
        print(f"⚠️ {name}({ticker}) 신호 계산 중 오류: {e}")
        return None


# ─────────────────────────────────────────
# 3. 비용 산입 백테스트
# ─────────────────────────────────────────

def run_backtest(ticker, market, settings, period_months=3):
    """
    과거 데이터로 이 전략이 실제로 얼마나 벌었는지 시뮬레이션한다.
    수수료, 슬리피지, 한국 주식 거래세를 모두 반영한 실수익(Net Profit)으로 계산.
    """
    기간_맵  = {3: "3mo", 6: "6mo", 12: "1y"}
    yf_기간  = 기간_맵.get(period_months, "3mo")

    df = get_price_data(ticker, period=yf_기간)
    if df is None or len(df) < 40:
        return None

    try:
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze() if "Volume" in df.columns else None
        open_ = df["Open"].squeeze() if "Open" in df.columns else None
        high = df["High"].squeeze() if "High" in df.columns else None
        low = df["Low"].squeeze() if "Low" in df.columns else None

        # ── 매수 신호 로직(점수제)과 동일한 조건으로 "진입"을 만들기 ──
        # ※ 매도(청산) 규칙은 기존처럼 RSI_SELL(과열)로 유지한다. (알림 로직에 '매도 규칙'이 명확히 없어서)
        rsi_series = calc_rsi(close, 14)

        ma_window = int(settings.get("MA_WINDOW", 20))
        ma20 = close.rolling(ma_window).mean()
        ma20_slope = ma20.diff()

        bb_center, _, _ = calc_bollinger_bands(close)
        macd_line, sig_line, hist = calc_macd(close)

        avg_vol = volume.rolling(20).mean() if volume is not None else None

        rsi_buy = settings.get("RSI_BUY", 50)
        threshold = int(settings.get("BUY_SCORE_THRESHOLD", 3))

        RSI_BUY    = settings.get("RSI_BUY",    50)
        RSI_SELL   = settings.get("RSI_SELL",   80)
        commission = settings.get("COMMISSION", 0.001)
        slippage   = settings.get("SLIPPAGE",   0.0005)
        거래세     = 0.002 if market == "KR" else 0.0

        매수_비용 = commission + slippage
        매도_비용 = commission + slippage + 거래세

        매수가       = None
        수익률_목록 = []

        # 점수제 조건은 지표 워밍업이 필요해서 30일 이후부터 체크
        start_i = 30
        for i in range(start_i, len(close)):
            현재가 = close.iloc[i]
            rsi = rsi_series.iloc[i]

            if 매수가 is None:
                # ── 점수제 매수 조건(6개 중 threshold 이상) ──
                cond1_rsi = bool(rsi < rsi_buy)
                cond2_ma = bool((현재가 > ma20.iloc[i]) and (ma20_slope.iloc[i] > 0))
                cond3_vol = bool(
                    (avg_vol is not None) and
                    (not np.isnan(avg_vol.iloc[i])) and
                    (volume.iloc[i] > avg_vol.iloc[i] * 1.3)
                )
                cond4_bb = bool(현재가 <= bb_center.iloc[i])
                cond5_macd = bool((hist.iloc[i] > 0) and (hist.iloc[i - 1] <= 0) and (macd_line.iloc[i] > sig_line.iloc[i]))

                # 변동성 돌파(오늘 시가 + 전일 변동폭*0.5)
                cond6_break = False
                if open_ is not None and high is not None and low is not None and i >= 1:
                    기준 = open_.iloc[i] + (high.iloc[i - 1] - low.iloc[i - 1]) * 0.5
                    cond6_break = bool(현재가 > 기준)

                score = sum([cond1_rsi, cond2_ma, cond3_vol, cond4_bb, cond5_macd, cond6_break])
                if score >= threshold:
                    매수가 = 현재가 * (1 + 매수_비용)
            else:
                # RSI가 너무 높아지면(과열) "매도"했다고 가정
                if rsi > RSI_SELL:
                    매도가 = 현재가 * (1 - 매도_비용)
                    수익률 = (매도가 - 매수가) / 매수가 * 100
                    수익률_목록.append(수익률)
                    매수가 = None

        # 거래가 한 번도 없으면(=백테스트가 유효한 판단을 못 했으면) 0으로 꾸며서 보여주지 말고 None 처리.
        # 그래야 리포트에서 "+0.0% | 승률 0%" 같은 오해를 줄일 수 있다.
        if not 수익률_목록:
            return None

        누적_수익률 = sum(수익률_목록)

        누적_곡선 = [100.0]
        for r in 수익률_목록:
            누적_곡선.append(누적_곡선[-1] * (1 + r / 100))
        고점 = 누적_곡선[0]
        mdd  = 0.0
        for v in 누적_곡선:
            고점 = max(고점, v)
            낙폭 = (v - 고점) / 고점 * 100
            mdd  = min(mdd, 낙폭)

        평균_수익률     = float(np.mean(수익률_목록))
        수익률_표준편차 = float(np.std(수익률_목록))
        sharpe = (평균_수익률 / 수익률_표준편차) if 수익률_표준편차 > 0 else 0.0

        승률 = (sum(1 for r in 수익률_목록 if r > 0) / len(수익률_목록)) * 100

        return {
            "수익률": round(누적_수익률, 1),
            "mdd":    round(mdd, 1),
            "sharpe": round(sharpe, 2),
            "승률":   round(승률, 0),
        }

    except Exception as e:
        print(f"⚠️ {ticker} 백테스트 오류: {e}")
        return None


# ─────────────────────────────────────────
# 4. ATR 기반 포지션 사이징
# ─────────────────────────────────────────

def calc_position_size(총자산, atr, market):
    """
    한 종목에서 최대 손실을 '총자산의 1%'로 제한하도록 매수 수량을 계산한다.
    공식: 매수 수량 = (총자산 * 0.01) / ATR
    """
    if atr is None or atr == 0 or (isinstance(atr, float) and np.isnan(atr)):
        return 0
    리스크_금액 = 총자산 * 0.01
    수량 = 리스크_금액 / atr

    if market in ("CRYPTO", "CRYPTO_KRW"):
        return round(수량, 6)
    else:
        return max(1, int(수량))
