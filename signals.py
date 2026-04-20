"""
signals.py — 마켓 레짐 필터 · 매수 신호 판단 · 백테스트
=========================================================
이 파일이 하는 일:
  1) get_market_regime()  : 현재 시장이 하락장인지 판단 (VIX 복합 필터)
  2) calc_signals()       : 종목별 5대 융합 매수 신호 계산
  3) run_backtest()       : 과거 성과 시뮬레이션 (수수료·세금 반영 실수익)
  4) calc_position_size() : ATR 기반 추천 매수 수량 계산 (리스크 1% 원칙)
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
    하락장에서 매수 신호를 그냥 따르면 손실이 크다.
    S&P500, KOSPI, VIX, BTC 4가지 기준을 복합적으로 체크한다.

    MA200이란? 200일 동안의 평균 주가. 이 선 아래에 있으면 장기 하락 추세.
    VIX란?    시장 참여자들이 느끼는 공포의 크기. 높을수록 불안·변동성이 크다.

    반환값: (하락_레짐 bool, 크립토_하락_레짐 bool, 경고_목록 list)
    """
    경고_목록     = []
    하락_레짐     = False
    크립토_하락_레짐 = False

    try:
        # ── 조건A: S&P500 MA200 ─────────────────────────
        # S&P500이 200일 평균선 아래 = 미국 시장 장기 하락 추세
        sp500_df = get_price_data("^GSPC", period="1y")
        if sp500_df is not None and len(sp500_df) >= 200:
            현재가_sp500 = sp500_df["Close"].squeeze().iloc[-1]
            ma200_sp500  = sp500_df["Close"].squeeze().rolling(200).mean().iloc[-1]
            if 현재가_sp500 < ma200_sp500:
                하락_레짐 = True
                경고_목록.append("S&P500 MA200 하회")

        # ── 조건B: KOSPI MA200 ──────────────────────────
        # KOSPI가 200일 평균선 아래 = 한국 시장 장기 하락 추세
        kospi_df = get_price_data("^KS11", period="1y")
        if kospi_df is not None and len(kospi_df) >= 200:
            현재가_kospi = kospi_df["Close"].squeeze().iloc[-1]
            ma200_kospi  = kospi_df["Close"].squeeze().rolling(200).mean().iloc[-1]
            if 현재가_kospi < ma200_kospi:
                하락_레짐 = True
                경고_목록.append("KOSPI MA200 하회")

        # ── 조건C: VIX 급등 ─────────────────────────────
        # VIX가 최근 20일 평균의 1.2배 이상 = 공포가 평소보다 20% 이상 급증
        vix_df = get_price_data("^VIX", period="3mo")
        if vix_df is not None and len(vix_df) >= 20:
            현재_vix     = vix_df["Close"].squeeze().iloc[-1]
            평균_vix_20일 = vix_df["Close"].squeeze().rolling(20).mean().iloc[-1]
            if 현재_vix > 평균_vix_20일 * 1.2:
                하락_레짐 = True
                경고_목록.append("VIX 급등")

        # ── 조건D: BTC MA200 (암호화폐 전용) ────────────
        # BTC가 200일 평균선 아래 = 크립토 시장 하락 레짐
        # KR/US 종목 판단에는 영향을 주지 않는다
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
# 2. 5대 융합 매수 신호 판단
# ─────────────────────────────────────────

def calc_signals(ticker, name, market, settings):
    """
    5대 융합 전략으로 매수 신호를 판단한다.

    [필수 조건 - 2개 모두 충족]
      ① RSI < RSI_BUY (기본 45): 과매도(저평가) 상태
      ② 현재가 > MA20 AND MA20 기울기 > 0: 단기 상승 추세(정배열)

    [보조 조건 - 3개 중 2개 이상 충족]
      ③ 거래량 > 20일 평균 * 1.5: 평소보다 거래 폭발
      ④ 현재가 <= 볼린저밴드 중심선(20일 평균선): 눌림/저점 구간(완화 조건)
      ⑤ MACD 히스토그램 음→양 전환: 모멘텀 반등 신호

    [변동성 돌파]
      ⑥ 현재가 > 당일 시가 + (전일 고가-저가) * 0.5: 강한 상승 에너지

    → 필수 2개 + 보조 2개 이상 + 변동성 돌파 모두 충족 시 매수 신호 발생
    """
    df = get_price_data(ticker, period="1y")
    if df is None or len(df) < 60:
        return None

    try:
        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        # ── 지표 계산 ───────────────────────────────────
        rsi = calc_rsi(close, 14).iloc[-1]

        # MA20: 최근 20일 종가 평균 (단기 추세 파악)
        ma_window = int(settings.get("MA_WINDOW", 20))
        ma20       = close.rolling(ma_window).mean()
        현재_ma20  = ma20.iloc[-1]
        # 기울기: 어제 MA20보다 오늘 MA20이 높으면 상승 추세 중
        ma20_기울기 = 현재_ma20 - ma20.iloc[-2]

        # 볼린저밴드 (중심선/하단 사용)
        _, bb_중심선, bb_하단 = calc_bollinger_bands(close)
        현재_bb_중심선 = bb_중심선.iloc[-1]
        현재_bb_하단  = bb_하단.iloc[-1]

        # MACD 지표
        macd선, 시그널선, 히스토그램 = calc_macd(close)
        현재_히스토그램 = 히스토그램.iloc[-1]
        이전_히스토그램 = 히스토그램.iloc[-2]

        # 거래량: 오늘 vs 20일 평균
        평균_거래량 = volume.rolling(20).mean().iloc[-1]
        오늘_거래량 = volume.iloc[-1]

        # 변동성 돌파 기준선
        오늘_시가  = df["Open"].squeeze().iloc[-1]
        전일_고가  = df["High"].squeeze().iloc[-2]
        전일_저가  = df["Low"].squeeze().iloc[-2]
        현재가     = close.iloc[-1]
        변동성_돌파_기준 = 오늘_시가 + (전일_고가 - 전일_저가) * 0.5

        # ── 조건 평가 ───────────────────────────────────
        rsi_buy = settings.get("RSI_BUY", 45)

        # [필수 1] RSI < 기준 → 저평가 상태
        조건1_rsi    = bool(rsi < rsi_buy)
        # [필수 2] 현재가 > MA20 AND MA20 상승 중 → 단기 정배열
        조건2_ma     = bool((현재가 > 현재_ma20) and (ma20_기울기 > 0))
        # [보조 3] 거래량 폭발 → 시장 관심 급증
        조건3_거래량  = bool(오늘_거래량 > 평균_거래량 * 1.5)
        # [보조 4] 볼린저 중심선(20일 평균선) 이하 → 눌림/저점 구간(완화 조건)
        조건4_볼린저  = bool(현재가 <= 현재_bb_중심선)
        # [보조 5] MACD 골든크로스 + 히스토그램 음→양 전환 → 모멘텀 반등
        조건5_macd    = bool(
            (현재_히스토그램 > 0) and
            (이전_히스토그램 <= 0) and
            (macd선.iloc[-1] > 시그널선.iloc[-1])
        )
        # [변동성 돌파] 장중 강한 상승 에너지
        조건6_변동성돌파 = bool(현재가 > 변동성_돌파_기준)

        # 보조 조건 충족 개수 (3개 중 2개 이상 필요)
        보조조건_충족수 = sum([조건3_거래량, 조건4_볼린저, 조건5_macd])

        # 최종 매수 신호: 필수 2개 AND 보조 2개 이상 AND 변동성 돌파
        매수신호 = (조건1_rsi and 조건2_ma and
                   (보조조건_충족수 >= 2) and 조건6_변동성돌파)

        # ATR 계산 (포지션 사이징에 사용)
        atr_변동폭 = calc_atr(df)

        return {
            "ticker":       ticker,
            "name":         name,
            "market":       market,
            "현재가":        현재가,
            "rsi":          rsi,
            "ma20":         현재_ma20,
            "bb_중심선":    현재_bb_중심선,
            "bb_하단":      현재_bb_하단,
            "atr":          atr_변동폭,
            "매수신호":      매수신호,
            "조건1_rsi":    조건1_rsi,
            "조건2_ma":     조건2_ma,
            "조건3_거래량":  조건3_거래량,
            "조건4_볼린저":  조건4_볼린저,
            "조건5_macd":   조건5_macd,
            "조건6_변동성돌파": 조건6_변동성돌파,
        }

    except Exception as e:
        print(f"⚠️ {name}({ticker}) 신호 계산 중 오류: {e}")
        return None


# ─────────────────────────────────────────
# 3. 비용 산입 백테스트 (기존 코드에서 가져옴 — 새 코드에 없던 기능)
# ─────────────────────────────────────────

def run_backtest(ticker, market, settings, period_months=3):
    """
    과거 데이터로 이 전략이 실제로 얼마나 벌었는지 시뮬레이션한다.
    수수료, 슬리피지, 한국 주식 거래세를 모두 반영한 '실수익(Net Profit)'으로 계산.

    [출력 지표 4가지]
    ① 누적 수익률(%): 해당 기간 동안 전략을 따랐을 때 총 수익
    ② MDD(최대 낙폭, %): 고점 대비 최대 하락폭. 클수록 위험한 전략
    ③ Sharpe Ratio: 수익 대비 위험 비율. 1.0 이상이면 양호, 2.0 이상이면 우수
    ④ 승률(%): 전체 매매 중 수익이 난 비율

    MDD란?         전략의 '최악의 순간'. -30% MDD는 한때 원금의 30%를 잃었다는 의미.
    Sharpe Ratio란? 위험 1단위당 얼마나 벌었는지. 은행 예금보다 나으면 1.0 이상.
    """
    기간_맵  = {3: "3mo", 6: "6mo", 12: "1y"}
    yf_기간  = 기간_맵.get(period_months, "3mo")

    df = get_price_data(ticker, period=yf_기간)
    if df is None or len(df) < 40:
        return None

    try:
        close = df["Close"].squeeze()

        rsi_series = calc_rsi(close)
        ma20       = close.rolling(20).mean()

        RSI_BUY    = settings.get("RSI_BUY",    45)
        RSI_SELL   = settings.get("RSI_SELL",   80)
        commission = settings.get("COMMISSION", 0.001)   # 수수료 0.1%
        slippage   = settings.get("SLIPPAGE",   0.0005)  # 슬리피지 0.05%

        # 한국 주식만 매도 시 증권거래세 0.20% 추가 차감
        # 미국 주식·암호화폐는 거래세 없음
        거래세     = 0.002 if market == "KR" else 0.0

        # 매수/매도 총 비용
        매수_비용 = commission + slippage
        매도_비용 = commission + slippage + 거래세

        # ── 매매 시뮬레이션 ────────────────────────────
        매수가          = None
        수익률_목록    = []

        for i in range(20, len(close)):
            현재가 = close.iloc[i]
            rsi    = rsi_series.iloc[i]
            ma     = ma20.iloc[i]

            if 매수가 is None:
                # 매수 조건: RSI < RSI_BUY AND 현재가 > MA20
                if rsi < RSI_BUY and 현재가 > ma:
                    # 실제 체결가 = 현재가 * (1 + 비용) → 약간 비싸게 체결됨
                    매수가 = 현재가 * (1 + 매수_비용)
            else:
                # 매도 조건: RSI > RSI_SELL
                if rsi > RSI_SELL:
                    # 실제 체결가 = 현재가 * (1 - 비용) → 약간 싸게 체결됨
                    매도가  = 현재가 * (1 - 매도_비용)
                    수익률  = (매도가 - 매수가) / 매수가 * 100
                    수익률_목록.append(수익률)
                    매수가  = None

        if not 수익률_목록:
            return {"수익률": 0, "mdd": 0, "sharpe": 0, "승률": 0}

        # ── 성과 지표 계산 ─────────────────────────────

        # ① 누적 수익률
        누적_수익률 = sum(수익률_목록)

        # ② MDD(최대 낙폭): 고점 대비 최대 하락폭
        누적_곡선 = [100.0]
        for r in 수익률_목록:
            누적_곡선.append(누적_곡선[-1] * (1 + r / 100))
        고점 = 누적_곡선[0]
        mdd  = 0.0
        for v in 누적_곡선:
            고점 = max(고점, v)
            낙폭 = (v - 고점) / 고점 * 100
            mdd  = min(mdd, 낙폭)

        # ③ Sharpe Ratio: 평균 수익률 / 수익률 표준편차
        평균_수익률      = float(np.mean(수익률_목록))
        수익률_표준편차  = float(np.std(수익률_목록))
        sharpe = (평균_수익률 / 수익률_표준편차) if 수익률_표준편차 > 0 else 0.0

        # ④ 승률: 이긴 매매 / 전체 매매
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
    이를 통해 한 종목이 폭락해도 전체 계좌가 치명상을 입지 않는다.

    공식: 매수 수량 = (총자산 * 0.01) / ATR
    ATR이 클수록(변동성이 클수록) → 더 적은 수량 → 동일한 리스크 유지

    시장구분:
      CRYPTO     → USD 기준 암호화폐 (바이낸스 등)
      CRYPTO_KRW → KRW 기준 암호화폐 (업비트 등)
                   업비트 BTC는 ATR이 USD 대비 약 1300배 크므로 수량이 자동으로 작아짐
      KR / US    → 주식 (정수 단위, 최소 1주)
    """
    if atr is None or (isinstance(atr, float) and (np.isnan(atr) or atr == 0)):
        return 0

    # 리스크 금액 = 총자산의 1%
    리스크_금액 = 총자산 * 0.01

    # 수량 = 리스크 금액 / ATR
    수량 = 리스크_금액 / atr

    if market in ("CRYPTO", "CRYPTO_KRW"):
        # 암호화폐는 소수점 단위 매수 가능 (예: 0.000148 BTC)
        return round(수량, 6)
    else:
        # 주식은 정수 단위 (최소 1주)
        return max(1, int(수량))
