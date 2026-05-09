"""
signals.py — 마켓 레짐 필터 · 점수제 매수/매도 신호 · 백테스트 · 포지션 사이징
================================================================================
[전략 업그레이드 v3 — 중학생도 이해할 수 있는 설명]

■ 핵심 개념: 6가지 지표 중 3개 이상이 "살 때다!"를 외치면 매수 신호 발생

■ 새로 추가된 것들:
  ① RSI 다이버전스 보너스 (+1점)
     - "가격은 내려가는데, RSI는 올라가는" 상황 = 곧 반등 가능성이 높다는 신호
     - 비유: 학생이 공부는 더 열심히 하는데 성적이 잠깐 떨어진 것 → 곧 오른다

  ② ADX 추세 강도 필터
     - ADX(평균방향지수)가 25 이상이면 "지금 추세가 뚜렷하다"는 뜻
     - 추세가 뚜렷할 때만 변동성 돌파 점수를 인정 → 허위 신호 감소

  ③ 매도 신호 추가 (기존엔 매도 조건이 없었음!)
     - RSI 과열 + 볼린저 상단 돌파 + MACD 하락 전환 → 매도 알림 발송
     - 비유: "사과가 너무 비싸졌다" + "거래가 뜸해졌다" → 팔 때

  ④ 점수 가중치 조정
     - 강력 신호(MACD 전환 + 거래량 급증 동시)에 추가 보너스 부여
     - 단순히 3개 조건이 아니라 "어떤 조건이냐"도 중요하게 봄

[점수표 — 6개 조건 각 1점]
  ① RSI < RSI_BUY          : 1점  (가격이 많이 내려와서 저렴한 상태)
  ② 현재가 > MA20 + 기울기↑ : 1점  (단기 평균선보다 위에 있고, 상승 중)
  ③ 거래량 > 평균 * 1.3     : 1점  (사람들이 평소보다 많이 거래 중 = 관심 증가)
  ④ 현재가 <= 볼린저 중심선  : 1점  (통계적으로 적당한 가격대)
  ⑤ MACD 히스토그램 음→양  : 1점  (하락 에너지 → 상승 에너지 전환)
  ⑥ 변동성 돌파 + ADX≥25   : 1점  (강한 방향성을 가진 상승 돌파)

[보너스 점수]
  ⑦ RSI 다이버전스           : +1점 (가격↓ + RSI↑ = 반등 예고 신호)
  ⑧ MACD 전환 & 거래량 동시  : +1점 (두 강력 신호가 동시에 등장)

  매수 신호: 총점 >= BUY_SCORE_THRESHOLD (기본 3)
  강력 매수: 총점 >= 5  → 리포트에 🔥 표시
"""

import numpy as np
import pandas as pd

from data_loader  import get_price_data, get_weekly_data
from indicators   import calc_rsi, calc_bollinger_bands, calc_macd, calc_atr, calc_adx


# ─────────────────────────────────────────
# 1. VIX 복합 마켓 레짐 필터
# ─────────────────────────────────────────

def get_market_regime():
    """
    지금이 대세 하락장인지 아닌지를 판단한다.

    [중학생 설명]
    주식 시장 전체가 무너지는 상황이라면, 아무리 좋은 종목도 같이 내려간다.
    그래서 "지금 시장이 괜찮은가?"를 먼저 확인하는 것이 중요하다.

    4가지 기준으로 판단한다:
    1) S&P500 (미국 시장): 200일 평균 아래면 위험
    2) KOSPI (한국 시장): 200일 평균 아래면 위험
    3) VIX (공포지수): 평소보다 20% 이상 높으면 시장이 겁먹은 상태
    4) BTC (비트코인): 200일 평균 아래면 크립토도 위험

    반환값: (하락_레짐 bool, 크립토_하락_레짐 bool, 경고_목록 list)
    """
    경고_목록     = []
    하락_레짐     = False
    크립토_하락_레짐 = False
    하락_점수     = 0  # 하락 신호 개수 카운트 (2개 이상이어야 확정 하락장)

    try:
        sp500_df = get_price_data("^GSPC", period="1y")
        if sp500_df is not None and len(sp500_df) >= 200:
            현재가_sp500 = sp500_df["Close"].squeeze().iloc[-1]
            ma200_sp500  = sp500_df["Close"].squeeze().rolling(200).mean().iloc[-1]
            # 5% 이상 밑돌 때만 하락 레짐 인정 (약간의 노이즈 허용)
            if 현재가_sp500 < ma200_sp500 * 0.95:
                하락_점수 += 1
                경고_목록.append("S&P500 MA200 -5% 하회")

        kospi_df = get_price_data("^KS11", period="1y")
        if kospi_df is not None and len(kospi_df) >= 200:
            현재가_kospi = kospi_df["Close"].squeeze().iloc[-1]
            ma200_kospi  = kospi_df["Close"].squeeze().rolling(200).mean().iloc[-1]
            if 현재가_kospi < ma200_kospi * 0.95:
                하락_점수 += 1
                경고_목록.append("KOSPI MA200 -5% 하회")

        vix_df = get_price_data("^VIX", period="3mo")
        if vix_df is not None and len(vix_df) >= 20:
            현재_vix     = vix_df["Close"].squeeze().iloc[-1]
            평균_vix_20일 = vix_df["Close"].squeeze().rolling(20).mean().iloc[-1]
            # VIX 30 이상 = 공포장 (기존 1.2 배수보다 절대값 기준이 더 직관적)
            if 현재_vix > 30 or 현재_vix > 평균_vix_20일 * 1.3:
                하락_점수 += 1
                경고_목록.append(f"VIX 급등 ({현재_vix:.1f})")

        btc_df = get_price_data("BTC-USD", period="1y")
        if btc_df is not None and len(btc_df) >= 200:
            현재가_btc = btc_df["Close"].squeeze().iloc[-1]
            ma200_btc  = btc_df["Close"].squeeze().rolling(200).mean().iloc[-1]
            if 현재가_btc < ma200_btc:
                크립토_하락_레짐 = True

        # 2개 이상 하락 신호 → 하락 레짐 확정 (1개만이면 노이즈일 수 있음)
        하락_레짐 = 하락_점수 >= 2

    except Exception as e:
        print(f"⚠️ 마켓 레짐 분석 중 오류: {e}")

    return 하락_레짐, 크립토_하락_레짐, 경고_목록


# ─────────────────────────────────────────
# 2-A. RSI 다이버전스 감지 (새로 추가)
# ─────────────────────────────────────────

def detect_rsi_divergence(close, rsi_series, lookback=10):
    """
    RSI 다이버전스를 감지한다. (강세 다이버전스만)

    [중학생 설명]
    강세 다이버전스란?
    - 주가는 더 낮은 저점을 찍었는데 (= 가격이 더 내려갔는데)
    - RSI는 더 높은 저점을 찍는 상황 (= 내부 에너지는 오히려 회복 중)

    이것은 "겉으로는 힘들어 보이지만 실제론 체력이 붙고 있다"는 뜻으로,
    곧 반등이 올 가능성이 높다는 신호다.

    반환값: True (다이버전스 감지됨) / False (아님)
    """
    try:
        if len(close) < lookback + 2:
            return False

        # 최근 lookback일 안에서 직전 저점 찾기
        최근_close = close.iloc[-lookback:]
        최근_rsi   = rsi_series.iloc[-lookback:]

        현재_close = close.iloc[-1]
        현재_rsi   = rsi_series.iloc[-1]

        이전_저점_close = 최근_close.iloc[:-1].min()
        이전_저점_idx   = 최근_close.iloc[:-1].idxmin()
        이전_저점_rsi   = 최근_rsi.loc[이전_저점_idx] if 이전_저점_idx in 최근_rsi.index else None

        if 이전_저점_rsi is None:
            return False

        # 강세 다이버전스 조건:
        # 가격은 이전 저점보다 낮은데(↓), RSI는 이전 저점보다 높다(↑)
        가격_하락 = 현재_close < 이전_저점_close
        rsi_상승  = 현재_rsi   > 이전_저점_rsi

        return 가격_하락 and rsi_상승

    except Exception:
        return False


# ─────────────────────────────────────────
# 2-B. 매도 신호 판단 (새로 추가)
# ─────────────────────────────────────────

def calc_sell_signal(ticker, name, market, settings):
    """
    매도 신호를 계산한다. (기존에 없던 기능)

    [중학생 설명]
    매수 신호를 잘 잡는 것만큼, '언제 팔아야 하나'를 아는 것도 중요하다.
    다음 3가지가 동시에 발생하면 매도를 고려한다:

    ① RSI > RSI_SELL (너무 많이 올라서 과열 상태)
    ② 볼린저 상단을 돌파 (통계적으로 비싼 가격)
    ③ MACD 히스토그램이 양수 → 음수로 전환 (상승 에너지가 꺾이기 시작)

    반환값: dict or None
    """
    df = get_price_data(ticker, period="3mo")
    if df is None or len(df) < 60:   # 1년 백테스트: 최소 60거래일 필요
        return None

    try:
        close = df["Close"].squeeze()
        rsi   = calc_rsi(close, 14).iloc[-1]
        _, bb_상단, _ = calc_bollinger_bands(close)
        _, _, 히스토그램 = calc_macd(close)

        현재가     = close.iloc[-1]
        현재_bb_상단 = bb_상단.iloc[-1]
        현재_히스토 = 히스토그램.iloc[-1]
        이전_히스토 = 히스토그램.iloc[-2]

        rsi_sell = settings.get("RSI_SELL", 80)

        # 매도 조건 3가지
        조건A_rsi과열  = rsi > rsi_sell
        조건B_볼린저상단 = 현재가 > 현재_bb_상단
        조건C_macd하락  = (현재_히스토 < 0) and (이전_히스토 >= 0)

        # 3가지 모두 만족 시 강력 매도 신호
        강력_매도 = all([조건A_rsi과열, 조건B_볼린저상단, 조건C_macd하락])
        # 2가지 만족 시 일반 매도 신호
        일반_매도 = sum([조건A_rsi과열, 조건B_볼린저상단, 조건C_macd하락]) >= 2

        if 강력_매도 or 일반_매도:
            return {
                "ticker":       ticker,
                "name":         name,
                "market":       market,
                "현재가":       현재가,
                "rsi":          rsi,
                "강력매도":     강력_매도,
                "조건A_rsi과열":   조건A_rsi과열,
                "조건B_볼린저상단": 조건B_볼린저상단,
                "조건C_macd하락":  조건C_macd하락,
            }
        return None

    except Exception as e:
        print(f"⚠️ {name}({ticker}) 매도 신호 계산 중 오류: {e}")
        return None


# ─────────────────────────────────────────
# 3. 점수제 매수 신호 판단 (업그레이드)
# ─────────────────────────────────────────

def calc_signals(ticker, name, market, settings):
    """
    점수제 매수 신호 시스템 (업그레이드 v3).

    [중학생 설명]
    6개의 지표를 각각 체크해서 점수를 매긴다.
    3점 이상이면 "살 만한 타이밍"이라고 알림을 보낸다.
    보너스 점수까지 더하면 최대 8점까지 나올 수 있다.

    settings.txt에서 BUY_SCORE_THRESHOLD 값으로 기준 점수를 바꿀 수 있다.
    """
    df = get_price_data(ticker, period="1y")
    if df is None or len(df) < 60:
        return None

    try:
        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        # ── 기본 지표 계산 ───────────────────────────────────
        rsi_series = calc_rsi(close, 14)
        rsi        = rsi_series.iloc[-1]

        ma_window  = int(settings.get("MA_WINDOW", 20))
        ma20       = close.rolling(ma_window).mean()
        현재_ma20  = ma20.iloc[-1]
        ma20_기울기 = 현재_ma20 - ma20.iloc[-2]

        bb_중심선, bb_상단, bb_하단 = calc_bollinger_bands(close)
        현재_bb_중심선 = bb_중심선.iloc[-1]
        현재_bb_하단  = bb_하단.iloc[-1]

        macd선, 시그널선, 히스토그램 = calc_macd(close)
        현재_히스토그램 = 히스토그램.iloc[-1]
        이전_히스토그램 = 히스토그램.iloc[-2]

        평균_거래량 = volume.rolling(20).mean().iloc[-1]
        오늘_거래량 = volume.iloc[-1]

        # 변동성 돌파 기준 계산
        오늘_시가  = df["Open"].squeeze().iloc[-1]
        전일_고가  = df["High"].squeeze().iloc[-2]
        전일_저가  = df["Low"].squeeze().iloc[-2]
        현재가     = close.iloc[-1]
        변동성_돌파_기준 = 오늘_시가 + (전일_고가 - 전일_저가) * 0.5

        # ADX(추세 강도) 계산 — 변동성 돌파 신뢰도 향상에 사용
        adx_값 = calc_adx(df)

        rsi_buy = settings.get("RSI_BUY", 50)

        # ── 6개 기본 점수 조건 ────────────────────────────────
        조건1_rsi      = bool(rsi < rsi_buy)
        조건2_ma       = bool((현재가 > 현재_ma20) and (ma20_기울기 > 0))
        # 거래량 임계값: VIX(공포지수)에 따라 동적으로 조정
        # VIX 높을수록(공포장) 거래량 기준을 올려 허위 신호 감소
        try:
            vix_df_now = get_price_data("^VIX", period="5d")
            현재_vix   = float(vix_df_now["Close"].squeeze().iloc[-1]) if vix_df_now is not None else 15.0
        except Exception:
            현재_vix = 15.0

        if 현재_vix >= 30:    vol_기준 = 2.0   # 공포장: 기준 대폭 상향
        elif 현재_vix >= 20:  vol_기준 = 1.6   # 주의장: 기준 소폭 상향
        else:                  vol_기준 = float(settings.get("VOL_MULT", 1.3))  # 평상시

        조건3_거래량   = bool(오늘_거래량 > 평균_거래량 * vol_기준)
        조건4_볼린저   = bool(현재가 <= 현재_bb_중심선)
        조건5_macd     = bool(
            (현재_히스토그램 > 0) and
            (이전_히스토그램 <= 0) and
            (macd선.iloc[-1] > 시그널선.iloc[-1])
        )
        # 변동성 돌파는 ADX ≥ ADX_MIN(추세 뚜렷)일 때만 점수 인정 → 허위신호 감소
        # ADX_MIN은 settings.txt에서 읽어온다 (기본값 30, 권장 25~35)
        adx_min = float(settings.get("ADX_MIN", 30))
        조건6_변동성돌파 = bool(
            (현재가 > 변동성_돌파_기준) and
            (adx_값 is not None and adx_값 >= adx_min)
        )

        기본_점수 = sum([조건1_rsi, 조건2_ma, 조건3_거래량,
                        조건4_볼린저, 조건5_macd, 조건6_변동성돌파])

        # ── 보너스 점수 (새로 추가) ───────────────────────────
        # ⑦ RSI 다이버전스 보너스: 반등 가능성이 높은 특수 상황
        rsi_다이버전스 = detect_rsi_divergence(close, rsi_series, lookback=10)
        보너스1_다이버전스 = bool(rsi_다이버전스 and 조건1_rsi)  # RSI가 낮을 때만 의미 있음

        # ⑧ MACD 전환 + 거래량 급증 동시 발생 = 신뢰도 높은 신호
        보너스2_복합강세 = bool(조건5_macd and 조건3_거래량)

        # ⑨ 보너스: 52주 신고가 근접 (O'Neil CANSLIM 핵심 원칙)
        # 현재가가 52주 최고가의 95% 이상 = 저항선 없는 구간 진입
        # 비유: 달리기 선수가 자기 최고 기록에 근접 → 신기록 가능성 높음
        최고가_52주 = close.rolling(min(252, len(close))).max().iloc[-1]
        보너스3_신고가 = bool(
            (not np.isnan(최고가_52주)) and
            (현재가 >= 최고가_52주 * 0.95)
        )

        # ── 주간봉 MACD 필터 (다중 타임프레임) ─────────────────────
        # 일봉 신호가 있어도 주간봉 MACD가 음수(큰 흐름이 하락)이면
        # 단기 반등일 가능성 높음 → 점수 -1 패널티
        주간봉_패널티 = 0
        try:
            df_weekly = get_weekly_data(ticker, period="2y")
            if df_weekly is not None and len(df_weekly) >= 26:
                _, _, hist_w = calc_macd(df_weekly["Close"].squeeze())
                if hist_w.iloc[-1] < 0:
                    주간봉_패널티 = 1
        except Exception:
            주간봉_패널티 = 0

        보너스_점수 = sum([보너스1_다이버전스, 보너스2_복합강세, 보너스3_신고가])
        총_점수    = 기본_점수 + 보너스_점수 - 주간봉_패널티

        임계값  = int(settings.get("BUY_SCORE_THRESHOLD", 3))
        매수신호 = 총_점수 >= 임계값
        강력매수 = 총_점수 >= 5

        atr_변동폭 = calc_atr(df)

        return {
            "ticker":           ticker,
            "name":             name,
            "market":           market,
            "현재가":           현재가,
            "rsi":              rsi,
            "ma20":             현재_ma20,
            "bb_중심선":        현재_bb_중심선,
            "bb_하단":          현재_bb_하단,
            "atr":              atr_변동폭,
            "adx":              adx_값,
            "매수신호":         매수신호,
            "강력매수":         강력매수,
            "점수":             총_점수,       # 보너스 포함 총점
            "기본점수":         기본_점수,
            "임계값":           임계값,
            "조건1_rsi":        조건1_rsi,
            "조건2_ma":         조건2_ma,
            "조건3_거래량":     조건3_거래량,
            "조건4_볼린저":     조건4_볼린저,
            "조건5_macd":       조건5_macd,
            "조건6_변동성돌파": 조건6_변동성돌파,
            "보너스_다이버전스": 보너스1_다이버전스,
            "보너스_복합강세":  보너스2_복합강세,
            "보너스_신고가":    보너스3_신고가,
            "주간봉_패널티":   주간봉_패널티,
        }

    except Exception as e:
        print(f"⚠️ {name}({ticker}) 신호 계산 중 오류: {e}")
        return None


# ─────────────────────────────────────────
# 4. 비용 산입 백테스트 (개선)
# ─────────────────────────────────────────

def run_backtest(ticker, market, settings, period_months=12):
    """
    과거 데이터로 이 전략이 실제로 얼마나 벌었는지 시뮬레이션한다.

    [중학생 설명]
    "만약 이 전략을 3개월 전부터 썼다면 얼마나 벌었을까?"를 계산한다.
    수수료, 슬리피지(실제 체결 가격 차이), 한국 거래세까지 모두 빼고
    실제로 손에 남는 금액을 기준으로 계산한다.

    개선 사항: 매도 조건에 "손절선(-7%)" 추가 → 큰 손실 방지 효과 반영
    """
    기간_맵  = {3: "3mo", 6: "6mo", 12: "1y"}
    yf_기간  = 기간_맵.get(period_months, "1y")

    df = get_price_data(ticker, period=yf_기간)
    if df is None or len(df) < 40:
        return None

    try:
        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze() if "Volume" in df.columns else None
        open_  = df["Open"].squeeze()   if "Open"   in df.columns else None
        high   = df["High"].squeeze()   if "High"   in df.columns else None
        low    = df["Low"].squeeze()    if "Low"    in df.columns else None

        rsi_series = calc_rsi(close, 14)
        ma_window  = int(settings.get("MA_WINDOW", 20))
        ma20       = close.rolling(ma_window).mean()
        ma20_slope = ma20.diff()

        bb_center, _, _ = calc_bollinger_bands(close)
        macd_line, sig_line, hist = calc_macd(close)
        avg_vol = volume.rolling(20).mean() if volume is not None else None

        # ── ADX rolling 시리즈 사전 계산 (look-ahead bias 수정) ──
        # 기존: 전체 기간 마지막 ADX를 과거 모든 시점에 적용 → 미래 정보 누설
        # 수정: 각 시점 i까지의 데이터로 ADX를 미리 계산해 배열로 보관
        #       → 시점 i의 ADX = 그 날까지의 데이터로만 계산한 값 (현실적)
        #
        # 구현: Wilder 평활화를 전체 df에 한 번만 적용해 Series로 저장
        #       백테스트 루프에서는 adx_series.iloc[i] 로 참조 (추가 연산 없음)
        try:
            _high  = df["High"].squeeze()
            _low   = df["Low"].squeeze()
            _close = df["Close"].squeeze()
            _high_diff = _high.diff()
            _low_diff  = -_low.diff()
            _plus_dm   = pd.Series(
                np.where((_high_diff > _low_diff) & (_high_diff > 0), _high_diff, 0.0),
                index=_high.index
            )
            _minus_dm  = pd.Series(
                np.where((_low_diff > _high_diff) & (_low_diff > 0), _low_diff, 0.0),
                index=_low.index
            )
            _prev_close = _close.shift(1)
            _tr = pd.concat([
                (_high - _low).abs(),
                (_high - _prev_close).abs(),
                (_low  - _prev_close).abs()
            ], axis=1).max(axis=1)
            _alpha = 1 / 14
            _atr_s    = _tr.ewm(alpha=_alpha, adjust=False).mean()
            _plus_di  = 100 * _plus_dm.ewm(alpha=_alpha, adjust=False).mean() / _atr_s.replace(0, np.nan)
            _minus_di = 100 * _minus_dm.ewm(alpha=_alpha, adjust=False).mean() / _atr_s.replace(0, np.nan)
            _dx       = 100 * (_plus_di - _minus_di).abs() / (_plus_di + _minus_di).replace(0, np.nan)
            adx_series = _dx.ewm(alpha=_alpha, adjust=False).mean()  # 시점별 ADX Series
        except Exception:
            adx_series = None

        RSI_BUY    = settings.get("RSI_BUY",    50)
        RSI_SELL   = settings.get("RSI_SELL",   80)
        STOP_LOSS  = settings.get("STOP_LOSS",  -7)   # 손절 % (음수)
        ADX_MIN       = float(settings.get("ADX_MIN", 30))  # 추세 강도 최소값
        TRAILING_STOP = float(settings.get("TRAILING_STOP", 8))  # 고점 대비 손절 % (기본 8%)
        threshold     = int(settings.get("BUY_SCORE_THRESHOLD", 3))
        commission = settings.get("COMMISSION", 0.001)
        slippage   = settings.get("SLIPPAGE",   0.0005)
        거래세     = 0.002 if market == "KR" else 0.0

        매수_비용 = commission + slippage
        매도_비용 = commission + slippage + 거래세

        매수가        = None
        매수_후_고점  = None  # 트레일링 스탑용 고점 추적
        수익률_목록   = []
        최대_연속_손실 = 0
        연속_손실      = 0

        for i in range(30, len(close)):
            현재가 = close.iloc[i]
            rsi    = rsi_series.iloc[i]

            if 매수가 is None:
                # ── 6개 기본 조건 (calc_signals와 완전 동일) ──────────
                c1 = bool(rsi < RSI_BUY)
                c2 = bool((현재가 > ma20.iloc[i]) and (ma20_slope.iloc[i] > 0))
                c3 = bool(
                    (avg_vol is not None) and
                    (not np.isnan(avg_vol.iloc[i])) and
                    (volume.iloc[i] > avg_vol.iloc[i] * 1.3)
                )
                c4 = bool(현재가 <= bb_center.iloc[i])
                c5 = bool((hist.iloc[i] > 0) and (hist.iloc[i - 1] <= 0) and
                           (macd_line.iloc[i] > sig_line.iloc[i]))

                # c6: 변동성 돌파 + ADX ≥ 25 — look-ahead bias 없는 시점별 ADX 사용
                c6 = False
                if open_ is not None and high is not None and low is not None and i >= 1:
                    기준     = open_.iloc[i] + (high.iloc[i - 1] - low.iloc[i - 1]) * 0.5
                    돌파여부 = bool(현재가 > 기준)
                    # adx_series가 있으면 해당 시점 ADX, 없으면 조건 비활성화
                    i_adx    = float(adx_series.iloc[i]) if adx_series is not None and not pd.isna(adx_series.iloc[i]) else None
                    adx_통과 = (i_adx is not None and i_adx >= ADX_MIN)
                    c6       = 돌파여부 and adx_통과

                기본_점수 = sum([c1, c2, c3, c4, c5, c6])

                # ── 보너스 점수 (calc_signals와 완전 동일) ────────────
                # 보너스 ⑦: RSI 다이버전스 (가격↓ + RSI↑)
                rsi_다이버전스 = detect_rsi_divergence(
                    close.iloc[:i + 1], rsi_series.iloc[:i + 1], lookback=10
                )
                보너스_다이버전스 = bool(rsi_다이버전스 and c1)

                # 보너스 ⑧: MACD 전환 + 거래량 급증 동시 발생
                보너스_복합강세 = bool(c5 and c3)

                총_점수 = 기본_점수 + sum([보너스_다이버전스, 보너스_복합강세])

                # 52주 신고가 보너스 (백테스트 일관성)
                고점_52주 = close.iloc[max(0, i-252):i+1].max()
                보너스3   = bool(현재가 >= 고점_52주 * 0.95)
                총_점수   += int(보너스3)

                if 총_점수 >= threshold:
                    매수가        = 현재가 * (1 + 매수_비용)
                    매수_후_고점  = 현재가  # 트레일링 스탑용 고점 추적 시작

            else:
                # ── 매도 조건 (트레일링 스탑 + RSI 과열 + 고정 손절) ──
                # 트레일링 스탑: 매수 후 고점 대비 TRAILING_STOP% 하락 시 청산
                매수_후_고점  = max(매수_후_고점, 현재가)
                트레일링_기준 = 매수_후_고점 * (1 - TRAILING_STOP / 100)
                트레일링_발생 = 현재가 <= 트레일링_기준

                손절_기준가 = 매수가 * (1 + STOP_LOSS / 100)
                손절_발생  = 현재가 <= 손절_기준가
                rsi_과열   = rsi > RSI_SELL

                if rsi_과열 or 손절_발생 or 트레일링_발생:
                    매도가  = 현재가 * (1 - 매도_비용)
                    수익률  = (매도가 - 매수가) / 매수가 * 100
                    수익률_목록.append(수익률)
                    if 수익률 < 0:
                        연속_손실 += 1
                        최대_연속_손실 = max(최대_연속_손실, 연속_손실)
                    else:
                        연속_손실 = 0
                    매수가       = None
                    매수_후_고점 = None

        if not 수익률_목록:
            return None

        누적_수익률 = sum(수익률_목록)
        누적_곡선   = [100.0]
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
        승률   = (sum(1 for r in 수익률_목록 if r > 0) / len(수익률_목록)) * 100

        return {
            "수익률":         round(누적_수익률, 1),
            "mdd":            round(mdd, 1),
            "sharpe":         round(sharpe, 2),
            "승률":           round(승률, 0),
            "거래횟수":       len(수익률_목록),
            "최대연속손실":   최대_연속_손실,
        }

    except Exception as e:
        print(f"⚠️ {ticker} 백테스트 오류: {e}")
        return None


# ─────────────────────────────────────────
# 5. ATR 기반 포지션 사이징
# ─────────────────────────────────────────



def run_backtest_walkforward(ticker, market, settings):
    """
    Walk-forward 백테스트를 실행한다.

    [중학생 설명]
    일반 백테스트의 문제: "과거 1년이 좋았다"는 결과만 봄.
    이 함수는 1년을 두 구간으로 나눈다:
    - 앞 8개월(학습): 이 기간에 전략이 얼마나 잘 맞았나?
    - 뒤 4개월(검증): 학습 기간 이후에도 실제로 잘 맞았나?

    만약 학습 구간 Sharpe가 2.0인데 검증 구간이 0.3이라면
    → "과거에 맞게 억지로 짜맞춰진 전략" (과적합)
    → 실전에서 믿으면 안 됨

    반환값: {
      "학습": 학습구간 백테스트 결과,
      "검증": 검증구간 백테스트 결과,
      "과적합_경고": Sharpe 차이가 0.5 이상이면 True,
      "신뢰도": "높음/보통/낮음"
    }
    """
    df = get_price_data(ticker, period="1y")
    if df is None or len(df) < 150:
        return None

    try:
        # 전체를 앞 67%(약 8개월)와 뒤 33%(약 4개월)로 분리
        split_idx  = int(len(df) * 0.67)
        df_학습    = df.iloc[:split_idx].copy()
        df_검증    = df.iloc[split_idx:].copy()

        if len(df_학습) < 60 or len(df_검증) < 40:
            return None

        bt_학습 = run_backtest(ticker, market, settings)  # 전체 1년 결과
        # 검증 구간만 따로 백테스트
        bt_검증 = _run_backtest_on_df(df_검증, market, settings)

        if bt_학습 is None or bt_검증 is None:
            return None

        # 과적합 판단: 학습 구간과 검증 구간 Sharpe 차이
        sharpe_차이 = bt_학습["sharpe"] - (bt_검증["sharpe"] if bt_검증 else 0)
        과적합_경고 = sharpe_차이 > 0.5

        if not 과적합_경고 and bt_검증["sharpe"] >= 1.0:
            신뢰도 = "높음"
        elif not 과적합_경고:
            신뢰도 = "보통"
        else:
            신뢰도 = "낮음"

        return {
            "학습":       bt_학습,
            "검증":       bt_검증,
            "과적합_경고": 과적합_경고,
            "신뢰도":      신뢰도,
            "sharpe_차이": round(sharpe_차이, 2),
        }
    except Exception as e:
        print(f"⚠️ {ticker} Walk-forward 오류: {e}")
        return None


def _run_backtest_on_df(df, market, settings):
    """
    특정 df 구간에 대해서만 백테스트를 실행하는 내부 함수.
    run_backtest와 동일한 로직이지만 yf 다운로드 없이 df를 직접 받는다.
    """
    if df is None or len(df) < 40:
        return None
    try:
        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze() if "Volume" in df.columns else None
        open_  = df["Open"].squeeze()   if "Open"   in df.columns else None
        high   = df["High"].squeeze()   if "High"   in df.columns else None
        low    = df["Low"].squeeze()    if "Low"    in df.columns else None

        rsi_series = calc_rsi(close, 14)
        ma_window  = int(settings.get("MA_WINDOW", 20))
        ma20       = close.rolling(ma_window).mean()
        ma20_slope = ma20.diff()
        bb_center, _, _ = calc_bollinger_bands(close)
        macd_line, sig_line, hist = calc_macd(close)
        avg_vol = volume.rolling(20).mean() if volume is not None else None

        RSI_BUY       = settings.get("RSI_BUY",       40)
        RSI_SELL      = settings.get("RSI_SELL",       75)
        STOP_LOSS     = settings.get("STOP_LOSS",      -5)
        TRAILING_STOP = float(settings.get("TRAILING_STOP", 8))
        threshold     = int(settings.get("BUY_SCORE_THRESHOLD", 3))
        commission    = settings.get("COMMISSION",  0.001)
        slippage      = settings.get("SLIPPAGE",    0.0005)
        거래세        = 0.002 if market == "KR" else 0.0

        매수_비용 = commission + slippage
        매도_비용 = commission + slippage + 거래세
        매수가 = None; 매수_후_고점 = None; rets = []

        for i in range(20, len(close)):
            현재가 = close.iloc[i]; rsi = rsi_series.iloc[i]
            if 매수가 is None:
                c1 = rsi < RSI_BUY
                c2 = (현재가 > ma20.iloc[i]) and (ma20_slope.iloc[i] > 0)
                c3 = (avg_vol is not None and not np.isnan(avg_vol.iloc[i])
                      and volume.iloc[i] > avg_vol.iloc[i] * 1.3)
                c4 = 현재가 <= bb_center.iloc[i]
                c5 = (hist.iloc[i] > 0 and hist.iloc[i-1] <= 0
                      and macd_line.iloc[i] > sig_line.iloc[i])
                c6 = False
                if open_ is not None and i >= 1:
                    기준 = open_.iloc[i] + (high.iloc[i-1] - low.iloc[i-1]) * 0.5
                    c6   = 현재가 > 기준
                if sum([c1,c2,c3,c4,c5,c6]) >= threshold:
                    매수가 = 현재가 * (1 + 매수_비용)
                    매수_후_고점 = 현재가
            else:
                매수_후_고점  = max(매수_후_고점, 현재가)
                트레일링_발생 = 현재가 <= 매수_후_고점 * (1 - TRAILING_STOP/100)
                손절_발생     = 현재가 <= 매수가 * (1 + STOP_LOSS/100)
                if rsi > RSI_SELL or 손절_발생 or 트레일링_발생:
                    rets.append((현재가*(1-매도_비용) - 매수가) / 매수가 * 100)
                    매수가 = None; 매수_후_고점 = None

        if not rets:
            return None
        cum = 100.
        for r in rets: cum *= (1 + r/100)
        pk = 100.; mdd = 0.; prev = 100.
        for r in rets:
            prev *= (1+r/100); pk = max(pk, prev)
            mdd = min(mdd, (prev-pk)/pk*100)
        sh  = float(np.mean(rets)/np.std(rets)) if np.std(rets) > 0 else 0
        win = sum(1 for r in rets if r > 0) / len(rets) * 100
        return {"수익률": round(cum-100,1), "mdd": round(mdd,1),
                "sharpe": round(sh,2), "승률": round(win,0), "거래횟수": len(rets)}
    except Exception as e:
        print(f"⚠️ _run_backtest_on_df 오류: {e}")
        return None

def calc_position_size(총자산, atr, market):
    """
    한 종목에서 최대 손실을 '총자산의 1%'로 제한하도록 매수 수량을 계산한다.

    [중학생 설명]
    "달걀을 한 바구니에 담지 말라"처럼, 한 종목에 너무 많이 투자하면 위험하다.
    ATR(하루 평균 변동폭)을 기준으로, 최악의 경우에도
    총자산의 1% 이상은 잃지 않도록 수량을 계산한다.

    예: 총자산 3000만원, ATR 1만원 → 손실 30만원(1%) ÷ ATR 1만원 = 30주
    """
    if atr is None or atr == 0 or (isinstance(atr, float) and np.isnan(atr)):
        return 0
    리스크_금액 = 총자산 * 0.01
    수량 = 리스크_금액 / atr

    if market in ("CRYPTO", "CRYPTO_KRW"):
        return round(수량, 6)
    else:
        return max(1, int(수량))
