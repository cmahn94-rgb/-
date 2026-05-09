"""
indicators.py — 기술 지표 계산 (RSI, 볼린저밴드, MACD, ATR, ADX)
=================================================================
이 파일이 하는 일:
  모든 기술 지표를 계산하는 순수 함수 모음.
  각 함수는 pandas Series 또는 DataFrame을 받아 지표값을 반환한다.

[지표 개념 요약 — 중학생도 이해할 수 있게]
  RSI        : 최근 14일간 오른 날/내린 날 비율. 30 미만 = 많이 빠진 상태(저평가 가능)
  볼린저밴드 : 평균가 ± 2*표준편차. 하단 터치 = 통계적으로 싼 가격
  MACD       : 단기(12일)-장기(26일) 이동평균 차이. 골든크로스 = 상승 전환 신호
  ATR        : 하루 평균 변동폭. 클수록 위험 = 더 적은 수량을 사야 안전
  ADX (신규) : 추세 강도 지표. 25 이상이면 "지금 추세가 뚜렷하다"는 의미
               - ADX가 낮으면(횡보장) 변동성 돌파 신호를 믿으면 안 됨
               - ADX가 높으면(추세장) 매수/매도 신호의 신뢰도가 올라감
"""

import pandas as pd
import numpy as np


def calc_rsi(series, period=14):
    """
    RSI(상대강도지수, Relative Strength Index)를 계산한다.

    [중학생 설명]
    최근 14일 동안 오른 날과 내린 날의 '힘의 비율'을 숫자로 나타낸 것.
    - 30 미만: 너무 많이 내려서 반등 가능성 있음 (저평가 구간)
    - 70 초과: 너무 많이 올라서 조정 가능성 있음 (과열 구간)
    - 50 기준: 45~50 미만이면 아직 충분히 안 올랐으니 매수 기회일 수 있다

    반환값: RSI 값들의 pandas Series (0~100 사이 숫자)
    """
    # 전날 대비 가격 변화량 계산
    delta = series.diff()

    # 오른 날(gain)과 내린 날(loss)을 분리
    gain = delta.clip(lower=0)    # 오른 날만 (내린 날은 0으로 처리)
    loss = -delta.clip(upper=0)   # 내린 날만 (오른 날은 0으로 처리)

    # 지수이동평균(EWM)으로 평균 상승폭·하락폭 계산
    # com=period-1 → 14일 기준의 지수 감쇠 계수
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # RS = 평균 상승폭 / 평균 하락폭
    rs = avg_gain / avg_loss

    # RSI 공식: 100 - (100 / (1 + RS))
    # RS가 클수록(많이 오른 날이 많을수록) RSI는 100에 가까워짐
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_bollinger_bands(series, window=20, num_std=2):
    """
    볼린저밴드(Bollinger Bands)를 계산한다.

    [중학생 설명]
    20일 평균가를 중심으로 "보통 이 정도 범위 안에서 움직인다"는 띠를 만든 것.
    - 상단밴드 위: 통계적으로 비싼 가격 (팔 타이밍 가능성)
    - 하단밴드 아래: 통계적으로 싼 가격 (살 타이밍 가능성)
    - 전체 거래일의 약 95%는 이 밴드 안에 있음

    반환값: (중심선, 상단밴드, 하단밴드) 각각 pandas Series
    """
    # 중심선(Middle Band): 20일 단순이동평균
    중심선 = series.rolling(window).mean()

    # 표준편차: 가격이 평균에서 얼마나 흩어져 있는지 (변동성 측정)
    표준편차 = series.rolling(window).std()

    # 상단밴드 = 평균 + 2*표준편차
    상단밴드 = 중심선 + (num_std * 표준편차)

    # 하단밴드 = 평균 - 2*표준편차
    하단밴드 = 중심선 - (num_std * 표준편차)

    return 중심선, 상단밴드, 하단밴드


def calc_macd(series, fast=12, slow=26, signal=9):
    """
    MACD(이동평균 수렴·발산, Moving Average Convergence Divergence)를 계산한다.

    [중학생 설명]
    단기(12일)와 장기(26일) 이동평균의 '차이'를 추적하는 지표.
    - MACD선이 시그널선 위로 올라서면 '골든크로스' = 상승 전환 신호
    - 히스토그램이 음수(-) → 양수(+)로 바뀌면 하락 에너지가 꺾인 것

    반환값: (MACD선, 시그널선, 히스토그램) 각각 pandas Series
    """
    # 단기 지수이동평균(12일): 최근 가격에 더 큰 비중
    ema_단기 = series.ewm(span=fast, adjust=False).mean()

    # 장기 지수이동평균(26일): 완만하게 추세를 따라감
    ema_장기 = series.ewm(span=slow, adjust=False).mean()

    # MACD선 = 단기 - 장기 (양수: 단기가 더 강하게 오르는 중)
    macd선 = ema_단기 - ema_장기

    # 시그널선 = MACD선의 9일 지수이동평균
    시그널선 = macd선.ewm(span=signal, adjust=False).mean()

    # 히스토그램 = MACD선 - 시그널선
    # 음수→양수 전환: 하락 에너지가 꺾이고 상승 에너지가 살아나는 신호
    히스토그램 = macd선 - 시그널선

    return macd선, 시그널선, 히스토그램


def calc_atr(df, period=20):
    """
    ATR(평균 진폭, Average True Range)을 계산한다.

    [중학생 설명]
    주가가 하루에 평균 얼마나 움직이는지를 나타내는 '변동성 자'다.
    - ATR이 크면 = 하루에 많이 흔들린다 = 리스크가 크다
    - 같은 1% 손실을 막으려면, ATR이 클수록 더 적은 수량을 사야 함
    - 암호화폐는 주식보다 ATR이 2~5배 크므로, 수량이 적게 나오는 것이 정상

    반환값: ATR 값 (숫자 하나, 원 또는 달러 단위)
    """
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    close = df["Close"].squeeze()

    prev_close = close.shift(1)  # 전일 종가

    # True Range = 다음 3가지 중 최대값
    tr1 = high - low                        # 오늘 고가 - 오늘 저가 (일반 변동폭)
    tr2 = (high - prev_close).abs()         # 오늘 고가 - 전일 종가 (갭 상승 포함)
    tr3 = (low  - prev_close).abs()         # 오늘 저가 - 전일 종가 (갭 하락 포함)

    # 세 값 중 최대값 = True Range (실제 체감 변동 범위)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR = True Range의 20일 평균
    atr_변동폭 = true_range.rolling(period).mean().iloc[-1]
    if pd.isna(atr_변동폭):
        return None
    return atr_변동폭


def calc_adx(df, period=14):
    """
    ADX(평균방향지수, Average Directional Index)를 계산한다. [신규 추가]

    [중학생 설명]
    지금 주가가 "뚜렷한 방향을 가지고 움직이는지"를 나타내는 지표.
    방향(오르는지/내리는지)이 아니라, "얼마나 강하게 추세가 있는가"만 본다.

    - ADX < 20: 추세 없음 (횡보장) → 매수/매도 신호가 자주 틀림
    - ADX 20~25: 약한 추세 시작
    - ADX ≥ 25: 뚜렷한 추세 → 매수/매도 신호 신뢰도 높음
    - ADX ≥ 40: 매우 강한 추세

    반환값: ADX 값 (숫자 하나, 0~100 사이) 또는 None
    """
    try:
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()
        close = df["Close"].squeeze()

        if len(close) < period * 2 + 5:
            return None

        # +DM (양의 방향 이동): 오늘 고가 - 어제 고가 (상승 압력)
        # -DM (음의 방향 이동): 어제 저가 - 오늘 저가 (하락 압력)
        high_diff = high.diff()
        low_diff  = -low.diff()

        # +DM: 상승폭이 하락폭보다 크고 양수일 때만 인정
        plus_dm  = pd.Series(
            np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0),
            index=high.index
        )
        # -DM: 하락폭이 상승폭보다 크고 양수일 때만 인정
        minus_dm = pd.Series(
            np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0),
            index=low.index
        )

        # True Range 계산 (ATR과 동일한 방식)
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)

        # 지수이동평균으로 평활화 (Wilder's smoothing)
        atr_smooth    = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di_smooth  = plus_dm.ewm(alpha=1/period, adjust=False).mean()
        minus_di_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()

        # +DI, -DI 계산 (방향 지표, 0~100)
        plus_di  = 100 * plus_di_smooth  / atr_smooth.replace(0, np.nan)
        minus_di = 100 * minus_di_smooth / atr_smooth.replace(0, np.nan)

        # DX = |+DI - -DI| / (+DI + -DI) * 100
        di_diff = (plus_di - minus_di).abs()
        di_sum  = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * di_diff / di_sum

        # ADX = DX의 지수이동평균 (추세 강도)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        결과값 = adx.iloc[-1]
        if pd.isna(결과값):
            return None
        return float(결과값)

    except Exception as e:
        # ADX 계산 실패 시 None 반환 (이 지표 하나 때문에 전체가 멈추지 않도록)
        return None
