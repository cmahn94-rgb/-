"""
indicators.py — 기술 지표 계산 (RSI, 볼린저밴드, MACD, ATR)
=============================================================
이 파일이 하는 일:
  모든 기술 지표를 계산하는 순수 함수 모음.
  각 함수는 pandas Series 또는 DataFrame을 받아 지표값을 반환한다.

[지표 개념 요약]
  RSI        : 최근 14일간 오른 날/내린 날 비율. 45 미만 = 저평가 구간
  볼린저밴드 : 평균가 ± 2*표준편차. 하단 터치 = 통계적 저점
  MACD       : 단기(12일)-장기(26일) 이동평균 차이. 골든크로스 = 상승 전환
  ATR        : 하루 평균 변동폭. 클수록 변동성이 크다 = 포지션 축소 필요
"""

import pandas as pd
import numpy as np


def calc_rsi(series, period=14):
    """
    RSI(상대강도지수, Relative Strength Index)를 계산한다.

    RSI란? 최근 14일간 오른 날/내린 날의 비율로 과매수·과매도를 판단하는 지표.
    - 30 미만: 과매도 (너무 많이 내려서 반등 가능성 있음)
    - 70 초과: 과매수 (너무 많이 올라서 조정 가능성 있음)
    - 45 미만 기준: 아직 충분히 안 올랐으니 매수 기회일 수 있다
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

    볼린저밴드란? 평균가 ± 2*표준편차로 만든 가격 범위.
    - 상단밴드 위: 과매수 구간 (고평가 가능성)
    - 하단밴드 아래: 과매도 구간 (저평가 가능성, 매수 기회)
    통계적으로 전체 거래일의 약 95%는 이 밴드 안에 있다.
    """
    # 중심선(Middle Band): 20일 단순이동평균
    중심선 = series.rolling(window).mean()

    # 표준편차: 가격이 평균에서 얼마나 흩어져 있는지 (변동성 측정)
    표준편차 = series.rolling(window).std()

    # 상단밴드 = 평균 + 2*표준편차 (여기에 닿으면 통계적으로 비싼 가격)
    상단밴드 = 중심선 + (num_std * 표준편차)

    # 하단밴드 = 평균 - 2*표준편차 (여기에 닿으면 통계적으로 싼 가격)
    하단밴드 = 중심선 - (num_std * 표준편차)

    return 중심선, 상단밴드, 하단밴드


def calc_macd(series, fast=12, slow=26, signal=9):
    """
    MACD(이동평균 수렴·발산, Moving Average Convergence Divergence)를 계산한다.

    MACD란? 단기(12일)와 장기(26일) 이동평균의 차이.
    - MACD선이 시그널선 위로 올라서면 '골든크로스' = 상승 전환 신호
    - 히스토그램이 음수 → 양수로 바뀌면 모멘텀(상승 에너지) 반등 신호
    """
    # 단기 지수이동평균(12일): 최근 가격에 더 큰 비중
    ema_단기 = series.ewm(span=fast, adjust=False).mean()

    # 장기 지수이동평균(26일): 좀 더 완만하게 추세를 따라감
    ema_장기 = series.ewm(span=slow, adjust=False).mean()

    # MACD선 = 단기 - 장기 (양수: 단기가 더 강하게 오르는 중)
    macd선 = ema_단기 - ema_장기

    # 시그널선 = MACD선의 9일 지수이동평균 (MACD의 평균)
    시그널선 = macd선.ewm(span=signal, adjust=False).mean()

    # 히스토그램 = MACD선 - 시그널선
    # 음수 → 양수 전환: 하락 에너지가 꺾이고 상승 에너지가 살아나는 신호
    히스토그램 = macd선 - 시그널선

    return macd선, 시그널선, 히스토그램


def calc_atr(df, period=20):
    """
    ATR(평균 진폭, Average True Range)을 계산한다.

    ATR이란? 주가가 하루에 평균 얼마나 움직이는지 나타내는 변동성 지표.
    - ATR이 클수록 하루 변동폭이 크다 = 리스크가 크다
    - 같은 1% 리스크를 지키려면, ATR이 클수록 더 적은 수량을 사야 한다.
    - 암호화폐는 주식보다 ATR이 2~5배 크므로 수량이 적게 나오는 것이 정상이다.
    """
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    close = df["Close"].squeeze()

    prev_close = close.shift(1)  # 전일 종가

    # True Range = 다음 3가지 중 최대값
    tr1 = high - low                        # 오늘 고가 - 오늘 저가 (일반 변동폭)
    tr2 = (high - prev_close).abs()         # 오늘 고가 - 전일 종가 (갭 상승 포함)
    tr3 = (low  - prev_close).abs()         # 오늘 저가 - 전일 종가 (갭 하락 포함)

    # 세 값 중 최대값 = True Range (실제 변동 범위)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR = True Range의 20일 평균
    atr_변동폭 = true_range.rolling(period).mean().iloc[-1]
    # ✅ NaN 방어: 데이터 부족 시 None 반환
    if pd.isna(atr_변동폭):
        return None
    
    return atr_변동폭
