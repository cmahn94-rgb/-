"""
market_phase.py — 시장 국면 5단계 감지 + 국면별 동적 임계값
=============================================================
[참고 봇(hybrid_trading_bot)의 핵심 아이디어 도입]

■ 문제: 고정 BUY_SCORE_THRESHOLD=4의 부작용
  - 급등장(강한상승): 4점 기준이 너무 높아 신호 0개 → 기회 놓침
  - 패닉장(급락): 4점 기준이 너무 낮아 위험한 종목도 진입

■ 해결: 국면에 따라 임계값을 자동으로 조정
  강한상승 → 2점 (기회 포착 우선)
  완만한상승 → 3점
  횡보 → 4점
  조정하락 → 5점
  급락패닉 → 6점 (거의 진입 안 함)

■ 국면 판단 기준:
  KOSPI/S&P500의 MA20, MA60, RSI, VIX를 복합 분석
  → 기존 마켓 레짐 필터(4중 하락 체크)와 통합
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Phase(Enum):
    STRONG_BULL = "🚀 강한상승"
    MILD_BULL   = "📈 완만한상승"
    SIDEWAYS    = "↔️ 횡보박스권"
    CORRECTION  = "📉 조정하락"
    PANIC       = "🔴 급락패닉"


# 국면별 설정
PHASE_CONFIG = {
    Phase.STRONG_BULL: {"score_threshold": 2, "description": "강한 상승장 — 모멘텀 전략, 임계값 완화"},
    Phase.MILD_BULL:   {"score_threshold": 3, "description": "완만한 상승 — 추세추종 + 눌림목"},
    Phase.SIDEWAYS:    {"score_threshold": 4, "description": "횡보 박스권 — 과매도 반등 위주"},
    Phase.CORRECTION:  {"score_threshold": 5, "description": "조정·하락 — 비중 축소, 반등 신호만"},
    Phase.PANIC:       {"score_threshold": 6, "description": "급락 패닉 — 현금 확대, 극단 과매도만"},
}


@dataclass
class PhaseResult:
    phase: Phase
    score_threshold: int   # 이 국면에서 사용할 매수 임계값
    description: str
    ma20: float = 0.0
    ma60: float = 0.0
    rsi:  float = 0.0
    vix:  float = 0.0
    bear_score: int = 0    # 하락 신호 개수 (기존 마켓 레짐 필터와 통합)


def detect_market_phase(market: str = "KR") -> PhaseResult:
    """
    현재 시장 국면을 감지하고 적절한 매수 임계값을 반환한다.

    [중학생 설명]
    주식 시장에는 계절처럼 '국면'이 있다.
    봄(강한상승), 여름(완만한상승), 가을(횡보), 겨울(조정), 혹한(패닉)처럼
    국면마다 다른 전략이 필요하다.
    지금이 봄이라면 조금 느슨하게 사도 되지만,
    혹한이라면 아주 확실한 신호가 있을 때만 사야 한다.

    반환값: PhaseResult (국면 + 임계값 + 세부 지표)
    """
    import yfinance as yf
    from indicators import calc_rsi as _calc_rsi

    # 시장별 지수 설정
    if market == "KR":
        idx_ticker = "^KS200"
        vix_ticker = "^VKOSPI"
    else:
        idx_ticker = "^GSPC"
        vix_ticker = "^VIX"

    ma20 = ma60 = rsi = vix = 0.0
    bear_score = 0

    try:
        # 지수 데이터
        df = yf.download(idx_ticker, period="1y", progress=False, auto_adjust=True)
        if df is not None and len(df) >= 60:
            close = df["Close"].squeeze()
            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma60 = float(close.rolling(60).mean().iloc[-1])
            ma20_5d = float(close.rolling(20).mean().iloc[-6])
            current = float(close.iloc[-1])
            rsi = float(_calc_rsi(close, 14).iloc[-1])

            # VIX
            vix_df = yf.download(vix_ticker, period="1mo", progress=False, auto_adjust=True)
            if vix_df is not None and len(vix_df) > 0:
                vix = float(vix_df["Close"].squeeze().iloc[-1])

            # 기존 하락 레짐 점수 계산 (4중 필터)
            bear_score = _calc_bear_score()

            # 국면 분류
            phase = _classify(current, ma20, ma60, ma20_5d, rsi, vix, bear_score)
        else:
            # 데이터 없으면 기본값 (횡보로 보수적 처리)
            phase = Phase.SIDEWAYS

    except Exception as e:
        print(f"⚠️ 시장 국면 감지 오류: {e}")
        phase = Phase.SIDEWAYS  # 오류 시 보수적 기본값

    cfg = PHASE_CONFIG[phase]
    return PhaseResult(
        phase=phase,
        score_threshold=cfg["score_threshold"],
        description=cfg["description"],
        ma20=ma20, ma60=ma60, rsi=rsi, vix=vix,
        bear_score=bear_score,
    )


def _classify(price, ma20, ma60, ma20_prev, rsi, vix, bear_score) -> Phase:
    """국면을 분류한다."""
    ma20_rising = ma20 > ma20_prev * 1.001

    # 복합 하락 신호 2개 이상 → 패닉 또는 조정
    if bear_score >= 2:
        if rsi < 30 and vix > 25:
            return Phase.PANIC
        return Phase.CORRECTION

    if price < ma60 and rsi < 30 and vix > 25:
        return Phase.PANIC
    if price < ma20 or (ma20 < ma60 * 0.99):
        return Phase.CORRECTION

    # 횡보: MA20-MA60 차이가 1.5% 미만이고 RSI 40~55
    if abs(ma20 - ma60) / ma60 < 0.015 and 40 <= rsi <= 55:
        return Phase.SIDEWAYS

    if price > ma20 > ma60:
        if rsi > 60 and ma20_rising:
            return Phase.STRONG_BULL
        return Phase.MILD_BULL

    return Phase.SIDEWAYS


def _calc_bear_score() -> int:
    """기존 마켓 레짐 필터: 4개 지표 중 하락 신호 개수를 반환"""
    import yfinance as yf
    score = 0
    try:
        sp = yf.download("^GSPC", period="1y",  progress=False, auto_adjust=True)["Close"].squeeze()
        ks = yf.download("^KS11", period="1y",  progress=False, auto_adjust=True)["Close"].squeeze()
        vx = yf.download("^VIX",  period="3mo", progress=False, auto_adjust=True)["Close"].squeeze()

        if len(sp) >= 200 and float(sp.iloc[-1]) < float(sp.rolling(200).mean().iloc[-1]) * 0.95:
            score += 1
        if len(ks) >= 200 and float(ks.iloc[-1]) < float(ks.rolling(200).mean().iloc[-1]) * 0.95:
            score += 1
        if len(vx) >= 20:
            cur_vix = float(vx.iloc[-1])
            avg_vix = float(vx.rolling(20).mean().iloc[-1])
            if cur_vix > 30 or cur_vix > avg_vix * 1.3:
                score += 1
    except Exception:
        pass
    return score
