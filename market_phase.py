"""
market_phase.py — 시장 국면 5단계 감지 + 국면별 동적 임계값
=============================================================
[v2 수정 내역]

🔴 버그 수정:
  1) ^KS200 → ^KS11 통일 (KS200은 yfinance 불안정, bear_score와 불일치)
  2) ^VKOSPI → ^VIX 대체 (VKOSPI yfinance 지원 불안정 → vix=0 고착 방지)
  3) ma20_5d iloc[-6] → 안전한 인덱스 접근 + nan 방어 코드 추가
  4) _calc_bear_score() 중복 다운로드 제거 → 데이터 재활용으로 통합

🟡 설계 개선:
  1) KR/US 시장 분리 감지: 23:30 미국장은 S&P500 기준 국면 별도 적용
  2) 횡보 조건 완화: MA 차이 1.5%→3%, RSI 범위 40~55→38~58
  3) bear_score 1개일 때 STRONG_BULL 억제 로직 추가
  4) API 호출 횟수 5~6회 → 3회로 축소 (실행 시간 단축)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from zoneinfo import ZoneInfo
from datetime import datetime

import numpy as np
import pandas as pd

# ④ 수정: get_price_data 사용 → bulk_download 캐시 활용
from data_loader import get_price_data as _get_price


class Phase(Enum):
    STRONG_BULL = "🚀 강한상승"
    MILD_BULL   = "📈 완만한상승"
    SIDEWAYS    = "↔️ 횡보박스권"
    CORRECTION  = "📉 조정하락"
    PANIC       = "🔴 급락패닉"


PHASE_CONFIG = {
    # [중요] RSI는 게이트 전용(점수 제외) → 기본점수 최대 5점(기존 6점).
    # 국면별 임계값을 1씩 낮춰 동일한 선별력 유지.
    # 리포트 표시 임계값 = signals.py 실제 판정 임계값 (불일치 없음).
    Phase.STRONG_BULL: {"score_threshold": 1, "description": "강한 상승장 — 모멘텀 전략, 임계값 완화"},
    Phase.MILD_BULL:   {"score_threshold": 2, "description": "완만한 상승 — 추세추종 + 눌림목"},
    Phase.SIDEWAYS:    {"score_threshold": 3, "description": "횡보 박스권 — 과매도 반등 위주"},
    Phase.CORRECTION:  {"score_threshold": 4, "description": "조정·하락 — 비중 축소, 반등 신호만"},
    Phase.PANIC:       {"score_threshold": 5, "description": "급락 패닉 — 현금 확대, 극단 과매도만"},
}


@dataclass
class PhaseResult:
    phase: Phase
    score_threshold: int
    description: str
    ma20: float = 0.0
    ma60: float = 0.0
    rsi:  float = 0.0
    vix:  float = 0.0
    bear_score: int = 0
    usdkrw: float = 0.0        # USD/KRW 환율 (0이면 수집 실패)
    usdkrw_trend: str = ""     # 상승/하락/횡보
    # 추가: 미국장 국면 (23:30 실행 시 US 기준 병렬 판단)
    us_phase: Phase | None = None


def detect_market_phase(market: str = "KR") -> PhaseResult:
    """
    현재 시장 국면을 감지하고 적절한 매수 임계값을 반환한다.

    [개선 내용]
    - KR 실행: KOSPI(^KS11) 기준 국면 감지
    - US 실행 (23:30): S&P500 기준 국면도 함께 감지 → 더 엄격한 쪽 적용
    - VIX: ^VKOSPI 불안정 → ^VIX(미국) 사용 (글로벌 공포 지수)
    - API 호출 통합: 중복 다운로드 제거 (5~6회 → 3회)

    반환값: PhaseResult (국면 + 임계값 + 세부 지표)
    """
    from indicators import calc_rsi as _calc_rsi
    # ④ yf.download 제거 — _get_price (get_price_data) 사용

    ma20 = ma60 = rsi = vix = 0.0
    bear_score = 0
    usdkrw = 0.0          # try 블록 밖에서 초기화 (locals() 버그 방지)
    usdkrw_trend = "횡보"  # try 실패 시 기본값 보장
    phase = Phase.SIDEWAYS  # 기본값 (오류 시 보수적)
    us_phase = None

    try:
        # ── 1) 공통: VIX + S&P500 — bulk_download 캐시 활용 ─────
        # ④ 수정: yf.download 직접 호출 → get_price_data (캐시 재사용)
        # scheduler_job.py의 bulk_download로 이미 캐시에 적재됨
        vix_df = _get_price("^VIX",  period="3mo")
        sp_df  = _get_price("^GSPC", period="1y")

        if vix_df is not None and len(vix_df) > 0:
            vix_close = vix_df["Close"].squeeze()
            vix = float(vix_close.iloc[-1]) if not pd.isna(vix_close.iloc[-1]) else 0.0

        # ── 2) KR 기준 국면 ──────────────────────────────────────
        ks_df = _get_price("^KS11", period="1y")

        if ks_df is not None and len(ks_df) >= 60:
            close = ks_df["Close"].squeeze()
            ma20  = _safe_float(close.rolling(20).mean().iloc[-1])
            ma60  = _safe_float(close.rolling(60).mean().iloc[-1])
            # [버그3 수정] iloc[-6] 대신 안전하게 최근 6일 평균의 5일 전 값 사용
            ma20_series = close.rolling(20).mean()
            ma20_5d = _safe_float(
                ma20_series.iloc[-6] if len(ma20_series) >= 6 else ma20_series.iloc[0]
            )
            current = _safe_float(close.iloc[-1])
            rsi_series = _calc_rsi(close, 14)
            rsi = _safe_float(rsi_series.iloc[-1])

            # [버그4 수정] bear_score를 이미 다운로드한 데이터로 계산 (중복 제거)
            bear_score = _calc_bear_score_from_data(sp_df, ks_df, vix_df)

            # USD/KRW 환율 수집 + bear_score 보정
            usdkrw, usdkrw_trend = _get_usdkrw()
            if usdkrw > 0:
                if usdkrw >= 1400 and usdkrw_trend == "상승":
                    bear_score += 1  # 환율 급등 = 외국인 이탈 압력
                    print(f"  ⚠️ 환율 {usdkrw:,.0f}원 상승 추세 → bear_score +1")
                elif usdkrw <= 1280 and usdkrw_trend == "하락":
                    bear_score = max(0, bear_score - 1)  # 환율 하락 = 외국인 유입
                    print(f"  ✅ 환율 {usdkrw:,.0f}원 하락 추세 → bear_score -1")

            phase = _classify(current, ma20, ma60, ma20_5d, rsi, vix, bear_score)
        else:
            print("⚠️ KOSPI 데이터 부족 → 횡보 기본값 적용")
            phase = Phase.SIDEWAYS

        # ── 3) US 기준 국면 (미국장 시간대에만) ─────────────────
        # 23:30 KST = 미국 장 중반. 미국 종목 분석 시 US 국면도 체크.
        now_h = datetime.now(ZoneInfo("Asia/Seoul")).hour
        is_us_session = (now_h >= 22 or now_h < 6)

        if is_us_session and sp_df is not None and len(sp_df) >= 60:
            sp_close = sp_df["Close"].squeeze()
            sp_ma20  = _safe_float(sp_close.rolling(20).mean().iloc[-1])
            sp_ma60  = _safe_float(sp_close.rolling(60).mean().iloc[-1])
            sp_ma20_series = sp_close.rolling(20).mean()
            sp_ma20_5d = _safe_float(
                sp_ma20_series.iloc[-6] if len(sp_ma20_series) >= 6 else sp_ma20_series.iloc[0]
            )
            sp_current = _safe_float(sp_close.iloc[-1])
            sp_rsi = _safe_float(_calc_rsi(sp_close, 14).iloc[-1])
            us_phase = _classify(sp_current, sp_ma20, sp_ma60, sp_ma20_5d, sp_rsi, vix, bear_score)

            # 미국장에서는 KR/US 중 더 보수적인 국면 적용
            phase = _more_conservative(phase, us_phase)

    except Exception as e:
        print(f"⚠️ 시장 국면 감지 오류: {e}")
        phase = Phase.SIDEWAYS

    cfg = PHASE_CONFIG[phase]
    return PhaseResult(
        phase=phase,
        score_threshold=cfg["score_threshold"],
        description=cfg["description"],
        ma20=ma20, ma60=ma60, rsi=rsi, vix=vix,
        bear_score=bear_score,
        usdkrw=usdkrw,            # 명시적 변수 참조 (locals() 제거)
        usdkrw_trend=usdkrw_trend,
        us_phase=us_phase,
    )




def _get_usdkrw() -> tuple[float, str]:
    """
    USD/KRW 환율과 추세를 수집한다.

    [중학생 설명]
    환율은 외국인 투자자가 한국 주식을 살 때 얼마나 불리한지를 보여준다.
    원화가 약해지면(환율 상승) 외국인이 한국 주식을 팔고 나가는 경향이 있다.
    반대로 원화가 강해지면(환율 하락) 외국인이 돈을 더 벌 수 있어 들어오는 경향.

    환율 기준 국면 보정:
    - 환율 1,400원 이상 + 상승 추세 → bear_score +1 (외국인 이탈 압력)
    - 환율 1,280원 이하 + 하락 추세 → bear_score -1 (외국인 유입 지지)
    - 그 외                         → bear_score 0 (중립)

    수집 순서: yfinance(KRW=X) → exchangerate-api.com (무료, 키 불필요)
    반환값: (환율 숫자, "상승"/"하락"/"횡보")
    """
    import requests

    rate = 0.0
    trend = "횡보"

    # ── 소스 1: get_price_data (캐시 활용) ───────────────
    try:
        df = _get_price("KRW=X", period="1mo")
        if df is not None and len(df) >= 5:
            close = df["Close"].squeeze()
            rate  = _safe_float(close.iloc[-1])
            # 5일 전 대비 추세 판단
            prev  = _safe_float(close.iloc[-6] if len(close) >= 6 else close.iloc[0])
            if rate > 0 and prev > 0:
                change = (rate - prev) / prev * 100
                trend = "상승" if change > 0.5 else ("하락" if change < -0.5 else "횡보")
            if rate > 100:  # 정상 범위 확인 (KRW는 1000~1500 사이)
                print(f"  💱 USD/KRW (yfinance): {rate:,.1f}원 ({trend})")
                return rate, trend
    except Exception:
        pass

    # ── 소스 2: exchangerate-api.com + 캐시 파일로 추세 계산 ──
    # ⑬ 수정: 이전 환율을 .usdkrw_cache에 저장 → 다음 호출에서 추세 계산 가능
    try:
        import os as _os_fx
        _cache_path = _os_fx.path.join(
            _os_fx.path.dirname(_os_fx.path.abspath(__file__)), ".usdkrw_cache"
        )
        resp = requests.get(
            "https://api.exchangerate-api.com/v4/latest/USD",
            timeout=8
        )
        if resp.status_code == 200:
            data = resp.json()
            rate = float(data.get("rates", {}).get("KRW", 0))
            if rate > 100:
                # 이전 환율 읽기
                prev_rate = 0.0
                try:
                    prev_rate = float(open(_cache_path).read().strip())
                except Exception:
                    pass
                # 추세 계산
                if prev_rate > 0:
                    chg = (rate - prev_rate) / prev_rate * 100
                    trend = "상승" if chg > 0.3 else ("하락" if chg < -0.3 else "횡보")
                # 현재 환율 캐시 저장
                try:
                    open(_cache_path, "w").write(str(rate))
                except Exception:
                    pass
                print(f"  💱 USD/KRW (exchangerate-api): {rate:,.1f}원 ({trend})")
                return rate, trend
    except Exception:
        pass

    print("  ⚠️ USD/KRW 환율 수집 실패 → 국면 판단에서 제외")
    return 0.0, "횡보"

def _safe_float(val) -> float:
    """NaN/None/Series를 안전하게 float으로 변환. 실패 시 0.0 반환."""
    try:
        v = float(val)
        return 0.0 if (v != v) else v  # nan 체크
    except Exception:
        return 0.0


def _classify(price, ma20, ma60, ma20_prev, rsi, vix, bear_score) -> Phase:
    """
    국면을 분류한다.

    [개선 내용]
    - bear_score=1 이어도 RSI/VIX 조건 추가 체크 (STRONG_BULL 억제)
    - 횡보 조건 완화: MA 차이 1.5%→3%, RSI 38~58 (기존 40~55)
    - 모든 인자 0.0 시 SIDEWAYS 반환 (방어)
    """
    # 모든 값이 0이면 데이터 없는 것 → 기본값
    if ma20 == 0.0 and ma60 == 0.0:
        return Phase.SIDEWAYS

    ma20_rising = (ma20 > ma20_prev * 1.001) if ma20_prev > 0 else False

    # 복합 하락 신호 2개 이상 → 패닉 또는 조정
    if bear_score >= 2:
        if rsi < 30 and vix > 25:
            return Phase.PANIC
        return Phase.CORRECTION

    # 극단적 패닉: MA60 아래 + RSI 과매도 + VIX 급등
    if price < ma60 and rsi < 30 and vix > 25:
        return Phase.PANIC

    # 조정: MA20 아래 OR MA20이 MA60 아래로 꺾임
    if price > 0 and ma20 > 0 and (price < ma20 or (ma60 > 0 and ma20 < ma60 * 0.99)):
        return Phase.CORRECTION

    # [설계1 개선] bear_score=1이면 STRONG_BULL 억제
    if bear_score >= 1:
        if price > ma20 > ma60:
            if rsi > 60 and ma20_rising:
                return Phase.MILD_BULL  # STRONG_BULL 대신 MILD_BULL
            return Phase.MILD_BULL
        return Phase.SIDEWAYS

    # [설계4 개선] 횡보: 조건 완화 (MA 차이 3%, RSI 38~58)
    if ma60 > 0 and abs(ma20 - ma60) / ma60 < 0.030 and 38 <= rsi <= 58:
        return Phase.SIDEWAYS

    # 상승 구간
    if price > ma20 > ma60:
        if rsi > 60 and ma20_rising:
            return Phase.STRONG_BULL
        return Phase.MILD_BULL

    return Phase.SIDEWAYS


def _more_conservative(phase_kr: Phase, phase_us: Phase | None) -> Phase:
    """
    KR과 US 국면 중 더 보수적인(임계값 높은) 쪽을 반환한다.

    [이유]
    미국장(23:30) 실행 시 나스닥이 폭락해도 KOSPI 기준으로만 판단하면
    MILD_BULL이 나와 미국 종목에 위험하게 진입할 수 있다.
    두 국면 중 더 엄격한 쪽을 적용해 안전성을 높인다.
    """
    if phase_us is None:
        return phase_kr
    # 임계값이 높을수록 보수적
    kr_thr = PHASE_CONFIG[phase_kr]["score_threshold"]
    us_thr = PHASE_CONFIG[phase_us]["score_threshold"]
    return phase_kr if kr_thr >= us_thr else phase_us


def _calc_bear_score_from_data(
    sp_df: pd.DataFrame | None,
    ks_df: pd.DataFrame | None,
    vix_df: pd.DataFrame | None,
) -> int:
    """
    이미 다운로드한 데이터 + 선행지표로 하락 레짐 점수를 계산한다.

    [선행지표 추가 — 수정 이유]
    MA200, KOSPI, VIX는 모두 후행 지표다.
    하락장이 시작되고 MA가 꺾인 다음에야 CORRECTION으로 바뀐다.
    이미 10~15% 빠진 뒤에 경고가 오는 문제를 개선하기 위해
    선행 지표 2가지를 추가한다:

    ① 10년물 국채금리(^TNX): 빠르게 오르면(전월 대비 +0.3%p 이상)
       기업 대출비용 상승 + 주식 밸류에이션 압박 → 선제 경고
       역사적으로 금리 급등 후 3~6개월 내 증시 조정 발생

    ② 시장 공포 지수 (^PCALL → ^VVIX → ^VIX3M → VIX모멘텀 순 폴백):
       투자자들이 하락 헤지(풋 옵션)를 얼마나 사는지를 나타냄
       0.8 이상 = 시장 참여자 다수가 하락에 베팅 중 → 선제 경고
       MA200/VIX보다 1~2주 앞서 반응하는 선행 지표
    """
    score = 0
    try:
        # S&P500: 현재가 < 200일 평균 * 0.95
        if sp_df is not None and len(sp_df) >= 200:
            sp = sp_df["Close"].squeeze()
            sp_ma200 = sp.rolling(200).mean()
            cur  = _safe_float(sp.iloc[-1])
            ma200 = _safe_float(sp_ma200.iloc[-1])
            if cur > 0 and ma200 > 0 and cur < ma200 * 0.95:
                score += 1

        # KOSPI: 현재가 < 200일 평균 * 0.95
        if ks_df is not None and len(ks_df) >= 200:
            ks = ks_df["Close"].squeeze()
            ks_ma200 = ks.rolling(200).mean()
            cur  = _safe_float(ks.iloc[-1])
            ma200 = _safe_float(ks_ma200.iloc[-1])
            if cur > 0 and ma200 > 0 and cur < ma200 * 0.95:
                score += 1

        # VIX: 현재 VIX > 30 또는 20일 평균의 1.3배 이상
        if vix_df is not None and len(vix_df) >= 20:
            vx = vix_df["Close"].squeeze()
            cur_vix = _safe_float(vx.iloc[-1])
            avg_vix = _safe_float(vx.rolling(20).mean().iloc[-1])
            if cur_vix > 0 and (cur_vix > 30 or (avg_vix > 0 and cur_vix > avg_vix * 1.3)):
                score += 1

    except Exception as e:
        print(f"⚠️ bear_score 후행지표 계산 오류: {e}")

    # ── 선행지표 ①: 10년물 국채금리 급등 ─────────────────────────
    try:
        tnx_df = _get_price("^TNX", period="3mo")
        if tnx_df is not None and len(tnx_df) >= 22:
            tnx = tnx_df["Close"].squeeze()
            현재_금리  = _safe_float(tnx.iloc[-1])
            한달전_금리 = _safe_float(tnx.iloc[-22])  # 약 1개월 전
            if 현재_금리 > 0 and 한달전_금리 > 0:
                금리_상승폭 = 현재_금리 - 한달전_금리  # %p 단위
                if 금리_상승폭 >= 0.3:  # 한 달 사이 0.3%p 이상 급등
                    score += 1
                    print(f"  ⚠️ 10Y 금리 급등 ({한달전_금리:.2f}% → {현재_금리:.2f}%, +{금리_상승폭:.2f}%p) → bear_score +1")
    except Exception as e:
        pass  # 금리 데이터 실패 시 무시 (yfinance 미지원 환경 대비)

    # ── 선행지표 ②: 시장 공포/헤지 수요 지표 ─────────────────────
    # ^PCCE(CBOE Equity Put/Call)가 yfinance에서 상장 폐지됨 → 대체 심볼 순서대로 시도
    # 대체 순서: ^PCALL(전체 P/C) → ^VVIX(VIX의 변동성) → VIX 급등 모멘텀
    try:
        공포_지수_심볼 = None
        공포_df = None
        for sym in ["^PCALL", "^VVIX", "^VIX3M"]:
            try:
                df_시도 = _get_price(sym, period="1mo")
                if df_시도 is not None and len(df_시도) >= 5:
                    공포_지수_심볼 = sym
                    공포_df = df_시도
                    break
            except Exception:
                continue

        if 공포_df is not None:
            공포 = 공포_df["Close"].squeeze()
            현재값 = _safe_float(공포.iloc[-1])
            평균값 = _safe_float(공포.rolling(5).mean().iloc[-1])
            if 공포_지수_심볼 == "^PCALL":
                발동 = 현재값 >= 0.8 or (평균값 > 0 and 현재값 > 평균값 * 1.2)
            elif 공포_지수_심볼 == "^VVIX":
                발동 = 현재값 >= 95 or (평균값 > 0 and 현재값 > 평균값 * 1.15)
            else:
                발동 = 현재값 >= 25
            if 발동:
                score += 1
                print(f"  ⚠️ 공포 지수 급등 ({공포_지수_심볼}={현재값:.2f}) → 하락 헤지 수요 증가 → bear_score +1")
        else:
            # 모든 공포 지수 실패 → VIX 5일 급등 모멘텀으로 대체
            if vix_df is not None and len(vix_df) >= 6:
                vx = vix_df["Close"].squeeze()
                vix_현재 = _safe_float(vx.iloc[-1])
                vix_5일전 = _safe_float(vx.iloc[-6])
                if vix_5일전 > 0 and vix_현재 / vix_5일전 >= 1.3:
                    score += 1
                    print(f"  ⚠️ VIX 5일 급등 ({vix_5일전:.1f}→{vix_현재:.1f}) → bear_score +1")
    except Exception:
        pass

    return score


# 하위 호환성 유지 (기존 코드가 _calc_bear_score()를 직접 호출하는 경우 대비)
def _calc_bear_score() -> int:
    """기존 호출 호환용. 새 코드에서는 _calc_bear_score_from_data() 사용."""
    sp_df  = _get_price("^GSPC", period="1y")
    ks_df  = _get_price("^KS11", period="1y")
    vix_df = _get_price("^VIX",  period="3mo")
    return _calc_bear_score_from_data(sp_df, ks_df, vix_df)
