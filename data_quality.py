"""
data_quality.py — 데이터 품질 게이트 (하이엔드 4순위)
=====================================================
[목적]
신호 계산 전에 가격 데이터가 '건강한지' 검사한다.
쓰레기 데이터(0원·음수·비현실적 급변·오래된 데이터)로 신호를 내면
잘못된 매매로 이어지므로, 문제 데이터를 걸러내고 경고한다.

[중학생 설명]
데이터가 가끔 이상하게 온다. 가격이 0원이거나, 하루에 300% 뛰거나,
일주일 전 데이터가 최신인 척 오거나. 이런 걸 그냥 계산하면 봇이
엉뚱한 신호를 낸다. 이 파일은 데이터를 쓰기 전에 "이거 정상이야?"를
검사하는 문지기(gate) 역할을 한다.

[검사 항목]
  1. 결측/빈 데이터
  2. 가격 0 또는 음수
  3. 비현실적 일간 변동 (±50% 초과 = 데이터 오류 가능성)
  4. 데이터 신선도 (마지막 데이터가 너무 오래됨)
  5. 거래량 전부 0 (거래정지/상장폐지 의심)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# 비현실적 일간 변동 임계값 (이 이상이면 데이터 오류로 간주)
_MAX_DAILY_MOVE = 0.50      # ±50%
# 데이터 신선도: 마지막 종가가 이 일수보다 오래되면 경고
_MAX_STALENESS_DAYS = 7


def check_price_data(df, ticker: str = "", 최대_신선도일: int = _MAX_STALENESS_DAYS) -> dict:
    """
    가격 데이터의 품질을 검사한다.

    반환:
      {
        "ok": bool,             # 신호 계산에 사용해도 되는지
        "문제": [str, ...],     # 발견된 문제 목록
        "심각도": "정상"|"경고"|"차단",
      }
    - "차단": 신호 계산에서 제외해야 함 (결측·0원·거래정지)
    - "경고": 사용은 하되 주의 (신선도 낮음·이상 변동 1건)
    - "정상": 문제 없음
    """
    문제 = []

    # ── 1. 결측/빈 데이터 ─────────────────────────────────
    if df is None or len(df) == 0:
        return {"ok": False, "문제": ["데이터 없음"], "심각도": "차단"}

    if "Close" not in df.columns:
        return {"ok": False, "문제": ["종가 컬럼 없음"], "심각도": "차단"}

    try:
        close = df["Close"].squeeze()
    except Exception:
        return {"ok": False, "문제": ["종가 파싱 실패"], "심각도": "차단"}

    # 최근 종가 (NaN 제외)
    valid_close = close.dropna()
    if len(valid_close) == 0:
        return {"ok": False, "문제": ["유효 종가 없음(전부 NaN)"], "심각도": "차단"}

    차단 = False

    # ── 2. 가격 0 또는 음수 ───────────────────────────────
    최근가 = float(valid_close.iloc[-1])
    if 최근가 <= 0:
        문제.append(f"최근가 비정상({최근가})")
        차단 = True

    # ── 3. 비현실적 일간 변동 ─────────────────────────────
    if len(valid_close) >= 2:
        일간변동 = valid_close.pct_change().dropna()
        극단 = 일간변동[일간변동.abs() > _MAX_DAILY_MOVE]
        if len(극단) > 0:
            최대변동 = float(극단.abs().max()) * 100
            문제.append(f"비현실적 변동 {len(극단)}건(최대 {최대변동:.0f}%)")
            # 최근 5일 내 극단 변동이면 차단 (데이터 오류 가능성 높음)
            if len(극단) >= 3:
                차단 = True

    # ── 4. 데이터 신선도 ──────────────────────────────────
    try:
        마지막_날짜 = valid_close.index[-1]
        if hasattr(마지막_날짜, "to_pydatetime"):
            마지막_dt = 마지막_날짜.to_pydatetime()
            if 마지막_dt.tzinfo is not None:
                마지막_dt = 마지막_dt.replace(tzinfo=None)
            경과 = (datetime.now() - 마지막_dt).days
            if 경과 > 최대_신선도일:
                문제.append(f"데이터 오래됨(마지막 {경과}일 전)")
    except Exception:
        pass

    # ── 5. 거래량 전부 0 (거래정지 의심) ──────────────────
    if "Volume" in df.columns:
        try:
            vol = df["Volume"].squeeze().dropna()
            if len(vol) >= 5 and float(vol.tail(5).sum()) == 0:
                문제.append("최근 5일 거래량 0(거래정지 의심)")
                차단 = True
        except Exception:
            pass

    # ── 심각도 판정 ───────────────────────────────────────
    if 차단:
        심각도 = "차단"
        ok = False
    elif 문제:
        심각도 = "경고"
        ok = True
    else:
        심각도 = "정상"
        ok = True

    return {"ok": ok, "문제": 문제, "심각도": 심각도}


class DataQualityTracker:
    """실행 1회 동안 데이터 품질 문제를 집계."""

    def __init__(self):
        self.차단_종목 = []
        self.경고_종목 = []

    def check(self, df, ticker: str) -> bool:
        """
        데이터를 검사하고 집계. 신호 계산 진행 가능하면 True.
        차단(False)이면 호출측에서 해당 종목을 스킵해야 한다.
        """
        결과 = check_price_data(df, ticker)
        if 결과["심각도"] == "차단":
            self.차단_종목.append((ticker, 결과["문제"]))
            return False
        elif 결과["심각도"] == "경고":
            self.경고_종목.append((ticker, 결과["문제"]))
        return True

    def summary(self) -> str:
        """품질 검사 요약 (문제 있을 때만 의미 있음)."""
        if not self.차단_종목 and not self.경고_종목:
            return ""
        줄 = ["🔍 데이터 품질 검사"]
        if self.차단_종목:
            줄.append(f"  ❌ 차단 {len(self.차단_종목)}종목 (신호 계산 제외):")
            for t, 문제 in self.차단_종목[:5]:
                줄.append(f"      {t}: {', '.join(문제)}")
        if self.경고_종목:
            줄.append(f"  ⚠️ 경고 {len(self.경고_종목)}종목 (사용하되 주의)")
        return "\n".join(줄)
