"""
momentum.py — 돌파 추종형 단기 모멘텀 전략 (안정 전략과 독립)
================================================================
[철학 — 안정 전략(signals.py)의 정반대]
  안정: RSI<40 과매도 눌림목을 "싸게" 줍는다 (수일~수주 보유)
  모멘텀: 강세 돌파를 "따라타고" 빠르게 나온다 (당일~2일 보유)

[진입 조건 5개 — RSI 게이트 없음]
  M1. 당일 종가가 20일 신고가 근처(≥98%)  → 단기 저항 돌파, 추가 상승 여력
  M2. 5일선 > 20일선 (단기 정배열)         → 단기 추세 살아있음
  M3. 거래량 ≥ 20일 평균 × 2.0배           → 돌파에 실제 자금 유입
  M4. 당일 상승률 +3% 이상                  → 오늘 이미 강한 종목
  M5. 상대강도(RS) 시장 대비 +5%p 이상      → 시장 빠져도 강한 주도주

[청산 — 안정봇보다 타이트]
  손절 -2.5% / 익절 +5% / 트레일링 -3% / 최대 보유 2일

[중요]
  - 가격 데이터는 get_price_data() 캐시 재사용 → 추가 API 호출 0
  - 안정 전략 코드(signals.py)는 일절 건드리지 않음
  - 백테스트 진입/청산 가정만 안정봇과 다름 (2일 보유, 손절 -2.5%, 익절 +5%)
  - 거래비용·벤치마크·통화기호·ATR 보정은 strategy_utils 공용 헬퍼 사용
    → 두 전략이 같은 거래비용 가정으로 공정하게 비교됨 (v5.14 리팩토링)

[고위험 경고]
  모멘텀 단타는 승률이 낮고(50% 안팎) 백테스트-실전 괴리가 크다.
  슬리피지·갭·체결 실패가 수익을 크게 갉아먹으므로 반드시 참고용으로 표시하고,
  페이퍼 트레이딩 검증 후 소액으로만 사용한다.
"""

import numpy as np
import pandas as pd

from data_loader    import get_price_data
from indicators     import calc_atr, calc_relative_strength
from strategy_utils import (get_trade_costs, get_benchmark_close,
                            get_currency_symbol, safe_atr)


# ── 모멘텀 전략 파라미터 (settings.txt에서 덮어쓸 수 있음) ──
M_NEAR_HIGH    = 0.98   # M1: 20일 신고가의 98% 이상
M_VOL_MULT     = 2.0    # M3: 거래량 20일 평균 대비 배수
M_DAY_GAIN     = 3.0    # M4: 당일 상승률 % 이상
M_RS_THRESHOLD = 5.0    # M5: 상대강도 시장 대비 %p 이상
M_STOP         = -2.5   # 손절 %
M_TARGET       = 5.0    # 익절 %
M_TRAIL        = 3.0    # 트레일링 스탑 %
M_HOLD_DAYS    = 2      # 최대 보유일


def calc_momentum_signal(ticker: str, name: str, market: str, settings: dict) -> dict | None:
    """
    모멘텀 단타 매수 신호를 계산한다.

    크립토·미국주식도 가능하지만 1차 대상은 한국주식(K자 장 대응).
    반환 dict (신호 없으면 충족=False로 반환, None은 데이터 부족):
      {
        "ticker","name","현재가","atr",
        "충족": bool,           # 5개 조건 모두 통과(매수신호)
        "조건": {M1~M5 bool},
        "충족수": int,           # 0~5
        "당일상승률","거래량배수","rs초과","신고가비율",
        "진입추천가","손절가","익절가",
      }
    """
    # settings.txt 값 우선 (없으면 모듈 기본 상수)
    near_high = float(settings.get("M_NEAR_HIGH", M_NEAR_HIGH))
    vol_mult  = float(settings.get("M_VOL_MULT",  M_VOL_MULT))
    day_gain  = float(settings.get("M_DAY_GAIN",  M_DAY_GAIN))
    rs_thr    = float(settings.get("M_RS_THRESHOLD", M_RS_THRESHOLD))
    stop      = float(settings.get("M_STOP",   M_STOP))
    target    = float(settings.get("M_TARGET", M_TARGET))

    # 가격 데이터: 캐시 재사용 (안정봇이 이미 다운로드 → 네트워크 0)
    df = get_price_data(ticker, period="1y")
    if df is None or len(df) < 25:
        return None

    try:
        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze() if "Volume" in df.columns else None
        if volume is None:
            return None

        현재가   = float(close.iloc[-1])
        전일종가 = float(close.iloc[-2])
        if 전일종가 <= 0:
            return None

        # ── M1: 20일 신고가 근접 ──────────────────────────
        고가20 = float(close.iloc[-20:].max())
        신고가비율 = 현재가 / 고가20 if 고가20 > 0 else 0
        M1 = bool(신고가비율 >= near_high)

        # ── M2: 5일선 > 20일선 (단기 정배열) ──────────────
        ma5  = float(close.rolling(5).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        M2 = bool(ma5 > ma20)

        # ── M3: 거래량 폭증 ───────────────────────────────
        평균거래량 = float(volume.rolling(20).mean().iloc[-1])
        오늘거래량 = float(volume.iloc[-1])
        거래량배수 = (오늘거래량 / 평균거래량) if 평균거래량 > 0 else 0
        M3 = bool(거래량배수 >= vol_mult)

        # ── M4: 당일 상승률 ───────────────────────────────
        당일상승률 = (현재가 / 전일종가 - 1) * 100
        M4 = bool(당일상승률 >= day_gain)

        # ── M5: 상대강도 (시장 대비) ──────────────────────
        # 안정봇이 settings에 주입한 벤치마크 종가 재사용 (네트워크 0)
        bench = get_benchmark_close(settings, market)
        rs = calc_relative_strength(close, bench, period=20)  # 단타라 20일 RS
        rs초과 = rs.get("rs_초과수익")
        M5 = bool(rs초과 is not None and rs초과 >= rs_thr)

        조건 = {"M1_신고가": M1, "M2_정배열": M2, "M3_거래량": M3,
                "M4_당일강세": M4, "M5_상대강도": M5}
        충족수 = sum(조건.values())

        # 5개 모두 충족해야 신호 (단타는 엄격하게)
        충족 = bool(충족수 == 5)

        # ATR + 진입가/청산가
        atr = safe_atr(calc_atr(df, period=14), 현재가)

        # 진입 추천가: 돌파 추종이므로 현재가 기준(추격). 살짝 위 지정가.
        진입추천가 = 현재가
        손절가 = 현재가 * (1 + stop / 100)
        익절가 = 현재가 * (1 + target / 100)

        return {
            "ticker": ticker, "name": name, "market": market,
            "현재가": 현재가, "atr": atr,
            "충족": 충족, "조건": 조건, "충족수": 충족수,
            "당일상승률": round(당일상승률, 2),
            "거래량배수": round(거래량배수, 2),
            "rs초과":    round(rs초과, 2) if rs초과 is not None else None,
            "신고가비율": round(신고가비율 * 100, 1),
            "진입추천가": 진입추천가,
            "손절가": 손절가,
            "익절가": 익절가,
        }
    except Exception:
        return None


def backtest_momentum(ticker: str, market: str, settings: dict,
                      period_months: int = 3) -> dict | None:
    """
    모멘텀 전략 백테스트 — 단타 가정(2일 보유, 손절 -2.5%, 익절 +5%).

    안정봇 백테스트와 별개 함수. 가격은 캐시 재사용(네트워크 0).
    최근 period_months(기본 3개월) 동안 매일 진입 조건을 점검해서,
    조건 충족 시 다음날 시가 매수 → 2일 내 청산 규칙으로 수익률 누적.

    반환: {"수익률","mdd","sharpe","승률","거래횟수"} | None
    """
    기간_맵 = {3: "3mo", 6: "6mo", 12: "1y"}
    df = get_price_data(ticker, period=기간_맵.get(period_months, "3mo"))
    if df is None or len(df) < 40:
        return None

    try:
        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze() if "Volume" in df.columns else None
        open_  = df["Open"].squeeze() if "Open" in df.columns else close
        if volume is None:
            return None

        ma5_s   = close.rolling(5).mean()
        ma20_s  = close.rolling(20).mean()
        high20_s = close.rolling(20).max()
        vol20_s = volume.rolling(20).mean()

        # 거래 비용 — 안정봇과 동일 가정 (strategy_utils로 통일, 공정 비교)
        매수비용, 매도비용 = get_trade_costs(settings, market)

        rets = []
        i = 25
        n = len(close)
        while i < n - 1:
            try:
                현재가   = float(close.iloc[i])
                전일종가 = float(close.iloc[i - 1])
                if 전일종가 <= 0:
                    i += 1; continue

                고가20   = float(high20_s.iloc[i])
                신고가ok = 고가20 > 0 and 현재가 >= 고가20 * M_NEAR_HIGH
                정배열ok = float(ma5_s.iloc[i]) > float(ma20_s.iloc[i])
                평균거래 = float(vol20_s.iloc[i])
                거래량ok = 평균거래 > 0 and float(volume.iloc[i]) >= 평균거래 * M_VOL_MULT
                당일강세 = (현재가 / 전일종가 - 1) * 100 >= M_DAY_GAIN

                # RS는 백테스트에서 생략(벤치마크 시점 정렬 복잡) → 4조건으로 근사
                if not (신고가ok and 정배열ok and 거래량ok and 당일강세):
                    i += 1; continue

                # 다음날 시가 매수 (현실적 진입)
                진입가 = float(open_.iloc[i + 1]) * (1 + 매수비용)
                if 진입가 <= 0:
                    i += 1; continue

                # 최대 2일 보유: 손절 -2.5% / 익절 +5% / 트레일링 -3%
                청산가 = None
                고점 = 진입가
                for d in range(1, M_HOLD_DAYS + 1):
                    idx = i + 1 + d
                    if idx >= n:
                        break
                    고 = float(df["High"].squeeze().iloc[idx]) if "High" in df.columns else float(close.iloc[idx])
                    저 = float(df["Low"].squeeze().iloc[idx])  if "Low"  in df.columns else float(close.iloc[idx])
                    종 = float(close.iloc[idx])
                    고점 = max(고점, 고)
                    # 익절
                    if 고 >= 진입가 * (1 + M_TARGET / 100):
                        청산가 = 진입가 * (1 + M_TARGET / 100); break
                    # 손절
                    if 저 <= 진입가 * (1 + M_STOP / 100):
                        청산가 = 진입가 * (1 + M_STOP / 100); break
                    # 트레일링
                    if 종 <= 고점 * (1 - M_TRAIL / 100):
                        청산가 = 종; break
                if 청산가 is None:
                    # 시한 초과 → 마지막 종가 청산
                    청산가 = float(close.iloc[min(i + 1 + M_HOLD_DAYS, n - 1)])

                청산가 *= (1 - 매도비용)
                rets.append(청산가 / 진입가 - 1)
                i += M_HOLD_DAYS + 1   # 청산 후 다음 탐색
            except Exception:
                i += 1

        if len(rets) < 3:
            return None

        rets = np.array(rets)
        cum = float(np.prod(1 + rets))
        수익률 = (cum - 1) * 100
        승률 = float((rets > 0).mean() * 100)
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252 / (M_HOLD_DAYS + 1))) if rets.std() > 0 else 0.0
        # MDD
        equity = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(equity)
        mdd = float(((equity - peak) / peak).min() * 100)

        return {
            "수익률": round(수익률, 1), "mdd": round(mdd, 1),
            "sharpe": round(sharpe, 2), "승률": round(승률, 0),
            "거래횟수": len(rets),
        }
    except Exception:
        return None
