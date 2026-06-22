"""
strategy_utils.py — 안정·모멘텀 두 전략이 공유하는 공용 유틸
==============================================================
[목적]
듀얼 전략(signals.py 안정 + momentum.py 모멘텀)에서 똑같이 쓰던
로직을 한 곳에 모아 중복을 없애고 '두 전략의 가정이 항상 일치'하도록 보장한다.

[중학생 설명]
같은 계산(거래비용, 벤치마크 선택, 통화기호)을 두 파일에 따로 적어두면,
한쪽만 고쳤을 때 두 전략이 서로 다른 기준으로 돌아가 버린다.
실제로 거래비용 기본값이 두 파일에서 달랐다(버그). 여기 하나로 통일한다.

[안정봇 호환]
signals.py(안정 전략)는 기존 동작을 그대로 유지하기 위해 이 모듈을
강제로 쓰지 않는다. momentum.py와 신규 코드가 이 모듈을 사용한다.
거래비용 기본값은 안정봇 기준(commission 0.001 등)으로 통일했다.
"""

import numpy as np


# ── 거래비용 기본값 (안정봇 기준으로 통일) ──
# 기존 momentum.py는 commission 0.00015 등 다른 값을 써서 두 전략의
# 백테스트가 불공정하게 비교됐다 → 안정봇 값으로 일치시킴.
_DEFAULT_COMMISSION = 0.001    # 수수료 0.1%
_DEFAULT_SLIPPAGE   = 0.0005   # 슬리피지 0.05%
_DEFAULT_KR_TAX     = 0.002    # 한국 거래세 0.2% (매도 시)


def get_trade_costs(settings: dict, market: str) -> tuple[float, float]:
    """
    매수/매도 거래비용을 반환한다. (두 전략 공통 가정)

    반환: (매수비용, 매도비용)
      매수비용 = 수수료 + 슬리피지
      매도비용 = 수수료 + 슬리피지 + 거래세(한국주식만)

    [중학생 설명]
    주식을 사고팔 때 수수료·세금이 빠진다. 백테스트가 이걸 무시하면
    "서류상 수익"이 실제보다 부풀려진다. 두 전략이 같은 비용을 쓰게 한다.
    """
    commission = float(settings.get("COMMISSION", _DEFAULT_COMMISSION))
    slippage   = float(settings.get("SLIPPAGE",   _DEFAULT_SLIPPAGE))
    거래세     = _DEFAULT_KR_TAX if market == "KR" else 0.0
    매수비용 = commission + slippage
    매도비용 = commission + slippage + 거래세
    return 매수비용, 매도비용


def get_benchmark_close(settings: dict, market: str):
    """
    상대강도(RS) 계산용 벤치마크 종가 시리즈를 반환한다.

    KR → KOSPI(BENCH_KR_CLOSE), US → S&P500(BENCH_US_CLOSE),
    크립토 등 → None (벤치마크 없음, RS 미적용)

    scheduler_job이 실행 시작 시 settings에 1회 주입한 값을 꺼내 쓴다.
    (종목마다 지수를 다시 다운로드하지 않음 → 네트워크 0)
    """
    if market == "US":
        return settings.get("BENCH_US_CLOSE")
    if market == "KR":
        return settings.get("BENCH_KR_CLOSE")
    return None


def get_currency_symbol(market: str) -> str:
    """시장에 맞는 통화 기호를 반환한다. (KR/크립토원화=₩, 그 외=$)"""
    return "₩" if market in ("KR", "CRYPTO_KRW") else "$"


def safe_atr(atr, 현재가: float, ratio: float = 0.02) -> float:
    """
    ATR 값이 비정상(None·NaN·0)이면 현재가의 ratio(기본 2%)로 대체한다.

    [중학생 설명]
    ATR(평균 변동폭)은 손절·포지션 크기 계산에 쓰는데, 데이터가 부족하면
    NaN이 나올 수 있다. 그럴 때 현재가의 2%를 임시 변동폭으로 써서
    계산이 멈추지 않게 한다.
    """
    try:
        v = float(atr)
        if np.isnan(v) or v <= 0:
            return 현재가 * ratio
        return v
    except (TypeError, ValueError):
        return 현재가 * ratio
