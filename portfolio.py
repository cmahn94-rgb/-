"""
portfolio.py — 포트폴리오 손익 알림
=====================================
이 파일이 하는 일:
  portfolio.txt의 보유 종목을 실시간 가격과 비교하여
  익절·손절·비정상 급등락 발생 시 알림 메시지를 생성한다.
"""

from data_loader import get_price_data
from indicators  import calc_atr


def check_portfolio_alerts(포트폴리오, settings):
    """
    보유 종목의 현재 손익을 확인하고, 조건 도달 시 알림 문자열을 반환한다.

    [알림 조건]
    · 익절 알림: 현재가 >= 평단가 * (1 + TARGET_PROFIT/100)
    · 손절 알림: 현재가 <= 평단가 * (1 + STOP_LOSS/100)
    · ATR 변동성 경고: 당일 변동폭 > ATR * 2.0 (비정상 급등락 감지)

    매도 목표가와 손절가는 '%' 표시뿐 아니라 '절대 가격'으로도 함께 출력한다.
    """
    알림_목록 = []

    # settings.txt에서 목표 수익률과 손절 기준 읽기
    TARGET = settings.get("TARGET_PROFIT", 25)   # 기본 25%
    STOP   = settings.get("STOP_LOSS",    -7)    # 기본 -7%

    for 종목 in 포트폴리오:
        ticker    = 종목["ticker"]
        보유수량  = 종목["quantity"]
        평단가    = 종목["avg_price"]
        통화      = 종목["currency"]

        df = get_price_data(ticker, period="5d")
        if df is None:
            continue

        현재가 = df["Close"].squeeze().iloc[-1]

        # 수익률 = (현재가 - 평단가) / 평단가 * 100
        수익률 = ((현재가 - 평단가) / 평단가) * 100

        # 목표가: 평단가에서 TARGET%만큼 오른 가격
        목표가 = 평단가 * (1 + TARGET / 100)
        # 손절가: 평단가에서 STOP%만큼 내린 가격
        손절가 = 평단가 * (1 + STOP / 100)

        # 통화 기호 결정
        통화_기호 = "₩" if 통화 == "KRW" else "$"

        # ── 익절 알림 ───────────────────────────────────
        if 현재가 >= 목표가:
            알림_목록.append(
                f"🎯 *{ticker} 익절 목표 달성!*\n"
                f"  현재가: {통화_기호}{현재가:,.0f} | 수익률: +{수익률:.1f}%\n"
                f"  목표가: {통화_기호}{목표가:,.0f} (+{TARGET:.0f}%) | "
                f"손절가: {통화_기호}{손절가:,.0f} ({STOP:.0f}%)"
            )

        # ── 손절 알림 ───────────────────────────────────
        elif 현재가 <= 손절가:
            알림_목록.append(
                f"🚨 *{ticker} 손절 기준 도달!*\n"
                f"  현재가: {통화_기호}{현재가:,.0f} | 수익률: {수익률:.1f}%\n"
                f"  목표가: {통화_기호}{목표가:,.0f} (+{TARGET:.0f}%) | "
                f"손절가: {통화_기호}{손절가:,.0f} ({STOP:.0f}%)"
            )

        # ── ATR 변동성 경고 ─────────────────────────────
        # 당일 변동폭이 ATR의 2배 이상 = 비정상적인 급등락
        else:
            atr         = calc_atr(df)
            일중_변동폭 = abs(df["High"].squeeze().iloc[-1] - df["Low"].squeeze().iloc[-1])
            if atr and 일중_변동폭 > atr * 2.0:
                알림_목록.append(
                    f"⚡ *{ticker} 비정상 급등락 감지!*\n"
                    f"  당일 변동폭: {통화_기호}{일중_변동폭:,.0f} "
                    f"(ATR의 {일중_변동폭/atr:.1f}배)"
                )

    return 알림_목록
