"""
portfolio.py — 포트폴리오 손익 알림 (강화판)
=============================================
[손절 기준 강화]
  기존: 평단가 기준 고정 손절 (STOP_LOSS%)만 존재
  추가: ATR 추적 손절 (Trailing Stop)
        → 고점 대비 ATR * 2.0 하락 시 손절 알림
        → 이미 수익권인 종목이 다시 하락할 때 보호

[알림 조건 - 3단계]
  1. 익절 알림:       현재가 >= 평단가 * (1 + TARGET_PROFIT%)
  2. 고정 손절 알림:  현재가 <= 평단가 * (1 + STOP_LOSS%)
  3. ATR 추적 손절:   최근 20일 고점 대비 현재가 하락폭 > ATR * 2.0
  4. ATR 변동성 경고: 당일 변동폭 > ATR * 2.0 (비정상 급등락)
"""

from data_loader import get_price_data
from indicators  import calc_atr


def check_portfolio_alerts(포트폴리오, settings):
    """
    보유 종목의 현재 손익을 확인하고, 조건 도달 시 알림 문자열을 반환한다.
    """
    알림_목록 = []

    TARGET = settings.get("TARGET_PROFIT", 25)
    STOP   = settings.get("STOP_LOSS",    -7)

    for 종목 in 포트폴리오:
        ticker   = 종목["ticker"]
        name     = 종목.get("name", ticker)  # 종목명 (없으면 티커로 대체)
        보유수량 = 종목["quantity"]
        평단가   = 종목["avg_price"]
        통화     = 종목["currency"]

        df = get_price_data(ticker, period="3mo")  # 추적 손절을 위해 3mo로 확장
        if df is None or len(df) < 5:
            continue

        현재가    = df["Close"].squeeze().iloc[-1]
        수익률    = ((현재가 - 평단가) / 평단가) * 100
        목표가    = 평단가 * (1 + TARGET / 100)
        손절가    = 평단가 * (1 + STOP / 100)
        통화_기호 = "₩" if 통화 == "KRW" else "$"
        평가액    = 현재가 * 보유수량

        # 알림은 익절→손절→기타 순서로 1개만(중복 알림 방지)

        # ── 익절 알림 ───────────────────────────────────
        if 현재가 >= 목표가:
            알림_목록.append(
                f"🎯 *{name}({ticker}) 익절 목표 달성!*\n"
                f"  현재가: {통화_기호}{현재가:,.0f} | 수익률: +{수익률:.1f}%\n"
                f"  평가액: {통화_기호}{평가액:,.0f} ({보유수량}주/개)\n"
                f"  목표가: {통화_기호}{목표가:,.0f} (+{TARGET:.0f}%) | "
                f"손절가: {통화_기호}{손절가:,.0f} ({STOP:.0f}%)"
            )

        # ── 고정 손절 알림 ──────────────────────────────
        elif 현재가 <= 손절가:
            알림_목록.append(
                f"🚨 *{name}({ticker}) 손절 기준 도달!*\n"
                f"  현재가: {통화_기호}{현재가:,.0f} | 수익률: {수익률:.1f}%\n"
                f"  평가액: {통화_기호}{평가액:,.0f} ({보유수량}주/개)\n"
                f"  목표가: {통화_기호}{목표가:,.0f} (+{TARGET:.0f}%) | "
                f"손절가: {통화_기호}{손절가:,.0f} ({STOP:.0f}%)"
            )

        else:
            atr         = calc_atr(df)
            일중_변동폭 = abs(df["High"].squeeze().iloc[-1] - df["Low"].squeeze().iloc[-1])

            # ── ATR 추적 손절 (Trailing Stop) ──────────
            # 최근 20일 고점 기준으로 ATR * 2.0 하락 시 알림
            # 수익권(+3% 이상)에서만 의미 있으므로 수익 조건 추가
            if atr and atr > 0 and 수익률 >= 3.0:
                # ATR은 '평소 흔들림 폭'이라서, 고점 대비 ATR×2 이상 빠지면 경고
                최근_고점      = df["High"].squeeze().tail(20).max()
                고점_대비_낙폭 = 최근_고점 - 현재가
                추적_손절_기준 = atr * 2.0
                if 고점_대비_낙폭 > 추적_손절_기준:
                    알림_목록.append(
                        f"📉 *{name}({ticker}) 추적 손절 알림!*\n"
                        f"  현재가: {통화_기호}{현재가:,.0f} | 수익률: {수익률:+.1f}%\n"
                        f"  최근 고점: {통화_기호}{최근_고점:,.0f}\n"
                        f"  고점 대비 낙폭: {통화_기호}{고점_대비_낙폭:,.0f} "
                        f"(ATR×2 기준: {통화_기호}{추적_손절_기준:,.0f})\n"
                        f"  ※ 고점 대비 ATR 2배 이상 하락 — 부분 매도 고려"
                    )

            # ── ATR 변동성 경고 (비정상 급등락) ─────────
            elif atr and 일중_변동폭 > atr * 2.0:
                # 하루 변동폭이 ATR×2보다 크면 비정상 흔들림으로 경고
                알림_목록.append(
                    f"⚡ *{name}({ticker}) 비정상 급등락 감지!*\n"
                    f"  현재가: {통화_기호}{현재가:,.0f} | 수익률: {수익률:+.1f}%\n"
                    f"  당일 변동폭: {통화_기호}{일중_변동폭:,.0f} "
                    f"(ATR의 {일중_변동폭/atr:.1f}배)"
                )

    return 알림_목록


def check_max_holding_days(settings, 통화_기호_맵=None):
    """
    portfolio_signals.txt '보유중' 종목에 대해 두 가지를 동시에 체크한다.

    [체크 1] 트레일링 스탑 실전 알림
    백테스트에만 있던 트레일링 스탑을 실전에서도 알려준다.
    진입 후 최고가(High 기준)를 조회해서 현재가가 고점 대비
    TRAILING_STOP% 이상 빠지면 즉시 알림.

    예) 진입가 10만원 → 고점 13만원 → 현재 11만 8천원
        (13만 - 11만8천) / 13만 = 9.2% 하락 → 8% 기준 초과 → 알림!

    [체크 2] 최대 보유일 초과 알림
    MAX_HOLDING_DAYS(기본 30일) 넘으면 청산 검토 권고.
    """
    from datetime import date as _date
    import os

    MAX_DAYS      = int(settings.get("MAX_HOLDING_DAYS", 30))
    TRAILING_STOP = float(settings.get("TRAILING_STOP", 8))
    알림_목록 = []

    # 파일 경로: GitHub Actions는 레포 루트에서 실행
    sig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "portfolio_signals.txt")
    if not os.path.exists(sig_path):
        sig_path = "portfolio_signals.txt"  # 폴백
    if not os.path.exists(sig_path):
        return 알림_목록

    try:
        from data_loader import get_price_data
        오늘 = _date.today()
        with open(sig_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue

            날짜_str, ticker, name, 점수, 진입가_str, 통화, _, 상태 = parts[:8]
            if 상태 != "보유중":
                continue

            try:
                진입일 = _date.fromisoformat(날짜_str)
                보유일 = (오늘 - 진입일).days
                진입가 = float(진입가_str)
            except Exception:
                continue

            기호 = "₩" if 통화 == "KRW" else "$"

            # 현재가 + 고가 조회 (진입일 이후 전체 기간)
            현재가 = None
            고점   = None
            try:
                # 진입 이후 기간 계산
                since_days = max(보유일 + 5, 10)
                period = f"{min(since_days, 365)}d"
                df = get_price_data(ticker, period=period)
                if df is not None and len(df) >= 1:
                    현재가 = float(df["Close"].squeeze().iloc[-1])
                    # 진입일 이후의 최고가 (High 컬럼)
                    if "High" in df.columns:
                        고점 = float(df["High"].squeeze().max())
                    else:
                        고점 = float(df["Close"].squeeze().max())
            except Exception:
                pass

            현재가_str = f"{기호}{현재가:,.0f}" if 현재가 else "조회실패"
            수익률_str = ""
            if 현재가 and 진입가 > 0:
                수익률 = (현재가 - 진입가) / 진입가 * 100
                수익률_str = f" | 수익률: {수익률:+.1f}%"

            # ── 체크 1: 트레일링 스탑 ─────────────────────────
            if 현재가 and 고점 and 고점 > 0:
                낙폭률 = (고점 - 현재가) / 고점 * 100
                if 낙폭률 >= TRAILING_STOP:
                    알림_목록.append(
                        f"🔻 *트레일링 스탑* {name}({ticker})\n"
                        f"  고점: {기호}{고점:,.0f} → 현재: {현재가_str}"
                        f"{수익률_str}\n"
                        f"  고점 대비 -{낙폭률:.1f}% 하락 "
                        f"(기준 -{TRAILING_STOP:.0f}%) — 즉시 청산 검토"
                    )

            # ── 체크 2: 최대 보유일 ───────────────────────────
            if 보유일 >= MAX_DAYS:
                알림_목록.append(
                    f"⏰ *최대 보유일 초과* {name}({ticker}) {보유일}일\n"
                    f"  진입가: {기호}{진입가:,.0f} → 현재: {현재가_str}"
                    f"{수익률_str}\n"
                    f"  {MAX_DAYS}일 초과 — 청산 또는 보유 연장 결정 필요"
                )

    except Exception as e:
        print(f"⚠️ 보유일/트레일링 확인 오류: {e}")

    return 알림_목록


def generate_weekly_report(settings):
    """
    매주 일요일: 지난 7일간 신호 발생 종목의 실제 수익률을 집계한다.

    [중학생 설명]
    "지난 주에 이 시스템이 추천한 종목들이 실제로 올랐는가?"를
    숫자로 확인한다. 이게 쌓이면 이 시스템이 얼마나 믿을 수 있는지 알 수 있다.

    출력 형식:
    📊 주간 신호 성과 리포트 (2026-05-05 ~ 2026-05-12)
    총 7개 신호 | 수익 4개(57%) | 손실 3개(43%)
    평균 수익률: +3.2% | 최고: +8.1%(AMD) | 최저: -2.4%(INTC)
    """
    from datetime import date as _date, timedelta
    import os

    알림_목록 = []
    오늘 = _date.today()

    # 일요일에만 실행 (weekday 6 = 일요일)
    if 오늘.weekday() != 6:
        return 알림_목록

    sig_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "portfolio_signals.txt")
    if not os.path.exists(sig_path2):
        sig_path2 = "portfolio_signals.txt"
    if not os.path.exists(sig_path2):
        return 알림_목록

    일주일전 = 오늘 - timedelta(days=7)
    결과들 = []

    try:
        with open(sig_path2, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue

            날짜_str, ticker, name, 점수, 진입가_str, 통화, _, 상태 = parts[:8]

            try:
                신호일 = _date.fromisoformat(날짜_str)
                진입가 = float(진입가_str)
            except Exception:
                continue

            # 지난 7일 신호만
            if not (일주일전 <= 신호일 <= 오늘):
                continue

            # 현재가 조회
            from data_loader import get_price_data
            df = get_price_data(ticker, period="5d")
            if df is None or len(df) < 1:
                continue

            현재가 = float(df["Close"].squeeze().iloc[-1])
            수익률 = (현재가 - 진입가) / 진입가 * 100
            # ⑩ 수정: 상태 구분 저장 (신호발생=시스템정확도, 보유중=실제손익)
            결과들.append((ticker, name, 수익률, 신호일, 상태))

        if not 결과들:
            return 알림_목록

        # ⑩ 수정: 신호발생(시스템 정확도)과 보유중(실제 손익) 분리 집계
        전체_결과 = 결과들
        실매수_결과 = [r for r in 결과들 if r[4] in ("보유중", "익절", "손절", "기간초과청산")]
        집계_대상 = 실매수_결과 if 실매수_결과 else 전체_결과
        집계_라벨 = "실매수 손익" if 실매수_결과 else "신호 정확도(미매수 포함)"

        수익_건 = [r for r in 집계_대상 if r[2] > 0]
        손실_건 = [r for r in 집계_대상 if r[2] <= 0]
        avg = sum(r[2] for r in 집계_대상) / len(집계_대상)
        최고 = max(집계_대상, key=lambda x: x[2])
        최저 = min(집계_대상, key=lambda x: x[2])
        승률 = len(수익_건) / len(집계_대상) * 100

        리포트 = (
            f"📊 *주간 신호 성과 리포트* ({집계_라벨})\n"
            f"기간: {일주일전} ~ {오늘}\n"
            f"총 {len(집계_대상)}개 | "
            f"수익 {len(수익_건)}개({승률:.0f}%) | "
            f"손실 {len(손실_건)}개({100-승률:.0f}%)\n"
            f"평균 수익률: {avg:+.1f}%\n"
            f"최고: {최고[2]:+.1f}% ({최고[1]}) | "
            f"최저: {최저[2]:+.1f}% ({최저[1]})\n"
        )

        리포트 += "\n상세:\n"
        for ticker, name, ret, dt, st in sorted(집계_대상, key=lambda x: x[2], reverse=True):
            아이콘 = "🟢" if ret > 0 else "🔴"
            리포트 += f"  {아이콘} {name}({ticker}): {ret:+.1f}% ({st}, 신호일 {dt})\n"

        알림_목록.append(리포트)

    except Exception as e:
        print(f"⚠️ 주간 리포트 생성 오류: {e}")

    return 알림_목록
