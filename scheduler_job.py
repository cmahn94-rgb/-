"""
scheduler_job.py — 리포트 조립 및 전체 분석 실행 (병렬 처리 적용)
===================================================================
이 파일이 하는 일:
  1) 모든 종목을 병렬로 동시에 분석 → 100종목도 빠르게 처리
  2) run_analysis()가 main.py의 스케줄러에 의해 하루 3회 호출된다.
  3) 주말·휴장일 처리: 주식은 건너뛰고, 암호화폐는 항상 실행한다.

[속도 개선 원리 - 3단계]
  1단계) 일괄 다운로드: 50종목씩 묶어 한 번에 다운로드 (300번 → 6번 요청)
  2단계) 병렬 분석:     ThreadPoolExecutor로 20개 동시 처리
  3단계) 지연 백테스트: 매수 신호 있는 종목에만 백테스트 실행 (낭비 제거)

[리포트 출력 순서]
  1. 레짐 경고 (하락장 해당 시에만)
  2. 매수 신호 종목 (없으면 '충족 종목 없음' 출력)
  3. 백테스트 성적 (각 신호 종목 하단에 인라인 표시)
  4. 포트폴리오 손익 현황
  5. 급등락 종목 뉴스
  6. 암호화폐 분석 (CRYPTO 종목이 있을 때만)
"""

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config         import load_settings
from stocks_loader  import load_stocks, load_portfolio
from signals        import get_market_regime, calc_signals, run_backtest, calc_position_size
from data_loader    import get_price_data, get_news_summary, clear_cache, bulk_download
from portfolio      import check_portfolio_alerts
from telegram_bot   import send_telegram

# 동시에 분석할 최대 종목 수
# 너무 크면 야후 파이낸스 서버에서 요청을 차단할 수 있으므로 20으로 제한
MAX_WORKERS = 20


# ─────────────────────────────────────────
# 휴장일 판단
# ─────────────────────────────────────────

def is_market_open():
    """
    오늘이 주식 거래일인지 확인한다.
    주말(토·일)은 주식 시장이 닫혀 있으므로 건너뛴다.
    암호화폐는 24시간 365일 거래되므로 항상 True 반환 → 별도 처리.
    """
    오늘 = datetime.now().weekday()
    # 0=월, 1=화, 2=수, 3=목, 4=금, 5=토, 6=일
    if 오늘 >= 5:
        print("📅 오늘은 주말 휴장입니다. 암호화폐만 분석합니다.")
        return False
    return True


# ─────────────────────────────────────────
# 단일 종목 분석 (병렬 실행 단위)
# ─────────────────────────────────────────

def analyze_one(종목, settings):
    """
    종목 1개를 분석하고 결과 딕셔너리를 반환한다.
    이 함수가 ThreadPoolExecutor에 의해 여러 종목에서 동시에 실행된다.
    """
    ticker = 종목["ticker"]
    name   = 종목["name"]
    market = 종목["market"]

    결과 = calc_signals(ticker, name, market, settings)
    if 결과 is None:
        return None

    현재가    = 결과["현재가"]
    atr       = 결과["atr"]
    # 통화 기호 및 총자산 결정
    # KR / CRYPTO_KRW → 원화(₩), 업비트 포함 KRW 기준 자산
    # US / CRYPTO     → 달러($)
    if market in ("KR", "CRYPTO_KRW"):
        통화_기호 = "₩"
        총자산    = 30_000_000   # 원화 기준 총자산 (설정 가능)
    else:
        통화_기호 = "$"
        총자산    = 20_000       # 달러 기준 총자산 (설정 가능)
    추천_수량 = calc_position_size(총자산, atr, market)

    TARGET = settings.get("TARGET_PROFIT", 25)
    STOP   = settings.get("STOP_LOSS",    -7)
    목표가 = 현재가 * (1 + TARGET / 100)
    손절가 = 현재가 * (1 + STOP   / 100)

    # 백테스트: 매수 신호 있는 종목에만 실행 (신호 없으면 건너뜀 → 속도 절약)
    # 신호 없는 종목의 백테스트는 어차피 리포트에 출력되지 않으므로 낭비다.
    bt_문구 = ""
    if 결과["매수신호"]:
        bt3 = run_backtest(ticker, market, settings, period_months=3)
        if bt3:
            bt_문구 = (
                f"  • 📊 백테스트(3개월, 비용 차감): {bt3['수익률']:+.1f}% | "
                f"MDD: {bt3['mdd']:.1f}% | "
                f"Sharpe: {bt3['sharpe']:.2f} | "
                f"승률: {bt3['승률']:.0f}%\n"
            )

    # 뉴스
    df_5d = get_price_data(ticker, period="5d")
    뉴스_목록, 변동률 = None, 0.0
    if df_5d is not None and len(df_5d) >= 2:
        전일_종가 = df_5d["Close"].squeeze().iloc[-2]
        뉴스_목록, 변동률 = get_news_summary(ticker, 현재가, 전일_종가)

    return {
        "결과":      결과,
        "현재가":    현재가,
        "atr":       atr,
        "통화_기호": 통화_기호,
        "추천_수량": 추천_수량,
        "TARGET":    TARGET,
        "STOP":      STOP,
        "목표가":    목표가,
        "손절가":    손절가,
        "bt_문구":   bt_문구,
        "뉴스_목록": 뉴스_목록,
        "변동률":    변동률,
        "market":    market,
        "name":      name,
        "ticker":    ticker,
    }


# ─────────────────────────────────────────
# 종목별 신호 블록 생성 (병렬 처리)
# ─────────────────────────────────────────

def build_report_sections(종목목록, settings, 크립토_하락_레짐):
    """
    모든 종목을 최대 20개씩 동시에 분석하고
    매수 신호 섹션, 크립토 섹션, 뉴스 섹션을 반환한다.

    ThreadPoolExecutor란?
      여러 함수를 동시에 실행하는 파이썬 내장 도구.
      with 블록이 끝나면 모든 작업이 완료될 때까지 자동으로 기다린다.
    """
    매수신호_섹션 = ""
    크립토_섹션   = ""
    뉴스_섹션     = ""

    # 결과를 원래 종목 순서대로 유지하기 위한 딕셔너리
    # (병렬 실행은 완료 순서가 다를 수 있으므로 ticker로 매핑)
    결과_맵 = {}

    print(f"  ⚡ {len(종목목록)}개 종목 병렬 분석 시작 (동시 {MAX_WORKERS}개)...")

    # ThreadPoolExecutor: 최대 MAX_WORKERS개 종목을 동시에 실행
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 각 종목에 대해 analyze_one 함수를 비동기로 제출
        future_map = {
            executor.submit(analyze_one, 종목, settings): 종목["ticker"]
            for 종목 in 종목목록
        }

        # 완료된 작업부터 순서대로 결과 수집
        완료수 = 0
        for future in as_completed(future_map):
            ticker = future_map[future]
            완료수 += 1
            try:
                분석결과 = future.result()
                결과_맵[ticker] = 분석결과
                print(f"  ✅ ({완료수}/{len(종목목록)}) {ticker} 완료")
            except Exception as e:
                print(f"  ⚠️ {ticker} 분석 실패: {e}")
                결과_맵[ticker] = None

    # 원래 종목 순서대로 리포트 블록 조립
    def 조건표시(v): return "✅" if v else "❌"

    for 종목 in 종목목록:
        ticker = 종목["ticker"]
        d = 결과_맵.get(ticker)
        if d is None or not d["결과"]["매수신호"]:
            continue

        결과      = d["결과"]
        통화_기호 = d["통화_기호"]
        추천_수량 = d["추천_수량"]
        atr       = d["atr"]
        market    = d["market"]
        name      = d["name"]
        TARGET    = d["TARGET"]
        STOP      = d["STOP"]
        목표가    = d["목표가"]
        손절가    = d["손절가"]
        bt_문구   = d["bt_문구"]
        뉴스_목록 = d["뉴스_목록"]
        변동률    = d["변동률"]
        현재가    = d["현재가"]

        # 수량 표시: 암호화폐는 소수점, 주식은 정수
        수량_표시 = f"{추천_수량:.6f} BTC" if market in ("CRYPTO", "CRYPTO_KRW") else f"{추천_수량}주"
        atr_표시  = f"{통화_기호}{atr:,.0f}"

        신호_블록 = (
            f"📈 *매수 신호*: {name} ({ticker})\n"
            f"  • RSI: {결과['rsi']:.1f} {조건표시(결과['조건1_rsi'])} | "
            f"MA20 {조건표시(결과['조건2_ma'])} | "
            f"거래량폭발 {조건표시(결과['조건3_거래량'])} | "
            f"볼린저중심선 {조건표시(결과['조건4_볼린저'])} | "
            f"MACD {조건표시(결과['조건5_macd'])}\n"
            f"  • 📐 추천 수량: {수량_표시} (ATR={atr_표시}, 리스크 1%)\n"
            f"  • 🎯 목표가: {통화_기호}{목표가:,.0f} (+{TARGET:.0f}%) | "
            f"손절가: {통화_기호}{손절가:,.0f} ({STOP:.0f}%)\n"
        )
        신호_블록 += bt_문구

        # CRYPTO / CRYPTO_KRW 모두 암호화폐 섹션으로 분류
        if market in ("CRYPTO", "CRYPTO_KRW"):
            if 크립토_하락_레짐:
                신호_블록 += "  ⚠️ *BTC 하락 레짐 감지*: 소액·분할 매수만 고려하세요.\n"
            크립토_섹션 += 신호_블록
        else:
            매수신호_섹션 += 신호_블록 + "─────────────────────────\n"

        if 뉴스_목록 and abs(변동률) >= 2.0:
            뉴스_섹션 += f"📰 *{name} 급등락 뉴스* (변동 {변동률:+.1f}%)\n"
            for 뉴스 in 뉴스_목록:
                뉴스_섹션 += f"  • \"{뉴스}\"\n"

    return 매수신호_섹션, 크립토_섹션, 뉴스_섹션


# ─────────────────────────────────────────
# 전체 분석 실행
# ─────────────────────────────────────────

def run_analysis(include_crypto=True):
    """
    전체 분석을 실행하고 텔레그램으로 리포트를 전송한다.
    main.py의 스케줄러가 하루 3회 이 함수를 호출한다.
    """
    지금 = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n🔍 분석 시작: {지금}")

    # 매 실행 시 캐시 초기화 (오래된 데이터 방지)
    clear_cache()

    settings   = load_settings()
    종목목록   = load_stocks()
    포트폴리오 = load_portfolio()

    if not include_crypto:
        종목목록 = [s for s in 종목목록 if s["market"] not in ("CRYPTO", "CRYPTO_KRW")]

    # ── [속도 개선 핵심] 일괄 사전 다운로드 ──────────────────
    # 분석에 필요한 모든 데이터를 미리 한꺼번에 받아 캐시에 저장한다.
    # 이후 calc_signals/run_backtest/뉴스 호출은 모두 캐시에서 즉시 반환.
    모든_티커 = [s["ticker"] for s in 종목목록]
    # 레짐 판단용 지수들도 미리 받아둠
    레짐_티커 = ["^GSPC", "^KS11", "^VIX", "BTC-USD"]
    전체_티커 = list(set(모든_티커 + 레짐_티커))

    print(f"  🚀 사전 일괄 다운로드 시작 (총 {len(전체_티커)}개 티커)...")
    bulk_download(전체_티커, period="1y")   # 신호 계산 + 레짐 판단용
    bulk_download(전체_티커, period="3mo")  # 백테스트용
    bulk_download(전체_티커, period="5d")   # 뉴스 변동률용
    print(f"  ✅ 사전 다운로드 완료 — 이후 분석은 캐시에서 즉시 실행")

    하락_레짐, 크립토_하락_레짐, 경고_목록 = get_market_regime()

    리포트  = f"⚔️ *퀀트 헤지펀드 리포트* | {지금}\n"
    리포트 += "─────────────────────────\n"

    if 하락_레짐:
        경고_문구 = " + ".join(경고_목록) if 경고_목록 else "복합 조건"
        리포트 += (
            "⚠️ *대세 하락장 경고*: 현금 비중 확대 권고. "
            "매수 신호가 발생해도 소액·분할 매수만 고려하세요.\n"
        )
        리포트 += f"(트리거: {경고_문구})\n"
        리포트 += "─────────────────────────\n"

    매수신호_섹션, 크립토_섹션, 뉴스_섹션 = build_report_sections(
        종목목록, settings, 크립토_하락_레짐
    )

    if 매수신호_섹션:
        리포트 += 매수신호_섹션
    else:
      # 시장 상황 요약 추가
        if 하락_레짐:
            시장_요약 = "⚠️ 대세 하락장입니다. 현금 비중을 높이고 관망하는 것을 추천합니다."
        elif 크립토_하락_레짐:
            시장_요약 = "₿ 크립토 시장이 하락 레짐입니다. 주식은 중립이나 소극적 매수가 적합합니다."
        else:
            시장_요약 = "현재 시장은 매수 신호가 나오지 않을 정도로 조용하거나 조건이 엄격한 상태입니다."

        리포트 += f"현재 매수 조건 충족 종목 없음 (주식/ETF)\n"
        리포트 += f"   {시장_요약}\n"
        리포트 += "─────────────────────────\n"

    포트폴리오_알림 = check_portfolio_alerts(포트폴리오, settings)
    if 포트폴리오_알림:
        리포트 += "💼 *포트폴리오 알림*\n"
        for 알림 in 포트폴리오_알림:
            리포트 += 알림 + "\n"
        리포트 += "─────────────────────────\n"

    if 뉴스_섹션:
        리포트 += 뉴스_섹션
        리포트 += "─────────────────────────\n"

    if include_crypto and 크립토_섹션:
        리포트 += "₿ *암호화폐 분석*\n"
        리포트 += 크립토_섹션
        리포트 += "─────────────────────────\n"

    send_telegram(리포트)
    print("✅ 분석 완료 및 리포트 전송")
