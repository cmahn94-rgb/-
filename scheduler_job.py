"""
scheduler_job.py — 리포트 조립 및 전체 분석 실행 (병렬 처리)
=============================================================
[변경사항]
  - 매수 신호 블록에 점수(N/6) 표시 추가
  - 5점 이상 강력 매수 🔥 표시
  - AI 시장 코멘트 섹션 추가 (ANTHROPIC_API_KEY 있을 때만)
  - AI 종목별 신호 이유 한 줄 추가
"""

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config        import load_settings
from stocks_loader import load_stocks, load_portfolio
from signals       import get_market_regime, calc_signals, run_backtest, calc_position_size
from data_loader   import get_price_data, get_news_summary, clear_cache, bulk_download
from portfolio     import check_portfolio_alerts
from telegram_bot  import send_telegram
from ai_analyst    import get_ai_market_commentary, get_ai_signal_reason

MAX_WORKERS = 20


def is_market_open():
    """
    (아주 단순 버전) 주식 시장이 열려 있는 날인지 확인한다.
    - 월~금: 열려 있다고 가정
    - 토/일: 휴장으로 보고, 주식 대신 암호화폐만 분석하도록 안내
    """
    오늘 = datetime.now().weekday()
    if 오늘 >= 5:
        print("📅 오늘은 주말 휴장입니다. 암호화폐만 분석합니다.")
        return False
    return True


def analyze_one(종목, settings):
    """
    종목 1개를 분석해서, 리포트에 필요한 값들을 한 번에 묶어서 돌려준다.

    왜 이렇게 하냐면:
    - 종목이 많을 때 `ThreadPoolExecutor`로 여러 종목을 동시에 분석하려고(=병렬 처리)
    - 리포트에서 쓰는 값(현재가/목표가/뉴스/백테스트 문구 등)을 여기서 미리 계산해두려고
    """
    ticker = 종목["ticker"]
    name   = 종목["name"]
    market = 종목["market"]

    결과 = calc_signals(ticker, name, market, settings)
    if 결과 is None:
        return None

    현재가 = 결과["현재가"]
    atr    = 결과["atr"]

    # 시장별로 "자산 규모(가정값)"와 "통화 기호"를 다르게 잡는다.
    # (초보자용 설명) 추천 수량을 계산하려면, '총자산이 얼마인지' 같은 기준이 필요해서
    # 여기서는 간단히 KR/US를 나눠 기본 값을 넣어둔 것이다.
    if market in ("KR", "CRYPTO_KRW"):
        통화_기호 = "₩"
        총자산    = 30_000_000
    else:
        통화_기호 = "$"
        총자산    = 20_000
    추천_수량 = calc_position_size(총자산, atr, market)

    # 목표가/손절가는 settings.txt의 % 값을 이용해 "현재가 기준으로" 계산한다.
    TARGET = settings.get("TARGET_PROFIT", 25)
    STOP   = settings.get("STOP_LOSS",    -7)
    목표가 = 현재가 * (1 + TARGET / 100)
    손절가 = 현재가 * (1 + STOP   / 100)

    bt_문구 = ""
    if 결과["매수신호"]:
        # 매수 신호가 뜬 종목만 백테스트를 돌린다(시간 절약).
        bt3 = run_backtest(ticker, market, settings, period_months=3)
        if bt3:
            bt_문구 = (
                f"  • 📊 백테스트(3개월, 비용 차감): {bt3['수익률']:+.1f}% | "
                f"MDD: {bt3['mdd']:.1f}% | "
                f"Sharpe: {bt3['sharpe']:.2f} | "
                f"승률: {bt3['승률']:.0f}%\n"
            )

    df_5d = get_price_data(ticker, period="5d")
    뉴스_목록, 변동률 = None, 0.0
    if df_5d is not None and len(df_5d) >= 2:
        # 최근 5일 데이터로 "전일 종가"를 구해서, 급등락 뉴스만 간단히 뽑는다.
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


def build_report_sections(종목목록, settings, 크립토_하락_레짐):
    매수신호_섹션 = ""
    크립토_섹션   = ""
    뉴스_섹션     = ""
    신호_종목_요약 = []   # AI 코멘트용

    결과_맵 = {}
    print(f"  ⚡ {len(종목목록)}개 종목 병렬 분석 시작 (동시 {MAX_WORKERS}개)...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(analyze_one, 종목, settings): 종목["ticker"]
            for 종목 in 종목목록
        }
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
        점수      = 결과["점수"]
        강력매수  = 결과["강력매수"]

        # AI 코멘트용 요약 수집
        신호_종목_요약.append({
            "name":   name,
            "ticker": ticker,
            "점수":   점수,
            "rsi":    결과["rsi"],
        })

        # AI 종목별 신호 이유 한 줄
        ai_이유 = get_ai_signal_reason(ticker, name, {
            "rsi":          결과["rsi"],
            "점수":          점수,
            "ma_정배열":     결과["조건2_ma"],
            "macd_전환":     결과["조건5_macd"],
            "거래량_증가":   결과["조건3_거래량"],
            "변동성돌파":    결과["조건6_변동성돌파"],
        })

        수량_표시  = f"{추천_수량:.6f}" if market in ("CRYPTO", "CRYPTO_KRW") else f"{추천_수량}주"
        atr_표시   = f"{통화_기호}{atr:,.0f}"
        강력_표시  = "🔥 *강력 매수*" if 강력매수 else "📈 *매수 신호*"
        점수_표시  = f"[{점수}/6점]"

        신호_블록 = (
            f"{강력_표시} {점수_표시}: {name} ({ticker})\n"
            f"  • RSI: {결과['rsi']:.1f} {조건표시(결과['조건1_rsi'])} | "
            f"MA20 {조건표시(결과['조건2_ma'])} | "
            f"거래량 {조건표시(결과['조건3_거래량'])} | "
            f"볼린저 {조건표시(결과['조건4_볼린저'])} | "
            f"MACD {조건표시(결과['조건5_macd'])} | "
            f"변동성돌파 {조건표시(결과['조건6_변동성돌파'])}\n"
            f"  • 📐 추천 수량: {수량_표시} (ATR={atr_표시}, 리스크 1%)\n"
            f"  • 🎯 목표가: {통화_기호}{목표가:,.0f} (+{TARGET:.0f}%) | "
            f"손절가: {통화_기호}{손절가:,.0f} ({STOP:.0f}%)\n"
        )
        if ai_이유:
            신호_블록 += f"  • 🤖 {ai_이유}\n"
        신호_블록 += bt_문구

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

    return 매수신호_섹션, 크립토_섹션, 뉴스_섹션, 신호_종목_요약


def run_analysis(include_crypto=True):
    """
    전체 분석의 '메인 함수'다.
    - 설정/종목/포트폴리오를 읽고
    - 필요한 가격 데이터를 미리 다운받고(속도 개선)
    - 시장 상태(하락장 등)를 판단한 뒤
    - 리포트를 만들어 텔레그램으로 보낸다
    """
    지금 = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n🔍 분석 시작: {지금}")

    clear_cache()

    settings   = load_settings()
    종목목록   = load_stocks()
    포트폴리오 = load_portfolio()

    if not include_crypto:
        # 암호화폐를 빼고 싶을 때(옵션): 종목 리스트에서 CRYPTO만 제외한다.
        종목목록 = [s for s in 종목목록 if s["market"] not in ("CRYPTO", "CRYPTO_KRW")]

    # 여러 번 다운로드하면 느리니까, 필요한 티커를 모아서 한 번에 받는다.
    모든_티커 = [s["ticker"] for s in 종목목록]
    레짐_티커 = ["^GSPC", "^KS11", "^VIX", "BTC-USD"]
    전체_티커 = list(set(모든_티커 + 레짐_티커))

    print(f"  🚀 사전 일괄 다운로드 시작 (총 {len(전체_티커)}개 티커)...")
    bulk_download(전체_티커, period="1y")
    bulk_download(전체_티커, period="3mo")
    bulk_download(전체_티커, period="5d")
    print(f"  ✅ 사전 다운로드 완료")

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

    매수신호_섹션, 크립토_섹션, 뉴스_섹션, 신호_종목_요약 = build_report_sections(
        종목목록, settings, 크립토_하락_레짐
    )

    # AI 시장 코멘트 (신호 종목이 있거나 하락장일 때만)
    ai_코멘트 = get_ai_market_commentary(
        하락_레짐, 크립토_하락_레짐, 경고_목록, 신호_종목_요약
    )
    if ai_코멘트:
        리포트 += ai_코멘트
        리포트 += "─────────────────────────\n"

    if 매수신호_섹션:
        리포트 += 매수신호_섹션
    else:
        if 하락_레짐:
            시장_요약 = "⚠️ 대세 하락장입니다. 현금 비중을 높이고 관망하는 것을 추천합니다."
        elif 크립토_하락_레짐:
            시장_요약 = "₿ 크립토 시장이 하락 레짐입니다. 주식은 중립이나 소극적 매수가 적합합니다."
        else:
            시장_요약 = "현재 매수 조건 충족 종목 없음 (점수 미달)"

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
