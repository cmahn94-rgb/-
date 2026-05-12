"""
scheduler_job.py — 리포트 조립 및 전체 분석 실행 (병렬 처리)
=============================================================
[업그레이드 v4 변경사항]
  - 매도 신호 섹션 추가 (기존엔 매도 알림이 전혀 없었음)
  - 보너스 점수 표시 (다이버전스 🔍, 복합강세 💥)
  - ADX 추세 강도 표시 추가
  - 백테스트에 거래 횟수 추가 표시
  - 점수 표시: [N/6점] → [N점] (보너스 포함 총점으로 변경)
  - Gemini API 429/503 오류 시 지수 백오프(재시도 대기) 적용
  - 우선순위 종합점수(0~100) 도입 — 신호 많을 때 무엇을 먼저 볼지 자동 정렬
    ┌ 신호점수  (0~25): 기술지표 총점 반영
    ├ RSI점수   (0~20): RSI 낮을수록(저평가) 높은 점수
    ├ Sharpe점수(0~25): 백테스트 위험조정수익률
    ├ 승률점수  (0~20): 백테스트 승률
    └ ADX점수   (0~10): 추세 강도 (뚜렷한 추세일수록 신뢰도 ↑)
    → 알림 상단에 "오늘의 TOP 3" 요약 표시
    → 전체 신호 리스트는 종합점수 내림차순 정렬
    → MAX_SIGNAL_DISPLAY: 한 번에 보여줄 최대 종목 수 (기본 7개)
"""

from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+ 내장 (한국시간 변환용)
from concurrent.futures import ThreadPoolExecutor, as_completed

from config        import load_settings
from stocks_loader import load_stocks, load_portfolio
from signals       import get_market_regime, calc_signals, calc_sell_signal, run_backtest, run_backtest_walkforward, calc_position_size
from market_phase  import detect_market_phase
from data_loader   import get_price_data, get_news_summary, clear_cache, bulk_download, get_today_open
from portfolio     import check_portfolio_alerts
from telegram_bot  import send_telegram
from ai_analyst    import (
    get_ai_market_commentary,
    get_market_news_briefing,
    get_ai_signal_reasons_batch,
    translate_to_korean_one_line_batch,
)

MAX_WORKERS = 20       # 동시에 분석할 종목 수 (너무 많으면 API 제한에 걸림)
MAX_SIGNAL_DISPLAY = 7  # 알림에 표시할 최대 매수 신호 종목 수
                        # 이 수를 초과하면 하위 종목은 '요약 목록'으로만 표시
                        # settings.txt의 MAX_SIGNAL_DISPLAY 값으로 덮어쓸 수 있음


def is_market_open():
    """
    주식 시장이 열려 있는 날인지 확인한다.

    [중학생 설명]
    월~금은 장이 열리고, 토/일은 주식 시장이 쉰다.
    주말엔 주식 분석을 건너뛰고 암호화폐만 분석한다.
    (암호화폐는 365일 24시간 거래 가능)
    """
    오늘 = datetime.now(ZoneInfo("Asia/Seoul")).weekday()  # KST 기준 요일
    if 오늘 >= 5:
        print("📅 오늘은 주말 휴장입니다. 암호화폐만 분석합니다.")
        return False
    return True


def analyze_one(종목, settings):
    """
    종목 1개를 분석해서, 리포트에 필요한 값들을 한 번에 묶어서 돌려준다.

    [중학생 설명]
    종목이 100개라면, 이 함수가 100번 호출된다.
    ThreadPoolExecutor가 이 함수를 동시에 20개씩 병렬로 실행해서 속도를 높인다.
    (순서대로 1개씩 하면 너무 느리기 때문)
    """
    ticker = 종목["ticker"]
    name   = 종목["name"]
    market = 종목["market"]

    결과 = calc_signals(ticker, name, market, settings)
    if 결과 is None:
        return None

    현재가 = 결과["현재가"]
    atr    = 결과["atr"]

    # 시장별 기준 자산 및 통화 기호 설정
    # (포지션 사이징을 위한 가정 자산 — 실제 자산은 settings.txt에서 변경 가능)
    if market in ("KR", "CRYPTO_KRW"):
        통화_기호 = "₩"
        총자산    = float(settings.get("TOTAL_ASSET_KRW", 30_000_000))
    else:
        통화_기호 = "$"
        총자산    = float(settings.get("TOTAL_ASSET_USD", 20_000))

    추천_수량 = calc_position_size(총자산, atr, market)

    # 목표가, 손절가 계산
    TARGET = settings.get("TARGET_PROFIT", 25)
    STOP   = settings.get("STOP_LOSS",    -7)
    목표가 = 현재가 * (1 + TARGET / 100)
    손절가 = 현재가 * (1 + STOP   / 100)

    # 매수 신호 종목만 백테스트 + walk-forward 실행
    bt_문구 = ""; bt1y = None
    if 결과["매수신호"]:
        bt1y = run_backtest(ticker, market, settings, period_months=12)
        wf   = run_backtest_walkforward(ticker, market, settings)
        if bt1y is not None:
            거래횟수_표시 = f" | 거래:{bt1y.get('거래횟수', '?')}회"
            bt_문구 = (
                f"  • 📊 백테스트(1년, 비용차감): {bt1y['수익률']:+.1f}% | "
                f"MDD: {bt1y['mdd']:.1f}% | "
                f"Sharpe: {bt1y['sharpe']:.2f} | "
                f"승률: {bt1y['승률']:.0f}%{거래횟수_표시}\n"
            )
        # Walk-forward 신뢰도 표시 + 과적합 시 매수 신호 차단
        if wf is not None:
            신뢰도_아이콘 = {"높음":"🟢", "보통":"🟡", "낮음":"🔴"}.get(wf["신뢰도"], "⚪")
            과적합_표시  = " ⚠️과적합주의" if wf["과적합_경고"] else ""
            검증_결과 = wf.get("검증")
            if 검증_결과 is None:
                sharpe_표시 = "거래부족"
            elif 검증_결과.get("거래횟수", 0) < 4:
                sharpe_표시 = f"거래부족({검증_결과['거래횟수']}회)"
            else:
                sharpe_표시 = f"{검증_결과['sharpe']:.2f}"
            bt_문구 += (
                f"  • 🔬 Walk-forward 검증(뒤 4개월): "
                f"{신뢰도_아이콘} 신뢰도 {wf['신뢰도']}{과적합_표시} "
                f"| 검증Sharpe: {sharpe_표시}\n"
            )
            # ★ 과적합 경고 + 검증 Sharpe 0.3 미만이면 매수 신호 차단
            # 백테스트가 실제 필터로 작동 (기존엔 표시만 했음)
            if wf["과적합_경고"] and 검증_결과 is not None:
                val_sh = 검증_결과.get("sharpe", 0)
                if val_sh < 0.3:
                    결과["매수신호"] = False
                    bt_문구 += f"  ⛔ WF 과적합 + 검증Sharpe {val_sh:.2f} < 0.3 → 매수신호 차단\n"
                    print(f"  ⛔ {name} WF과적합 차단 (Sharpe {val_sh:.2f})")

    # 최근 5일 데이터로 전일 대비 변동률과 뉴스 가져오기
    df_5d = get_price_data(ticker, period="5d")
    뉴스_목록, 변동률 = None, 0.0
    갭_비율 = 0.0
    if df_5d is not None and len(df_5d) >= 2:
        전일_종가 = df_5d["Close"].squeeze().iloc[-2]
        뉴스_목록, 변동률 = get_news_summary(ticker, 현재가, 전일_종가, name=name)
        # 당일 시가 갭 확인 (장 시작 직후 알림에만 유효)
        당일_시가 = get_today_open(ticker)
        if 당일_시가 and 전일_종가:
            갭_비율 = (당일_시가 - 전일_종가) / 전일_종가 * 100

    return {
        "결과":      결과,
        "현재가":    현재가,
        "atr":       atr,
        "통화_기호": 통화_기호,
        "추천_수량": 추천_수량,
        "market":    market,
        "name":      name,
        "TARGET":    TARGET,
        "STOP":      STOP,
        "목표가":    목표가,
        "손절가":    손절가,
        "bt_문구":   bt_문구,
        "bt_결과":   bt1y if 결과["매수신호"] else None,  # 우선순위 점수 계산용
        "뉴스_목록": 뉴스_목록,
        "변동률":    변동률,
        "갭_비율":   갭_비율,
    }


def calc_priority_score(결과: dict, bt: dict | None) -> tuple[float, dict]:
    """
    매수 신호 종목의 우선순위 종합점수(0~100점)를 계산한다.

    [중학생 설명]
    신호가 10개 동시에 뜨면 "어떤 걸 먼저 봐야 하나?"가 문제다.
    이 함수는 5가지 기준으로 각 종목에 점수를 매겨서
    가장 좋은 종목이 알림 맨 위에 오도록 순서를 정한다.

    점수 구성 (합계 최대 100점):
    ① 신호점수  (0~25): 기술지표 총점. 점수 높을수록 강한 신호
    ② RSI점수   (0~20): RSI 낮을수록(저평가) 반등 여지가 크다
    ③ Sharpe점수(0~25): 백테스트 위험 대비 수익률. 높을수록 안정적
    ④ 승률점수  (0~20): 백테스트 승률. 높을수록 이 전략이 잘 맞는 종목
    ⑤ ADX점수   (0~10): 추세 강도. 뚜렷한 추세일수록 신뢰도 ↑

    반환: (종합점수, {항목별 점수 dict})
    """
    세부 = {}

    # ① 신호점수 (0~25): 기본6 + 보너스2 = 최대8점 → 25점으로 환산
    신호점수_원 = 결과.get("점수", 0)
    세부["신호점수"] = round(신호점수_원 / 8 * 25, 1)

    # ② RSI점수 (0~20): RSI ≤ 20 → 20점, RSI = 50 → 8점, RSI > 60 → 0점
    rsi = 결과.get("rsi", 50)
    if rsi <= 20:    세부["rsi점수"] = 20.0
    elif rsi <= 30:  세부["rsi점수"] = round(20 - (rsi - 20) * 0.4, 1)
    elif rsi <= 40:  세부["rsi점수"] = round(16 - (rsi - 30) * 0.4, 1)
    elif rsi <= 50:  세부["rsi점수"] = round(12 - (rsi - 40) * 0.4, 1)
    else:             세부["rsi점수"] = 0.0

    # ③ Sharpe점수 (0~25): 백테스트 없으면 0점
    sharpe = (bt or {}).get("sharpe", 0)
    if sharpe >= 2.0:    세부["sharpe점수"] = 25.0
    elif sharpe >= 1.5:  세부["sharpe점수"] = round(20 + (sharpe - 1.5) * 10, 1)
    elif sharpe >= 1.0:  세부["sharpe점수"] = round(15 + (sharpe - 1.0) * 10, 1)
    elif sharpe >= 0.5:  세부["sharpe점수"] = round(8  + (sharpe - 0.5) * 14, 1)
    else:                 세부["sharpe점수"] = 0.0

    # ④ 승률점수 (0~20): 백테스트 없으면 0점
    승률 = (bt or {}).get("승률", 0)
    if 승률 >= 70:    세부["승률점수"] = 20.0
    elif 승률 >= 60:  세부["승률점수"] = round(14 + (승률 - 60) * 0.6, 1)
    elif 승률 >= 50:  세부["승률점수"] = round(8  + (승률 - 50) * 0.6, 1)
    else:              세부["승률점수"] = 0.0

    # ⑤ ADX점수 (0~10): ADX ≥ 40 → 10점, 25~40 → 비례, 25 미만 → 0점
    adx = 결과.get("adx")
    if adx is None:   세부["adx점수"] = 3.0   # 데이터 없으면 중간값
    elif adx >= 40:   세부["adx점수"] = 10.0
    elif adx >= 25:   세부["adx점수"] = round((adx - 25) / 15 * 10, 1)
    else:              세부["adx점수"] = 0.0

    종합 = round(sum(세부.values()), 1)
    return 종합, 세부


def build_report_sections(종목목록, settings, 크립토_하락_레짐, 포트폴리오=None):
    """
    모든 종목을 병렬 분석하고, 매수/매도/크립토 섹션을 각각 만든다.

    [중학생 설명]
    1) 모든 종목을 동시에 20개씩 분석 (병렬 처리)
    2) 매수 신호 종목 → 매수 신호 섹션
    3) 매도 신호 종목 → 매도 신호 섹션 (신규!)
    4) 암호화폐 → 크립토 섹션
    5) AI 한 줄 코멘트 추가
    """
    매수신호_섹션 = ""
    매도신호_섹션 = ""   # ← 신규: 매도 알림 섹션
    크립토_섹션   = ""
    신호_종목_요약 = []  # AI 시장 코멘트용

    결과_맵 = {}
    print(f"  ⚡ {len(종목목록)}개 종목 병렬 분석 시작 (동시 {MAX_WORKERS}개)...")

    # ── 병렬 분석 ───────────────────────────────────────────
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

    # ── 매도 신호 병렬 분석 (신규) ──────────────────────────
    # 매도 신호는 매수 신호와 별도로 계산 (빠른 처리를 위해 독립 실행)
    매도_결과_맵 = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        매도_future_map = {
            executor.submit(calc_sell_signal, s["ticker"], s["name"], s["market"], settings): s["ticker"]
            for s in 종목목록
        }
        for future in as_completed(매도_future_map):
            ticker = 매도_future_map[future]
            try:
                매도_결과_맵[ticker] = future.result()
            except Exception:
                매도_결과_맵[ticker] = None

    # ── 유틸 함수 ────────────────────────────────────────────
    def 조건표시(v):
        """조건 달성 여부(bool)를 ✅/❌ 이모지로 변환한다. 리포트 가독성용."""
        return "✅" if v else "❌"

    # ── 매수·매도 동시 종목 필터 ─────────────────────────────
    # 강력매도(3개 조건 모두 충족)일 때만 매수 목록에서 제거한다.
    # 일반 매도(2개 충족)는 제거하지 않음 — 급등장에서 RSI+볼린저만으로
    # 매도 조건이 너무 쉽게 충족되어 매수 종목 대부분이 제거되는 문제 방지
    매도_티커_셋 = {
        ticker for ticker, d in 매도_결과_맵.items()
        if d is not None and d.get("강력매도") is True  # 강력매도만
    }
    if 매도_티커_셋:
        print(f"  ⛔ 강력매도 종목 매수 제외: {매도_티커_셋}")

    # ── 우선순위 점수 계산 → 정렬 ────────────────────────────
    # 매수 신호 종목을 종합점수(0~100) 기준으로 정렬해서
    # 알림 상단에 가장 유망한 종목이 오도록 한다.
    신호_종목_순서 = []
    for 종목 in 종목목록:
        ticker = 종목["ticker"]
        d = 결과_맵.get(ticker)
        if d is None or not d["결과"]["매수신호"]:
            continue
        # 강력매도 종목 매수 제외
        if ticker in 매도_티커_셋:
            continue
        # 갭업 +5% 이상 종목은 매수 제외 (급등장 현실 반영)
        갭 = d.get("갭_비율", 0.0)
        if 갭 >= 5.0:
            print(f"  ⛔ 갭업 {갭:.1f}% 초과 매수 제외: {ticker}")
            continue
        우선순위점수, 우선순위세부 = calc_priority_score(
            d["결과"],
            d.get("bt_결과")
        )
        # 갭업 2~5% 구간은 우선순위 점수 패널티 (2% 미만은 노이즈)
        if 갭 >= 2.0:
            갭_패널티 = round((갭 - 2.0) * 3, 1)  # 2%→0점, 5%→9점 패널티
            우선순위점수 = max(0, 우선순위점수 - 갭_패널티)
        신호_종목_순서.append((ticker, 우선순위점수, 우선순위세부))

    # 종합점수 높은 순으로 정렬
    신호_종목_순서.sort(key=lambda x: x[1], reverse=True)

    # settings.txt 또는 기본값으로 최대 표시 수 결정
    max_display = int(settings.get("MAX_SIGNAL_DISPLAY", MAX_SIGNAL_DISPLAY))

    # ── TOP 3 요약 헤더 생성 ────────────────────────────────
    # 신호가 3개 이상일 때만 "오늘의 TOP 3" 요약을 맨 앞에 표시
    # → 알림을 열자마자 무엇을 먼저 볼지 바로 알 수 있다
    top3_헤더 = ""
    주식_신호 = [(t, s, b) for t, s, b in 신호_종목_순서
                 if 결과_맵[t]["market"] not in ("CRYPTO", "CRYPTO_KRW")]
    if len(주식_신호) >= 3:
        top3_헤더 = "🏆 *오늘의 TOP 3 우선순위*\n"
        for rank, (ticker, 점수, 세부) in enumerate(주식_신호[:3], 1):
            d = 결과_맵[ticker]
            메달 = ["🥇","🥈","🥉"][rank - 1]
            top3_헤더 += (
                f"  {메달} {d['name']} ({ticker}) "
                f"종합 {점수:.0f}점 "
                f"| 신호 {세부['신호점수']:.0f} "
                f"RSI {세부['rsi점수']:.0f} "
                f"Sharpe {세부['sharpe점수']:.0f} "
                f"승률 {세부['승률점수']:.0f} "
                f"ADX {세부['adx점수']:.0f}\n"
            )
        top3_헤더 += "─────────────────────────\n"

    # ── AI 배치: 종목별 신호 이유 한 줄씩 ──────────────────
    reason_inputs = []
    for ticker, _, _ in 신호_종목_순서:
        d = 결과_맵.get(ticker)
        if d is None:
            continue
        r = d["결과"]
        reason_inputs.append({
            "ticker":     ticker,
            "name":       d["name"],
            "rsi":        float(r.get("rsi", 0.0)),
            "score":      int(r.get("점수", 0)),
            "ma":         bool(r.get("조건2_ma")),
            "macd":       bool(r.get("조건5_macd")),
            "vol":        bool(r.get("조건3_거래량")),
            "breakout":   bool(r.get("조건6_변동성돌파")),
            "divergence": bool(r.get("보너스_다이버전스")),
        })
    # Gemini 호출 전 10초 대기
    # 이유: 103개 병렬 분석 중 Alpha Vantage 뉴스를 최대 12개 동시 호출함
    # → 분당 API 한도가 이미 채워진 상태에서 Gemini 추가 호출 → 429
    # → 10초 대기로 이전 AV 호출의 분당 카운트가 리셋된 뒤 Gemini 호출
    import time as _time
    if reason_inputs:  # 신호 종목이 있을 때만 대기 (없으면 Gemini 호출 없음)
        _time.sleep(10)
    ai_reason_map = get_ai_signal_reasons_batch(reason_inputs)

    # ── 매수 신호 섹션 조립 (우선순위 정렬 순서로) ──────────
    표시된_주식_수 = 0
    숨겨진_종목_목록 = []   # max_display 초과 종목 → 간략 요약으로 처리

    for rank, (ticker, 우선순위점수, 우선순위세부) in enumerate(신호_종목_순서, 1):
        d = 결과_맵.get(ticker)
        if d is None:
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
        갭_비율   = d.get("갭_비율", 0.0)
        현재가    = d["현재가"]
        점수      = 결과["점수"]
        강력매수  = 결과["강력매수"]
        adx_값    = 결과.get("adx")

        # ── max_display 초과 종목 처리 ──────────────────────
        # 주식 종목이 max_display를 초과하면 상세 블록 대신
        # 간략 1줄 요약으로만 표시 → 알림이 너무 길어지는 것 방지
        if market not in ("CRYPTO", "CRYPTO_KRW"):
            표시된_주식_수 += 1
            if 표시된_주식_수 > max_display:
                숨겨진_종목_목록.append(
                    f"  {rank}. {name} ({ticker}) "
                    f"| 종합 {우선순위점수:.0f}점 "
                    f"| 신호점수 {점수}점 "
                    f"| RSI {결과['rsi']:.1f}"
                )
                continue   # 상세 블록 건너뛰고 다음 종목으로

        신호_종목_요약.append({
            "name":   name,
            "ticker": ticker,
            "점수":   점수,
            "rsi":    결과["rsi"],
        })

        ai_이유 = ai_reason_map.get(ticker, "")

        수량_표시  = f"{추천_수량:.6f}" if market in ("CRYPTO", "CRYPTO_KRW") else f"{추천_수량}주"
        atr_표시   = f"{통화_기호}{atr:,.0f}"
        # 신호 강도 표시: 강력매수/일반매수 구분
        # 강력매수 = 기본5점 이상 OR (4점+보너스 있음)
        if 강력매수 and (결과.get("보너스_다이버전스") or
                         결과.get("보너스_복합강세") or
                         결과.get("보너스_신고가")):
            강력_표시 = "🔥 *강력 매수*"  # 보너스 조건 포함 강력
        elif 강력매수:
            강력_표시 = "🔥 *강력 매수*"  # 기본 지표만으로 5점+
        else:
            강력_표시 = "📈 *매수 신호*"

        # 순위 뱃지: 1~3위는 메달 이모지, 4위 이하는 번호
        순위_뱃지 = ["🥇","🥈","🥉"][rank - 1] if rank <= 3 else f"#{rank}"

        # 보너스 점수 아이콘 표시
        보너스_아이콘 = ""
        if 결과.get("보너스_다이버전스"):
            보너스_아이콘 += " 🔍다이버전스"
        if 결과.get("보너스_복합강세"):
            보너스_아이콘 += " 💥복합강세"
        if 결과.get("보너스_stoch"):
            보너스_아이콘 += " 📊스토캐스틱"
        if 결과.get("보너스_cci"):
            보너스_아이콘 += " 🎯CCI과매도"

        # ADX 추세 강도 표시
        adx_표시 = ""
        if adx_값 is not None:
            if adx_값 >= 40:
                adx_표시 = f" | ADX {adx_값:.0f}🔥"
            elif adx_값 >= 25:
                adx_표시 = f" | ADX {adx_값:.0f}↑"
            else:
                adx_표시 = f" | ADX {adx_값:.0f}(횡보)"

        # 점수 표시: 신호점수 + 종합 우선순위 점수 함께 표시
        점수_표시 = f"[신호 {점수}점{보너스_아이콘} | 우선순위 {우선순위점수:.0f}점]"

        신호_블록 = (
            f"{순위_뱃지} {강력_표시} {점수_표시}: {name} ({ticker}){adx_표시}\n"
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
        # 분할 목표가 표시
        TARGET_1 = settings.get("TARGET_1", 12)
        TARGET_2 = settings.get("TARGET_2", 25)
        TRAILING = settings.get("TRAILING_STOP", 8)
        목표가_1 = 현재가 * (1 + TARGET_1 / 100)
        목표가_2 = 현재가 * (1 + TARGET_2 / 100)
        신호_블록 = 신호_블록.replace(
            f"  • 🎯 목표가: {통화_기호}{목표가:,.0f} (+{TARGET:.0f}%) | "
            f"손절가: {통화_기호}{손절가:,.0f} ({STOP:.0f}%)\n",
            f"  • 🎯 1차 {TARGET_1:.0f}%: {통화_기호}{목표가_1:,.0f} | "
            f"2차 {TARGET_2:.0f}%: {통화_기호}{목표가_2:,.0f} | "
            f"손절: {통화_기호}{손절가:,.0f} ({STOP:.0f}%) | "
            f"트레일링: -{TRAILING:.0f}%\n"
        )
        # 갭업/갭다운 경고 (당일 시가가 전일 종가 대비 ±2% 이상)
        if abs(갭_비율) >= 2.0:
            갭_방향 = "갭업" if 갭_비율 > 0 else "갭다운"
            갭_아이콘 = "⚠️" if 갭_비율 > 0 else "🔻"
            신호_블록 += (
                f"  {갭_아이콘} {갭_방향} {갭_비율:+.1f}% 감지 — "
                f"{'고점 진입 주의, 눌림목 기다릴 것 권장' if 갭_비율 > 0 else '추가 하락 가능성, 분할 매수 권장'}\n"
            )
        if ai_이유:
            신호_블록 += f"  • 🤖 {ai_이유}\n"
        신호_블록 += bt_문구

        # 뉴스 블록 (매수 신호 종목만)
        if 뉴스_목록:
            신호_블록 += f"  • 📰 뉴스 (변동 {변동률:+.1f}%):\n"

            # source 필드 기반 번역 대상 분류
            # gemini_grounding → 이미 한국어로 작성됨 → 번역 불필요
            # alphavantage / yfinance → 영어 원문 → 번역 필요
            texts, sentiments, 번역필요_flags = [], [], []
            for item in 뉴스_목록:
                text      = item.get("text")      if isinstance(item, dict) else str(item)
                sentiment = item.get("sentiment") if isinstance(item, dict) else "중립"
                source    = item.get("source", "yfinance") if isinstance(item, dict) else "yfinance"
                texts.append(text)
                sentiments.append(sentiment)
                번역필요_flags.append(source in ("alphavantage", "yfinance"))

            # 번역 필요한 텍스트만 묶어서 배치 번역 (Gemini API 절약)
            need_translate = [t for t, f in zip(texts, 번역필요_flags) if f]
            translated     = translate_to_korean_one_line_batch(need_translate) if need_translate else []
            it = iter(translated)

            for text, sentiment, need in zip(texts, sentiments, 번역필요_flags):
                if need:
                    ko = next(it, "")
                    # 번역 성공 시 한국어로 교체, 실패 시 원문 유지
                    if ko and ko.strip():
                        text = ko.strip()
                신호_블록 += f"    - [{sentiment}] {text}\n"

        if market in ("CRYPTO", "CRYPTO_KRW"):
            if 크립토_하락_레짐:
                신호_블록 += "  ⚠️ *BTC 하락 레짐 감지*: 소액·분할 매수만 고려하세요.\n"
            크립토_섹션 += 신호_블록
        else:
            매수신호_섹션 += 신호_블록 + "─────────────────────────\n"

    # ── 매도 신호 섹션 조립 (보유 종목에만 표시) ─────────────────
    # portfolio.txt에 등록된 종목에서만 매도 신호를 표시한다.
    # 보유하지 않는 종목의 매도 신호는 의미 없는 알림이므로 제거.
    보유_티커_셋 = {p["ticker"] for p in 포트폴리오} if 포트폴리오 else set()

    for 종목 in 종목목록:
        ticker = 종목["ticker"]
        d = 매도_결과_맵.get(ticker)
        if d is None:
            continue
        # 포트폴리오에 없는 종목은 매도 신호 생략
        if 보유_티커_셋 and ticker not in 보유_티커_셋:
            continue

        market    = d["market"]
        name      = d["name"]
        현재가    = d["현재가"]
        rsi       = d["rsi"]
        강력매도  = d["강력매도"]

        if market in ("CRYPTO", "CRYPTO_KRW"):
            통화_기호 = "₩" if market == "CRYPTO_KRW" else "$"
        else:
            통화_기호 = "₩" if market == "KR" else "$"

        강도_표시 = "🚨 *강력 매도*" if 강력매도 else "📉 *매도 신호*"

        매도_블록 = (
            f"{강도_표시}: {name} ({ticker})\n"
            f"  • RSI: {rsi:.1f} | 현재가: {통화_기호}{현재가:,.0f}\n"
            f"  • 과열 {조건표시(d['조건A_rsi과열'])} | "
            f"볼린저상단 {조건표시(d['조건B_볼린저상단'])} | "
            f"MACD하락 {조건표시(d['조건C_macd하락'])}\n"
        )

        # 매도 신호 종목의 뉴스도 표시 (analyze_one()에서 이미 수집한 데이터 재사용)
        # 결과_맵에서 해당 종목 뉴스를 꺼내서 번역 후 추가
        매도_뉴스 = 결과_맵.get(ticker)
        if 매도_뉴스 and 매도_뉴스.get("뉴스_목록"):
            매도_변동률 = 매도_뉴스.get("변동률", 0.0)
            매도_블록 += f"  • 📰 뉴스 (변동 {매도_변동률:+.1f}%):\n"

            sell_texts, sell_sentiments, sell_번역필요 = [], [], []
            for item in 매도_뉴스["뉴스_목록"]:
                text      = item.get("text", "")      if isinstance(item, dict) else str(item)
                sentiment = item.get("sentiment", "중립") if isinstance(item, dict) else "중립"
                source    = item.get("source", "yfinance") if isinstance(item, dict) else "yfinance"
                sell_texts.append(text)
                sell_sentiments.append(sentiment)
                sell_번역필요.append(source in ("alphavantage", "yfinance"))

            sell_need = [t for t, f in zip(sell_texts, sell_번역필요) if f]
            sell_translated = translate_to_korean_one_line_batch(sell_need) if sell_need else []
            sell_it = iter(sell_translated)

            for text, sentiment, need in zip(sell_texts, sell_sentiments, sell_번역필요):
                if need:
                    ko = next(sell_it, "")
                    if ko and ko.strip():
                        text = ko.strip()
                매도_블록 += f"    - [{sentiment}] {text}\n"

        if not market in ("CRYPTO", "CRYPTO_KRW"):
            매도신호_섹션 += 매도_블록 + "─────────────────────────\n"

    # ── 숨겨진 종목 간략 요약 추가 ────────────────────────────
    # max_display 초과로 상세 블록이 생략된 종목을
    # 알림 하단에 1줄씩 요약해서 존재 자체는 알 수 있게 한다.
    if 숨겨진_종목_목록:
        매수신호_섹션 += (
            f"\n📋 *추가 신호 종목 {len(숨겨진_종목_목록)}개* "
            f"(우선순위 {max_display + 1}위~, 간략 표시)\n"
        )
        for 줄 in 숨겨진_종목_목록:
            매수신호_섹션 += 줄 + "\n"
        매수신호_섹션 += "─────────────────────────\n"

    # ── TOP3 헤더를 매수 섹션 맨 앞에 붙이기 ───────────────
    # 헤더는 이미 top3_헤더 변수에 만들어져 있음
    매수신호_섹션 = top3_헤더 + 매수신호_섹션

    return 매수신호_섹션, 크립토_섹션, 매도신호_섹션, 신호_종목_요약


def _record_signals(신호_종목_요약: list):
    """
    신호 발생 종목을 portfolio_signals.txt에 자동 기록한다.

    [중학생 설명]
    "오늘 이 종목에서 신호가 났다"는 기록을 파일에 저장한다.
    일주일 뒤, 한 달 뒤에 실제로 올랐는지 확인하면
    이 시스템이 얼마나 정확한지 직접 측정할 수 있다.

    기록 형식: 날짜,티커,이름,신호점수,상태
    """
    if not 신호_종목_요약:
        return
    try:
        오늘 = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")  # KST 기준
        with open("portfolio_signals.txt", "a", encoding="utf-8") as f:
            for s in 신호_종목_요약:
                ticker = s.get("ticker", "")
                name   = s.get("name",   "")
                점수   = s.get("점수",   0)
                f.write(f"{오늘},{ticker},{name},{점수},신호발생\n")
        print(f"  📝 신호 {len(신호_종목_요약)}개 기록 완료 (portfolio_signals.txt)")
    except Exception as e:
        print(f"  ⚠️ 신호 기록 실패: {e}")


def run_analysis(include_crypto=True):
    """
    전체 분석의 '메인 함수'.
    설정 로드 → 데이터 다운로드 → 시장 레짐 판단 → 리포트 생성 → 텔레그램 전송

    [중학생 설명]
    이 함수 하나가 모든 과정을 순서대로 실행한다:
    1) settings.txt에서 전략 설정값 읽기
    2) stocks.txt에서 감시할 종목 목록 읽기
    3) 필요한 주가 데이터를 한꺼번에 미리 다운로드 (속도 개선)
    4) 지금이 하락장인지 판단
    5) 매수/매도 신호 분석 및 리포트 생성
    6) 텔레그램으로 알림 전송
    """
    # GitHub Actions는 UTC 기준으로 실행됨 → KST(UTC+9)로 변환
    지금 = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M KST")
    print(f"\n🔍 분석 시작: {지금}")

    clear_cache()  # 이전 실행에서 남은 캐시 삭제 (최신 데이터 보장)

    settings   = load_settings()
    종목목록   = load_stocks()
    포트폴리오 = load_portfolio()

    if not include_crypto:
        종목목록 = [s for s in 종목목록 if s["market"] not in ("CRYPTO", "CRYPTO_KRW")]

    # 데이터 사전 다운로드 (종목별로 따로 받으면 느리니까 한꺼번에 받음)
    모든_티커 = [s["ticker"] for s in 종목목록]
    레짐_티커 = ["^GSPC", "^KS11", "^VIX", "BTC-USD"]
    전체_티커 = list(set(모든_티커 + 레짐_티커))

    print(f"  🚀 사전 일괄 다운로드 시작 (총 {len(전체_티커)}개 티커)...")
    bulk_download(전체_티커, period="1y")
    bulk_download(전체_티커, period="3mo")
    bulk_download(전체_티커, period="5d")
    print(f"  ✅ 사전 다운로드 완료")

    # ── 시장 국면 감지 → 동적 임계값 설정 ──────────────────────
    # 고정 BUY_SCORE_THRESHOLD 대신 국면에 따라 2~6점으로 자동 조정
    # 강한상승=2점, 완만한상승=3점, 횡보=4점, 조정=5점, 패닉=6점
    print("  🔭 시장 국면 감지 중...")
    try:
        phase_result = detect_market_phase(market="KR")
        # settings의 BUY_SCORE_THRESHOLD를 국면별 값으로 덮어씀
        settings["BUY_SCORE_THRESHOLD"] = phase_result.score_threshold
        print(f"  {phase_result.phase.value} | 임계값: {phase_result.score_threshold}점 | "
              f"RSI:{phase_result.rsi:.0f} VIX:{phase_result.vix:.1f}")
    except Exception as e:
        print(f"  ⚠️ 국면 감지 실패 → settings 기본값 사용: {e}")
        phase_result = None

    하락_레짐, 크립토_하락_레짐, 경고_목록 = get_market_regime()

    # ── 리포트 헤더 ──────────────────────────────────────────
    리포트  = f"⚔️ *퀀트 헤지펀드 리포트* | {지금}\n"
    # 국면 정보 + 자금 배분 가이드 표시
    if phase_result is not None:
        # 국면별 자금 배분 권장 비율 (참고 봇 PHASE_CONFIG 기반)
        배분_가이드 = {
            "🚀 강한상승":  "주식 60% / 단기매매 35% / 현금 5%",
            "📈 완만한상승": "주식 60% / 단기매매 30% / 현금 10%",
            "↔️ 횡보박스권": "주식 40% / 단기매매 40% / 현금 20%",
            "📉 조정하락":  "주식 20% / 단기매매 20% / 현금 60%",
            "🔴 급락패닉":  "주식 0% / 단기매매 10% / 현금 90%",
        }
        배분 = 배분_가이드.get(phase_result.phase.value, "")
        리포트 += (
            f"시장 국면: {phase_result.phase.value} "
            f"| 매수 임계값: {phase_result.score_threshold}점\n"
            f"자금배분 권장: {배분}\n"
        )
    리포트 += "─────────────────────────\n"

    # ── 시장 주요 이슈 브리핑 (알림 최상단) ─────────────────
    # Gemini Google Search로 실시간 검색한 오늘의 이슈 3가지
    # 토스 앱처럼 '지금 시장에서 가장 중요한 것 3개'를 먼저 보여줌
    브리핑 = get_market_news_briefing()
    if 브리핑:
        리포트 += 브리핑
        리포트 += "─────────────────────────\n"

    if 하락_레짐:
        경고_문구 = " + ".join(경고_목록) if 경고_목록 else "복합 조건"
        리포트 += (
            "⚠️ *대세 하락장 경고*: 현금 비중 확대 권고. "
            "매수 신호가 발생해도 소액·분할 매수만 고려하세요.\n"
        )
        리포트 += f"(트리거: {경고_문구})\n"
        리포트 += "─────────────────────────\n"

    # ── 섹션 생성 ────────────────────────────────────────────
    매수신호_섹션, 크립토_섹션, 매도신호_섹션, 신호_종목_요약 = build_report_sections(
        종목목록, settings, 크립토_하락_레짐, 포트폴리오
    )

    # AI 시장 코멘트 — 신호이유 배치 후 5초 대기 (429 방지)
    import time as _time; _time.sleep(5)
    ai_코멘트 = get_ai_market_commentary(
        하락_레짐, 크립토_하락_레짐, 경고_목록, 신호_종목_요약
    )
    if ai_코멘트:
        리포트 += ai_코멘트
        리포트 += "─────────────────────────\n"

    # 매수 신호 섹션
    if 매수신호_섹션:
        리포트 += 매수신호_섹션
    else:
        if 하락_레짐:
            시장_요약 = "⚠️ 대세 하락장입니다. 현금 비중을 높이고 관망을 추천합니다."
        elif 크립토_하락_레짐:
            시장_요약 = "₿ 크립토 시장이 하락 레짐입니다. 소극적 매수가 적합합니다."
        else:
            시장_요약 = "현재 매수 조건 충족 종목 없음 (점수 미달)"
        리포트 += f"현재 매수 조건 충족 종목 없음 (주식/ETF)\n   {시장_요약}\n"
        리포트 += "─────────────────────────\n"

    # 매도 신호 섹션 (신규)
    if 매도신호_섹션:
        리포트 += "📉 *매도 신호 종목*\n"
        리포트 += 매도신호_섹션

    # 포트폴리오 알림 섹션
    포트폴리오_알림 = check_portfolio_alerts(포트폴리오, settings)
    if 포트폴리오_알림:
        리포트 += "💼 *포트폴리오 알림*\n"
        for 알림 in 포트폴리오_알림:
            리포트 += 알림 + "\n"
        리포트 += "─────────────────────────\n"

    # 암호화폐 섹션
    if include_crypto and 크립토_섹션:
        리포트 += "₿ *암호화폐 분석*\n"
        리포트 += 크립토_섹션
        리포트 += "─────────────────────────\n"

    # ── 신호 발생 종목 자동 기록 (실전 vs 백테스트 비교용) ────────
    # portfolio_signals.txt에 오늘 신호 종목과 예상 진입가를 저장한다.
    # 나중에 실제 수익률과 비교해 시스템 신뢰도를 측정할 수 있다.
    _record_signals(신호_종목_요약)

    # 텔레그램 전송
    send_telegram(리포트)
    print("✅ 분석 완료 및 리포트 전송")
