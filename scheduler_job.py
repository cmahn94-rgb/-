"""
scheduler_job.py — 리포트 조립 및 전체 분석 실행
=================================================
[v5.1 주요 기능]

■ 실행 흐름
  bulk_download → yfinance crumb 워밍업 → 시장 국면 감지
  → 병렬 신호 분석 (worker 4) → 상관관계 필터 → 리포트 조립 → 텔레그램 전송

■ 리스크 관리 (v5.1 신규)
  - 상관관계 필터: 신호 종목 간 60일 수익률 상관계수 ≥ CORRELATION_MAX(0.7) 시 하위 종목 제외
  - 최대 보유 수: MAX_POSITIONS(8) 초과 종목 제외
  - Sharpe 음수 차단: 1년 백테스트 Sharpe < 0 → 매수신호 False
  - WF 신뢰도 낮음 차단: 검증 Sharpe < 0 또는 과적합 → 매수신호 False
  - 표본 하한: 백테스트 거래 5회 미만 → Sharpe·승률 점수 0 처리

■ 우선순위 종합점수 (0~100점)
  ① 신호점수  (0~25): 기술지표 점수 + 독립 축 다양성 가중
  ② RSI점수   (0~20): RSI 낮을수록 (20 이하 = 20점, 40 미만 = ~12점)
  ③ Sharpe점수(0~25): 백테스트 위험조정수익률 (표본 5회 이상)
  ④ 승률점수  (0~20): 백테스트 승률 (표본 5회 이상)
  ⑤ ADX점수  (0~10): 추세 강도

■ 신호 블록 출력 순서 (종목당)
  순위뱃지 + 신호유형 + 점수/우선순위
  → 독립 축 충족 현황 (추세/모멘텀/평균회귀/수급)
  → 기술지표 6개 ✅❌
  → 추천수량 (ATR 리스크 1%)
  → 목표가 1차/2차 + 손절 + 트레일링
  → 진입 추천가 (볼린저하단 vs 현재가-ATR)
  → FA 데이터 (PER/PBR/ROE/FCF/목표주가)
  → 수급 (KR 전용: 기관/외국인 연속매수일, 동시매수 여부)
  → AI 한 줄 코멘트
  → 백테스트 + WF 검증
  → 뉴스
"""

from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+ 내장 (한국시간 변환용)
from concurrent.futures import ThreadPoolExecutor, as_completed
import time  # ⑧ 수정: 함수 내 반복 import 제거, 상단에서 한 번만
import os
import numpy as np  # 상관관계 계산용

from config        import load_settings
from stocks_loader import load_stocks, load_portfolio
from signals       import get_market_regime, calc_signals, calc_sell_signal, run_backtest, run_backtest_walkforward, calc_position_size
from market_phase  import detect_market_phase
from data_loader   import get_price_data, get_news_summary, clear_cache, bulk_download, bulk_download_weekly, get_today_open, _warmup_yfinance_crumb, get_vix_from_fred
from portfolio     import check_portfolio_alerts, check_max_holding_days, generate_weekly_report
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

    # ── 데이터 품질 게이트 (하이엔드): 신호 계산 전 검사 ──────────
    # 0원·비현실적 급변·거래정지 데이터면 신호 계산에서 제외.
    try:
        import data_quality as _dq
        _df_check = get_price_data(ticker, period="3mo")
        _q = _dq.check_price_data(_df_check, ticker)
        if _q["심각도"] == "차단":
            _dq_tracker = settings.get("_DQ_TRACKER")
            if _dq_tracker is not None:
                _dq_tracker.차단_종목.append((ticker, _q["문제"]))
            return None   # 불량 데이터 → 신호 계산 스킵
    except Exception:
        pass  # 품질 검사 실패는 무시하고 계속 (게이트가 봇을 막으면 안 됨)

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

            # [치명적 결함 수정 3] Sharpe 음수 종목 차단
            # 1년 백테스트에서 Sharpe < 0 = 리스크 대비 수익이 마이너스
            # TMUS(-0.43), NFLX(-0.47)처럼 신호점수 1위여도 실제 손실 전략 차단
            if bt1y["sharpe"] < 0:
                결과["매수신호"] = False
                bt_문구 += f"  ⛔ Sharpe {bt1y['sharpe']:.2f} < 0 → 백테스트 손실 전략 차단\n"
                print(f"  ⛔ {name} Sharpe 음수 차단 ({bt1y['sharpe']:.2f})")

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

            # [치명적 결함 수정 2] WF 신뢰도 낮음 = 완전 차단 (거래 부족 포함)
            # 기존: 거래 4회 미만이면 "차단 보류" → 통과됨
            # 수정: 거래 부족도 검증 불가 = 신뢰도 낮음으로 동일하게 차단
            # 근거: 검증 불가능한 전략에 실돈을 넣으면 안 됨
            if wf["신뢰도"] == "낮음" and 결과.get("매수신호"):
                결과["매수신호"] = False
                bt_문구 += f"  ⛔ WF 신뢰도 낮음 → 매수신호 차단 (검증 미통과)\n"
                print(f"  ⛔ {name} WF신뢰도낮음 차단 (Sharpe: {sharpe_표시})")

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
        "bt_결과":   bt1y,  # ⑨ 수정: WF 차단 여부와 무관하게 bt 결과 보존 (우선순위 점수 정확도 향상)
        "뉴스_목록": 뉴스_목록,
        "변동률":    변동률,
        "갭_비율":   갭_비율,
    }


def filter_correlated_signals(신호_종목_순서, 결과_맵, settings):
    """
    상관관계 기반 섹터 집중 차단 (1순위 리스크 관리).

    [중학생 설명]
    삼성전자·SK하이닉스·한미반도체가 동시에 매수 신호를 내면
    겉보기엔 3종목 분산이지만 실제론 전부 '반도체 한 방향'에 베팅하는 것.
    이 셋의 주가는 거의 똑같이 움직여서(상관계수 0.8+) 하나 무너지면 셋 다 무너진다.

    이 함수는 우선순위 높은 종목부터 순서대로 살펴보면서,
    이미 채택된 종목과 상관계수가 임계값(기본 0.7) 이상이면
    더 낮은 우선순위 종목을 '집중 위험'으로 제외한다.

    [계산 방식]
    - 최근 60일 일간 수익률(%)의 피어슨 상관계수
    - 같은 통화권(KR끼리, US끼리)만 비교 (KR-US 비교는 무의미)
    - 크립토는 종목이 1개뿐이라 제외 대상 아님

    반환: (채택된 종목 순서 리스트, 제외된 종목 dict {ticker: 사유})
    """
    상관_임계값 = float(settings.get("CORRELATION_MAX", 0.7))
    최대_종목수 = int(settings.get("MAX_POSITIONS", 8))

    # 종목별 최근 60일 일간 수익률 시리즈 준비
    수익률_맵 = {}
    for ticker, _, _ in 신호_종목_순서:
        d = 결과_맵.get(ticker)
        if d is None:
            continue
        market = d["market"]
        # 크립토는 비교 대상에서 제외 (1종목)
        if market in ("CRYPTO", "CRYPTO_KRW"):
            continue
        try:
            df = get_price_data(ticker, period="3mo")
            if df is not None and len(df) >= 40:
                close = df["Close"].squeeze()
                일간수익률 = close.pct_change().dropna().tail(60)
                if len(일간수익률) >= 30:
                    수익률_맵[ticker] = 일간수익률
        except Exception:
            pass

    채택 = []
    채택_수익률 = []  # (ticker, market, 수익률시리즈)
    제외 = {}

    for ticker, 점수, 세부 in 신호_종목_순서:
        d = 결과_맵.get(ticker)
        if d is None:
            continue
        market = d["market"]
        name   = d.get("name", ticker)

        # 크립토는 상관관계 필터 면제 (항상 채택)
        if market in ("CRYPTO", "CRYPTO_KRW"):
            채택.append((ticker, 점수, 세부))
            continue

        # 최대 보유 종목 수 초과 시 제외
        if len([c for c in 채택 if 결과_맵[c[0]]["market"] not in ("CRYPTO", "CRYPTO_KRW")]) >= 최대_종목수:
            제외[ticker] = f"최대 보유 {최대_종목수}종목 초과"
            continue

        내_수익률 = 수익률_맵.get(ticker)
        if 내_수익률 is None:
            # 수익률 데이터 없으면 상관관계 판단 불가 → 그냥 채택
            채택.append((ticker, 점수, 세부))
            채택_수익률.append((ticker, market, None))
            continue

        # 이미 채택된 같은 통화권 종목과 상관계수 검사
        충돌_종목 = None
        최대_상관 = 0.0
        for c_ticker, c_market, c_수익률 in 채택_수익률:
            if c_market != market or c_수익률 is None:
                continue
            # 두 시리즈를 같은 날짜로 정렬 후 상관계수
            공통 = 내_수익률.index.intersection(c_수익률.index)
            if len(공통) < 30:
                continue
            corr = float(np.corrcoef(내_수익률.loc[공통], c_수익률.loc[공통])[0, 1])
            if not np.isnan(corr) and corr >= 상관_임계값 and corr > 최대_상관:
                최대_상관 = corr
                충돌_종목 = c_ticker

        if 충돌_종목 is not None:
            충돌_이름 = 결과_맵[충돌_종목].get("name", 충돌_종목)
            제외[ticker] = f"{충돌_이름}와 상관계수 {최대_상관:.2f} (집중 위험)"
            print(f"  ⛔ {name} 상관관계 차단: {충돌_이름}와 {최대_상관:.2f}")
        else:
            채택.append((ticker, 점수, 세부))
            채택_수익률.append((ticker, market, 내_수익률))

    return 채택, 제외


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

    # ① 신호점수 (0~25): 기본 점수 + 독립 축 다양성 가중
    # [4순위] 단순 점수 합산이 아니라 '서로 다른 축이 몇 개 충족됐나'를 반영
    #         같은 종가 정보 4번 카운트한 4점 < 독립 축 3개 충족한 3점
    신호점수_원 = 결과.get("점수", 0)
    축_개수     = 결과.get("축_개수", 0)
    # 기본 환산(최대 9점 → 25점) 후, 축 다양성으로 ±보정
    기본_환산 = 신호점수_원 / 9 * 20  # 최대 20점까지만 점수로
    축_보너스 = 축_개수 / 5 * 5        # 5개 축(RS 포함) 모두 충족 시 +5점
    세부["신호점수"] = round(min(25, 기본_환산 + 축_보너스), 1)

    # ② RSI점수 (0~20): RSI 40 미만만 신호로 통과하므로 그 안에서 세분화
    # RSI 20 이하: 극단 과매도 = 20점 (최고점)
    # RSI 30 이하: 강한 과매도 = 16점
    # RSI 40 미만: 일반 과매도 = 10~14점
    # (RSI 40 이상은 signals.py에서 이미 차단되므로 0점 케이스 없음)
    rsi = 결과.get("rsi", 50)
    if rsi <= 20:    세부["rsi점수"] = 20.0
    elif rsi <= 25:  세부["rsi점수"] = round(20 - (rsi - 20) * 0.8, 1)
    elif rsi <= 30:  세부["rsi점수"] = round(16 - (rsi - 25) * 0.4, 1)
    elif rsi <= 35:  세부["rsi점수"] = round(14 - (rsi - 30) * 0.4, 1)
    else:            세부["rsi점수"] = round(12 - (rsi - 35) * 0.4, 1)

    # ③ Sharpe점수 (0~25): 백테스트 없으면 0점
    # [2순위] 거래 표본 부족(5회 미만) 시 Sharpe·승률 통계 신뢰 불가 → 0점
    #         거래 3회짜리 Sharpe 2.0은 운이지 실력이 아님
    거래횟수 = (bt or {}).get("거래횟수", 0)
    표본_충분 = 거래횟수 >= 5

    sharpe = (bt or {}).get("sharpe", 0)
    if not 표본_충분:
        세부["sharpe점수"] = 0.0
        세부["표본부족"] = True
    elif sharpe >= 2.0:    세부["sharpe점수"] = 25.0
    elif sharpe >= 1.5:  세부["sharpe점수"] = round(20 + (sharpe - 1.5) * 10, 1)
    elif sharpe >= 1.0:  세부["sharpe점수"] = round(15 + (sharpe - 1.0) * 10, 1)
    elif sharpe >= 0.5:  세부["sharpe점수"] = round(8  + (sharpe - 0.5) * 14, 1)
    else:                 세부["sharpe점수"] = 0.0

    # ④ 승률점수 (0~20): 백테스트 없거나 표본 부족이면 0점
    승률 = (bt or {}).get("승률", 0)
    if not 표본_충분:
        세부["승률점수"] = 0.0
    elif 승률 >= 70:    세부["승률점수"] = 20.0
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

    # 데이터 품질 트래커 초기화 (하이엔드) — analyze_one이 차단 종목을 집계
    try:
        import data_quality as _dq_mod
        settings["_DQ_TRACKER"] = _dq_mod.DataQualityTracker()
    except Exception:
        settings["_DQ_TRACKER"] = None

    결과_맵 = {}
    # ⑥ 수정: DART API 키 있으면 병렬 수 제한 (동시 20개 → 8개)
    # 52개 종목 × DART 동시 호출 → 503 방지
    import os as _os_tmp
    # .info() 동시 요청을 줄여 401 방지: DART 있을때 4, 없을때 MAX_WORKERS
    _workers = 4 if _os_tmp.getenv("DART_API_KEY") else MAX_WORKERS
    print(f"  ⚡ {len(종목목록)}개 종목 병렬 분석 시작 (동시 {_workers}개)...")

    # ── 병렬 분석 ───────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=_workers) as executor:
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

    # ── 모멘텀 전략 분석 (신규, 듀얼 전략) ───────────────────
    # 안정 전략과 독립. 가격 데이터는 캐시 재사용(네트워크 0).
    # K자 양극화 장에서 안정봇이 못 잡는 돌파 강세주를 포착한다.
    from momentum import calc_momentum_signal, backtest_momentum
    from market_phase import classify_market_regime

    모멘텀_결과_맵 = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        모멘텀_future = {
            executor.submit(calc_momentum_signal, s["ticker"], s["name"], s["market"], settings): s["ticker"]
            for s in 종목목록 if s["market"] in ("KR", "US")  # 크립토 제외
        }
        for future in as_completed(모멘텀_future):
            ticker = 모멘텀_future[future]
            try:
                모멘텀_결과_맵[ticker] = future.result()
            except Exception:
                모멘텀_결과_맵[ticker] = None

    # 장세 판단: 이미 분석한 결과로 breadth 집계 (네트워크 0)
    _장세입력 = [{"변동률": v.get("변동률", 0.0)}
                for v in 결과_맵.values() if v and isinstance(v, dict)]
    장세 = classify_market_regime(_장세입력, current_vix=settings.get("CURRENT_VIX"))
    print(f"  📊 장세: {장세['유형']} (breadth {장세['breadth']}%, "
          f"안정={장세['안정_가중']}/모멘텀={장세['모멘텀_가중']})")

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
        # 갭업 2~5% 구간 패널티
        if 갭 >= 2.0:
            갭_패널티 = round((갭 - 2.0) * 3, 1)
            우선순위점수 = max(0, 우선순위점수 - 갭_패널티)
        # 실적 임박 종목 우선순위 -5점 (변동성 리스크)
        fa_res = d["결과"].get("fa", {})
        if fa_res and fa_res.get("earnings", {}).get("임박_경고"):
            우선순위점수 = max(0, 우선순위점수 - 5)
        # FA 애널리스트 목표주가 괴리율 보너스 반영
        if fa_res:
            fa_보너스 = float(fa_res.get("fa_보너스", 0))
            우선순위점수 = min(100, 우선순위점수 + fa_보너스)
        신호_종목_순서.append((ticker, 우선순위점수, 우선순위세부))

    # 종합점수 높은 순으로 정렬
    신호_종목_순서.sort(key=lambda x: x[1], reverse=True)

    # ── 1순위 리스크 관리: 상관관계 기반 섹터 집중 차단 ──────────
    # 우선순위 높은 종목부터 채택하되, 이미 채택된 종목과
    # 상관계수가 높으면(같은 섹터·같은 방향) 제외
    신호_종목_순서, 상관_제외_맵 = filter_correlated_signals(
        신호_종목_순서, 결과_맵, settings
    )
    if 상관_제외_맵:
        print(f"  🔗 상관관계/집중 차단 {len(상관_제외_맵)}종목: {list(상관_제외_맵.keys())}")
        # 리포트 상단에 차단 내역 표시 (매수 섹션 앞에 추가)
        차단_줄 = f"🔗 *집중 위험 차단 {len(상관_제외_맵)}종목* (상관관계 높음 — 분산 유지)\n"
        for 차단_ticker, 차단_사유 in 상관_제외_맵.items():
            차단_이름 = 결과_맵.get(차단_ticker, {}).get("name", 차단_ticker)
            차단_줄 += f"  ⛔ {차단_이름}({차단_ticker}): {차단_사유}\n"
        차단_줄 += "─────────────────────────\n"
        매수신호_섹션 = 차단_줄 + 매수신호_섹션

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
                f"| 신호 {d['결과']['점수']} "
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
    if reason_inputs:  # ⑧ 수정: import time 상단 이동
        time.sleep(10)
    ai_reason_map = get_ai_signal_reasons_batch(reason_inputs)

    # ── 매수 신호 섹션 조립 (우선순위 정렬 순서로) ──────────
    표시된_주식_수 = 0
    크립토_rank   = 0   # 크립토 섹션 독립 번호 카운터
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
            "name":    name,
            "ticker":  ticker,
            "점수":    점수,
            "rsi":     결과["rsi"],
            "현재가":  d["현재가"],
            "market":  market,
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

        # 순위 뱃지: 크립토는 독립 번호, 주식은 전체 우선순위 기준
        if market in ("CRYPTO", "CRYPTO_KRW"):
            크립토_rank += 1
            순위_뱃지 = f"#{크립토_rank}"
        else:
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
        )

        # 독립 팩터 축 표시 (중복 제거 결과)
        # "RSI 5개 충족"이 아니라 "독립 축 N개 충족"인지를 한눈에 보여줌
        축_개수_val = 결과.get("축_개수", 0)
        추세_충족   = bool(결과.get("조건2_ma") or 결과.get("조건6_변동성돌파"))
        모멘텀_충족 = bool(결과.get("조건5_macd"))
        회귀_충족   = bool(결과.get("조건1_rsi") or 결과.get("조건4_볼린저"))
        수급_충족   = bool(결과.get("조건3_거래량"))
        _rs        = 결과.get("상대강도", {}) or {}
        rs_충족    = bool(_rs.get("아웃퍼폼"))
        축_표시 = (
            f"  • 팩터축: 추세{'✅' if 추세_충족 else '❌'} "
            f"모멘텀{'✅' if 모멘텀_충족 else '❌'} "
            f"평균회귀{'✅' if 회귀_충족 else '❌'} "
            f"수급{'✅' if 수급_충족 else '❌'} "
            f"상대강도{'✅' if rs_충족 else '❌'} "
            f"({축_개수_val}/5 독립축)\n"
        )
        신호_블록 += 축_표시

        # 상대강도(RS) 상세 — 시장 대비 초과수익
        _rs_초과 = _rs.get("rs_초과수익")
        if _rs_초과 is not None:
            _종목수익 = _rs.get("종목_수익률")
            _시장수익 = _rs.get("시장_수익률")
            _벤치명   = "S&P500" if 결과.get("market") == "US" else "KOSPI"
            _강도     = "🔥주도주" if _rs.get("강한_아웃퍼폼") else ("우위" if _rs_초과 > 0 else "열위")
            신호_블록 += (
                f"  • 📐 상대강도: {_rs_초과:+.1f}%p vs {_벤치명} "
                f"(종목 {_종목수익:+.1f}% / 시장 {_시장수익:+.1f}%, 60일) {_강도}\n"
            )

        신호_블록 += (
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

        # ── 진입 추천가 계산 및 표시 ──────────────────────────
        # [치명적 결함 수정 - 진입 타이밍]
        # 신호 발생 = 지금 당장 시가 매수가 아님
        # 하락 눌림목에서 반등을 노리는 전략이므로
        # "어디서 사야 하는가"를 계산해서 표시
        #
        # 진입 추천가 계산 방식:
        # 기준1. 볼린저 밴드 하단 (통계적 저점 지지선)
        # 기준2. 현재가 - ATR (하루 변동폭만큼 눌렸을 때)
        # 두 값 중 현재가에 더 가까운(높은) 쪽을 추천 진입가로 제시
        # → 너무 낮은 지정가를 피하고 실제 체결 가능한 수준 제시
        bb_하단 = 결과.get("bb_하단", 0)
        atr_val  = atr if atr and atr > 0 else 0

        진입가_bb  = bb_하단 if bb_하단 and bb_하단 > 0 else 0
        진입가_atr = (현재가 - atr_val) if atr_val > 0 else 0

        # 현재가 이하에서 가장 높은 값 선택 (체결 가능성 최대화)
        후보들 = [v for v in [진입가_bb, 진입가_atr] if 0 < v < 현재가]
        if 후보들:
            추천_진입가 = max(후보들)
            진입가_표시 = f"{통화_기호}{추천_진입가:,.0f}"
            # 현재가 대비 괴리율
            괴리율 = (추천_진입가 - 현재가) / 현재가 * 100
            if abs(괴리율) < 0.5:
                진입_메모 = "현재가 근처 (즉시 진입 가능)"
            else:
                진입_메모 = f"현재가 대비 {괴리율:+.1f}% (지정가 대기)"
            신호_블록 += (
                f"  • 📌 진입 추천가: {진입가_표시} ({진입_메모})\n"
                f"     └ 볼린저하단 {통화_기호}{진입가_bb:,.0f} | "
                f"현재가-ATR {통화_기호}{진입가_atr:,.0f}\n"
            )
        # 갭업/갭다운 경고
        if abs(갭_비율) >= 2.0:
            갭_방향 = "갭업" if 갭_비율 > 0 else "갭다운"
            갭_아이콘 = "⚠️" if 갭_비율 > 0 else "🔻"
            신호_블록 += (
                f"  {갭_아이콘} {갭_방향} {갭_비율:+.1f}% 감지 — "
                f"{'고점 진입 주의, 눌림목 기다릴 것 권장' if 갭_비율 > 0 else '추가 하락 가능성, 분할 매수 권장'}\n"
            )

        # ── 기본적 분석 표시 ────────────────────────────────
        fa = 결과.get("fa", {})
        실적_임박 = 결과.get("실적_임박", False)
        if fa:
            fa_표시 = fa.get("fa_표시", "")
            if fa_표시:
                신호_블록 += f"  • 📊 FA: {fa_표시}\n"
            elif market in ("US",) and not fa_표시:
                신호_블록 += "  • 📊 FA: 기본적 분석 조회 실패 (yfinance 인증 오류)\n"
            if 실적_임박:
                ed = fa.get("earnings", {}).get("표시문구", "")
                신호_블록 += f"  {ed}\n" if ed else ""

        # ── 기관/외국인 수급 표시 (한국 주식 전용) ────────────
        수급 = 결과.get("수급", {})
        if market == "KR" and 수급 and not 수급.get("데이터_없음"):
            수급_문구 = 수급.get("표시문구", "")
            외국인_연속 = 수급.get("외국인_연속", 0)
            기관_연속   = 수급.get("기관_연속", 0)
            동시순매수  = 수급.get("동시순매수", False)

            # 아이콘: 동시순매수(💚) / 외국인만(🟦) / 기관만(🟧) / 매도(🔻)
            if 동시순매수:
                수급_아이콘 = "💚"
            elif 외국인_연속 > 0 or 기관_연속 > 0:
                수급_아이콘 = "📈"
            elif 외국인_연속 < 0 and 기관_연속 < 0:
                수급_아이콘 = "🔻"
            else:
                수급_아이콘 = "➡️"

            if 수급_문구:
                신호_블록 += f"  • {수급_아이콘} 수급: {수급_문구}\n"
        elif market == "KR" and (not 수급 or 수급.get("데이터_없음")):
            신호_블록 += "  • ➡️ 수급: 데이터 조회 실패 (KRX/네이버 API)\n"

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
                번역필요_flags.append(source in ("alphavantage", "yfinance", "gnews", "newsapi"))

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
    # 매수 신호가 발생한 종목은 매도 섹션에서 제외 (동시 표출 방지)
    매수_티커_셋 = {t for t, _, _ in 신호_종목_순서}

    for 종목 in 종목목록:
        ticker = 종목["ticker"]
        d = 매도_결과_맵.get(ticker)
        if d is None:
            continue
        # 포트폴리오에 없는 종목은 매도 신호 생략
        if 보유_티커_셋 and ticker not in 보유_티커_셋:
            continue
        # 매수 신호가 동시에 뜬 종목은 매도 섹션에서 제외
        if ticker in 매수_티커_셋:
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
                sell_번역필요.append(source in ("alphavantage", "yfinance", "gnews", "newsapi"))

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

    # ── ⚡ 모멘텀 전략 섹션 조립 (듀얼 전략) ─────────────────
    모멘텀_섹션 = _build_momentum_section(
        모멘텀_결과_맵, 장세, settings, backtest_momentum
    )

    return (매수신호_섹션, 크립토_섹션, 매도신호_섹션,
            신호_종목_요약, 모멘텀_섹션, 장세)


def _build_momentum_section(모멘텀_결과_맵, 장세, settings, backtest_momentum) -> str:
    """
    모멘텀 신호를 리포트 텍스트로 조립한다.

    장세가 '끔'(횡보·하락)이면 모멘텀 섹션을 비활성으로 표시하고,
    충족 종목만 백테스트(Sharpe<0 차단) 후 우선순위로 정렬해 보여준다.
    """
    # 장세 진단 배너 (항상 표시)
    유형_아이콘 = {"K자양극화": "⚡", "추세장": "📈",
                 "횡보장": "➖", "하락장": "📉"}.get(장세["유형"], "•")
    배너 = (
        f"\n━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *장세 진단: {유형_아이콘} {장세['유형']}*\n"
        f"  breadth {장세['breadth']}% "
        f"(상승 {장세['상승종목']} / 하락 {장세['하락종목']}) | VIX {장세['vix']}\n"
        f"  🛡️ 안정 {장세['안정_가중']} / ⚡ 모멘텀 {장세['모멘텀_가중']}\n"
        f"  💡 {장세['한줄진단']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    # 모멘텀이 '끔'이면 비활성 안내만
    if 장세["모멘텀_가중"] == "끔":
        return (배너 +
                f"\n⚡ *모멘텀 전략 — 비활성*\n"
                f"  현재 장세({장세['유형']})에서는 가짜 돌파가 많아 "
                f"모멘텀 전략을 끕니다.\n")

    # 충족 종목 추출
    충족종목 = [(t, r) for t, r in 모멘텀_결과_맵.items()
              if r and r.get("충족")]

    if not 충족종목:
        return (배너 +
                f"\n⚡ *모멘텀 신호 — 0개* ⚠️고위험·참고용\n"
                f"  현재 돌파 조건(5/5)을 만족하는 종목이 없습니다.\n")

    # 백테스트 + Sharpe<0 차단 → 우선순위(당일상승률×RS) 정렬
    검증된 = []
    for t, r in 충족종목:
        bt = backtest_momentum(t, r["market"], settings, period_months=3)
        if bt and bt["sharpe"] < 0:
            continue  # 백테스트 손실 전략 차단
        우선 = (r.get("당일상승률", 0) or 0) + (r.get("rs초과", 0) or 0)
        검증된.append((t, r, bt, 우선))
    검증된.sort(key=lambda x: x[3], reverse=True)

    if not 검증된:
        return (배너 +
                f"\n⚡ *모멘텀 신호 — 0개* ⚠️고위험·참고용\n"
                f"  돌파 종목은 있으나 백테스트 검증(Sharpe<0)을 통과하지 못했습니다.\n")

    섹션 = (배너 +
            f"\n⚡ *모멘텀 신호 — {len(검증된)}개* ⚠️고위험·참고용·단타\n"
            f"─────────────────────────\n")

    from strategy_utils import get_currency_symbol
    메달 = ["🥇", "🥈", "🥉"] + ["•"] * 10
    for i, (t, r, bt, _) in enumerate(검증된[:5]):
        cur = get_currency_symbol(r["market"])
        섹션 += (
            f"{메달[i]} *{r['name']}* ({t})\n"
            f"  • 진입트리거: 20일신고가 {r['신고가비율']}% | "
            f"당일 +{r['당일상승률']}% | 거래량 {r['거래량배수']}배 | "
            f"RS {r['rs초과']:+.1f}%p ({r['충족수']}/5)\n"
            f"  • 📌 진입(돌파추격): {cur}{r['진입추천가']:,.0f} → "
            f"익절 +5% {cur}{r['익절가']:,.0f} / 손절 -2.5% {cur}{r['손절가']:,.0f}\n"
            f"  • ⏱️ 보유 시한: {2}일 내 청산 (단타)\n"
        )
        if bt:
            섹션 += (
                f"  • 📊 백테스트(3개월·2일보유): {bt['수익률']:+.1f}% | "
                f"MDD {bt['mdd']:.1f}% | Sharpe {bt['sharpe']:.2f} | "
                f"승률 {bt['승률']:.0f}% | 거래 {bt['거래횟수']}회\n"
            )
        섹션 += "─────────────────────────\n"

    섹션 += (
        f"⚠️ 단타는 백테스트-실전 괴리가 큽니다(슬리피지·갭). "
        f"손절 -2.5% 엄수, 페이퍼 검증 후 소액 사용.\n"
    )
    return 섹션


def _record_signals(신호_종목_요약: list):
    """
    신호 발생 종목을 portfolio_signals.txt에 자동 기록한다.

    [형식] 날짜,티커,종목명,신호점수,진입예상가,통화,보유일수,상태
    [상태] 신호발생 → 보유중(수동 변경) → 익절/손절/기간초과청산
    보유중으로 상태를 직접 변경하면 30일 카운트 시작 → 기간초과 알림
    """
    if not 신호_종목_요약:
        return
    try:
        오늘 = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
        import os as _os2
        _sp = _os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)),
                             "portfolio_signals.txt")
        with open(_sp, "a", encoding="utf-8") as f:
            for s in 신호_종목_요약:
                ticker = s.get("ticker",  "")
                name   = s.get("name",    "")
                점수   = s.get("점수",    0)
                진입가 = round(float(s.get("현재가", 0)), 2)
                통화   = "KRW" if s.get("market","") in ("KR","CRYPTO_KRW") else "USD"
                f.write(f"{오늘},{ticker},{name},{점수},{진입가},{통화},0,신호발생\n")
        print(f"  📝 신호 {len(신호_종목_요약)}개 기록 완료 (portfolio_signals.txt)")
    except Exception as e:
        print(f"  ⚠️ 신호 기록 실패: {e}")


def run_analysis(include_crypto=True, include_markets=None):
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

    include_markets: 분석할 시장 목록 (예: ["KR", "CRYPTO", "CRYPTO_KRW"])
                     None이면 전체 분석 (기존 동작 유지)
    """
    # GitHub Actions는 UTC 기준으로 실행됨 → KST(UTC+9)로 변환
    지금 = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M KST")
    print(f"\n🔍 분석 시작: {지금}")

    clear_cache()  # 이전 실행에서 남은 캐시 삭제 (최신 데이터 보장)

    settings   = load_settings()
    종목목록   = load_stocks()
    포트폴리오 = load_portfolio()

    # 시장 필터: include_markets 지정 시 해당 시장 종목만 분석
    if include_markets is not None:
        종목목록 = [s for s in 종목목록 if s["market"] in include_markets]
        include_crypto = any(m in include_markets for m in ("CRYPTO", "CRYPTO_KRW"))
        print(f"  🎯 시장 필터 적용: {include_markets} → {len(종목목록)}개 종목")

    if not include_crypto:
        종목목록 = [s for s in 종목목록 if s["market"] not in ("CRYPTO", "CRYPTO_KRW")]

    # 데이터 사전 다운로드 (종목별로 따로 받으면 느리니까 한꺼번에 받음)
    모든_티커 = [s["ticker"] for s in 종목목록]
    레짐_티커 = ["^GSPC", "^KS11", "^VIX", "BTC-USD"]
    전체_티커 = list(set(모든_티커 + 레짐_티커))

    print(f"  🚀 사전 일괄 다운로드 시작 (총 {len(전체_티커)}개 티커)...")
    # crumb 사전 획득: 병렬 .info() 호출 전에 yfinance 세션 워밍업
    _warmup_yfinance_crumb()
    # crumb이 Yahoo 서버에 완전히 등록되도록 3초 대기
    # 너무 빨리 bulk_download를 시작하면 crumb 없는 요청이 섞여 401 발생
    time.sleep(3)
    bulk_download(전체_티커, period="1y")
    bulk_download(전체_티커, period="3mo")
    bulk_download(전체_티커, period="5d")
    print(f"  ✅ 사전 다운로드 완료")

    # 주간봉 데이터 일괄 사전 다운로드 (실행 시간 단축)
    # 기존: calc_signals()가 종목마다 개별 다운로드 → 느림
    # 개선: 한 번에 다운로드 → 이후 캐시에서 즉시 반환
    print("  🚀 주간봉 데이터 사전 다운로드 중...")
    bulk_download_weekly(전체_티커, period="2y")
    print("  ✅ 주간봉 다운로드 완료")

    # ── VIX 1회 조회 → settings에 주입 (중복 다운로드 방지) ────
    # [v5.9 수정] ^VIX가 GitHub Actions IP에서 차단되는 문제 발견 (로그: "possibly delisted")
    # ^VIX는 폐지된 것이 아니라 yfinance 인덱스 API 엔드포인트가 GitHub IP에서 막힌 것.
    # FRED(미국 연방준비은행) 공개 API를 폴백으로 추가 → IP 차단 없음, 키 불필요.
    # 조회 순서: yfinance ^VIX → FRED VIXCLS → 기본값 15.0
    settings["CURRENT_VIX"] = 15.0  # 기본값 (모두 실패 시)

    # 1순위: yfinance ^VIX (캐시에 있으면 빠름)
    try:
        _vix_df = get_price_data("^VIX", period="5d")
        if _vix_df is not None and len(_vix_df) > 0:
            _vix_val = float(_vix_df["Close"].squeeze().iloc[-1])
            if _vix_val > 0:
                settings["CURRENT_VIX"] = _vix_val
    except Exception:
        pass

    # 2순위: FRED VIXCLS API (IP 차단 없음, 키 불필요) — data_loader 헬퍼 재사용
    _vix_src = "yfinance"
    if settings["CURRENT_VIX"] == 15.0:
        _fred_vix = get_vix_from_fred()
        if _fred_vix is not None and _fred_vix > 0:
            settings["CURRENT_VIX"] = _fred_vix
            _vix_src = "FRED"
        else:
            _vix_src = "기본값"

    print(f"  💹 VIX: {settings['CURRENT_VIX']:.1f} ({_vix_src})")

    # ── 벤치마크 지수 1회 조회 → settings에 주입 (상대강도 RS 계산용) ──
    # VIX와 동일한 패턴: 103종목마다 지수를 개별 다운로드하지 않고
    # 여기서 1회 받아 캐시. KR=KOSPI(^KS11), US=S&P500(^GSPC)
    # calc_signals()가 종목 시장에 맞는 벤치마크 종가를 꺼내 RS를 계산한다.
    try:
        _ks_df = get_price_data("^KS11", period="1y")
        settings["BENCH_KR_CLOSE"] = (
            _ks_df["Close"].squeeze()
            if _ks_df is not None and len(_ks_df) > 60 else None
        )
    except Exception:
        settings["BENCH_KR_CLOSE"] = None
    try:
        _sp_df = get_price_data("^GSPC", period="1y")
        settings["BENCH_US_CLOSE"] = (
            _sp_df["Close"].squeeze()
            if _sp_df is not None and len(_sp_df) > 60 else None
        )
    except Exception:
        settings["BENCH_US_CLOSE"] = None
    _kr_ok = "OK" if settings.get("BENCH_KR_CLOSE") is not None else "X"
    _us_ok = "OK" if settings.get("BENCH_US_CLOSE") is not None else "X"
    print(f"  ┐ 상대강도 벤치마크: KOSPI {_kr_ok} / S&P500 {_us_ok}")

    # ── 시장 국면 감지 → 동적 임계값 설정 ──────────────────────
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
        # 환율 정보 표시
        환율_표시 = ""
        if phase_result.usdkrw > 0:
            환율_아이콘 = "📈" if phase_result.usdkrw_trend == "상승" else (
                          "📉" if phase_result.usdkrw_trend == "하락" else "➡️")
            환율_표시 = f" | 💱 {phase_result.usdkrw:,.0f}원 {환율_아이콘}"
        리포트 += (
            f"시장 국면: {phase_result.phase.value} "
            f"| 매수 임계값: {phase_result.score_threshold}점{환율_표시}\n"
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
    (매수신호_섹션, 크립토_섹션, 매도신호_섹션, 신호_종목_요약,
     모멘텀_섹션, 장세) = build_report_sections(
        종목목록, settings, 크립토_하락_레짐, 포트폴리오
    )

    # AI 시장 코멘트 — 신호이유 배치 후 5초 대기 (429 방지)
    time.sleep(5)  # ⑧ 수정: 상단 import time 사용
    ai_코멘트 = get_ai_market_commentary(
        하락_레짐, 크립토_하락_레짐, 경고_목록, 신호_종목_요약
    )
    if ai_코멘트:
        리포트 += ai_코멘트
        리포트 += "─────────────────────────\n"

    # 🛡️ 안정 전략 섹션 (헤더를 항상 표시 → 모멘텀 ⚡ 섹션과 대칭)
    # 신호가 없을 때도 "오류가 아니라 정상적으로 없음"임을 명확히 알린다.
    리포트 += "\n🛡️ *안정 전략 — 눌림목·중기 (매수 신호)*\n"
    if 매수신호_섹션:
        리포트 += 매수신호_섹션
    else:
        if 하락_레짐:
            이유 = "⚠️ 대세 하락장 — 현금 비중을 높이고 관망을 추천합니다."
        elif 크립토_하락_레짐:
            이유 = "현재 안정 매수 조건(과매도 + 추세 정배열)을 만족하는 종목이 없습니다."
        else:
            이유 = "현재 안정 매수 조건(과매도 + 추세 정배열)을 만족하는 종목이 없습니다."
        리포트 += (
            f"✅ 안정 매수 신호 없음 (오류 아님 — 조건 미충족)\n"
            f"   {이유}\n"
            f"   📌 안정 전략은 무리하게 진입하지 않는 것이 정상 동작입니다.\n"
        )
        리포트 += "─────────────────────────\n"

    # ⚡ 모멘텀 전략 섹션 (듀얼 전략, 장세 배너 포함)
    if 모멘텀_섹션:
        리포트 += 모멘텀_섹션

    # 📈 실전 성과 추적 (하이엔드): 누적 승률 표시
    try:
        import performance_tracker as _pt
        _summary = _pt.get_performance_summary(최근일수=90)
        리포트 += "\n" + _pt.format_performance_line(_summary)
        리포트 += "─────────────────────────\n"
    except Exception:
        pass

    # 매도 신호 섹션 (신규)
    if 매도신호_섹션:
        리포트 += "📉 *매도 신호 종목*\n"
        리포트 += 매도신호_섹션

    # 포트폴리오 알림 섹션
    포트폴리오_알림 = check_portfolio_alerts(포트폴리오, settings)

    # 최대 보유일(30일) 초과 종목 알림
    보유일_알림 = check_max_holding_days(settings)
    포트폴리오_알림.extend(보유일_알림)

    # 주간 성과 리포트 (일요일에만 자동 발송)
    주간_리포트_알림 = generate_weekly_report(settings)
    포트폴리오_알림.extend(주간_리포트_알림)
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
    _record_signals(신호_종목_요약)

    # ── 성과 추적 시스템 (하이엔드): 신호 기록 + 이전 신호 채점 ──────
    # 인라인 로직은 performance_tracker.record_and_grade로 추출 (v5.19 리팩토링)
    try:
        import performance_tracker as _pt
        _채점 = _pt.record_and_grade(신호_종목_요약, 모멘텀_결과_맵, get_price_data)
        if _채점["채점"] > 0:
            print(f"  📈 성과 채점: {_채점['채점']}건 "
                  f"({_채점['승']}승 {_채점['패']}패 {_채점['중립']}중립)")
    except Exception as _e:
        print(f"  ⚠️ 성과 추적 오류: {_e}")

    # ── 수급 소스 진단 요약 1회 출력 (한국 flow 축 작동 여부 판별) ──
    try:
        from fundamental import get_supply_demand_diagnostics
        print(get_supply_demand_diagnostics())
    except Exception:
        pass

    # ── 데이터 품질 요약 출력 (하이엔드) ──────────────────────────
    try:
        _dq_t = settings.get("_DQ_TRACKER")
        if _dq_t is not None:
            _dq_summary = _dq_t.summary()
            if _dq_summary:
                print(_dq_summary)
    except Exception:
        pass

    # ── 실행 헬스 체크 + 연속 실패 감지 (하이엔드 관측성) ──────────
    # 수급 진단 결과를 헬스 추적에 반영하고, 3일 연속 실패 소스를 경고.
    헬스_경고 = []
    try:
        import observability as _obs
        from fundamental import _supply_demand_source_count as _sdc
        # 수급·VIX 헬스 반영 (observability.record_data_source_health로 추출)
        _obs.record_data_source_health(_sdc, settings.get("CURRENT_VIX", 15.0))
        print(_obs.health.summary())
        헬스_경고 = _obs.health.persist_and_check_streaks(streak_threshold=3)
        for _w in 헬스_경고:
            print(f"  {_w}")
        # 연속 실패 경고는 리포트(텔레그램)에도 노출 → 즉시 인지
        if 헬스_경고:
            리포트 += "\n⚠️ *시스템 경고*\n"
            for _w in 헬스_경고:
                리포트 += f"  {_w}\n"
            리포트 += "─────────────────────────\n"
    except Exception as _e:
        print(f"  ⚠️ 헬스 체크 오류: {_e}")

    # ── HTML 리포트 생성 + 링크 파일 저장 ───────────────────────
    # GH_PAGES_URL이 있으면 텔레그램 전송을 workflow로 위임.
    # 이유: python main.py 실행 직후 git push가 되고, Pages 배포는 1~2분 더 걸림.
    #       링크를 먼저 보내면 아직 파일이 없어서 404가 남.
    #       git push 완료 후 workflow에서 링크를 전송해야 정상 접속됨.
    gh_pages_url = os.environ.get("GH_PAGES_URL", "").rstrip("/")
    try:
        from report_generator import generate_html_report
        phase_str = phase_result.phase.value if phase_result else ""
        파일명, _ = generate_html_report(리포트, phase_str, 지금)

        if gh_pages_url:
            # 링크를 파일로 저장 → workflow의 텔레그램 전송 step에서 읽어서 전송
            링크_메시지 = (
                f"⚔️ 퀀트 리포트 준비됐습니다\n"
                f"📊 [{파일명}]({gh_pages_url}/{파일명})\n"
                f"📋 [전체 목록]({gh_pages_url}/index.html)\n"
                f"🕐 {지금}"
            )
            with open("docs/.pending_link.txt", "w", encoding="utf-8") as f:
                f.write(링크_메시지)
            print(f"  🌐 HTML 링크 저장 (git push 후 전송 예정): {gh_pages_url}/{파일명}")
            # GH_PAGES_URL이 있으면 텔레그램에 텍스트 리포트 전송 안 함
            # (링크만 전송 — workflow에서 처리)
            print("✅ 분석 완료 (텔레그램 전송은 git push 후 workflow에서 처리)")
            return
        else:
            # GH_PAGES_URL 없으면 기존처럼 텍스트 리포트 전송
            리포트 += (
                f"\n─────────────────────────\n"
                f"📄 HTML 리포트 생성됨 (링크 미설정)\n"
            )
    except Exception as e:
        print(f"  ⚠️ HTML 리포트 생성 실패 (무시): {e}")

    # GH_PAGES_URL 없을 때만 여기 도달 (텍스트 리포트 전송)
    send_telegram(리포트)
    print("✅ 분석 완료 및 리포트 전송")
