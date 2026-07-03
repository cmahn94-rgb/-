"""
performance_tracker.py — 신호 성과 추적 시스템 (하이엔드 1순위)
================================================================
[목적]
봇이 낸 신호가 '실제로 맞았는지'를 기록하고 채점한다.
백테스트(과거 시뮬레이션)와 별개로, 실전에서 발생한 신호의
사후 결과를 누적해 "이 전략의 실제 승률"을 측정한다.

[중학생 설명]
지금까진 "이 종목 사라"고 알림만 보내고 끝이었다. 그게 맞았는지 틀렸는지
아무도 안 봤다. 이제는 신호를 낼 때마다 기록해두고, 며칠 뒤 실제 주가로
"목표 도달했나? 손절됐나?"를 채점한다. 그러면 "안정 전략 실제 승률 63%"
같은 진짜 성적표가 나온다. 이게 있어야 전략을 데이터로 개선할 수 있다.

[동작]
  1. 신호 발생 시 → record_signals()로 signal_log.json에 기록
  2. 다음 실행 시 → grade_pending_signals()로 미채점 신호를 현재가로 채점
  3. 리포트에 → get_performance_summary()로 누적 성적표 표시

[채점 규칙]
  - 안정 전략: 목표가(+12%) 도달=승, 손절가(-5%) 도달=패, 30일 초과=중립청산
  - 모멘텀 전략: 익절(+5%) 도달=승, 손절(-2.5%) 도달=패, 2일 초과=시한청산
  - 판정은 기록된 진입가 대비 '이후 고가/저가'로 결정

[저장 형식] signal_log.json
  {"signals": [{id, 날짜, ticker, name, market, 전략, 진입가, 목표가, 손절가,
                보유상한일, 상태, 채점일, 결과, 수익률}, ...]}
"""

import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_KST = ZoneInfo("Asia/Seoul")
_LOG_FILENAME = "signal_log.json"


def _log_path() -> str:
    """signal_log.json의 절대 경로 (이 파일과 같은 폴더)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _LOG_FILENAME)


def _load_log() -> dict:
    """로그 파일을 읽는다. 없으면 빈 구조 반환."""
    path = _log_path()
    if not os.path.exists(path):
        return {"signals": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "signals" not in data:
            data["signals"] = []
        return data
    except Exception:
        return {"signals": []}


def _save_log(data: dict) -> bool:
    """로그 파일을 저장한다."""
    try:
        with open(_log_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def record_signals(신호목록: list, 전략: str = "안정") -> int:
    """
    신호를 로그에 기록한다. (중복 방지: 같은 날·같은 종목·같은 전략은 1회)

    Args:
        신호목록: [{ticker, name, market, 진입가, 목표가, 손절가, 보유상한일}, ...]
        전략: "안정" 또는 "모멘텀"

    반환: 새로 기록한 신호 수
    """
    if not 신호목록:
        return 0

    data = _load_log()
    오늘 = datetime.now(_KST).strftime("%Y-%m-%d")

    # 중복 체크용 기존 키 집합
    기존키 = {
        f"{s['날짜']}|{s['ticker']}|{s['전략']}"
        for s in data["signals"]
    }

    추가 = 0
    for sig in 신호목록:
        ticker = sig.get("ticker", "")
        if not ticker:
            continue
        키 = f"{오늘}|{ticker}|{전략}"
        if 키 in 기존키:
            continue

        data["signals"].append({
            "id":         f"{오늘}_{ticker}_{전략}",
            "날짜":       오늘,
            "ticker":     ticker,
            "name":       sig.get("name", ""),
            "market":     sig.get("market", ""),
            "전략":       전략,
            "진입가":     round(float(sig.get("진입가", 0)), 2),
            "목표가":     round(float(sig.get("목표가", 0)), 2),
            "손절가":     round(float(sig.get("손절가", 0)), 2),
            "보유상한일": int(sig.get("보유상한일", 30 if 전략 == "안정" else 2)),
            "상태":       "보유중",
            "채점일":     None,
            "결과":       None,      # "승"/"패"/"중립"
            "수익률":     None,
        })
        기존키.add(키)
        추가 += 1

    if 추가:
        _save_log(data)
    return 추가


def grade_pending_signals(가격조회함수) -> dict:
    """
    미채점('보유중') 신호를 현재가로 채점한다.

    Args:
        가격조회함수: ticker를 받아 최근 가격 DataFrame(High/Low/Close)을
                     반환하는 함수 (data_loader.get_price_data 등).
                     캐시를 쓰므로 추가 네트워크 부담 적음.

    반환: {"채점": N, "승": N, "패": N, "중립": N}
    """
    data = _load_log()
    오늘_dt = datetime.now(_KST)
    통계 = {"채점": 0, "승": 0, "패": 0, "중립": 0, "알림": []}

    for sig in data["signals"]:
        if sig["상태"] != "보유중":
            continue

        try:
            발생일 = datetime.strptime(sig["날짜"], "%Y-%m-%d").replace(tzinfo=_KST)
        except Exception:
            continue
        경과일 = (오늘_dt - 발생일).days
        # 보유상한 전이면 아직 채점하지 않음 (단, 목표/손절 도달은 조기 채점)
        상한도달 = 경과일 >= sig["보유상한일"]

        df = None
        try:
            df = 가격조회함수(sig["ticker"], period="3mo")
        except Exception:
            df = None
        if df is None or len(df) == 0:
            continue

        # 발생일 이후 구간의 고가/저가로 판정 (v5.20: 날짜 인덱스 직접 비교로 정밀화)
        try:
            close = df["Close"].squeeze()
            high  = df["High"].squeeze() if "High" in df.columns else close
            low   = df["Low"].squeeze()  if "Low"  in df.columns else close
            현재 = float(close.iloc[-1])

            # 인덱스가 날짜형이면 발생일 이후만 정확히 필터링
            발생일_naive = 발생일.replace(tzinfo=None)
            mask = None
            try:
                idx = close.index
                if hasattr(idx, "tz") and idx.tz is not None:
                    idx_cmp = idx.tz_localize(None)
                else:
                    idx_cmp = idx
                mask = idx_cmp > 발생일_naive
            except Exception:
                mask = None

            if mask is not None and hasattr(mask, "sum") and mask.sum() > 0:
                이후_고 = float(high[mask].max())
                이후_저 = float(low[mask].min())
            else:
                # 폴백: 경과일 기반 tail 근사
                n = max(1, 경과일)
                이후_고 = float(high.tail(n).max())
                이후_저 = float(low.tail(n).min())
        except Exception:
            continue

        진입 = sig["진입가"]
        목표 = sig["목표가"]
        손절 = sig["손절가"]
        if 진입 <= 0:
            continue

        결과 = None
        # 목표 도달 (고가가 목표 이상)
        if 목표 > 0 and 이후_고 >= 목표:
            결과 = "승"; 수익률 = (목표 / 진입 - 1) * 100
        # 손절 도달 (저가가 손절 이하)
        elif 손절 > 0 and 이후_저 <= 손절:
            결과 = "패"; 수익률 = (손절 / 진입 - 1) * 100
        # 보유상한 초과 → 현재가로 중립 청산
        elif 상한도달:
            수익률 = (현재 / 진입 - 1) * 100
            결과 = "승" if 수익률 > 0 else ("패" if 수익률 < 0 else "중립")
            결과 = "중립" if abs(수익률) < 0.5 else 결과

        if 결과:
            sig["상태"]   = "채점완료"
            sig["채점일"] = 오늘_dt.strftime("%Y-%m-%d")
            sig["결과"]   = 결과
            sig["수익률"] = round(수익률, 2)
            통계["채점"] += 1
            통계[결과] = 통계.get(결과, 0) + 1
            # 실시간 알림용: 방금 채점된 신호 상세 (v5.20)
            통계["알림"].append({
                "name": sig.get("name", sig["ticker"]),
                "ticker": sig["ticker"], "전략": sig.get("전략", ""),
                "결과": 결과, "수익률": round(수익률, 2),
            })

    if 통계["채점"]:
        _save_log(data)
    return 통계


def get_performance_summary(최근일수: int = 90) -> dict:
    """
    누적 성과 통계를 전략별로 반환한다. (리포트 표시용)

    Args:
        최근일수: 이 기간 내 채점된 신호만 집계 (기본 90일)

    반환:
      {
        "안정": {"채점수","승","패","중립","승률","평균수익률"},
        "모멘텀": {...},
        "전체_채점": N, "보유중": N,
      }
    """
    data = _load_log()
    cutoff = datetime.now(_KST) - timedelta(days=최근일수)

    결과 = {
        "안정":   {"채점수": 0, "승": 0, "패": 0, "중립": 0, "승률": 0.0, "평균수익률": 0.0},
        "모멘텀": {"채점수": 0, "승": 0, "패": 0, "중립": 0, "승률": 0.0, "평균수익률": 0.0},
        "전체_채점": 0, "보유중": 0,
    }
    수익_합 = {"안정": 0.0, "모멘텀": 0.0}

    for sig in data["signals"]:
        전략 = sig.get("전략", "안정")
        if 전략 not in 결과:
            continue

        if sig["상태"] == "보유중":
            결과["보유중"] += 1
            continue
        if sig["상태"] != "채점완료" or not sig.get("채점일"):
            continue

        try:
            채점일 = datetime.strptime(sig["채점일"], "%Y-%m-%d").replace(tzinfo=_KST)
            if 채점일 < cutoff:
                continue
        except Exception:
            continue

        r = 결과[전략]
        r["채점수"] += 1
        결과["전체_채점"] += 1
        결과_값 = sig.get("결과", "중립")
        r[결과_값] = r.get(결과_값, 0) + 1
        수익_합[전략] += float(sig.get("수익률", 0) or 0)

    for 전략 in ("안정", "모멘텀"):
        r = 결과[전략]
        판정 = r["승"] + r["패"]  # 중립 제외한 승패
        if 판정 > 0:
            r["승률"] = round(r["승"] / 판정 * 100, 1)
        if r["채점수"] > 0:
            r["평균수익률"] = round(수익_합[전략] / r["채점수"], 2)

    return 결과


def format_performance_line(summary: dict) -> str:
    """성과 요약을 리포트용 텍스트 한 블록으로 변환."""
    안정 = summary["안정"]
    모멘텀 = summary["모멘텀"]
    if summary["전체_채점"] == 0:
        return (
            "📈 *실전 성과 추적* (누적)\n"
            f"  아직 채점된 신호가 없습니다 (보유중 {summary['보유중']}건).\n"
            "  신호 발생 후 목표/손절 도달 또는 보유상한 경과 시 채점됩니다.\n"
        )
    줄 = ["📈 *실전 성과 추적* (최근 90일 채점 기준)"]
    if 안정["채점수"] > 0:
        줄.append(
            f"  🛡️ 안정: 승률 {안정['승률']}% "
            f"({안정['승']}승 {안정['패']}패 {안정['중립']}중립) | "
            f"평균 {안정['평균수익률']:+.1f}%"
        )
    if 모멘텀["채점수"] > 0:
        줄.append(
            f"  ⚡ 모멘텀: 승률 {모멘텀['승률']}% "
            f"({모멘텀['승']}승 {모멘텀['패']}패 {모멘텀['중립']}중립) | "
            f"평균 {모멘텀['평균수익률']:+.1f}%"
        )
    줄.append(f"  📌 보유중(미채점) {summary['보유중']}건")
    return "\n".join(줄) + "\n"


def record_and_grade(신호_종목_요약: list, 모멘텀_결과_맵: dict,
                     가격조회함수) -> dict:
    """
    신호 기록 + 채점을 한 번에 처리하는 통합 헬퍼. (scheduler_job 호출용)

    scheduler_job의 인라인 코드를 여기로 옮겨 run_analysis를 깔끔하게 유지.
    안정 신호(신호_종목_요약)와 모멘텀 신호(충족 종목)를 각각 형식 변환해
    기록한 뒤, 이전 신호를 채점한다.

    반환: grade_pending_signals의 채점 통계 {"채점","승","패","중립"}
    """
    # 안정 신호 → 성과추적 형식
    안정_기록 = [{
        "ticker": s.get("ticker", ""), "name": s.get("name", ""),
        "market": s.get("market", ""),
        "진입가": s.get("현재가", 0),
        "목표가": s.get("목표가", 0) or (s.get("현재가", 0) * 1.12),
        "손절가": s.get("손절가", 0) or (s.get("현재가", 0) * 0.95),
        "보유상한일": 30,
    } for s in (신호_종목_요약 or [])]
    record_signals(안정_기록, "안정")

    # 모멘텀 신호 (충족 종목만) → 성과추적 형식
    모멘텀_기록 = [{
        "ticker": t, "name": r.get("name", ""), "market": r.get("market", ""),
        "진입가": r.get("진입추천가", 0),
        "목표가": r.get("익절가", 0),
        "손절가": r.get("손절가", 0),
        "보유상한일": 2,
    } for t, r in (모멘텀_결과_맵 or {}).items() if r and r.get("충족")]
    record_signals(모멘텀_기록, "모멘텀")

    # 이전 신호 채점 (캐시된 가격 재사용)
    return grade_pending_signals(가격조회함수)


def format_grading_alerts(채점통계: dict) -> str:
    """
    방금 목표/손절에 도달한 신호를 리포트용 알림 텍스트로 만든다. (v5.20)

    [중학생 설명]
    "3일 전 추천한 삼성전자가 목표가 도달! +12%" 같은 실시간 결과 알림.
    실제 매매에 바로 도움이 되도록, 채점되는 순간 리포트에 띄운다.
    """
    알림 = 채점통계.get("알림", [])
    if not 알림:
        return ""
    줄 = ["🔔 *신호 결과 업데이트*"]
    for a in 알림[:8]:
        아이콘 = "🎯" if a["결과"] == "승" else ("🛑" if a["결과"] == "패" else "⏹️")
        전략아이콘 = "🛡️" if a["전략"] == "안정" else "⚡"
        판정 = {"승": "목표 도달", "패": "손절", "중립": "청산"}.get(a["결과"], a["결과"])
        줄.append(f"  {아이콘} {전략아이콘} {a['name']}: {판정} ({a['수익률']:+.1f}%)")
    return "\n".join(줄) + "\n"
