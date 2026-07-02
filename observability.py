"""
observability.py — 관측성 / 실행 헬스 추적 (하이엔드 2순위)
============================================================
[목적]
봇 실행 중 각 데이터 소스·단계의 성공/실패를 집계하고,
"오늘 뭐가 실패했는지"를 한눈에 보여준다. 또한 특정 소스가
여러 날 연속 실패하면 감지해 경고한다.

[중학생 설명]
지금까지는 데이터 소스가 하나 죽어도 봇이 조용히 넘어갔다(폴백은 좋지만,
'뭐가 죽었는지' 알 수가 없었다). 이제는 실행 때마다 "수급 3건 실패,
뉴스 1건 타임아웃" 같은 건강검진 결과를 남긴다. 그리고 어떤 소스가
3일 연속 죽으면 "이거 손봐야 한다"고 경고한다.

[사용법]
  from observability import health
  health.record("supply", ok=True)      # 성공 기록
  health.record("news", ok=False, detail="타임아웃")  # 실패 기록
  print(health.summary())                # 실행 끝에 요약 출력
  health.persist_and_check_streaks()     # 연속 실패 감지 (일별 저장)

[저장] health_log.json — 날짜별 소스 성공률 (연속 실패 감지용)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_KST = ZoneInfo("Asia/Seoul")
_HEALTH_FILE = "health_log.json"


# ── 로깅 설정 (print → logging 전환의 기반) ──────────────────
def setup_logger(name: str = "quantbot", level: int = logging.INFO) -> logging.Logger:
    """
    표준 logging 로거를 설정한다. GitHub Actions 로그에 레벨과 함께 출력.

    [중학생 설명]
    print는 다 똑같아 보이지만, logging은 "정보/경고/에러"를 구분한다.
    나중에 로그에서 ERROR만 걸러보거나, 심각도별로 처리할 수 있다.
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # 중복 핸들러 방지
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    return logger


class HealthTracker:
    """실행 1회 동안 소스별 성공/실패를 집계하는 추적기."""

    def __init__(self):
        self._counts: dict[str, dict] = {}

    def record(self, source: str, ok: bool, detail: str = ""):
        """소스의 성공/실패 1건을 기록한다."""
        if source not in self._counts:
            self._counts[source] = {"성공": 0, "실패": 0, "실패상세": []}
        c = self._counts[source]
        if ok:
            c["성공"] += 1
        else:
            c["실패"] += 1
            if detail and len(c["실패상세"]) < 3:
                c["실패상세"].append(detail)

    def get_source_health(self) -> dict:
        """소스별 성공률 dict 반환."""
        result = {}
        for src, c in self._counts.items():
            총 = c["성공"] + c["실패"]
            result[src] = {
                "성공": c["성공"], "실패": c["실패"],
                "성공률": round(c["성공"] / 총 * 100, 1) if 총 > 0 else 0.0,
                "실패상세": c["실패상세"],
            }
        return result

    def summary(self) -> str:
        """실행 헬스 요약 텍스트 (콘솔·리포트용)."""
        health = self.get_source_health()
        if not health:
            return "🏥 실행 헬스: 기록된 소스 없음"

        줄 = ["🏥 ===== 실행 헬스 체크 ====="]
        문제있음 = False
        for src, h in sorted(health.items()):
            아이콘 = "✅" if h["실패"] == 0 else ("⚠️" if h["성공"] > 0 else "❌")
            if h["실패"] > 0:
                문제있음 = True
            줄.append(f"  {아이콘} {src}: {h['성공']}성공/{h['실패']}실패 "
                      f"({h['성공률']}%)")
            for d in h["실패상세"]:
                줄.append(f"      └ {d}")
        if not 문제있음:
            줄.append("  🟢 모든 소스 정상")
        return "\n".join(줄)

    def persist_and_check_streaks(self, streak_threshold: int = 3) -> list[str]:
        """
        오늘의 소스별 성공/실패를 health_log.json에 저장하고,
        연속 실패(streak_threshold일 이상) 소스를 감지해 경고 리스트 반환.

        [중학생 설명]
        하루만 실패하면 일시적 문제일 수 있다. 하지만 3일 연속 실패하면
        진짜 고장난 것이다. 이걸 자동으로 잡아낸다.
        """
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _HEALTH_FILE)
        오늘 = datetime.now(_KST).strftime("%Y-%m-%d")

        # 로드
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"days": {}}
        if "days" not in data:
            data["days"] = {}

        # 오늘 기록 저장 (소스별 실패 여부만)
        오늘_상태 = {}
        for src, h in self.get_source_health().items():
            # 성공 0 + 실패>0 = 완전 실패로 간주
            오늘_상태[src] = "fail" if (h["성공"] == 0 and h["실패"] > 0) else "ok"
        data["days"][오늘] = 오늘_상태

        # 오래된 기록 정리 (14일치만 유지)
        cutoff = (datetime.now(_KST) - timedelta(days=14)).strftime("%Y-%m-%d")
        data["days"] = {d: v for d, v in data["days"].items() if d >= cutoff}

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # 연속 실패 감지: 최근 streak_threshold일 연속 fail인 소스
        경고 = []
        최근일자 = sorted(data["days"].keys(), reverse=True)[:streak_threshold]
        if len(최근일자) >= streak_threshold:
            # 모든 소스 후보
            모든소스 = set()
            for d in 최근일자:
                모든소스.update(data["days"][d].keys())
            for src in 모든소스:
                if all(data["days"][d].get(src) == "fail" for d in 최근일자):
                    경고.append(
                        f"🚨 '{src}' 소스가 {streak_threshold}일 연속 실패 — 점검 필요"
                    )
        return 경고


def record_data_source_health(supply_counts: dict, vix_value: float) -> None:
    """
    수급·VIX 소스의 성공/실패를 전역 health에 반영한다. (scheduler_job 호출용)

    scheduler_job 인라인 코드를 여기로 옮겨 run_analysis를 깔끔하게 유지.

    Args:
        supply_counts: fundamental._supply_demand_source_count
        vix_value: settings["CURRENT_VIX"] (15.0=기본값=실패로 간주)
    """
    확보 = (supply_counts.get("naver_mobile", 0) + supply_counts.get("pykrx", 0) +
           supply_counts.get("krx", 0) + supply_counts.get("naver", 0) +
           supply_counts.get("fdr", 0))
    실패 = supply_counts.get("전체실패", 0)
    for _ in range(확보):
        health.record("수급", ok=True)
    for _ in range(실패):
        health.record("수급", ok=False, detail="모든 폴백 실패")
    health.record("VIX", ok=(vix_value != 15.0))


# 전역 인스턴스 (실행당 1개)
health = HealthTracker()
logger = setup_logger()
