"""
main.py — 프로그램 진입점
1) 설정 파일 자동 생성
2) .env 로드
3) 분석 1회 실행 후 종료

[GitHub Actions cron 지연 대응]
GitHub cron은 정시 보장이 없다. 부하에 따라 수십 분~수 시간씩 밀릴 수 있다.
따라서 "예정된 시간"이 아니라 "실제 실행된 KST 시각"을 보고 모드를 결정한다.

예약 시간 → 실제 실행 예시:
  09:00 KST 예약 → 09:42에 실행  → morning 모드 (08:30~10:30 허용)
  15:30 KST 예약 → 16:15에 실행  → afternoon 모드 (14:30~17:30 허용)
  23:30 KST 예약 → 다음날 02:00  → night 모드 (21:00~04:00 허용)

허용 윈도우를 넓게 잡아서 cron이 1~2시간 밀려도 올바른 모드로 실행된다.
"""

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from config import create_default_files, load_env, check_required_keys
from scheduler_job import run_analysis


def get_run_mode() -> str:
    """
    실제 실행된 KST 시각을 보고 모드를 결정한다.

    [cron 지연 대응 설계]
    GitHub Actions cron은 정시 보장이 없어 1~2시간 지연이 흔하다.
    각 모드의 허용 윈도우를 넉넉하게 잡아서 지연에도 올바른 모드가 선택된다.

    윈도우 설계:
    - morning  : 08:30~10:30 KST  (장 시작 전후, 1시간 여유)
    - afternoon: 14:30~17:30 KST  (장 마감 전후, 1시간 여유)
    - night    : 21:00~04:00 KST  (미국장 중반, 자정 넘어도 포함)
    - all      : 그 외 (수동 실행, 예외 시간대)

    세 윈도우 사이에 겹치는 시간이 없으므로 모드 중복 없음:
      10:30~14:30 → all
      17:30~21:00 → all
    """
    kst_hour = datetime.now(ZoneInfo("Asia/Seoul")).hour
    kst_minute = datetime.now(ZoneInfo("Asia/Seoul")).minute

    # morning: 08:30 ~ 10:30
    if (kst_hour == 8 and kst_minute >= 30) or (kst_hour == 9) or (kst_hour == 10 and kst_minute <= 30):
        return "morning"

    # afternoon: 14:30 ~ 17:30
    if (kst_hour == 14 and kst_minute >= 30) or (kst_hour in (15, 16)) or (kst_hour == 17 and kst_minute <= 30):
        return "afternoon"

    # night: 21:00 ~ 04:00 (자정 넘어도 포함)
    if kst_hour >= 21 or kst_hour < 4:
        return "night"

    # 그 외: 수동 실행 또는 cron이 2시간 이상 밀린 극단 케이스 → 전체 분석
    return "all"


if __name__ == "__main__":
    # Windows PowerShell(기본 cp949)에서 이모지/특수문자 출력 시 UnicodeEncodeError가 날 수 있어
    # 표준 출력/에러를 UTF-8로 맞춘다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    print("=" * 55)
    print("퀀트 헤지펀드 자동 알림 시스템 시작")
    print("=" * 55)

    create_default_files()
    load_env()
    check_required_keys()   # 하이엔드: 필수 키 누락 시작 단계 점검

    mode = get_run_mode()
    kst_now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%H:%M KST")

    print(f"\n▶ 분석 1회 실행... [{kst_now} → {mode} 모드]")

    if mode == "morning":
        print("  📌 한국 장 시작 시간대 → 국내주식 + 크립토 분석")
        run_analysis(include_markets=["KR", "CRYPTO", "CRYPTO_KRW"])

    elif mode == "afternoon":
        print("  📌 한국 장 마감 시간대 → 국내주식 + 크립토 분석")
        run_analysis(include_markets=["KR", "CRYPTO", "CRYPTO_KRW"])

    elif mode == "night":
        print("  📌 미국 장 중반 시간대 → 전체 분석")
        run_analysis(include_markets=None)  # None = 전체

    else:
        print("  📌 예외 시간대(수동 실행 등) → 전체 분석")
        run_analysis(include_markets=None)

    print("\n✅ 실행 완료")
