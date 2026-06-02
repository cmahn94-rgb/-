"""
main.py — 프로그램 진입점
1) 설정 파일 자동 생성
2) .env 로드
3) 분석 1회 실행 후 종료
"""

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from config import create_default_files, load_env
from scheduler_job import run_analysis

def get_run_mode() -> str:
    """
    KST 현재 시각을 보고 이번 실행이 어떤 시간대인지 판단한다.

    실행 시간대 3가지:
    - "morning"  : 08:30~10:00 KST → 한국 장 시작 (KR + 크립토만)
    - "afternoon": 14:30~17:00 KST → 한국 장 마감 (KR + 크립토만)
    - "night"    : 21:00~다음날 02:00 KST → 미국 장 중반 (전체)

    시간대에 해당하지 않으면 "all"로 폴백해서 전체 분석.
    (수동 실행 또는 cron 지연이 심할 때도 빈 리포트가 오지 않도록)
    """
    kst_hour = datetime.now(ZoneInfo("Asia/Seoul")).hour
    if 8 <= kst_hour < 10:
        return "morning"
    elif 14 <= kst_hour < 17:
        return "afternoon"
    elif kst_hour >= 21 or kst_hour < 2:
        return "night"
    else:
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

    mode = get_run_mode()
    kst_now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%H:%M KST")

    print(f"\n▶ 분석 1회 실행... [{kst_now} → {mode} 모드]")

    if mode == "morning":
        # 09:00 KST — 한국 장 시작: 국내주식 + 크립토만
        print("  📌 한국 장 시작 시간대 → 국내주식 + 크립토 분석")
        run_analysis(include_markets=["KR", "CRYPTO", "CRYPTO_KRW"])

    elif mode == "afternoon":
        # 15:30 KST — 한국 장 마감: 국내주식 + 크립토만
        print("  📌 한국 장 마감 시간대 → 국내주식 + 크립토 분석")
        run_analysis(include_markets=["KR", "CRYPTO", "CRYPTO_KRW"])

    elif mode == "night":
        # 23:30 KST — 미국 장 중반: 전체 (KR + US + 크립토)
        print("  📌 미국 장 중반 시간대 → 전체 분석")
        run_analysis(include_markets=None)  # None = 전체

    else:
        # 수동 실행 등 예외 시간대: 전체 분석
        print("  📌 예외 시간대(수동 실행 등) → 전체 분석")
        run_analysis(include_markets=None)

    print("\n✅ 실행 완료")
