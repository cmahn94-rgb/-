"""
main.py — 프로그램 진입점
1) 설정 파일 자동 생성
2) .env 로드
3) 분석 1회 실행 후 종료
"""

import sys

from config import create_default_files, load_env
from scheduler_job import run_analysis

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

    print("\n▶ 분석 1회 실행...")
    run_analysis(include_crypto=True)

    print("\n✅ 실행 완료")
