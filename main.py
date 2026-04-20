"""
main.py — 프로그램 진입점 및 스케줄러
========================================
이 파일이 하는 일:
  1) 최초 실행 시 설정 파일(.env, settings.txt 등) 자동 생성
  2) 즉시 1회 분석 실행 (프로그램 시작하자마자 결과 확인 가능)
  3) 하루 3회 자동 실행 스케줄 등록:
       09:00 KST — 한국장 시가 + 미국장 전일 종가 통합 분석
       15:20 KST — 한국장 마감 직전 최종 점검
       23:30 KST — 미국장 개장 시가 분석 (코인 포함)
  4) 주말에는 주식 분석을 건너뛰고 암호화폐만 분석

[실행 방법]
  pip install -r requirements.txt
  python main.py
"""

import time
import schedule

from config        import create_default_files, load_env
from scheduler_job import run_analysis


def scheduled_run_full():
    """
    09:00 / 15:20 KST 스케줄: 주식 + 암호화폐 전체 분석.
    주말이면 주식 종목은 건너뛰고 암호화폐만 분석한다.
    (암호화폐는 24시간 365일 거래되므로 항상 실행)
    """
    run_analysis(include_crypto=True)


def scheduled_run_crypto():
    """
    23:30 KST 스케줄: 미국장 개장 시간대 분석.
    주식 + 암호화폐 전체 분석 실행.
    """
    run_analysis(include_crypto=True)


if __name__ == "__main__":
    print("=" * 55)
    print("⚔️  퀀트 헤지펀드 자동 알림 시스템 시작")
    print("=" * 55)

    # 1단계: 설정 파일 자동 생성 (없을 때만)
    create_default_files()

    # 2단계: .env 로드 (텔레그램 토큰 읽기)
    load_env()

    # 3단계: 즉시 1회 실행 (시작하자마자 분석 결과 확인)
    print("\n▶ 즉시 분석 1회 실행...")
    run_analysis()

    # 4단계: 하루 3회 자동 실행 스케줄 등록
    # 09:00 KST: 한국장 시가 + 미국장 전일 종가 통합 분석
    schedule.every().day.at("09:00").do(scheduled_run_full)
    # 15:20 KST: 한국장 마감 직전 최종 점검
    schedule.every().day.at("15:20").do(scheduled_run_full)
    # 23:30 KST: 미국장 개장 시가 분석 (코인 중심)
    schedule.every().day.at("23:30").do(scheduled_run_crypto)

    print("\n⏰ 스케줄 등록 완료:")
    print("  • 09:00 KST — 한국장 시가 + 미국장 전일 종가 분석")
    print("  • 15:20 KST — 한국장 마감 직전 점검")
    print("  • 23:30 KST — 미국장 개장 + 코인 분석")
    print("\n🔄 프로그램 실행 중. 종료하려면 Ctrl+C를 누르세요.\n")

    # 5단계: 스케줄 루프 (프로그램이 계속 돌면서 예약 시간에 실행)
    while True:
        schedule.run_pending()
        time.sleep(30)  # 30초마다 스케줄 확인
