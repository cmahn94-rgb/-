"""
telegram_bot.py — 텔레그램 리포트 전송
=========================================
이 파일이 하는 일:
  분석 리포트를 텔레그램 봇 API를 통해 사용자에게 전송한다.

  · 텔레그램 토큰이 미설정 상태면 콘솔에 출력 (테스트용)
  · 메시지가 4000자 초과 시 자동 분할 전송 (텔레그램 API 제한)
  · 네트워크 오류 발생 시 프로그램 중단 없이 경고만 출력
"""

import os
import requests
from dotenv import load_dotenv


def send_telegram(message):
    """
    텔레그램 봇으로 분석 리포트를 전송한다.

    .env 파일에 TELEGRAM_TOKEN과 TELEGRAM_CHAT_ID가 설정되어 있어야 한다.
    설정되지 않은 경우(초기 상태)는 콘솔에만 출력한다.
    """
    # .env 파일에서 토큰 로드
    load_dotenv()
    token   = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    # 토큰이 기본값('여기에...')이거나 비어있으면 콘솔 출력 모드
    if not token or "여기에" in token:
        print("\n" + "=" * 50)
        print("[텔레그램 미설정 — 콘솔 출력 모드]")
        print(message)
        print("=" * 50)
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # 텔레그램 메시지 최대 길이는 4096자 → 4000자씩 분할 전송
    for i in range(0, len(message), 4000):
        chunk = message[i:i + 4000]
        try:
            res = requests.post(
                url,
                json={
                    "chat_id":    chat_id,
                    "text":       chunk,
                    "parse_mode": "Markdown"  # 볼드(*텍스트*) 서식 적용
                },
                timeout=10
            )
            if res.status_code != 200:
                print(f"⚠️ 텔레그램 전송 오류: {res.text}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ 텔레그램 전송 실패 (네트워크 오류): {e}")
