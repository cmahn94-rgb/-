"""
telegram_bot.py — 텔레그램 리포트 전송
=========================================
이 파일이 하는 일:
  분석 리포트를 텔레그램 봇 API를 통해 사용자에게 전송한다.

  · 텔레그램 토큰이 미설정 상태면 콘솔에 출력 (테스트용)
  · 메시지가 4000자 초과 시 자동 분할 전송
  · GitHub Actions 로그에서도 리포트 내용을 확인할 수 있도록 개선
"""

import os
import requests
from dotenv import load_dotenv


def send_telegram(message):
    """
    텔레그램 봇으로 분석 리포트를 전송한다.
    동시에 GitHub Actions 로그에도 리포트 내용을 출력한다.
    """
    # .env 파일에서 토큰 로드
    load_dotenv()
    token   = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    # chat_id=받는 곳, token=봇 비밀번호(둘 중 하나라도 없으면 전송 불가)

    # ── 로그에 리포트 내용 항상 출력 (디버깅용) ─────────────────────
    print("\n" + "=" * 60)
    print("📤 전송할 리포트 내용 (GitHub Actions 로그)")
    print("=" * 60)
    print(message)
    print("=" * 60 + "\n")

    # 토큰이 기본값('여기에...')이거나 비어있으면 콘솔 출력 모드만 하고 종료
    if not token or "여기에" in token:
        print("ℹ️ 텔레그램 토큰이 설정되지 않았습니다. 콘솔 출력 모드만 실행했습니다.")
        return

    # 텔레그램 전송 시작
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    print(f"📨 텔레그램으로 전송 시작... (chat_id: {chat_id})")

    # 텔레그램은 글자 제한이 있어 4000자씩 잘라서 보냄
    chunks = [message[i:i + 4000] for i in range(0, len(message), 4000)]
    
    success_count = 0
    for idx, chunk in enumerate(chunks, 1):
        try:
            res = requests.post(
                url,
                json={
                    "chat_id":    chat_id,
                    "text":       chunk,
                    "parse_mode": "Markdown"
                },
                timeout=15
            )
            
            if res.status_code == 200:
                success_count += 1
                print(f"✅ 텔레그램 전송 성공 ({idx}/{len(chunks)})")
            else:
                print(f"⚠️ 텔레그램 전송 실패 ({idx}/{len(chunks)}): {res.status_code}")
                print(f"   응답: {res.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️ 텔레그램 전송 중 네트워크 오류 ({idx}/{len(chunks)}): {e}")

    if success_count == len(chunks):
        print("🎉 모든 리포트가 텔레그램으로 성공적으로 전송되었습니다.")
    else:
        print(f"⚠️ {success_count}/{len(chunks)} 개의 청크만 전송되었습니다.")
