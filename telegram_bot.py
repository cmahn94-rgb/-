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


def _escape_markdown(text: str) -> str:
    """
    텔레그램 Markdown 모드에서 충돌하는 특수문자를 이스케이프한다.

    [중학생 설명]
    텔레그램은 *굵게*, _기울게_, `코드` 같은 마크다운 문법을 지원하는데,
    종목명(삼성_SDI)이나 뉴스 제목에 _ * ` [ 같은 문자가 있으면
    텔레그램이 오해해서 전송 자체가 실패(400 에러)할 수 있다.
    이 함수는 그런 문자 앞에 이스케이프 문자(역슬래시)를 붙여서 안전하게 만든다.

    주의: *굵게* 표시를 의도적으로 사용한 줄은 건드리지 않는다.
    """
    # 텔레그램 Markdown v1에서 충돌하는 문자: _ [ ]
    # * 와 ` 는 의도적 포맷팅에 사용하므로 건드리지 않음
    # 뉴스 제목이나 AI 코멘트 안의 대괄호·언더스코어만 이스케이프
    result = []
    in_bold = False  # * 안에 있는지 추적
    i = 0
    while i < len(text):
        c = text[i]
        if c == "*":
            in_bold = not in_bold
            result.append(c)
        elif c in ("_", "[", "]") and not in_bold:
            result.append("\\" + c)
        else:
            result.append(c)
        i += 1
    return "".join(result)


def send_telegram(message):
    """
    텔레그램 봇으로 분석 리포트를 전송한다.
    동시에 GitHub Actions 로그에도 리포트 내용을 출력한다.

    [업그레이드] Markdown 특수문자 이스케이프 자동 처리
    → 종목명·뉴스 제목의 _, [ ] 등으로 인한 400 오류 방지
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

    # Markdown 특수문자 이스케이프 (_, [ ] 등이 400 오류 유발 방지)
    safe_message = _escape_markdown(message)

    # 텔레그램은 글자 제한이 있어 분할 전송
    # 수정: 4000자 단순 슬라이스 → 줄바꿈 기준으로 분할 (문장 중간 잘림 방지)
    def _split_by_lines(text: str, limit: int = 4000) -> list[str]:
        """줄바꿈 경계에서 분할 — AI 코멘트 등이 문장 중간에 끊기지 않도록"""
        if len(text) <= limit:
            return [text]
        parts = []
        lines = text.split("\n")
        current = ""
        for line in lines:
            # 한 줄 자체가 limit을 초과하면 강제 슬라이스 (극단적 케이스)
            if len(line) > limit:
                if current:
                    parts.append(current)
                    current = ""
                for i in range(0, len(line), limit):
                    parts.append(line[i:i + limit])
                continue
            candidate = current + ("\n" if current else "") + line
            if len(candidate) > limit:
                if current:
                    parts.append(current)
                current = line
            else:
                current = candidate
        if current:
            parts.append(current)
        return parts

    chunks = _split_by_lines(safe_message, limit=4000)
    
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
