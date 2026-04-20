"""
config.py — 설정 파일 자동 생성 및 로드
=========================================
이 파일이 하는 일:
  1) 최초 실행 시 .env, settings.txt, stocks.txt, portfolio.txt, .gitignore 자동 생성
  2) settings.txt에서 전략 파라미터(RSI 기준, 목표수익률 등)를 읽어 반환
  3) .env에서 텔레그램 토큰을 안전하게 로드
"""

import os
from dotenv import load_dotenv


# ─────────────────────────────────────────
# 파일 자동 생성 (최초 실행 시에만)
# ─────────────────────────────────────────

def create_default_files():
    """
    프로그램 실행에 필요한 설정 파일들을 자동으로 만들어준다.
    파일이 이미 있으면 덮어쓰지 않는다 (사용자 설정 보호).
    """

    # .env 파일: 텔레그램 봇 토큰과 채팅 ID를 저장하는 '비밀 파일'
    # 이 파일은 절대 GitHub 등에 올리면 안 된다 → .gitignore에 등록
    if not os.path.exists(".env"):
        with open(".env", "w", encoding="utf-8") as f:
            f.write("TELEGRAM_TOKEN=여기에_텔레그램_봇_토큰_입력\n")
            f.write("TELEGRAM_CHAT_ID=여기에_채팅_ID_입력\n")
        print("✅ .env 파일 생성 완료 — 텔레그램 토큰과 채팅 ID를 입력하세요.")

    # settings.txt: RSI 기준값, 목표 수익률, 손절 기준 등 전략 파라미터
    # 이 파일만 수정하면 전략 전체가 바뀐다 (코드 수정 불필요)
    if not os.path.exists("settings.txt"):
        with open("settings.txt", "w", encoding="utf-8") as f:
            f.write("RSI_BUY=45          # RSI가 이 값 미만이면 과매도(저평가) 신호\n")
            f.write("RSI_SELL=80         # RSI가 이 값 초과면 과매수(고평가) 신호\n")
            f.write("TARGET_PROFIT=25    # 목표 수익률 (%)\n")
            f.write("STOP_LOSS=-7        # 손절 기준 (%)\n")
            f.write("MA_WINDOW=20        # 이동평균선 기간 (일)\n")
            f.write("COMMISSION=0.001    # 수수료 0.1%\n")
            f.write("SLIPPAGE=0.0005     # 슬리피지 0.05%\n")
        print("✅ settings.txt 파일 생성 완료")

    # stocks.txt: 감시할 종목 목록 (티커, 이름, 시장구분)
    # 시장구분: KR=한국주식 / US=미국주식 / CRYPTO=암호화폐
    if not os.path.exists("stocks.txt"):
        with open("stocks.txt", "w", encoding="utf-8") as f:
            f.write("005930.KS,삼성전자,KR\n")
            f.write("000660.KS,SK하이닉스,KR\n")
            f.write("NVDA,엔비디아,US\n")
            f.write("AAPL,애플,US\n")
            f.write("BTC-USD,비트코인,CRYPTO\n")
            f.write("ETH-USD,이더리움,CRYPTO\n")
        print("✅ stocks.txt 파일 생성 완료")

    # portfolio.txt: 내가 실제로 보유 중인 종목 정보
    # 형식: 티커, 보유수량, 평단가, 통화 (암호화폐는 소수점 수량 허용)
    if not os.path.exists("portfolio.txt"):
        with open("portfolio.txt", "w", encoding="utf-8") as f:
            f.write("005930.KS,10,75000,KRW\n")
            f.write("NVDA,5,880.00,USD\n")
            f.write("BTC-USD,0.05,62000.00,USD\n")
        print("✅ portfolio.txt 파일 생성 완료")

    # .gitignore: Git 버전 관리에서 제외할 파일 목록
    # .gitignore란? 비밀 키와 자산 정보가 GitHub 등에 공개되는 사고를 방지한다.
    if not os.path.exists(".gitignore"):
        with open(".gitignore", "w", encoding="utf-8") as f:
            f.write("# 비밀 키와 개인 자산 정보는 절대 Git에 올리지 않는다\n")
            f.write(".env\n")
            f.write("portfolio.txt\n")
            f.write("__pycache__/\n")
            f.write("*.pyc\n")
        print("✅ .gitignore 파일 생성 완료 (.env, portfolio.txt 보호 등록)")


# ─────────────────────────────────────────
# 설정값 읽기
# ─────────────────────────────────────────

def load_settings():
    """
    settings.txt에서 전략 파라미터를 읽어 딕셔너리로 반환한다.
    파일이 없거나 오류가 있으면 빈 딕셔너리를 반환한다.
    """
    settings = {}
    if not os.path.exists("settings.txt"):
        print("⚠️ settings.txt 파일이 없습니다. 기본값을 사용합니다.")
        return settings

    with open("settings.txt", "r", encoding="utf-8") as f:
        for line in f:
            # '#' 이후는 주석 → 제거 후 파싱
            line = line.split("#")[0].strip()
            if "=" in line:
                key, val = line.split("=", 1)
                try:
                    settings[key.strip()] = float(val.strip())
                except ValueError:
                    pass  # 숫자로 변환 불가한 줄은 무시
    return settings


def load_env():
    """
    .env 파일에서 환경변수(텔레그램 토큰 등)를 로드한다.
    load_dotenv()를 호출해야 os.getenv()로 값을 읽을 수 있다.
    """
    load_dotenv()


def get_env_value(key, default=None):
    """환경변수 값을 안전하게 가져온다. 없으면 default 반환."""
    value = os.getenv(key)
    return value if value else default
