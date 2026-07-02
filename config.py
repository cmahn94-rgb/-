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
    GitHub Actions 환경(Secrets)과 로컬 환경을 모두 지원한다.
    """

    # .env 파일: 텔레그램 봇 토큰과 채팅 ID를 저장하는 '비밀 파일'
    # 이 파일은 절대 GitHub 등에 올리면 안 된다 → .gitignore에 등록
    # GitHub Actions에서는 Secrets를 사용하므로, 환경 변수가 없을 때만 생성한다.
    if not os.path.exists(".env"):
        # 로컬 실행 환경인지 확인 (환경 변수에 토큰이 없거나 기본값인 경우)
        if not os.getenv("TELEGRAM_TOKEN") or "여기에" in os.getenv("TELEGRAM_TOKEN", ""):
            with open(".env", "w", encoding="utf-8") as f:
                f.write("TELEGRAM_TOKEN=여기에_텔레그램_봇_토큰_입력\n")
                f.write("TELEGRAM_CHAT_ID=여기에_채팅_ID_입력\n")
                f.write("GEMINI_API_KEY=여기에_Gemini_API_KEY_입력(선택)\n")
            print("✅ .env 파일 생성 완료 — 텔레그램 토큰과 채팅 ID를 입력하세요.")
        else:
            print("ℹ️ GitHub Actions 환경: Secrets로 등록된 환경 변수를 사용합니다.")
          
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

    # 하이엔드: 설정값 검증 (잘못된 값이 조용히 통과하지 않게)
    validate_settings(settings)
    return settings


def validate_settings(settings: dict) -> list:
    """
    settings.txt 값의 유효성을 검사한다. (하이엔드 5순위)

    [중학생 설명]
    RSI 기준이 음수이거나, 손절이 양수(+)이거나, 목표수익률이 0이면
    봇이 이상하게 동작한다. 이런 잘못된 값을 시작 단계에서 잡아
    경고하고, 위험한 값은 안전한 기본값으로 되돌린다.

    반환: 발견된 문제 메시지 리스트 (없으면 빈 리스트)
    """
    문제 = []

    # ── 범위 검증 규칙: (키, 최소, 최대, 기본값, 설명) ──
    규칙 = [
        ("RSI_OVERSOLD",   1,   99,   30,   "RSI 과매도 기준"),
        ("RSI_OVERBOUGHT", 1,   99,   70,   "RSI 과매수 기준"),
        ("TARGET_1",       0.1, 200,  12,   "1차 목표수익률"),
        ("TARGET_2",       0.1, 500,  25,   "2차 목표수익률"),
        ("TRAILING_STOP",  0.1, 50,   8,    "트레일링 스탑"),
        ("MAX_HOLDING_DAYS", 1, 365,  30,   "최대 보유일"),
        ("CORRELATION_MAX", 0.1, 1.0, 0.7,  "상관계수 상한"),
        ("MAX_POSITIONS",  1,   50,   8,    "최대 동시 보유"),
        ("M_STOP",         -50, -0.1, -2.5, "모멘텀 손절(음수여야 함)"),
        ("M_TARGET",       0.1, 100,  5.0,  "모멘텀 익절"),
        ("M_VOL_MULT",     1.0, 20,   2.0,  "모멘텀 거래량 배수"),
    ]

    for 키, 최소, 최대, 기본, 설명 in 규칙:
        if 키 not in settings:
            continue
        값 = settings[키]
        if not (최소 <= 값 <= 최대):
            문제.append(f"{키}({설명})={값} 범위 이탈 [{최소}~{최대}] → 기본값 {기본} 적용")
            settings[키] = 기본

    # ── 논리 검증: 과매도 < 과매수 ──
    if ("RSI_OVERSOLD" in settings and "RSI_OVERBOUGHT" in settings
            and settings["RSI_OVERSOLD"] >= settings["RSI_OVERBOUGHT"]):
        문제.append("RSI 과매도 기준이 과매수보다 크거나 같음 → 기본값(30/70) 적용")
        settings["RSI_OVERSOLD"] = 30
        settings["RSI_OVERBOUGHT"] = 70

    if 문제:
        print("⚙️ 설정 검증 경고:")
        for m in 문제:
            print(f"   ⚠️ {m}")
    return 문제


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


def check_required_keys() -> list:
    """
    봇 실행에 필요한 API 키/환경변수 존재 여부를 점검한다. (하이엔드 5순위)

    [중학생 설명]
    텔레그램 토큰이 없으면 알림을 못 보내고, 키가 하나도 없으면 뉴스가
    안 나온다. 시작할 때 "뭐가 없는지"를 명확히 알려줘서, 나중에
    조용히 실패하는 걸 막는다.

    반환: 누락 경고 메시지 리스트
    """
    경고 = []

    # 필수: 텔레그램 (없으면 알림 자체가 안 감)
    if not os.getenv("TELEGRAM_TOKEN"):
        경고.append("❌ TELEGRAM_TOKEN 없음 — 알림 전송 불가")
    if not os.getenv("TELEGRAM_CHAT_ID"):
        경고.append("❌ TELEGRAM_CHAT_ID 없음 — 알림 대상 불명")

    # 선택: 뉴스/AI 키 (하나도 없으면 뉴스 품질 저하)
    뉴스_키 = ["GEMINI_API_KEY", "GROQ_API_KEY", "GNEWS_API_KEY",
             "NEWSAPI_KEY", "ALPHAVANTAGE_API_KEY"]
    if not any(os.getenv(k) for k in 뉴스_키):
        경고.append("⚠️ 뉴스/AI 키 없음 — RSS만으로 동작 (품질 저하 가능)")

    if 경고:
        print("🔑 API 키 점검:")
        for m in 경고:
            print(f"   {m}")
    return 경고
