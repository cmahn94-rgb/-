"""
stocks_loader.py — 종목 목록 및 포트폴리오 파일 읽기
=====================================================
이 파일이 하는 일:
  1) stocks.txt → 감시할 종목 목록 반환
  2) portfolio.txt → 내 보유 종목 정보 반환
"""

import os


def load_stocks():
    """
    stocks.txt에서 감시 종목 목록을 읽어 리스트로 반환한다.
    형식: 티커,이름,시장구분 (KR / US / CRYPTO)
    파일이 없거나 형식이 맞지 않는 줄은 건너뛴다.
    """
    종목목록 = []
    if not os.path.exists("stocks.txt"):
        print("⚠️ stocks.txt 파일이 없습니다.")
        return 종목목록

    with open("stocks.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 빈 줄이나 주석(#으로 시작)은 건너뜀
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 3:
                종목목록.append({
                    "ticker": parts[0].strip(),   # 예: 005930.KS, NVDA, BTC-USD
                    "name":   parts[1].strip(),   # 예: 삼성전자
                    "market": parts[2].strip()    # KR / US / CRYPTO
                })
    return 종목목록


def load_portfolio():
    """
    portfolio.txt에서 보유 종목 정보를 읽어 리스트로 반환한다.
    형식: 티커,보유수량,평단가,통화 (KRW / USD)
    암호화폐는 소수점 수량 허용 (예: 0.05 BTC → float으로 파싱).
    """
    포트폴리오 = []
    if not os.path.exists("portfolio.txt"):
        print("⚠️ portfolio.txt 파일이 없습니다.")
        return 포트폴리오

    with open("portfolio.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 4:
                try:
                    포트폴리오.append({
                        "ticker":    parts[0].strip(),
                        "quantity":  float(parts[1].strip()),   # 소수점 수량 허용
                        "avg_price": float(parts[2].strip()),   # 평단가
                        "currency":  parts[3].strip()           # KRW / USD
                    })
                except ValueError:
                    print(f"⚠️ portfolio.txt 파싱 오류 (줄 무시): {line}")
    return 포트폴리오
