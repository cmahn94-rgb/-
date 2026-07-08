"""
check_supply.py — 네이버 모바일 수급 API 해외 IP 접근성 진단
============================================================
[목적]
기존 수급 소스(KRX·네이버 데스크톱·pykrx·FDR)는 GitHub Actions 해외 IP에서
전부 차단됐다(naver=0 확인). 그런데 RSS는 잘 됐다.
→ 네이버 "모바일" 증권 API(m.stock.naver.com)는 데스크톱과 다른 경로라
   해외 IP에서 열려있을 수도 있다. 여러 후보 엔드포인트를 1회씩 호출해서
   외국인/기관 일별 순매수(수치)를 주는 게 있는지 확인한다.

[중요]
RSS는 '뉴스 텍스트'였지만, 이건 '외국인 순매수 금액' 같은 수치 데이터다.
하나라도 살아있으면 수급 flow 축을 실제로 복구할 수 있다.

[사용법]
  Actions → "Check Supply" → Run workflow → 로그 확인
  또는 로컬: python check_supply.py

[판단]
  ✅ 외국인/기관 수치가 나오는 엔드포인트 발견 → 수급 복구 가능
  ❌ 전부 차단/빈값 → 한국 IP(self-hosted) 외엔 방법 없음 확정
"""

import json
import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.0 Mobile/15E148 Safari/604.1"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://m.stock.naver.com/",
}

CODE = "005930"  # 삼성전자


def try_endpoint(label: str, url: str, parser) -> dict:
    """엔드포인트 1개 호출 → 수급 수치 추출 시도."""
    try:
        r = requests.get(url, headers=_HEADERS, timeout=12)
    except Exception as e:
        return {"label": label, "ok": False, "status": "ERR", "msg": str(e)[:70], "sample": ""}

    if r.status_code != 200:
        return {"label": label, "ok": False, "status": r.status_code, "msg": "non-200", "sample": r.text[:60]}

    try:
        sample = parser(r)
        if sample:
            return {"label": label, "ok": True, "status": 200, "msg": "수급 수치 확보", "sample": sample}
        return {"label": label, "ok": False, "status": 200, "msg": "200이나 수급필드 없음", "sample": r.text[:80]}
    except Exception as e:
        return {"label": label, "ok": False, "status": 200, "msg": f"파싱실패: {str(e)[:50]}", "sample": r.text[:80]}


def _parse_trend(r):
    """trend 계열: 외국인/기관 순매수 필드 탐색."""
    data = r.json()
    # 응답이 list거나 {"result":[...]} 형태일 수 있음
    rows = data if isinstance(data, list) else (data.get("result") or data.get("trends") or [])
    if not rows:
        return ""
    row = rows[0]
    # [v5.22] 실제 필드명(스키마)을 로그에 노출 — 봇 파서와 대조용
    print(f"       └ rows[0] 키: {list(row.keys())[:15]}")
    frgn = row.get("frgn_pure_buy_quant") or row.get("foreignerPureBuyQuant") or row.get("foreignerNetBuy")
    organ = row.get("organ_pure_buy_quant") or row.get("organPureBuyQuant") or row.get("organNetBuy")
    date = row.get("bizdate") or row.get("localTradedAt") or row.get("date") or "?"
    if frgn is not None or organ is not None:
        return f"{date} 외국인={frgn} 기관={organ}"
    return ""


def _parse_chart_info(r):
    """front-api chart: 외국인소진율 포함 여부."""
    txt = r.text.strip()
    # JSONP/배열 형태 → 외국인소진율 키워드 존재 여부로 판정
    if "외국인" in txt or "frgn" in txt.lower() or "foreigner" in txt.lower():
        return txt[:80]
    return ""


if __name__ == "__main__":
    print("=" * 62)
    print("  네이버 모바일 수급 API 해외 IP 접근성 진단 (삼성전자)")
    print("=" * 62)

    endpoints = [
        ("trend(m.stock)",
         f"https://m.stock.naver.com/api/stock/{CODE}/trend", _parse_trend),
        ("trend(api.stock)",
         f"https://api.stock.naver.com/stock/{CODE}/trend", _parse_trend),
        ("foreignerOrgan(m)",
         f"https://m.stock.naver.com/api/stock/{CODE}/foreignerOrgan", _parse_trend),
        ("frgn(api.stock)",
         f"https://api.stock.naver.com/stock/{CODE}/foreignerOrgan", _parse_trend),
        ("chart-info(외국인소진율)",
         f"https://m.stock.naver.com/front-api/external/chart/domestic/info"
         f"?symbol={CODE}&requestType=1&timeframe=day", _parse_chart_info),
    ]

    print("\n[엔드포인트 테스트]")
    any_ok = False
    for label, url, parser in endpoints:
        res = try_endpoint(label, url, parser)
        icon = "✅" if res["ok"] else "❌"
        print(f"  {icon} {label:24s} HTTP {res['status']} | {res['msg']}")
        if res["sample"]:
            print(f"       └ {res['sample']}")
        if res["ok"]:
            any_ok = True

    print("\n" + "=" * 62)
    print("  판단")
    print("=" * 62)
    if any_ok:
        print("  ✅ 수급 수치를 주는 엔드포인트 발견!")
        print("     → 위 ✅ 경로로 수급 flow 축 복구 가능. 통합 진행 권장.")
    else:
        print("  ❌ 모바일 API도 전부 차단/빈값")
        print("     → 한국 IP(self-hosted runner) 외엔 수급 복구 방법 없음 확정.")
        print("     → RSS 기반 키워드 보조신호만 차선책으로 남음.")
    print("=" * 62)
