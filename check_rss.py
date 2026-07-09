"""
check_rss.py — 구글 뉴스 RSS 접근성 진단 (GitHub Actions IP 차단 여부 확인)
============================================================================
[목적]
수급 데이터처럼 구글 뉴스 RSS도 해외 IP(GitHub Actions)에서 막히는지
실제로 1회 호출해서 확인한다. 이 결과에 따라 RSS를 뉴스 1순위로 올릴지,
폴백으로만 둘지, 아예 포기할지 결정한다.

[사용법 — check_keys.yml과 같은 방식]
  Actions → "Check RSS" workflow → Run workflow
  또는 로컬: python check_rss.py

[판단 기준]
  ✅ 한국·미국 종목 모두 기사 N개  → RSS를 한국 뉴스 1순위로 채택
  ⚠️ 미국만 되고 한국 0개          → 한국 전용으론 부적합, 영어만 폴백
  ❌ 둘 다 0개 또는 차단/타임아웃   → 수급처럼 해외 IP 차단 → RSS 포기
"""

import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


def fetch_rss(name: str, query: str, hl: str, gl: str, ceid: str) -> dict:
    """구글 뉴스 RSS를 표준 라이브러리만으로 호출·파싱 (feedparser 불필요)."""
    base = "https://news.google.com/rss/search"
    q = urllib.parse.quote(query)
    url = f"{base}?q={q}&hl={hl}&gl={gl}&ceid={urllib.parse.quote(ceid)}"

    req = urllib.request.Request(url, headers={
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
        "Accept-Language": hl,
    })

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
            raw = resp.read()
    except Exception as e:
        return {"name": name, "ok": False, "count": 0, "status": "ERR", "msg": str(e)[:80], "samples": []}

    if status != 200:
        return {"name": name, "ok": False, "count": 0, "status": status, "msg": "non-200", "samples": []}

    # RSS XML 파싱: <item><title>...</title><source>...</source><pubDate>...</pubDate>
    try:
        root = ET.fromstring(raw)
        items = root.findall(".//item")
        samples = []
        for it in items[:3]:
            title = (it.findtext("title") or "").strip()
            src_el = it.find("source")
            source = (src_el.text or "").strip() if src_el is not None else "?"
            samples.append(f"[{source}] {title[:50]}")
        return {
            "name": name, "ok": len(items) > 0, "count": len(items),
            "status": status, "msg": "OK", "samples": samples,
        }
    except Exception as e:
        return {"name": name, "ok": False, "count": 0, "status": status,
                "msg": f"파싱실패: {str(e)[:60]}", "samples": []}


if __name__ == "__main__":
    print("=" * 60)
    print("  구글 뉴스 RSS 접근성 진단 (해외 IP 차단 여부)")
    print("=" * 60)

    tests = [
        # (라벨, 검색어, hl, gl, ceid)
        # v5.16 개선: 평상시 검색어(종목명만) + 급등락 시 이벤트 지향 검색어 둘 다 검증
        ("평상시-삼성전자", "삼성전자", "ko", "KR", "KR:ko"),
        ("이벤트-삼성전자", "삼성전자 실적 OR 수주 OR 계약 OR 공시", "ko", "KR", "KR:ko"),
        ("평상시-SK하이닉스", "SK하이닉스", "ko", "KR", "KR:ko"),
    ]

    results = []
    print("\n[RSS 호출 테스트]")
    for name, query, hl, gl, ceid in tests:
        r = fetch_rss(name, query, hl, gl, ceid)
        results.append(r)
        icon = "✅" if r["ok"] else "❌"
        print(f"  {icon} {name:18s} 기사 {r['count']:2d}개  (HTTP {r['status']}, {r['msg']})")
        for s in r["samples"]:
            print(f"       └ {s}")

    # ── 종합 판단 ──────────────────────────────────────────
    # [v5.23] 봇과 동일한 requests 라이브러리 경로 테스트
    # urllib(위)=성공인데 requests=실패라면 → 라이브러리/TLS 지문 차단이 원인
    print("\n[requests 라이브러리 경로 테스트 — 봇과 동일]")
    try:
        import requests as _rq
        _url = ("https://news.google.com/rss/search?q=" +
                urllib.parse.quote("삼성전자") +
                "&hl=ko&gl=KR&ceid=" + urllib.parse.quote("KR:ko"))
        _hdr = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36"),
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
            "Accept-Language": "ko-KR,ko;q=0.9",
        }
        _r = _rq.get(_url, headers=_hdr, timeout=15)
        _n = _r.text.count("<item>") if _r.status_code == 200 else 0
        print(f"  {'✅' if _r.status_code == 200 and _n > 0 else '❌'} "
              f"requests: HTTP {_r.status_code} | 기사 {_n}개 | 최종URL {_r.url[:60]}")
    except Exception as _e:
        print(f"  ❌ requests 예외: {type(_e).__name__}: {_e}")

    평상시_ok = any(r["ok"] for r in results if r["name"].startswith("평상시"))
    이벤트_ok = any(r["ok"] for r in results if r["name"].startswith("이벤트"))

    print("\n" + "=" * 60)
    print("  판단 (v5.16 개선 검색어)")
    print("=" * 60)
    if 평상시_ok and 이벤트_ok:
        print("  ✅ 평상시·이벤트 검색어 모두 작동 → 개선된 RSS 그대로 사용 가능")
    elif 평상시_ok and not 이벤트_ok:
        print("  ⚠️ 평상시 검색어만 작동, 이벤트 검색어(OR 연산자) 차단/빈결과")
        print("     → data_loader.py에서 이벤트 검색어를 종목명만으로 단순화 권장")
    elif 이벤트_ok and not 평상시_ok:
        print("  ⚠️ 이벤트 검색어만 작동 (드문 경우)")
    else:
        print("  ❌ 둘 다 차단/실패 → 수급처럼 해외 IP 차단. RSS 도입 보류 권장")
    print("=" * 60)
