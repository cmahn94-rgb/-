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
        ("한국-삼성전자", "삼성전자 주가", "ko", "KR", "KR:ko"),
        ("한국-SK하이닉스", "SK하이닉스 주가", "ko", "KR", "KR:ko"),
        ("미국-Apple",   "Apple stock", "en-US", "US", "US:en"),
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
    kr_ok = any(r["ok"] for r in results if r["name"].startswith("한국"))
    us_ok = any(r["ok"] for r in results if r["name"].startswith("미국"))

    print("\n" + "=" * 60)
    print("  판단")
    print("=" * 60)
    if kr_ok and us_ok:
        print("  ✅ 한국·미국 모두 작동 → RSS를 한국 뉴스 1순위로 채택 가능")
    elif us_ok and not kr_ok:
        print("  ⚠️ 미국만 작동, 한국 0개 → 한국 전용으론 부적합")
    elif kr_ok and not us_ok:
        print("  ⚠️ 한국만 작동 → 한국 뉴스용으로 채택, 미국은 기존 소스 유지")
    else:
        print("  ❌ 둘 다 차단/실패 → 수급처럼 해외 IP 차단. RSS 도입 보류 권장")
    print("=" * 60)
