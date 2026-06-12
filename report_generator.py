"""
report_generator.py — 텔레그램 리포트 텍스트를 HTML 대시보드로 변환
====================================================================
[역할]
run_analysis()가 생성한 리포트 문자열을 받아
GitHub Pages에 올릴 수 있는 HTML 파일 2개를 생성한다.

[생성 파일]
  docs/YYYY-MM-DD-HHMM.html  : 이번 실행 리포트 (히스토리 누적)
  docs/index.html             : 리포트 목록 페이지 (최신순, 클릭하면 해당 리포트)

[GitHub Pages 설정 방법]
  GitHub 레포지토리 → Settings → Pages
  → Source: Deploy from a branch
  → Branch: main / docs
  → Save
  → 잠시 후 https://[계정명].github.io/[레포명] 으로 접속 가능

[히스토리 삭제 방법]
  특정 파일 삭제: git rm docs/2026-06-01-0900.html && git commit && git push
  전체 초기화  : git rm docs/*.html && git commit -m "히스토리 초기화" && git push
                 (index.html은 다음 실행 때 자동 재생성)
"""

from __future__ import annotations
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo


# ── 설정 ──────────────────────────────────────────────────
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
MAX_INDEX_ENTRIES = 90  # 인덱스 페이지에 표시할 최대 항목 수 (약 1달치)


def generate_html_report(
    리포트_텍스트: str,
    phase_str: str = "",
    kst_now: str = "",
) -> tuple[str, str]:
    """
    텔레그램 리포트 텍스트를 HTML로 변환하고 docs/ 폴더에 저장한다.

    반환: (리포트_파일명, 인덱스_URL_경로)
    예 : ("2026-06-02-0902.html", "docs/index.html")
    """
    os.makedirs(DOCS_DIR, exist_ok=True)

    # 파일명 생성 (YYYY-MM-DD-HHMM)
    now_kst  = datetime.now(ZoneInfo("Asia/Seoul"))
    파일명   = now_kst.strftime("%Y-%m-%d-%H%M") + ".html"
    파일경로 = os.path.join(DOCS_DIR, 파일명)
    표시시각 = kst_now or now_kst.strftime("%Y-%m-%d %H:%M KST")

    # 국면 문자열 파싱
    phase_emoji = "📈"
    phase_label = phase_str or ""
    for emoji in ["🚀", "📈", "↔️", "📉", "🔴"]:
        if emoji in phase_label:
            phase_emoji = emoji
            break

    # 리포트 텍스트 → HTML 변환
    html_body = _convert_report_to_html(리포트_텍스트)

    # 전체 HTML 조립
    html = _wrap_html(html_body, 표시시각, phase_label, phase_emoji)

    # 리포트 파일 저장
    with open(파일경로, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  📄 HTML 리포트 저장: docs/{파일명}")

    # 인덱스 페이지 갱신
    _update_index(파일명, 표시시각, phase_label, 리포트_텍스트)

    return 파일명, "docs/index.html"


def _convert_report_to_html(text: str) -> str:
    """
    텔레그램 마크다운 텍스트를 HTML 블록으로 변환한다.

    [변환 규칙]
    - *굵게* → <b>굵게</b>
    - `코드` → <code>코드</code>
    - ─────── → <hr>
    - 빈 줄   → <br>
    - 이모지+섹션 헤더 → <h3>
    - 들여쓰기 줄 → <div class="item">
    """
    lines  = text.split("\n")
    result = []

    # 섹션 헤더 패턴 (이모지로 시작하는 굵은 줄)
    header_prefixes = (
        "⚔️", "🏆", "📈", "📉", "💼", "₿", "📰", "🤖",
        "🔗", "🥇", "🥈", "🥉", "#", "🔥",
    )

    for line in lines:
        stripped = line.strip()

        # 구분선
        if set(stripped) <= set("─-—") and len(stripped) >= 5:
            result.append('<hr class="div">')
            continue

        # 빈 줄
        if not stripped:
            result.append('<div class="gap"></div>')
            continue

        # 인라인 마크다운 변환
        html_line = _inline_md(stripped)

        # 섹션 헤더 (이모지로 시작)
        if any(stripped.startswith(p) for p in header_prefixes) and len(stripped) < 80:
            result.append(f'<div class="section-header">{html_line}</div>')
            continue

        # 들여쓰기 항목
        if line.startswith("  ") or line.startswith("    "):
            result.append(f'<div class="item">{html_line}</div>')
            continue

        # 일반 줄
        result.append(f'<div class="line">{html_line}</div>')

    return "\n".join(result)


def _inline_md(text: str) -> str:
    """인라인 마크다운(*굵게*, `코드`)을 HTML로 변환 + 특수문자 이스케이프."""
    # HTML 이스케이프 (< > & 만)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # *굵게* → <b>굵게</b>  (마크다운 bold)
    text = re.sub(r"\*([^*]+)\*", r"<b>\1</b>", text)

    # `코드` → <code>코드</code>
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # ✅ ❌ 색상 처리
    text = text.replace("✅", '<span class="ok">✅</span>')
    text = text.replace("❌", '<span class="no">❌</span>')

    return text


def _wrap_html(body: str, 시각: str, phase: str, phase_emoji: str) -> str:
    """HTML 전체 템플릿에 body를 넣어 반환한다."""
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="robots" content="noindex">
<title>퀀트 리포트 — {시각}</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#0e1621;--surface:#17212b;--bubble:#182533;--border:#1f3045;
  --accent:#5cb8ff;--gold:#ffd54f;--green:#5dc97a;--red:#f06060;
  --orange:#ffa040;--yellow:#ffcc44;--muted:#4a6680;--text:#c8d8e8;--dim:#7a96b0;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);font-family:'Noto Sans KR',sans-serif;color:var(--text);min-height:100vh;}}

.topbar{{
  background:#111c27;border-bottom:1px solid var(--border);
  padding:14px 20px;display:flex;align-items:center;gap:12px;
  position:sticky;top:0;z-index:9;
}}
.topbar-icon{{width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#2AABEE,#1a7fc0);display:flex;align-items:center;justify-content:center;font-size:17px;flex-shrink:0;}}
.topbar h1{{font-size:14px;color:#ddeeff;font-weight:600;}}
.topbar p{{font-size:11px;color:var(--muted);margin-top:1px;}}
.topbar-right{{margin-left:auto;display:flex;gap:8px;align-items:center;}}
.phase-pill{{background:rgba(255,213,79,.14);border:1px solid rgba(255,213,79,.3);color:var(--gold);font-family:'JetBrains Mono',monospace;font-size:11px;padding:3px 10px;border-radius:20px;font-weight:600;}}
.idx-btn{{background:rgba(92,184,255,.12);border:1px solid rgba(92,184,255,.25);color:var(--accent);font-size:11px;padding:4px 11px;border-radius:8px;text-decoration:none;font-family:'JetBrains Mono',monospace;white-space:nowrap;}}
.idx-btn:hover{{background:rgba(92,184,255,.2);}}

.wrap{{max-width:600px;margin:0 auto;padding:16px 14px 60px;}}

.card{{background:var(--bubble);border:1px solid var(--border);border-radius:12px;padding:14px 16px;margin-bottom:12px;line-height:1.65;font-size:13px;}}

.section-header{{font-size:14px;font-weight:700;color:#ddeeff;margin:10px 0 5px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,.06);}}
.item{{color:var(--dim);font-size:12.5px;padding:2px 0 2px 10px;border-left:2px solid rgba(255,255,255,.06);margin:2px 0;}}
.line{{color:var(--text);font-size:13px;padding:2px 0;}}
.gap{{height:5px;}}
hr.div{{border:none;border-top:1px solid rgba(92,184,255,.1);margin:8px 0;}}

b{{color:#ddeeff;}}
code{{font-family:'JetBrains Mono',monospace;font-size:11px;background:rgba(255,255,255,.08);padding:1px 5px;border-radius:3px;color:var(--accent);}}
.ok{{color:var(--green);}} .no{{color:#906060;}}

.footer{{text-align:center;font-size:11px;color:var(--muted);padding:30px 0;font-family:'JetBrains Mono',monospace;}}
</style>
</head>
<body>
<div class="topbar">
  <div class="topbar-icon">⚔️</div>
  <div>
    <h1>퀀트 헤지펀드 리포트</h1>
    <p>{시각}</p>
  </div>
  <div class="topbar-right">
    <span class="phase-pill">{phase_emoji} {phase}</span>
    <a class="idx-btn" href="index.html">← 목록</a>
  </div>
</div>

<div class="wrap">
  <div class="card">
{body}
  </div>
</div>

<div class="footer">⚔️ 퀀트 헤지펀드 봇 · {시각}</div>
</body>
</html>"""


def _update_index(
    새_파일명: str,
    새_시각: str,
    새_phase: str,
    리포트_텍스트: str,
) -> None:
    """
    docs/index.html 을 갱신한다.
    기존 항목 목록을 읽어서 맨 앞에 새 항목을 추가한다.
    MAX_INDEX_ENTRIES 초과 시 오래된 것부터 제거.
    """
    index_path = os.path.join(DOCS_DIR, "index.html")

    # 기존 항목 파싱
    entries: list[tuple[str, str, str]] = []  # (파일명, 시각, phase)
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            content = f.read()
        # data-file 속성으로 기존 항목 추출
        for m in re.finditer(
            r'data-file="([^"]+)"[^>]*data-time="([^"]+)"[^>]*data-phase="([^"]*)"',
            content
        ):
            entries.append((m.group(1), m.group(2), m.group(3)))

    # 새 항목 맨 앞에 추가 (중복 제거)
    entries = [(f, t, p) for f, t, p in entries if f != 새_파일명]
    entries.insert(0, (새_파일명, 새_시각, 새_phase))

    # MAX_INDEX_ENTRIES 초과분 제거 (인덱스 목록에서)
    제거_대상 = entries[MAX_INDEX_ENTRIES:]   # 잘려나가는 오래된 항목
    entries = entries[:MAX_INDEX_ENTRIES]

    # [v5.9] 오래된 HTML 파일을 디스크에서도 삭제 (docs 폴더 무한 증가 방지)
    # 기존엔 인덱스 목록만 잘랐고 파일은 계속 쌓였음 → 레포 비대화/Pages 배포 지연
    for 옛파일, _, _ in 제거_대상:
        try:
            옛경로 = os.path.join(DOCS_DIR, 옛파일)
            if os.path.exists(옛경로):
                os.remove(옛경로)
                print(f"  🗑️  오래된 리포트 삭제: {옛파일}")
        except Exception:
            pass

    # 신호 개수 파싱 (요약용)
    buy_count  = 리포트_텍스트.count("📈 매수 신호") + 리포트_텍스트.count("🔥 강력 매수")
    sell_count = 리포트_텍스트.count("📉 매도 신호")
    요약 = f"매수 {buy_count}개"
    if sell_count > 0:
        요약 += f" · 매도 {sell_count}개"

    # 항목 HTML 생성
    items_html = ""
    for i, (fname, time_str, phase_str) in enumerate(entries):
        # 날짜/시각 파싱
        try:
            dt    = datetime.strptime(fname.replace(".html", ""), "%Y-%m-%d-%H%M")
            날짜  = dt.strftime("%Y.%m.%d")
            시각  = dt.strftime("%H:%M")
            요일  = ["월", "화", "수", "목", "금", "토", "일"][dt.weekday()]
        except Exception:
            날짜 = fname.replace(".html", "")
            시각 = ""
            요일 = ""

        phase_emoji = "📈"
        for emoji in ["🚀", "📈", "↔️", "📉", "🔴"]:
            if emoji in phase_str:
                phase_emoji = emoji
                break

        new_badge = ' <span class="new-badge">NEW</span>' if i == 0 else ""
        summary   = 요약 if i == 0 else ""

        items_html += f"""
    <a class="entry" href="{fname}"
       data-file="{fname}" data-time="{time_str}" data-phase="{phase_str}">
      <div class="entry-left">
        <div class="entry-date">{날짜} <span class="entry-dow">{요일}</span>{new_badge}</div>
        <div class="entry-time">{시각} KST</div>
      </div>
      <div class="entry-right">
        <span class="entry-phase">{phase_emoji} {phase_str}</span>
        {f'<span class="entry-summary">{summary}</span>' if summary else ''}
      </div>
    </a>"""

    total = len(entries)
    index_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="robots" content="noindex">
<title>퀀트 헤지펀드 — 리포트 히스토리</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#0e1621;--surface:#17212b;--bubble:#182533;--border:#1f3045;
  --accent:#5cb8ff;--gold:#ffd54f;--green:#5dc97a;--muted:#4a6680;--text:#c8d8e8;--dim:#7a96b0;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);font-family:'Noto Sans KR',sans-serif;color:var(--text);min-height:100vh;}}

.topbar{{background:#111c27;border-bottom:1px solid var(--border);padding:14px 20px;display:flex;align-items:center;gap:12px;}}
.topbar-icon{{width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#2AABEE,#1a7fc0);display:flex;align-items:center;justify-content:center;font-size:17px;}}
.topbar h1{{font-size:14px;color:#ddeeff;font-weight:600;}}
.topbar p{{font-size:11px;color:var(--muted);margin-top:1px;}}
.count-badge{{margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--dim);background:rgba(255,255,255,.04);border:1px solid var(--border);padding:3px 10px;border-radius:12px;}}

.wrap{{max-width:520px;margin:0 auto;padding:16px 14px 60px;}}
.section-title{{font-size:11px;color:var(--muted);font-family:'JetBrains Mono',monospace;letter-spacing:.06em;margin:12px 0 8px;text-transform:uppercase;}}

.entry{{
  display:flex;align-items:center;justify-content:space-between;
  background:var(--bubble);border:1px solid var(--border);
  border-radius:10px;padding:12px 14px;margin-bottom:6px;
  text-decoration:none;color:inherit;
  transition:border-color .15s,background .15s;
}}
.entry:hover{{border-color:rgba(92,184,255,.35);background:#1d2e42;}}
.entry:first-of-type{{border-color:rgba(92,184,255,.2);}}

.entry-left{{display:flex;flex-direction:column;gap:3px;}}
.entry-date{{font-size:14px;font-weight:600;color:#ddeeff;display:flex;align-items:center;gap:6px;}}
.entry-dow{{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);background:rgba(255,255,255,.06);padding:1px 5px;border-radius:4px;}}
.new-badge{{font-family:'JetBrains Mono',monospace;font-size:9px;background:rgba(92,184,255,.15);color:var(--accent);border:1px solid rgba(92,184,255,.25);padding:1px 6px;border-radius:8px;}}
.entry-time{{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);}}

.entry-right{{display:flex;flex-direction:column;align-items:flex-end;gap:4px;}}
.entry-phase{{font-family:'JetBrains Mono',monospace;font-size:11px;background:rgba(255,213,79,.1);color:var(--gold);border:1px solid rgba(255,213,79,.2);padding:2px 8px;border-radius:12px;}}
.entry-summary{{font-size:11px;color:var(--dim);}}

.delete-guide{{background:rgba(240,96,96,.06);border:1px solid rgba(240,96,96,.18);border-radius:10px;padding:12px 14px;margin-top:16px;font-size:12px;color:var(--dim);line-height:1.7;}}
.delete-guide b{{color:#ddeeff;}}
code{{font-family:'JetBrains Mono',monospace;font-size:10.5px;background:rgba(255,255,255,.08);padding:1px 5px;border-radius:3px;color:var(--accent);display:inline-block;margin:2px 0;}}

.footer{{text-align:center;font-size:11px;color:var(--muted);padding:30px 0;font-family:'JetBrains Mono',monospace;}}
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-icon">⚔️</div>
  <div>
    <h1>퀀트 헤지펀드 리포트 히스토리</h1>
    <p>클릭하면 해당 날짜 리포트가 열립니다</p>
  </div>
  <div class="count-badge">총 {total}개</div>
</div>

<div class="wrap">
  <div class="section-title">📋 리포트 목록 (최신순)</div>
{items_html}

  <div class="delete-guide">
    <b>🗑️ 히스토리 삭제 방법</b><br><br>
    <b>특정 날짜 삭제:</b><br>
    <code>git rm docs/2026-06-01-0900.html</code><br>
    <code>git commit -m "리포트 삭제"</code><br>
    <code>git push</code><br><br>
    <b>전체 초기화:</b><br>
    <code>git rm docs/*.html</code><br>
    <code>git commit -m "히스토리 초기화"</code><br>
    <code>git push</code><br><br>
    <span style="color:var(--muted);font-size:11px">※ 다음 실행 시 index.html이 자동 재생성됩니다</span>
  </div>
</div>

<div class="footer">⚔️ 퀀트 헤지펀드 봇 · 최근 {MAX_INDEX_ENTRIES}개 보관</div>
</body>
</html>"""

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
    print(f"  📋 인덱스 갱신: docs/index.html ({total}개 항목)")
