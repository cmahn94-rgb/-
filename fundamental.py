"""
fundamental.py — 기본적 분석 보조 모듈 (v1)
=============================================
기술적 분석을 보조하는 5가지 기본적 분석 지표를 제공한다.

[5가지 기본적 분석]
① PER + PBR 저평가 필터   — 기술적 신호 + 가치 저평가 동시 확인
② 애널리스트 목표주가      — 전문가 consensus 대비 현재 가격 괴리율
③ 실적 발표일 캘린더       — 실적 직전 매수 위험 경고
④ ROE + 매출성장률         — 기업 품질 필터 (적자·저성장 기업 제외)
⑤ FCF 수익률               — 회계 조작 불가한 실질 현금 창출력

[데이터 소스]
- 미국주식: yfinance .info (무료, 안정적)
- 한국주식: yfinance .info (불안정) + DART API (무료, 키 필요)
  → DART_API_KEY: https://opendart.fss.or.kr (무료 가입 즉시 발급)
  → GitHub Secrets: DART_API_KEY

[통합 방식]
- 보너스/패널티: signals.py calc_signals() 반환값에 포함
- 알림 표시: scheduler_job.py analyze_one()에서 신호 블록에 추가
- 실적 경고: 우선순위 점수 -5점 + 경고 문구 표시

[한국주식 한계]
- PER/PBR: yfinance KR 데이터 불안정 → 있으면 사용, 없으면 생략
- 목표주가: 무료 소스 없음 → 미국주식 전용
- ROE/매출성장: DART API로 보완 (DART_API_KEY 없으면 yfinance 시도)
"""

from __future__ import annotations
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────
# 캐시 (실행당 1회만 API 호출)
# ─────────────────────────────────────────
_info_cache: dict[str, dict] = {}
_dart_cache: dict[str, dict] = {}


def _dart_get(url: str, params: dict, timeout: int = 10) -> dict | None:
    """
    DART API GET 요청 래퍼 — 503/429 시 1회 재시도.
    병렬 ThreadPoolExecutor에서 52개 종목이 동시 호출할 때
    순간 과호출로 503이 날 수 있어 0.05초 딜레이 + 1회 재시도.
    """
    for attempt in range(2):
        try:
            time.sleep(0.05 * (attempt + 1))  # 1차: 0.05초, 재시도: 0.1초
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 503) and attempt == 0:
                time.sleep(1.0)  # 과호출 시 1초 대기 후 재시도
                continue
            return None
        except Exception:
            if attempt == 0:
                time.sleep(0.5)
    return None


def _fetch_quote_summary(ticker: str) -> dict:
    """
    yfinance YfData 싱글톤의 crumb/세션을 재사용해서
    v10/quoteSummary를 직접 호출한다.

    [중학생 설명]
    야후 파이낸스 .info()가 막혔을 때 우회 통로다.
    yfinance가 이미 받아둔 crumb(인증 토큰)을 빌려서
    직접 API를 호출하므로 새로운 인증 과정이 없어 빠르고 안정적이다.

    [버전 보호]
    YfData는 yfinance 내부 클래스 (공개 API 아님).
    업그레이드 시 클래스명·메서드명이 바뀔 수 있으므로
    3단계 폴백으로 보호한다:
      1) YfData().get_raw_json (현재 yfinance 0.2.x)
      2) YfData()._data.get  (일부 버전 대안)
      3) requests 직접 호출 (yfinance 완전 독립)
    """
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
    params = {
        "modules": "financialData,defaultKeyStatistics,summaryDetail,assetProfile",
        "corsDomain": "finance.yahoo.com",
        "formatted": "false",
    }

    def _parse_result(result):
        if not result:
            return {}
        data = result.get("quoteSummary", {}).get("result", [{}])
        if not data:
            return {}
        merged = {}
        for module in data[0].values():
            if isinstance(module, dict):
                merged.update(module)
        return {k: v for k, v in merged.items() if v is not None}

    # 1단계: YfData().get_raw_json (기존 방식)
    try:
        from yfinance.data import YfData
        yfdata = YfData()
        if hasattr(yfdata, "get_raw_json"):
            result = yfdata.get_raw_json(url, params=params)
            parsed = _parse_result(result)
            if parsed:
                return parsed
    except Exception:
        pass

    # 2단계: requests 직접 호출 — yfinance 내부 구조 변경과 무관하게 동작
    try:
        import requests as _req
        import yfinance as _yf
        # yfinance가 캐시한 쿠키/헤더를 최대한 재활용
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }
        resp = _req.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            parsed = _parse_result(resp.json())
            if parsed:
                return parsed
    except Exception:
        pass

    return {}


def _get_yf_info(ticker: str) -> dict:
    """
    yfinance .info를 캐시 + 재시도 + 직접 API 폴백으로 가져온다.

    [중학생 설명]
    야후 파이낸스 API는 GitHub Actions IP에서 가끔 401 오류를 낸다.
    아래 3단계로 이 문제를 해결한다:
      1단계: 캐시에 이미 있으면 즉시 반환 (API 호출 없음)
      2단계: yfinance .info() 를 최대 3번 재시도 (각 실패 후 딜레이)
      3단계: .info() 완전 실패 시 quoteSummary 직접 호출로 폴백
    
    핵심: 외부 curl_cffi 세션을 새로 만들지 않는다.
    yfinance 1.4.1은 내부적으로 curl_cffi + crumb을 관리하므로
    새 세션을 넘기면 오히려 crumb 없는 상태로 요청해서 401이 더 많이 남.
    """
    if ticker in _info_cache:
        return _info_cache[ticker]

    import yfinance as yf

    # 최대 3회 재시도: 1초 → 2초 → 4초 백오프
    info = {}
    for attempt in range(3):
        try:
            info = yf.Ticker(ticker).info or {}
            info = {k: v for k, v in info.items() if v is not None}
            if info:
                break  # 성공
        except Exception:
            pass
        if attempt < 2:
            time.sleep(2 ** attempt)  # 1s, 2s

    # .info()가 비어있으면 quoteSummary 직접 호출로 폴백
    if not info:
        info = _fetch_quote_summary(ticker)

    _info_cache[ticker] = info
    return info


def _safe(val, default=None):
    """None/NaN을 기본값으로 변환."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if (f != f) else f  # nan check
    except Exception:
        return default


# ─────────────────────────────────────────
# ① PER + PBR 저평가 필터
# ─────────────────────────────────────────

def get_valuation(ticker: str, market: str) -> dict:
    """
    PER(주가수익비율)과 PBR(주가순자산비율)을 가져온다.

    [중학생 설명]
    PER: 이 회사 주식을 사면 몇 년치 이익을 사는 것인가?
         PER 10 = 10년치 이익을 주고 사는 셈. 낮을수록 싸다.
    PBR: 이 회사를 지금 당장 팔면 주가보다 자산이 얼마나 되는가?
         PBR 0.8 = 회사 순자산보다 20% 싸게 거래 중 = 매력적

    반환값: {
        "per": float | None,
        "pbr": float | None,
        "per_보너스": bool,   PER < 업종 평균 추정치 * 0.8 → True
        "pbr_보너스": bool,   PBR < 1.0 → True
        "per_패널티": bool,   PER > 50 (고평가) → True
        "표시문구":   str,
    }
    """
    info = _get_yf_info(ticker)
    per  = _safe(info.get("trailingPE"))
    pbr  = _safe(info.get("priceToBook"))

    per_보너스 = False
    pbr_보너스 = False
    per_패널티 = False

    if per is not None:
        # ⑪ 수정: yfinance에 sectorPE 필드 없음 → 항상 기본값 20 사용되던 문제
        # 보수적 고정 기준 25배 사용 (업종 무관 일반 기업 평균)
        # PER < 20 → 저평가 보너스, PER > 50 → 고평가 패널티
        if per > 0:
            per_보너스 = per < 20.0   # 25 * 0.8 = 20
            per_패널티 = per > 50

    if pbr is not None and pbr > 0:
        pbr_보너스 = pbr < 1.0

    # 표시 문구 생성
    parts = []
    if per is not None:  parts.append(f"PER {per:.1f}")
    if pbr is not None and pbr > 0:  parts.append(f"PBR {pbr:.2f}")
    표시 = " | ".join(parts) if parts else ""

    return {
        "per":       per,
        "pbr":       pbr,
        "per_보너스": per_보너스,
        "pbr_보너스": pbr_보너스,
        "per_패널티": per_패널티,
        "표시문구":   표시,
    }


# ─────────────────────────────────────────
# ② 애널리스트 목표주가 괴리율
# ─────────────────────────────────────────

def get_analyst_target(ticker: str, 현재가: float, market: str) -> dict:
    """
    애널리스트 평균 목표주가와 현재가 대비 괴리율을 가져온다.

    [중학생 설명]
    증권사 전문가들이 "이 주식은 앞으로 이 가격까지 오를 것"이라고 예측한 값.
    현재가가 목표주가보다 30% 낮다면? 전문가들이 30% 오를 것으로 보고 있다는 뜻.
    단, 애널리스트도 틀릴 수 있고 의견 수가 적으면 신뢰도가 낮다.

    미국주식 전용 (한국주식: 무료 소스 없음)

    반환값: {
        "목표주가":    float | None,
        "괴리율":      float | None,   (목표주가 - 현재가) / 현재가 * 100
        "추천":        str,            "강매수"/"매수"/"보유"/"매도" 등
        "의견수":      int,
        "점수보너스":  float,          괴리율 > 20% → +5점, > 10% → +2점
        "표시문구":    str,
    }
    """
    if market not in ("US",):
        return {"목표주가": None, "괴리율": None, "추천": "",
                "의견수": 0, "점수보너스": 0, "표시문구": ""}

    info = _get_yf_info(ticker)
    목표주가 = _safe(info.get("targetMeanPrice"))
    추천_숫자 = _safe(info.get("recommendationMean"))
    의견수   = int(_safe(info.get("numberOfAnalystOpinions"), 0))

    괴리율     = None
    점수보너스 = 0.0
    추천_문자  = ""

    if 목표주가 and 현재가 > 0:
        괴리율 = (목표주가 - 현재가) / 현재가 * 100
        if   괴리율 > 20: 점수보너스 = 5.0
        elif 괴리율 > 10: 점수보너스 = 2.0

    # 추천 숫자 → 문자 변환 (1=강매수, 2=매수, 3=보유, 4=매도, 5=강매도)
    if 추천_숫자:
        if   추천_숫자 <= 1.5: 추천_문자 = "강매수"
        elif 추천_숫자 <= 2.5: 추천_문자 = "매수"
        elif 추천_숫자 <= 3.5: 추천_문자 = "보유"
        elif 추천_숫자 <= 4.5: 추천_문자 = "매도"
        else:                   추천_문자 = "강매도"

    # 표시 문구
    parts = []
    if 목표주가:
        통화 = "$" if market == "US" else "₩"
        parts.append(f"목표주가 {통화}{목표주가:,.0f}")
    if 괴리율 is not None:
        parts.append(f"괴리율 {괴리율:+.1f}%")
    if 추천_문자:
        신뢰 = f" ({의견수}명)" if 의견수 >= 3 else " ⚠️의견부족"
        parts.append(f"애널의견 {추천_문자}{신뢰}")
    표시 = " | ".join(parts)

    return {
        "목표주가":   목표주가,
        "괴리율":     괴리율,
        "추천":       추천_문자,
        "의견수":     의견수,
        "점수보너스": 점수보너스,
        "표시문구":   표시,
    }


# ─────────────────────────────────────────
# ③ 실적 발표일 캘린더
# ─────────────────────────────────────────

def get_earnings_alert(ticker: str, market: str) -> dict:
    """
    실적 발표일을 확인하고 발표 임박 여부를 반환한다.

    [중학생 설명]
    회사가 "지난 3개월 동안 얼마 벌었는지" 발표하는 날이 실적 발표일.
    이 날 전후로 주가가 크게 흔들린다.
    발표 3일 전에 매수하면 "발표가 나쁘면 바로 손절"이라는 위험이 생긴다.
    → 발표 D-5 이내면 경고 표시 + 우선순위 -5점

    미국: yfinance earningsDate 사용
    한국: DART 공시 API 사용 (DART_API_KEY 필요)

    반환값: {
        "발표일":        str | None,   "2026-05-15" 형식
        "D_day":         int | None,   오늘로부터 몇일 후 (-면 지남)
        "임박_경고":     bool,
        "점수패널티":    float,        임박 시 -5점
        "표시문구":      str,
    }
    """
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    발표일_str = None
    d_day      = None
    임박       = False
    패널티     = 0.0

    # ── 미국주식: yfinance ──────────────────────────────
    if market == "US":
        try:
            info = _get_yf_info(ticker)
            ed   = info.get("earningsDate")
            if ed:
                # earningsDate는 타임스탬프 또는 리스트
                if isinstance(ed, (list, tuple)):
                    ed = ed[0]
                if isinstance(ed, (int, float)):
                    from datetime import timezone
                    발표일 = datetime.fromtimestamp(ed, tz=timezone.utc)\
                                     .astimezone(ZoneInfo("Asia/Seoul"))
                else:
                    발표일 = pd.Timestamp(ed).to_pydatetime()
                발표일_str = 발표일.strftime("%Y-%m-%d")
                d_day = (발표일.date() - now.date()).days
                if 0 <= d_day <= 5:
                    임박   = True
                    패널티 = -5.0
        except Exception:
            pass

    # ── 한국주식: DART API ──────────────────────────────
    elif market == "KR":
        dart_result = _get_dart_earnings_date(ticker)
        if dart_result:
            발표일_str = dart_result
            try:
                발표일 = datetime.strptime(발표일_str, "%Y-%m-%d")\
                                 .replace(tzinfo=ZoneInfo("Asia/Seoul"))
                d_day = (발표일.date() - now.date()).days
                if 0 <= d_day <= 5:
                    임박   = True
                    패널티 = -5.0
            except Exception:
                pass

    # 표시 문구
    표시 = ""
    if 발표일_str and d_day is not None:
        if 임박:
            표시 = f"⚠️ 실적발표 D-{d_day} ({발표일_str}) — 진입 주의"
        elif d_day < 0:
            표시 = f"실적발표 {abs(d_day)}일 전 ({발표일_str})"
        else:
            표시 = f"실적발표 D+{d_day} ({발표일_str})"

    return {
        "발표일":     발표일_str,
        "D_day":      d_day,
        "임박_경고":  임박,
        "점수패널티": 패널티,
        "표시문구":   표시,
    }


def _get_dart_corp_code(ticker: str) -> str | None:
    """종목코드(6자리)로 DART 고유번호(corp_code)를 조회한다."""
    dart_key = os.getenv("DART_API_KEY", "")
    if not dart_key:
        return None

    # 종목코드 추출: 005930.KS → 005930
    code = ticker.replace(".KS", "").replace(".KQ", "").zfill(6)
    cache_key = f"dart_corp_{code}"
    if cache_key in _dart_cache:
        return _dart_cache[cache_key]

    try:
        data = _dart_get(
            "https://opendart.fss.or.kr/api/company.json",
            params={"crtfc_key": dart_key, "stock_code": code},
        )
        if data:
            corp_code = data.get("corp_code")
            _dart_cache[cache_key] = corp_code
            return corp_code
    except Exception:
        pass
    return None


def _get_dart_earnings_date(ticker: str) -> str | None:
    """DART에서 최근 실적 공시일을 가져온다."""
    dart_key = os.getenv("DART_API_KEY", "")
    if not dart_key:
        return None

    # [버그수정] list.json은 stock_code가 아닌 corp_code(8자리) 필요
    corp_code = _get_dart_corp_code(ticker)
    if not corp_code:
        return None

    try:
        now = datetime.now()
        bgn = (now - timedelta(days=30)).strftime("%Y%m%d")
        end = (now + timedelta(days=60)).strftime("%Y%m%d")

        data = _dart_get(
            "https://opendart.fss.or.kr/api/list.json",
            params={
                "crtfc_key": dart_key,
                "corp_code":  corp_code,
                "bgn_de":     bgn,
                "end_de":     end,
                "pblntf_ty":  "A",
                "page_count": 10,
            },
        )
        if not data:
            return None
        items = data.get("list", [])
        # 실적 관련 공시 키워드 필터
        keywords = ["사업보고서", "분기보고서", "반기보고서"]
        for item in items:
            if any(kw in item.get("report_nm", "") for kw in keywords):
                return item.get("rcept_dt", "")[:8]  # YYYYMMDD
    except Exception:
        pass
    return None


# ─────────────────────────────────────────
# ④ ROE + 매출성장률 품질 필터
# ─────────────────────────────────────────

def get_quality_metrics(ticker: str, market: str) -> dict:
    """
    ROE(자기자본이익률)와 매출성장률로 기업 품질을 평가한다.

    [중학생 설명]
    ROE: 내가 투자한 돈으로 회사가 얼마나 잘 벌고 있나?
         ROE 15% = 100원 투자하면 15원 이익 = 워런 버핏 기준 우량 기업
    매출성장률: 작년보다 매출이 얼마나 늘었나?
         20% 이상 = 빠르게 성장하는 회사

    보너스: ROE > 15% AND 매출성장 > 10% → +1점
    패널티: ROE < 0 (적자 기업) → -1점

    반환값: {
        "roe":           float | None,   0~1 사이 (0.15 = 15%)
        "매출성장률":    float | None,   0~1 사이 (0.20 = 20%)
        "영업이익률":    float | None,
        "부채비율":      float | None,
        "품질_보너스":   bool,
        "적자_패널티":   bool,
        "표시문구":      str,
    }
    """
    roe = 매출성장 = 영업이익률 = 부채비율 = None
    품질_보너스 = False
    적자_패널티 = False

    # ── 미국주식: yfinance ──────────────────────────────
    if market == "US":
        info = _get_yf_info(ticker)
        roe       = _safe(info.get("returnOnEquity"))
        매출성장   = _safe(info.get("revenueGrowth"))
        영업이익률 = _safe(info.get("operatingMargins"))
        d2e        = _safe(info.get("debtToEquity"))
        부채비율   = (d2e / 100) if d2e else None

    # ── 한국주식: DART API 우선, 실패 시 yfinance ───────
    elif market == "KR":
        dart_data = _get_dart_financials(ticker)
        if dart_data:
            roe       = dart_data.get("roe")
            매출성장   = dart_data.get("revenue_growth")
            영업이익률 = dart_data.get("operating_margin")
            부채비율   = dart_data.get("debt_ratio")
        else:
            # DART 실패 시 yfinance 시도 (불안정하지만 없는 것보다 나음)
            info = _get_yf_info(ticker)
            roe       = _safe(info.get("returnOnEquity"))
            매출성장   = _safe(info.get("revenueGrowth"))
            영업이익률 = _safe(info.get("operatingMargins"))

    # 보너스/패널티 판단
    if roe is not None:
        if roe < 0:
            적자_패널티 = True
        elif roe >= 0.15 and 매출성장 and 매출성장 >= 0.10:
            품질_보너스 = True

    # 표시 문구
    parts = []
    if roe       is not None: parts.append(f"ROE {roe*100:.1f}%")
    if 매출성장  is not None: parts.append(f"매출성장 {매출성장*100:+.1f}%")
    if 영업이익률 is not None: parts.append(f"영업이익률 {영업이익률*100:.1f}%")
    표시 = " | ".join(parts)

    return {
        "roe":        roe,
        "매출성장률": 매출성장,
        "영업이익률": 영업이익률,
        "부채비율":   부채비율,
        "품질_보너스": 품질_보너스,
        "적자_패널티": 적자_패널티,
        "표시문구":    표시,
    }


def _get_dart_financials(ticker: str) -> dict | None:
    """
    DART API로 한국주식 재무지표를 가져온다.
    ROE, 매출성장률, 영업이익률, 부채비율을 반환한다.
    """
    dart_key = os.getenv("DART_API_KEY", "")
    if not dart_key:
        return None

    stock_code = ticker.replace(".KS", "").replace(".KQ", "").zfill(6)
    cache_key  = f"dart_fin_{stock_code}"
    if cache_key in _dart_cache:
        return _dart_cache[cache_key]

    # fnlttSinglAcntAll은 corp_code(8자리) 필요
    corp_code = _get_dart_corp_code(ticker)
    if not corp_code:
        _dart_cache[cache_key] = None
        return None

    try:
        now = datetime.now()
        year = now.year - 1 if now.month < 5 else now.year

        resp = requests.get(
            "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
            params={
                "crtfc_key": dart_key,
                "corp_code":  corp_code,  # 8자리
                "bsns_year":  str(year),
                "reprt_code": "11011",
                "fs_div":     "CFS",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            _dart_cache[cache_key] = None
            return None

        items = resp.json().get("list", [])
        if not items:
            _dart_cache[cache_key] = None
            return None

        # 계정과목별로 데이터 추출
        def _find(accounts: list[str]) -> float | None:
            for item in items:
                nm = item.get("account_nm", "")
                if any(a in nm for a in accounts):
                    try:
                        v = item.get("thstrm_amount", "").replace(",", "")
                        if v and v != "-":
                            return float(v)
                    except Exception:
                        pass
            return None

        매출     = _find(["매출액", "수익(매출액)", "매출", "영업수익"])
        영업이익 = _find(["영업이익", "영업손익"])
        순이익   = _find(["당기순이익", "당기순손익"])
        자본     = _find(["자본총계", "자본합계"])
        부채     = _find(["부채총계", "부채합계"])

        result = {}
        if 순이익 and 자본 and 자본 != 0:
            result["roe"] = 순이익 / 자본
        if 영업이익 and 매출 and 매출 != 0:
            result["operating_margin"] = 영업이익 / 매출
        if 부채 and 자본 and 자본 != 0:
            result["debt_ratio"] = 부채 / 자본

        # 전년도 매출 가져와서 성장률 계산
        resp2 = requests.get(
            "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
            params={
                "crtfc_key": dart_key,
                "corp_code":  corp_code,  # 8자리
                "bsns_year":  str(year - 1),
                "reprt_code": "11011",
                "fs_div":     "CFS",
            },
            timeout=10,
        )
        if resp2.status_code == 200:
            prev_items = resp2.json().get("list", [])
            def _find_prev(accounts):
                for item in prev_items:
                    nm = item.get("account_nm", "")
                    if any(a in nm for a in accounts):
                        try:
                            v = item.get("thstrm_amount","").replace(",","")
                            if v and v != "-":
                                return float(v)
                        except Exception:
                            pass
                return None

            prev_매출 = _find_prev(["매출액","수익(매출액)","매출","영업수익"])
            if 매출 and prev_매출 and prev_매출 != 0:
                result["revenue_growth"] = (매출 - prev_매출) / abs(prev_매출)

        _dart_cache[cache_key] = result if result else None
        return result if result else None

    except Exception as e:
        print(f"⚠️ DART 재무데이터 오류 ({ticker}): {e}")
        _dart_cache[cache_key] = None
        return None


# ─────────────────────────────────────────
# ⑤ FCF(잉여현금흐름) 수익률
# ─────────────────────────────────────────

def get_fcf_yield(ticker: str, market: str) -> dict:
    """
    FCF(잉여현금흐름) 수익률을 계산한다.

    [중학생 설명]
    FCF = 영업으로 번 돈 - 공장/설비 투자 비용
    → 회사가 실제로 주머니에 쥐는 현금
    PER은 회계 숫자를 조작할 수 있지만 FCF는 조작이 어렵다.

    FCF 수익률 = FCF / 시가총액
    5% 이상 = 시가총액의 5%를 현금으로 버는 기업 = 매력적
    음수    = 돈 잃는 기업 = 위험

    반환값: {
        "fcf":        float | None,   원 단위
        "시가총액":   float | None,
        "fcf_수익률": float | None,   0.05 = 5%
        "fcf_보너스": bool,           fcf_yield > 0.05
        "fcf_패널티": bool,           fcf < 0
        "표시문구":   str,
    }
    """
    fcf = 시가총액 = fcf_수익률 = None
    fcf_보너스 = False
    fcf_패널티 = False

    # ── 미국주식: yfinance ──────────────────────────────
    if market == "US":
        info  = _get_yf_info(ticker)
        fcf   = _safe(info.get("freeCashflow"))
        시가총액 = _safe(info.get("marketCap"))

    # ── 한국주식: DART 현금흐름표 ────────────────────────
    elif market == "KR":
        dart_key = os.getenv("DART_API_KEY", "")
        if dart_key:
            fcf, 시가총액 = _get_dart_fcf(ticker)
        else:
            # DART 없으면 yfinance 시도
            info  = _get_yf_info(ticker)
            fcf   = _safe(info.get("freeCashflow"))
            시가총액 = _safe(info.get("marketCap"))

    # FCF 수익률 계산
    if fcf is not None and 시가총액 and 시가총액 > 0:
        fcf_수익률 = fcf / 시가총액
        fcf_보너스 = fcf_수익률 > 0.05
        fcf_패널티 = fcf < 0

    # 표시 문구
    표시 = ""
    if fcf_수익률 is not None:
        표시 = f"FCF수익률 {fcf_수익률*100:.1f}%"
        if fcf_패널티:
            표시 += " ⚠️FCF음수"

    return {
        "fcf":        fcf,
        "시가총액":   시가총액,
        "fcf_수익률": fcf_수익률,
        "fcf_보너스": fcf_보너스,
        "fcf_패널티": fcf_패널티,
        "표시문구":   표시,
    }


def _get_dart_fcf(ticker: str) -> tuple[float | None, float | None]:
    """DART 현금흐름표에서 FCF와 시가총액을 계산한다."""
    dart_key = os.getenv("DART_API_KEY", "")
    if not dart_key:
        return None, None

    stock_code = ticker.replace(".KS","").replace(".KQ","").zfill(6)
    cache_key  = f"dart_fcf_{stock_code}"
    if cache_key in _dart_cache:
        return _dart_cache[cache_key]

    corp_code = _get_dart_corp_code(ticker)
    if not corp_code:
        _dart_cache[cache_key] = (None, None)
        return None, None

    try:
        now  = datetime.now()
        year = now.year - 1 if now.month < 5 else now.year

        resp = requests.get(
            "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
            params={
                "crtfc_key": dart_key,
                "corp_code":  corp_code,  # 8자리
                "bsns_year":  str(year),
                "reprt_code": "11011",
                "fs_div":     "CFS",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            _dart_cache[cache_key] = (None, None)
            return None, None

        items = resp.json().get("list", [])

        def _find(accounts):
            for item in items:
                nm = item.get("account_nm","")
                if any(a in nm for a in accounts):
                    try:
                        v = item.get("thstrm_amount","").replace(",","")
                        if v and v != "-":
                            return float(v)
                    except Exception:
                        pass
            return None

        영업CF   = _find(["영업활동 현금흐름", "영업활동으로 인한 현금흐름"])
        자본적지출 = _find(["유형자산의 취득", "유형자산 취득"])

        fcf = None
        if 영업CF is not None and 자본적지출 is not None:
            # 자본적지출은 음수로 표시됨
            fcf = 영업CF + 자본적지출  # 음수 + 음수 = 빼는 효과

        # 시가총액: DART에 없으므로 yfinance에서 가져옴 (캐시 활용)
        시가총액 = None
        try:
            info = _get_yf_info(ticker)  # 캐시 재활용 → 추가 API 호출 없음
            시가총액 = _safe(info.get("marketCap"))
        except Exception:
            pass

        result = (fcf, 시가총액)
        _dart_cache[cache_key] = result
        return result

    except Exception as e:
        print(f"⚠️ DART FCF 오류 ({ticker}): {e}")
        _dart_cache[cache_key] = (None, None)
        return None, None


# ─────────────────────────────────────────
# 통합 함수: 5가지를 한 번에 조회
# ─────────────────────────────────────────

def get_fundamentals(ticker: str, name: str, market: str,
                     현재가: float) -> dict:
    """
    5가지 기본적 분석 지표를 한 번에 조회한다.
    signals.py calc_signals() 반환값에 포함시키고
    scheduler_job.py에서 알림 표시와 점수 보정에 활용한다.

    반환값: {
        "valuation":  PER/PBR 결과,
        "analyst":    목표주가 결과,
        "earnings":   실적발표일 결과,
        "quality":    ROE/매출성장 결과,
        "fcf":        FCF 수익률 결과,
        "fa_보너스":  float,  기본적 분석 보너스 점수 합산
        "fa_패널티":  float,  기본적 분석 패널티 점수 합산
        "fa_표시":    str,    알림에 표시할 한 줄 요약
    }
    """
    # 🔴 문제1 수정: 암호화폐는 기본적 분석 대상 아님 → 즉시 빈 결과 반환
    # (UPBIT:BTC를 yfinance에 조회해서 404 발생하던 문제 해결)
    if market in ("CRYPTO", "CRYPTO_KRW"):
        return {
            "valuation": {"per": None, "pbr": None, "per_보너스": False,
                          "pbr_보너스": False, "per_패널티": False, "표시문구": ""},
            "analyst":   {"목표주가": None, "괴리율": None, "추천": "",
                          "의견수": 0, "점수보너스": 0, "표시문구": ""},
            "earnings":  {"발표일": None, "D_day": None, "임박_경고": False,
                          "점수패널티": 0, "표시문구": ""},
            "quality":   {"roe": None, "매출성장률": None, "품질_보너스": False,
                          "적자_패널티": False, "표시문구": ""},
            "fcf":       {"fcf_수익률": None, "fcf_보너스": False,
                          "fcf_패널티": False, "표시문구": ""},
            "fa_보너스": 0, "fa_패널티": 0, "fa_표시": "",
        }

    # 한국주식은 DART 호출 2~3회 발생 가능 → 0.1초 딜레이
    if market == "KR" and os.getenv("DART_API_KEY"):
        time.sleep(0.1)

    valuation = get_valuation(ticker, market)
    analyst   = get_analyst_target(ticker, 현재가, market)
    earnings  = get_earnings_alert(ticker, market)
    quality   = get_quality_metrics(ticker, market)
    fcf       = get_fcf_yield(ticker, market)

    # 보너스 합산
    fa_보너스 = sum([
        1.0 if valuation["per_보너스"]  else 0,
        0.5 if valuation["pbr_보너스"]  else 0,
        analyst["점수보너스"],
        1.0 if quality["품질_보너스"]   else 0,
        1.0 if fcf["fcf_보너스"]        else 0,
    ])

    # 패널티 합산
    fa_패널티 = sum([
        -1.0 if valuation["per_패널티"] else 0,
        earnings["점수패널티"],          # 실적 임박 시 -5.0
        -1.0 if quality["적자_패널티"]  else 0,
        -1.0 if fcf["fcf_패널티"]       else 0,
    ])

    # 알림 표시 줄 생성
    표시_parts = []
    if valuation["표시문구"]:  표시_parts.append(valuation["표시문구"])
    if quality["표시문구"]:    표시_parts.append(quality["표시문구"])
    if fcf["표시문구"]:        표시_parts.append(fcf["표시문구"])
    if analyst["표시문구"]:    표시_parts.append(analyst["표시문구"])
    if earnings["표시문구"]:   표시_parts.append(earnings["표시문구"])

    fa_표시 = " | ".join(표시_parts) if 표시_parts else ""

    return {
        "valuation": valuation,
        "analyst":   analyst,
        "earnings":  earnings,
        "quality":   quality,
        "fcf":       fcf,
        "fa_보너스": fa_보너스,
        "fa_패널티": fa_패널티,
        "fa_표시":   fa_표시,
    }


# ═══════════════════════════════════════════════════════════
# 한국 기관/외국인 수급 팩터
# ═══════════════════════════════════════════════════════════
"""
[추가 배경]
한국 주식시장은 기관·외국인 수급이 주가에 미치는 영향이 매우 크다.
개인(개미)이 살 때 기관·외국인이 팔면 단기 반등이 꺾이는 패턴이 반복됨.
반대로 기관+외국인이 동시 순매수할 때 상승 확률이 크게 높아진다.

[데이터 소스]
KRX 정보데이터시스템 (data.krx.co.kr) — 공식, 무료, 인증 불필요
폴백: 네이버금융 비공식 JSON API

[수급 팩터 판단 기준]
① 외국인 순매수 연속 3일 이상   → 스마트머니 유입 (보너스 +1)
② 기관 순매수 연속 3일 이상     → 대형 매수세 확인 (보너스 +1)
③ 외국인+기관 동시 순매수       → 가장 강력한 신호 (보너스 +1)
④ 외국인 보유율 5일 추세 상승   → 장기 자금 유입 중 (보너스 +0.5)
⑤ 외국인/기관 동시 순매도 연속  → 매수 신호 신뢰도 하락 (패널티 -1)
"""

# 수급 데이터 캐시 (종목별, 실행당 1회 호출)
_supply_demand_cache: dict[str, dict] = {}

# 종목코드 → KRX ISIN 코드 변환 캐시
_isin_cache: dict[str, str] = {}


def _get_isin(ticker_ks: str) -> str | None:
    """
    005930.KS → KR7005930003 형태의 ISIN 코드 변환.
    KRX API 요청 시 ISIN 코드가 필요하다.
    """
    code = ticker_ks.replace(".KS", "").replace(".KQ", "").strip()
    if code in _isin_cache:
        return _isin_cache[code]
    try:
        url = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        params = {
            "bld": "dbms/comm/finder/finder_stkisu",
            "mktsel": "ALL",
            "searchText": code,
            "typeNo": "0",
        }
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer":    "https://data.krx.co.kr",
            "Origin":     "https://data.krx.co.kr",
            "Accept":     "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
            "X-Requested-With": "XMLHttpRequest",
        }
        r = requests.post(url, data=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        items = data.get("block1", [])
        for item in items:
            if item.get("short_isin_cd", "") == code:
                isin = item.get("isin_cd", "")
                _isin_cache[code] = isin
                return isin
    except Exception:
        pass
    return None


def _fetch_krx_supply_demand(ticker_ks: str, days: int = 10) -> pd.DataFrame | None:
    """
    KRX 정보데이터시스템에서 투자자별 순매수 데이터를 가져온다.

    반환 컬럼:
    - 날짜, 외국인_순매수, 기관합계_순매수, 개인_순매수
    - 외국인_보유율
    """
    try:
        isin = _get_isin(ticker_ks)
        if not isin:
            return None

        today = datetime.now(ZoneInfo("Asia/Seoul"))
        # 주말·공휴일 여유분 포함해서 days*2 달력일 조회
        start = today - timedelta(days=days * 2)
        url = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        params = {
            "bld":        "dbms/MDC/STAT/standard/MDCSTAT02303",
            "locale":     "ko_KR",
            "isuCd":      isin,
            "strtDd":     start.strftime("%Y%m%d"),
            "endDd":      today.strftime("%Y%m%d"),
            "share":      "1",
            "money":      "1",
            "csvxls_isNo": "false",
        }
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer":          "https://data.krx.co.kr",
            "Origin":           "https://data.krx.co.kr",
            "Content-Type":     "application/x-www-form-urlencoded",
            "Accept":           "application/json, text/javascript, */*; q=0.01",
            "Accept-Language":  "ko-KR,ko;q=0.9,en-US;q=0.8",
            "X-Requested-With": "XMLHttpRequest",
        }
        # 재시도: KRX는 첫 요청이 실패해도 한 번 더 시도하면 성공하는 경우 있음
        r = None
        for _attempt in range(2):
            try:
                r = requests.post(url, data=params, headers=headers, timeout=12)
                if r.status_code == 200:
                    break
            except Exception:
                if _attempt == 0:
                    time.sleep(1)
        if r is None or r.status_code != 200:
            return None
        if r.status_code != 200:
            return None

        data = r.json()
        rows = data.get("output", [])
        if not rows:
            return None

        records = []
        for row in rows:
            try:
                # KRX 반환 필드명 (숫자에 쉼표 포함)
                def _int(v):
                    return int(str(v).replace(",", "").replace(" ", "") or 0)

                records.append({
                    "날짜":            row.get("TRD_DD", ""),
                    "외국인_순매수":    _int(row.get("FRGN_NETBUY_TRDVOL", 0)),
                    "기관합계_순매수":  _int(row.get("ORGN_NETBUY_TRDVOL", 0)),
                    "개인_순매수":      _int(row.get("INDV_NETBUY_TRDVOL", 0)),
                    "외국인_보유율":    float(str(row.get("FRGN_HLD_QTY_RT", 0)).replace(",", "") or 0),
                })
            except Exception:
                continue

        if not records:
            return None

        df = pd.DataFrame(records)
        df["날짜"] = pd.to_datetime(df["날짜"], format="%Y/%m/%d", errors="coerce")
        df = df.dropna(subset=["날짜"]).sort_values("날짜").tail(days)
        return df if len(df) >= 3 else None

    except Exception:
        return None


def _fetch_naver_supply_demand(ticker_ks: str) -> pd.DataFrame | None:
    """
    네이버금융 비공식 API 폴백.
    KRX API 실패 시 사용. 외국인 순매수/보유율 데이터 제공.
    """
    try:
        code = ticker_ks.replace(".KS", "").replace(".KQ", "").strip()
        url = f"https://finance.naver.com/item/frgn.naver?code={code}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer":         "https://finance.naver.com",
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
        }
        # 네이버도 한 번 재시도
        r = None
        for _attempt in range(2):
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    break
            except Exception:
                if _attempt == 0:
                    time.sleep(1)
        if r is None or r.status_code != 200:
            return None
        if r.status_code != 200:
            return None

        # BeautifulSoup으로 테이블 파싱
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"class": "type2"})
        if not table:
            return None

        records = []
        for row in table.find_all("tr")[2:12]:  # 최근 10일
            cols = row.find_all("td")
            if len(cols) < 4:
                continue
            try:
                def _int(v):
                    t = v.get_text(strip=True).replace(",", "").replace("+", "")
                    return int(t) if t and t != "-" else 0

                def _float(v):
                    t = v.get_text(strip=True).replace(",", "")
                    return float(t) if t and t != "-" else 0.0

                records.append({
                    "날짜":           cols[0].get_text(strip=True),
                    "외국인_순매수":   _int(cols[2]),
                    "기관합계_순매수": _int(cols[3]) if len(cols) > 3 else 0,
                    "개인_순매수":     0,
                    "외국인_보유율":   _float(cols[1]) if len(cols) > 1 else 0.0,
                })
            except Exception:
                continue

        if not records:
            return None

        df = pd.DataFrame(records)
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
        df = df.dropna(subset=["날짜"]).sort_values("날짜")
        return df if len(df) >= 3 else None

    except Exception:
        return None


def _fetch_fdr_supply_demand(ticker_ks: str) -> pd.DataFrame | None:
    """
    FinanceDataReader로 투자자별 매매 데이터를 가져온다.

    KRX API와 네이버금융 모두 실패할 경우 사용하는 3차 폴백.
    FDR은 KRX/네이버 데이터를 내부적으로 활용하므로
    GitHub Actions IP에서도 동작할 가능성이 있다.
    """
    try:
        import FinanceDataReader as fdr
        code = ticker_ks.replace(".KS", "").replace(".KQ", "").strip()
        today = datetime.now(ZoneInfo("Asia/Seoul"))
        start = today - timedelta(days=30)
        df = fdr.DataReader(
            f"KRX:{code}",
            start.strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d")
        )
        if df is None or df.empty:
            return None
        # FDR 컬럼명 표준화
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if "foreign" in cl or "외국" in cl or "frgn" in cl:
                col_map[col] = "외국인_순매수"
            elif "institution" in cl or "기관" in cl or "organ" in cl:
                col_map[col] = "기관합계_순매수"
        if not col_map:
            return None
        df = df.rename(columns=col_map)
        if "외국인_순매수" not in df.columns:
            return None
        if "기관합계_순매수" not in df.columns:
            df["기관합계_순매수"] = 0
        df["개인_순매수"] = 0
        df["외국인_보유율"] = 0.0
        df["날짜"] = df.index
        df = df[["날짜", "외국인_순매수", "기관합계_순매수", "개인_순매수", "외국인_보유율"]]
        df = df.dropna(subset=["날짜"]).sort_values("날짜").tail(10)
        return df if len(df) >= 3 else None
    except Exception:
        return None


def _fetch_yfinance_institutional(ticker_ks: str) -> dict | None:
    """
    yfinance .info에서 기관 보유율 데이터를 가져온다.

    일별 수급이 아닌 분기별 보유율이지만,
    KRX/네이버/FDR 모두 실패했을 때 최소한의 정보를 제공한다.
    heldPercentInstitutions: 기관 보유 비율 (0.0~1.0)
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker_ks).info or {}
        inst_pct = info.get("heldPercentInstitutions")
        if inst_pct is None:
            return None
        return {"기관_보유율": round(inst_pct * 100, 1)}
    except Exception:
        return None


def get_kr_supply_demand(ticker: str) -> dict:
    """
    한국 주식 기관/외국인 수급 팩터를 분석한다.
    KRX API → 네이버금융 순으로 폴백.

    반환 dict:
    {
      "보너스":          int,   # 수급 점수 합산 (+1~+3 / -1)
      "표시문구":        str,   # 텔레그램 알림용 한 줄 요약
      "외국인_연속":     int,   # 외국인 순매수 연속 일수 (음수=연속 매도)
      "기관_연속":       int,   # 기관 순매수 연속 일수
      "동시순매수":      bool,  # 외국인+기관 동시 순매수 여부
      "외국인_보유율_추세": str, # "상승" / "하락" / "횡보"
      "데이터_없음":     bool,  # API 실패 시 True
    }
    """
    _빈_결과 = {
        "보너스": 0, "표시문구": "",
        "외국인_연속": 0, "기관_연속": 0,
        "동시순매수": False, "외국인_보유율_추세": "횡보",
        "데이터_없음": True,
    }

    # KR 종목만 처리
    if not (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        return _빈_결과

    # 캐시 확인
    if ticker in _supply_demand_cache:
        return _supply_demand_cache[ticker]

    # 데이터 수집: KRX → 네이버 → FDR → yfinance 기관보유율 4중 폴백
    df = _fetch_krx_supply_demand(ticker, days=10)
    if df is None:
        df = _fetch_naver_supply_demand(ticker)
    if df is None:
        df = _fetch_fdr_supply_demand(ticker)
    if df is None:
        # 4차 폴백: yfinance 기관 보유율 (일별 수급 없음, 방향성만 표시)
        yf_inst = _fetch_yfinance_institutional(ticker)
        if yf_inst:
            기관_보유율 = yf_inst["기관_보유율"]
            결과 = {
                "보너스": 0,
                "표시문구": f"기관보유율 {기관_보유율:.1f}% (분기 기준, 일별수급 조회실패)",
                "외국인_연속": 0, "기관_연속": 0,
                "동시순매수": False, "외국인_보유율_추세": "횡보",
                "데이터_없음": True,
            }
            _supply_demand_cache[ticker] = 결과
            return 결과
        _supply_demand_cache[ticker] = _빈_결과
        return _빈_결과

    try:
        # ── 연속 순매수 일수 계산 ─────────────────────────
        def _연속일수(series: pd.Series) -> int:
            """
            가장 최근 날짜부터 거슬러 올라가며 연속 양수(매수)·음수(매도) 일수 반환.
            양수면 연속 매수, 음수면 연속 매도.
            """
            vals = series.tolist()[::-1]  # 최신 → 과거
            if not vals:
                return 0
            방향 = 1 if vals[0] > 0 else -1
            count = 0
            for v in vals:
                if (방향 > 0 and v > 0) or (방향 < 0 and v < 0):
                    count += 1
                else:
                    break
            return count * 방향

        외국인_연속 = _연속일수(df["외국인_순매수"])
        기관_연속   = _연속일수(df["기관합계_순매수"])

        # 오늘(최신) 기준 동시 순매수
        최신 = df.iloc[-1]
        동시순매수 = bool(최신["외국인_순매수"] > 0 and 최신["기관합계_순매수"] > 0)
        동시순매도 = bool(최신["외국인_순매수"] < 0 and 최신["기관합계_순매수"] < 0)

        # 외국인 보유율 5일 추세
        if "외국인_보유율" in df.columns and len(df) >= 5:
            보유율_5일전 = df["외국인_보유율"].iloc[-5]
            보유율_현재  = df["외국인_보유율"].iloc[-1]
            diff = 보유율_현재 - 보유율_5일전
            if diff > 0.2:
                보유율_추세 = "상승"
            elif diff < -0.2:
                보유율_추세 = "하락"
            else:
                보유율_추세 = "횡보"
        else:
            보유율_추세 = "횡보"

        # ── 수급 점수 계산 ─────────────────────────────────
        보너스 = 0
        표시_파츠 = []

        # ① 외국인 연속 3일 이상 순매수
        if 외국인_연속 >= 3:
            보너스 += 1
            표시_파츠.append(f"외국인 {외국인_연속}일 연속매수")

        # ② 기관 연속 3일 이상 순매수
        if 기관_연속 >= 3:
            보너스 += 1
            표시_파츠.append(f"기관 {기관_연속}일 연속매수")

        # ③ 외국인+기관 동시 순매수 (당일)
        if 동시순매수:
            보너스 += 1
            표시_파츠.append("외국인+기관 동시매수")

        # ④ 외국인 보유율 상승 추세
        if 보유율_추세 == "상승":
            보너스 += 0  # 단독으로는 점수 부여 안 함, 표시만
            표시_파츠.append(f"외국인보유율↑{df['외국인_보유율'].iloc[-1]:.1f}%")

        # ⑤ 외국인+기관 동시 연속 매도 → 패널티
        if 외국인_연속 <= -3 and 기관_연속 <= -3:
            보너스 -= 1
            표시_파츠.append(f"⚠️외국인·기관 동반매도{abs(외국인_연속)}일")
        elif 동시순매도:
            보너스 -= 1
            표시_파츠.append("⚠️외국인·기관 동시매도")

        표시문구 = " | ".join(표시_파츠) if 표시_파츠 else ""

        결과 = {
            "보너스":             보너스,
            "표시문구":           표시문구,
            "외국인_연속":        외국인_연속,
            "기관_연속":          기관_연속,
            "동시순매수":         동시순매수,
            "외국인_보유율_추세": 보유율_추세,
            "데이터_없음":        False,
        }
        _supply_demand_cache[ticker] = 결과
        return 결과

    except Exception as e:
        print(f"⚠️ {ticker} 수급 분석 오류: {e}")
        _supply_demand_cache[ticker] = _빈_결과
        return _빈_결과
