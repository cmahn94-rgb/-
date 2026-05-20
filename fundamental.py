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


def _get_yf_info(ticker: str) -> dict:
    """yfinance .info를 캐시와 함께 가져온다. 실패 시 빈 dict 반환."""
    if ticker in _info_cache:
        return _info_cache[ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        # None인 값 제거
        info = {k: v for k, v in info.items() if v is not None}
        _info_cache[ticker] = info
        return info
    except Exception:
        _info_cache[ticker] = {}
        return {}


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
        # 업종 평균 PER 추정: 섹터별 대략적 기준
        # (정확한 업종 평균은 유료 데이터 필요 → 보수적 20배 기준 사용)
        sector_avg_per = float(info.get("sectorPE", 20))
        if per > 0:
            per_보너스 = per < sector_avg_per * 0.8
            per_패널티 = per > 50

    if pbr is not None and pbr > 0:
        pbr_보너스 = pbr < 1.0

    # 표시 문구 생성
    parts = []
    if per is not None:  parts.append(f"PER {per:.1f}")
    if pbr is not None:  parts.append(f"PBR {pbr:.2f}")
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
        resp = requests.get(
            "https://opendart.fss.or.kr/api/company.json",
            params={"crtfc_key": dart_key, "stock_code": code},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
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

        resp = requests.get(
            "https://opendart.fss.or.kr/api/list.json",
            params={
                "crtfc_key": dart_key,
                "corp_code":  corp_code,  # 8자리 고유코드
                "bgn_de":     bgn,
                "end_de":     end,
                "pblntf_ty":  "A",
                "page_count": 10,
            },
            timeout=8,
        )
        if resp.status_code != 200:
            return None
        items = resp.json().get("list", [])
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

        # 시가총액: DART에 없으므로 yfinance에서 가져옴
        시가총액 = None
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
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
