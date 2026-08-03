"""
test_core.py — 핵심 로직 회귀 방지 테스트 (하이엔드 3순위)
===========================================================
[목적]
지금까지 수동으로 하던 검증을 pytest로 고정한다.
리팩토링·기능 추가 시 이 테스트가 통과하면 기존 동작이 안 깨진 것.

[실행]
  pip install pytest
  pytest tests/ -v

[커버 범위]
  - 지표 계산 (RSI/ATR/상대강도)
  - 모멘텀 신호 5조건
  - 장세 분류 (K자/추세/횡보/하락)
  - RSS 파싱 (제목·요약·필터)
  - 성과 추적 (기록·채점·통계)
  - 공용 유틸 (거래비용·통화·ATR보정)
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# 1. 공용 유틸 (strategy_utils)
# ═══════════════════════════════════════════════════════════
class TestStrategyUtils:
    def test_trade_costs_kr(self):
        from strategy_utils import get_trade_costs
        매수, 매도 = get_trade_costs({}, "KR")
        assert abs(매수 - 0.0015) < 1e-9
        assert abs(매도 - 0.0035) < 1e-9   # 거래세 0.002 포함

    def test_trade_costs_us(self):
        from strategy_utils import get_trade_costs
        매수, 매도 = get_trade_costs({}, "US")
        assert abs(매도 - 0.0015) < 1e-9   # 거래세 없음

    def test_currency_symbol(self):
        from strategy_utils import get_currency_symbol
        assert get_currency_symbol("KR") == "₩"
        assert get_currency_symbol("US") == "$"
        assert get_currency_symbol("CRYPTO_KRW") == "₩"

    def test_safe_atr(self):
        from strategy_utils import safe_atr
        assert safe_atr(float("nan"), 1000) == 20.0   # 2% 폴백
        assert safe_atr(None, 1000) == 20.0
        assert safe_atr(35.5, 1000) == 35.5           # 정상값 유지

    def test_benchmark_selection(self):
        from strategy_utils import get_benchmark_close
        s = {"BENCH_KR_CLOSE": pd.Series([1, 2, 3]),
             "BENCH_US_CLOSE": pd.Series([4, 5, 6])}
        assert get_benchmark_close(s, "KR").iloc[0] == 1
        assert get_benchmark_close(s, "US").iloc[0] == 4
        assert get_benchmark_close(s, "CRYPTO_KRW") is None


# ═══════════════════════════════════════════════════════════
# 2. 지표 계산 (indicators)
# ═══════════════════════════════════════════════════════════
class TestIndicators:
    def test_relative_strength_outperform(self):
        from indicators import calc_relative_strength
        종목 = pd.Series([100 * (1 + 0.30 * i / 60) for i in range(61)])
        시장 = pd.Series([100 * (1 + 0.10 * i / 60) for i in range(61)])
        r = calc_relative_strength(종목, 시장, period=60)
        assert r["아웃퍼폼"] is True
        assert r["강한_아웃퍼폼"] is True   # +20%p 초과

    def test_relative_strength_underperform(self):
        from indicators import calc_relative_strength
        종목 = pd.Series([100 * (1 + 0.05 * i / 60) for i in range(61)])
        시장 = pd.Series([100 * (1 + 0.15 * i / 60) for i in range(61)])
        r = calc_relative_strength(종목, 시장, period=60)
        assert r["아웃퍼폼"] is False

    def test_relative_strength_no_benchmark(self):
        from indicators import calc_relative_strength
        종목 = pd.Series([100] * 61)
        r = calc_relative_strength(종목, None, period=60)
        assert r["아웃퍼폼"] is False


# ═══════════════════════════════════════════════════════════
# 3. 모멘텀 신호 (momentum)
# ═══════════════════════════════════════════════════════════
class TestMomentum:
    def _breakout_df(self):
        dates = pd.date_range("2026-05-01", periods=60)
        base = np.linspace(100, 118, 60)
        base[-1] = base[-2] * 1.04           # 당일 +4%
        close = pd.Series(base, index=dates)
        volume = pd.Series([1000000] * 59 + [2800000], index=dates)  # 2.8배
        return pd.DataFrame({
            "Open": close.shift(1).fillna(close), "High": close * 1.01,
            "Low": close * 0.99, "Close": close, "Volume": volume,
        })

    def test_momentum_breakout_signal(self):
        df = self._breakout_df()
        dates = df.index
        settings = {"BENCH_KR_CLOSE": pd.Series(np.linspace(100, 105, 60), index=dates)}
        with patch("momentum.get_price_data", return_value=df):
            from momentum import calc_momentum_signal
            r = calc_momentum_signal("005930.KS", "삼성전자", "KR", settings)
        assert r is not None
        assert r["조건"]["M3_거래량"] is True
        assert r["조건"]["M4_당일강세"] is True

    def test_momentum_insufficient_data(self):
        short_df = pd.DataFrame({"Close": [100, 101], "Volume": [1, 2]})
        with patch("momentum.get_price_data", return_value=short_df):
            from momentum import calc_momentum_signal
            r = calc_momentum_signal("005930.KS", "삼성", "KR", {})
        assert r is None


# ═══════════════════════════════════════════════════════════
# 4. 장세 분류 (market_phase)
# ═══════════════════════════════════════════════════════════
class TestRegime:
    def test_k_polarization(self):
        from market_phase import classify_market_regime
        k = [{"변동률": v} for v in ([4.2, 6.1, 3.5] + [-1.2] * 15 + [-2.0] * 5)]
        r = classify_market_regime(k, current_vix=24)
        assert r["유형"] == "K자양극화"
        assert r["모멘텀_가중"] == "주력"

    def test_trend_market(self):
        from market_phase import classify_market_regime
        t = [{"변동률": v} for v in ([1.5, 2.1, 0.8, 1.2, 2.5, 0.5] * 4 + [-0.5] * 8)]
        r = classify_market_regime(t, current_vix=16)
        assert r["유형"] == "추세장"

    def test_bear_market(self):
        from market_phase import classify_market_regime
        b = [{"변동률": v} for v in ([-2.5] * 18 + [-4.0] * 5 + [0.5] * 2)]
        r = classify_market_regime(b, current_vix=30)
        assert r["유형"] == "하락장"
        assert r["모멘텀_가중"] == "끔"

    def test_custom_thresholds(self):
        from market_phase import classify_market_regime
        # 커스텀 임계값으로 K자 기준 완화
        r = classify_market_regime(
            [{"변동률": v} for v in [5.0] + [-1.0] * 10 + [1.0] * 4],
            current_vix=20, settings={"REGIME_K_BREADTH": 50})
        assert r["유형"] == "K자양극화"

    def test_sideways(self):
        from market_phase import classify_market_regime
        s = [{"변동률": v} for v in ([0.3, -0.2, 0.1, -0.4] * 6)]
        r = classify_market_regime(s, current_vix=15)
        assert r["유형"] == "횡보장"


# ═══════════════════════════════════════════════════════════
# 5. RSS 파싱 (data_loader)
# ═══════════════════════════════════════════════════════════
class TestRSS:
    def test_clean_title(self):
        from data_loader import _clean_rss_title
        assert _clean_rss_title("삼성전자 실적 발표 - 연합뉴스") == "삼성전자 실적 발표"

    def test_strip_html(self):
        from data_loader import _strip_html
        r = _strip_html('<a>삼성전자 <b>급등</b></a>&quot;호재&quot;')
        assert "<" not in r and "&quot;" not in r
        assert '"호재"' in r

    def test_rss_filters(self):
        from data_loader import _get_news_google_rss
        now = datetime.now(timezone.utc)
        어제 = (now - timedelta(days=1)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        오래 = (now - timedelta(days=5)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        xml = f'''<?xml version="1.0"?><rss><channel>
<item><title>삼성전자 3분기 영업이익 10조 돌파 실적 - 연합</title>
  <description>반도체 호조</description><pubDate>{어제}</pubDate><source>연합</source></item>
<item><title>특징주 삼성</title>
  <description>x</description><pubDate>{어제}</pubDate><source>A</source></item>
<item><title>삼성전자 외국인 5일 연속 순매수 지속 기록</title>
  <description>수급</description><pubDate>{오래}</pubDate><source>B</source></item>
</channel></rss>'''
        resp = MagicMock(); resp.status_code = 200; resp.content = xml.encode()
        with patch("data_loader.requests.get", return_value=resp):
            result = _get_news_google_rss("005930.KS", "삼성전자", 변동률=2.0)
        titles = [r["title"] for r in result]
        assert not any("특징주" in t for t in titles)   # 광고성 제외
        assert not any("외국인 5일" in t for t in titles)  # 오래된 제외

    def test_rss_us_ticker_excluded(self):
        from data_loader import _get_news_google_rss
        assert _get_news_google_rss("AAPL", "Apple", 0) == []


# ═══════════════════════════════════════════════════════════
# 6. 성과 추적 (performance_tracker)
# ═══════════════════════════════════════════════════════════
class TestPerformanceTracker:
    def _setup_temp(self, tmp_name):
        import performance_tracker as pt
        pt._LOG_FILENAME = tmp_name
        if os.path.exists(pt._log_path()):
            os.remove(pt._log_path())
        return pt

    def test_record_and_dedup(self):
        pt = self._setup_temp("test_pt_1.json")
        try:
            신호 = [{"ticker": "005930.KS", "name": "삼성", "market": "KR",
                    "진입가": 70000, "목표가": 78400, "손절가": 66500, "보유상한일": 30}]
            assert pt.record_signals(신호, "안정") == 1
            assert pt.record_signals(신호, "안정") == 0   # 중복 방지
        finally:
            if os.path.exists(pt._log_path()):
                os.remove(pt._log_path())

    def test_grading_alerts(self):
        from performance_tracker import format_grading_alerts
        통계 = {"채점": 1, "승": 1, "패": 0, "중립": 0, "알림": [
            {"name": "삼성", "ticker": "005930.KS", "전략": "안정",
             "결과": "승", "수익률": 12.3}]}
        alert = format_grading_alerts(통계)
        assert "목표 도달" in alert and "삼성" in alert
        assert format_grading_alerts({"알림": []}) == ""

    def test_grading(self):
        pt = self._setup_temp("test_pt_2.json")
        try:
            from zoneinfo import ZoneInfo
            KST = ZoneInfo("Asia/Seoul")
            발생 = (datetime.now(KST) - timedelta(days=5)).strftime("%Y-%m-%d")
            pt._save_log({"signals": [{
                "id": "t1", "날짜": 발생, "ticker": "005930.KS", "name": "삼성",
                "market": "KR", "전략": "안정", "진입가": 70000, "목표가": 78400,
                "손절가": 66500, "보유상한일": 30, "상태": "보유중",
                "채점일": None, "결과": None, "수익률": None,
            }]})

            def mock_price(ticker, period="3mo"):
                dates = pd.date_range(datetime.now() - timedelta(days=10), periods=10, tz=KST)
                # v5.25: 지정가(70000)까지 하락해 '체결'된 뒤 목표(78400) 도달하는 흐름.
                # (체결되지 않으면 미체결로 분류되는 것이 정상 동작)
                close = pd.Series([72000, 71000, 70500, 71000, 72000,
                                   73000, 68000, 75000, 79000, 80000], index=dates)
                return pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99}, index=dates)

            통계 = pt.grade_pending_signals(mock_price)
            assert 통계["채점"] == 1
            assert 통계["승"] == 1   # 목표 78400 도달
        finally:
            if os.path.exists(pt._log_path()):
                os.remove(pt._log_path())


# ═══════════════════════════════════════════════════════════
# 7. 관측성 (observability)
# ═══════════════════════════════════════════════════════════
class TestObservability:
    def test_health_tracking(self):
        from observability import HealthTracker
        h = HealthTracker()
        h.record("supply", ok=True)
        h.record("supply", ok=True)
        h.record("supply", ok=False, detail="타임아웃")
        health = h.get_source_health()
        assert health["supply"]["성공"] == 2
        assert health["supply"]["실패"] == 1
        assert health["supply"]["성공률"] == 66.7

    def test_health_summary_all_ok(self):
        from observability import HealthTracker
        h = HealthTracker()
        h.record("news", ok=True)
        assert "정상" in h.summary()


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])


# ═══════════════════════════════════════════════════════════
# 8. 통합 일관성 (스코프/언팩 버그 방지 — v5.21에서 발견된 실버그 재발 방지)
# ═══════════════════════════════════════════════════════════
class TestIntegrationConsistency:
    def test_build_report_sections_return_matches_unpack(self):
        """build_report_sections의 return 개수와 호출부 언팩 개수가 일치해야 한다.
        (실제 CI에서 '모멘텀_결과_맵 is not defined' NameError가 났던 버그 방지)"""
        import ast, os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scheduler_job.py")
        tree = ast.parse(open(path, encoding="utf-8").read())

        return_arity = None
        unpack_arity = None
        for node in ast.walk(tree):
            # 함수 내부 return 튜플 크기
            if isinstance(node, ast.FunctionDef) and node.name == "build_report_sections":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Tuple):
                        return_arity = len(sub.value.elts)
            # 호출부 언팩 크기: (a, b, ...) = build_report_sections(...)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = node.value.func
                if isinstance(fn, ast.Name) and fn.id == "build_report_sections":
                    tgt = node.targets[0]
                    if isinstance(tgt, ast.Tuple):
                        unpack_arity = len(tgt.elts)

        assert return_arity is not None, "return 튜플 못 찾음"
        assert unpack_arity is not None, "언팩 호출부 못 찾음"
        assert return_arity == unpack_arity, (
            f"return {return_arity}개 vs 언팩 {unpack_arity}개 불일치")

    def test_run_analysis_no_undefined_momentum_map(self):
        """run_analysis 영역에서 쓰는 모멘텀_결과_맵이 언팩으로 정의돼 있어야 한다."""
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scheduler_job.py")
        src = open(path, encoding="utf-8").read()
        # 언팩 라인에 모멘텀_결과_맵 포함 확인
        assert "모멘텀_결과_맵) = build_report_sections(" in src.replace("\n", "").replace(" ", "") \
            or "모멘텀_결과_맵)=build_report_sections(" in src.replace("\n", "").replace(" ", "")



# ═══════════════════════════════════════════════════════════
# 9. 워크플로우 YAML 유효성 (v5.24 — run_analysis.yml YAML 깨짐 재발 방지)
# ═══════════════════════════════════════════════════════════
class TestWorkflowYAML:
    """GitHub Actions 워크플로우 YAML이 파싱 가능한지 검증.
    (v5.21에서 run_analysis.yml에 멀티라인 문자열을 넣어 YAML이 깨지고
    워크플로우가 아예 실행 불가였던 버그 방지 — bash 멀티라인은 $'\\n' 사용)"""

    def _workflow_dir(self):
        import os
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".github", "workflows")

    def test_all_workflows_parse(self):
        import os
        try:
            import yaml
        except ImportError:
            import pytest
            pytest.skip("pyyaml 미설치")
        wd = self._workflow_dir()
        if not os.path.isdir(wd):
            import pytest
            pytest.skip("워크플로우 폴더 없음")
        for fn in os.listdir(wd):
            if fn.endswith((".yml", ".yaml")):
                path = os.path.join(wd, fn)
                # 예외 없이 파싱되면 통과
                yaml.safe_load(open(path, encoding="utf-8"))

    def test_run_analysis_has_schedule(self):
        import os
        try:
            import yaml
        except ImportError:
            import pytest
            pytest.skip("pyyaml 미설치")
        path = os.path.join(self._workflow_dir(), "run_analysis.yml")
        if not os.path.exists(path):
            import pytest
            pytest.skip("run_analysis.yml 없음")
        d = yaml.safe_load(open(path, encoding="utf-8"))
        # YAML에서 'on'은 True로 파싱될 수 있음
        on = d.get("on", d.get(True, {}))
        assert "schedule" in on, "run_analysis에 schedule 트리거가 있어야 함"


# ═══════════════════════════════════════════════════════════
# 10. 진입가/목표/손절 정합성 (v5.25 — 손절가>진입가 모순 재발 방지)
# ═══════════════════════════════════════════════════════════
class TestEntryPriceConsistency:
    """급락으로 ATR이 폭증해도 손절 < 진입 < 목표 순서가 깨지지 않아야 한다.
    (실제 SK하이닉스 사례: ATR이 주가의 16.8% → 손절가가 진입가보다 14% 위)"""

    @staticmethod
    def _levels(현재가, atr, gap_cap=5.0, bb하단=0, T1=12, T2=25, STOP=-5):
        진입_atr = 현재가 - atr if atr > 0 else 0
        후보 = [v for v in (bb하단, 진입_atr) if 0 < v < 현재가]
        진입 = max(후보) if 후보 else 현재가
        진입 = max(진입, 현재가 * (1 - gap_cap / 100))
        return (진입, 진입 * (1 + T1 / 100),
                진입 * (1 + T2 / 100), 진입 * (1 + STOP / 100))

    def test_extreme_atr_no_contradiction(self):
        진입, t1, t2, 손절 = self._levels(1_401_000, 235_150)   # ATR 16.8%
        assert 손절 < 진입 < t1 < t2

    def test_gap_cap_applied(self):
        진입, _, _, _ = self._levels(1_401_000, 235_150, gap_cap=5.0)
        assert 진입 == 1_401_000 * 0.95   # 상한이 걸려야 함

    def test_normal_atr_unchanged(self):
        진입, _, _, 손절 = self._levels(70_000, 1_400)   # ATR 2%
        assert abs(진입 - 68_600) < 1     # 원래 로직(현재가-ATR) 유지
        assert 손절 < 진입

    def test_analyze_one_returns_entry_fields(self):
        """analyze_one 반환 dict에 진입가 필드가 있어야 리포트/성과추적이 일치한다."""
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scheduler_job.py")
        src = open(path, encoding="utf-8").read()
        for key in ('"추천_진입가"', '"진입가_bb"', '"진입_가드"'):
            assert key in src, f"{key} 누락"


class TestUnfilledGrading:
    """지정가 미체결 신호를 승/패로 채점하면 승률이 왜곡된다 (v5.25)."""

    def test_unfilled_excluded(self):
        import os, pandas as pd
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        import performance_tracker as pt
        KST = ZoneInfo("Asia/Seoul")
        pt._LOG_FILENAME = "test_unfilled.json"
        if os.path.exists(pt._log_path()):
            os.remove(pt._log_path())
        try:
            발생 = (datetime.now(KST) - timedelta(days=40)).strftime("%Y-%m-%d")
            pt._save_log({"signals": [{
                "id": "nf", "날짜": 발생, "ticker": "NOFILL.KS", "name": "미체결",
                "market": "KR", "전략": "안정", "진입가": 95, "목표가": 106,
                "손절가": 90, "보유상한일": 30, "상태": "보유중",
                "채점일": None, "결과": None, "수익률": None}]})

            def mp(t, period="3mo"):
                d = pd.date_range(datetime.now() - timedelta(days=35),
                                  periods=35, tz=KST)
                c = pd.Series([100 + i * 0.3 for i in range(35)], index=d)
                return pd.DataFrame({"Close": c, "High": c * 1.005,
                                     "Low": c * 0.995}, index=d)

            통계 = pt.grade_pending_signals(mp)
            assert 통계.get("미체결", 0) == 1
            assert 통계["승"] == 0 and 통계["패"] == 0
        finally:
            if os.path.exists(pt._log_path()):
                os.remove(pt._log_path())


# ═══════════════════════════════════════════════════════════
# 11. 채점기 정확성 (v5.26 — 승률 0% 사태 재발 방지)
# ═══════════════════════════════════════════════════════════
class TestGradingAccuracy:
    """장중 저가가 손절선을 스쳤다는 이유만으로 '패' 처리하면
    한국장에서 승률이 구조적으로 0%가 된다 (실제 0승 39패 발생).
    손절은 종가 기준, 목표는 장중 고가 기준, 판정은 '먼저 온 쪽'."""

    def _grade(self, 종가들, 저가배수=0.995, 경과=35,
               진입=100, 목표=112, 손절=95, 상한=30):
        import os, pandas as pd
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        import performance_tracker as pt
        KST = ZoneInfo("Asia/Seoul")
        pt._LOG_FILENAME = "test_grading_acc.json"
        if os.path.exists(pt._log_path()):
            os.remove(pt._log_path())
        try:
            발생 = (datetime.now(KST) - timedelta(days=경과)).strftime("%Y-%m-%d")
            pt._save_log({"signals": [{
                "id": "x", "날짜": 발생, "ticker": "T.KS", "name": "t",
                "market": "KR", "전략": "안정", "진입가": 진입, "목표가": 목표,
                "손절가": 손절, "보유상한일": 상한, "상태": "보유중",
                "채점일": None, "결과": None, "수익률": None}]})

            def mp(t, period="3mo"):
                d = pd.date_range(datetime.now() - timedelta(days=len(종가들)),
                                  periods=len(종가들), tz=KST)
                c = pd.Series(종가들, index=d)
                return pd.DataFrame({"Close": c, "High": c * 1.005,
                                     "Low": c * 저가배수}, index=d)

            pt.grade_pending_signals(mp)
            return pt._load_log()["signals"][0]
        finally:
            if os.path.exists(pt._log_path()):
                os.remove(pt._log_path())

    def test_intraday_dip_not_counted_as_loss(self):
        """장중 -6% 스침 + 종가는 손절선 위 + 이후 회복 → 패가 아니어야 한다."""
        s = self._grade([100, 98, 96, 97, 100, 104, 107, 109, 110, 110] + [110] * 15,
                        저가배수=0.94)
        assert s["결과"] == "승"

    def test_real_decline_is_loss(self):
        """종가가 손절선 아래로 내려가면 정상적으로 패."""
        s = self._grade([100, 97, 94, 92, 90, 89, 88, 88, 88, 88] + [88] * 15)
        assert s["결과"] == "패"

    def test_target_hit_is_win(self):
        s = self._grade([100, 98, 101, 105, 109, 113, 115, 115, 115, 115] + [115] * 15,
                        경과=25)
        assert s["결과"] == "승"

    def test_pending_not_graded_early(self):
        """보유 기간이 남았고 목표·손절 미도달이면 성급히 채점하지 않는다."""
        s = self._grade([100, 99, 101, 102, 103] + [103] * 10, 경과=5)
        assert s["상태"] == "보유중"

    def test_grading_version_tagged(self):
        s = self._grade([100, 98, 101, 105, 109, 113, 115] + [115] * 15, 경과=25)
        assert s.get("채점버전")


class TestLegacyMigration:
    """구버전 채점 기록은 통계에서 분리하되 보존해야 한다 (v5.26)."""

    def test_migration_separates_and_preserves(self):
        import os, json
        import performance_tracker as pt
        pt._LOG_FILENAME = "test_migration.json"
        legacy_p = os.path.join(
            os.path.dirname(os.path.abspath(pt.__file__)), "signal_log_legacy.json")
        for p in (pt._log_path(), legacy_p):
            if os.path.exists(p):
                os.remove(p)
        try:
            sigs = [{"id": f"a{i}", "날짜": "2026-07-01", "ticker": f"T{i}.KS",
                     "name": "t", "market": "KR", "전략": "안정", "진입가": 100,
                     "목표가": 112, "손절가": 95, "보유상한일": 30,
                     "상태": "채점완료", "채점일": "2026-08-01",
                     "결과": "패", "수익률": -5.0} for i in range(5)]
            sigs.append({"id": "h", "날짜": "2026-08-02", "ticker": "H.KS",
                         "name": "h", "market": "KR", "전략": "안정", "진입가": 100,
                         "목표가": 112, "손절가": 95, "보유상한일": 30,
                         "상태": "보유중", "채점일": None,
                         "결과": None, "수익률": None})
            pt._save_log({"signals": sigs})

            r = pt.migrate_legacy_log()
            assert r["이관"] == 5 and r["유지"] == 1
            # 통계에서 제외됐는지
            summary = pt.get_performance_summary()
            assert summary["전체_채점"] == 0 and summary["보유중"] == 1
            # 아카이브에 보존됐는지
            arch = json.load(open(legacy_p, encoding="utf-8"))
            assert len(arch["signals"]) == 5
        finally:
            for p in (pt._log_path(), legacy_p):
                if os.path.exists(p):
                    os.remove(p)
