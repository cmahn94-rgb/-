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
                close = pd.Series([70000, 71000, 79000, 75000, 76000,
                                   77000, 78000, 79500, 79000, 80000], index=dates)
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
