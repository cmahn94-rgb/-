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
