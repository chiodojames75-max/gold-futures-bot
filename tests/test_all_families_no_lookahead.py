"""Every strategy family: rewriting the future must not change any trade that closed before it."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oro import backtest  # noqa: E402
from oro.config import OroConfig, RiskConfig  # noqa: E402
from oro.data import add_indicators, add_session_columns, resample, synthetic_gold  # noqa: E402
from oro.strategies import REGISTRY  # noqa: E402

BASE = None


def _base():
    global BASE
    if BASE is None:
        BASE = resample(synthetic_gold(days=60, seed=21), 5)
    return BASE


def _cfg():
    c = OroConfig()
    c.risk = RiskConfig(risk_per_trade=1e9, max_contracts=1, daily_loss_stop=1e9, daily_profit_stop=1e9, mll_buffer=0)
    return c


@pytest.mark.parametrize("name", list(REGISTRY))
def test_no_lookahead(name):
    raw = _base()
    cut = raw.index[int(len(raw) * 0.6)]
    a = backtest.run(add_indicators(add_session_columns(raw)), REGISTRY[name](), _cfg(), enforce_mll=False).trades
    raw2 = raw.copy()
    later = raw2.index > cut
    raw2.loc[later, ["open", "high", "low", "close"]] *= 1.07
    b = backtest.run(add_indicators(add_session_columns(raw2)), REGISTRY[name](), _cfg(), enforce_mll=False).trades
    if a.empty:
        pytest.skip("no trades on this sample")
    pa = a[a.exit_time < cut].reset_index(drop=True)
    pb = b[b.exit_time < cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(pa, pb)
