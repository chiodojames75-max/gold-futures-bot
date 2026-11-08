"""Paper-mode smoke test of the live loop against a fake TopstepX (no network)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oro.config import OroConfig  # noqa: E402
from oro.data import resample, synthetic_gold  # noqa: E402
from oro.live import runner as R  # noqa: E402


class FakeAPI:
    def __init__(self, bars):
        self.all, self.n = bars, 300

    def find_contract(self, sym): return {"id": "CON.F.US.MGC.Z26", "name": "MGCZ6"}
    def accounts(self): return [{"id": 1, "name": "50K-TEST", "canTrade": True, "balance": 50000.0}]
    def bars(self, cid, minutes): return self.all.iloc[: self.n]


def test_paper_loop_runs_and_journals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bars = resample(synthetic_gold(days=15, seed=5), 5)
    fake = FakeAPI(bars)
    monkeypatch.setattr(R, "TopstepX", lambda *a, **k: fake)
    for k in ("TOPSTEPX_BASE_URL", "TOPSTEPX_USERNAME", "TOPSTEPX_API_KEY"):
        monkeypatch.setenv(k, "x")
    run = R.Runner("comex_orb", live=False, cfg=OroConfig())
    while fake.n < len(bars):
        run.step()
        fake.n += 1
    j = pd.read_csv("journal.csv")
    assert (j.event == "entry").sum() >= 3
    assert (j.event == "exit").sum() >= (j.event == "entry").sum() - 1
    assert os.path.exists("oro_state.json")
