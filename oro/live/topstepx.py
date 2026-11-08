"""
Minimal TopstepX / ProjectX Gateway REST client.

Verified against the public ProjectX docs (Oct 2026):
  POST /api/Auth/loginKey        {userName, apiKey} -> {token, success, errorCode}
  POST /api/Account/search       {onlyActiveAccounts}
  POST /api/Contract/available   {live}
  POST /api/Order/place          {accountId, contractId, type, side, size, ...}  type 2 = Market, side 1 = Ask(sell)

Marked VERIFY: field names / codes that come from community SDKs, not the page I could read.
Check each against https://gateway.docs.projectx.com before the first live run.

Rules this client must run under (Topstep): run on YOUR computer, no VPS, no VPN, no HFT.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# Order type / side codes
MARKET, LIMIT, STOP = 2, 1, 4        # verified against gateway.docs.projectx.com (Order/place), Oct 20 2026
BUY, SELL = 0, 1                     # verified: 0 = Bid (buy), 1 = Ask (sell)


class TopstepXError(RuntimeError):
    pass


class TopstepX:
    def __init__(self, base_url: str, username: str, api_key: str, timeout: float = 10.0):
        if not base_url:
            raise ValueError("Set TOPSTEPX_BASE_URL (find the production gateway URL in the ProjectX docs "
                             "'connection URLs' page, or in your TopstepX API settings).")
        self.base = base_url.rstrip("/")
        self.username, self.api_key, self.timeout = username, api_key, timeout
        self.token = None
        self.token_time = 0.0
        self.s = requests.Session()

    # ---------- plumbing ----------
    def _post(self, path: str, body: dict, auth: bool = True) -> dict:
        if auth and (self.token is None or time.time() - self.token_time > 20 * 3600):
            self.login()
        headers = {"Content-Type": "application/json", "accept": "text/plain"}
        if auth:
            headers["Authorization"] = f"Bearer {self.token}"
        for attempt in range(3):
            try:
                r = self.s.post(self.base + path, json=body, headers=headers, timeout=self.timeout)
                if r.status_code == 401 and auth and attempt == 0:
                    self.login()
                    headers["Authorization"] = f"Bearer {self.token}"
                    continue
                r.raise_for_status()
                data = r.json()
                if not data.get("success", False):
                    raise TopstepXError(f"{path} failed: code={data.get('errorCode')} msg={data.get('errorMessage')}")
                return data
            except requests.RequestException as e:
                if attempt == 2:
                    raise TopstepXError(f"{path} network error: {e}") from e
                time.sleep(1.5 * (attempt + 1))
        raise TopstepXError(f"{path} failed after retries")

    def login(self) -> None:
        data = self._post("/api/Auth/loginKey", {"userName": self.username, "apiKey": self.api_key}, auth=False)
        self.token, self.token_time = data["token"], time.time()

    # ---------- reference data ----------
    def accounts(self) -> list[dict]:
        return self._post("/api/Account/search", {"onlyActiveAccounts": True})["accounts"]

    def find_contract(self, symbol: str = "MGC", live: bool = False) -> dict:
        """Front-month contract whose name starts with the symbol, e.g. MGCZ6."""
        cons = self._post("/api/Contract/available", {"live": live})["contracts"]
        matches = [c for c in cons if str(c.get("name", "")).upper().startswith(symbol.upper())
                   and c.get("activeContract", True)]
        if not matches:
            raise TopstepXError(f"No active contract found for {symbol}")
        return matches[0]

    # ---------- market data ----------
    def bars(self, contract_id: str, minutes: int = 5, lookback_hours: int = 72, live: bool = False) -> pd.DataFrame:
        """Verified: /api/History/retrieveBars, unit 2 = minute, max 20,000 bars per request, bars t/o/h/l/c/v."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)
        data = self._post("/api/History/retrieveBars", {
            "contractId": contract_id, "live": live,
            "startTime": start.isoformat(), "endTime": end.isoformat(),
            "unit": 2, "unitNumber": minutes, "limit": 20000, "includePartialBar": False,
        })
        rows = data.get("bars", [])
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).rename(columns={"t": "time", "o": "open", "h": "high", "l": "low",
                                                "c": "close", "v": "volume"})
        df.index = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/New_York")
        return df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()

    def daily_closes(self, contract_id: str, days: int = 420, live: bool = False) -> pd.Series:
        """Daily closes for the trend filter. Verified: unit 4 = day."""
        end = datetime.now(timezone.utc); start = end - timedelta(days=days)
        data = self._post("/api/History/retrieveBars", {"contractId": contract_id, "live": live, "startTime": start.isoformat(),
                                                        "endTime": end.isoformat(), "unit": 4, "unitNumber": 1, "limit": 1000, "includePartialBar": False})
        rows = data.get("bars", [])
        if not rows:
            return pd.Series(dtype=float)
        s = pd.Series({pd.to_datetime(r["t"], utc=True).tz_convert("America/New_York").date(): float(r["c"]) for r in rows})
        return s.sort_index()

    def daily_ranges(self, contract_id: str, days: int = 220, live: bool = False) -> pd.Series:
        """Daily high-low for the quiet-market filter. Verified: unit 4 = day."""
        end = datetime.now(timezone.utc); start = end - timedelta(days=days)
        data = self._post("/api/History/retrieveBars", {"contractId": contract_id, "live": live, "startTime": start.isoformat(),
                                                        "endTime": end.isoformat(), "unit": 4, "unitNumber": 1, "limit": 1000, "includePartialBar": False})
        rows = data.get("bars", [])
        s = pd.Series({pd.to_datetime(r["t"], utc=True).tz_convert("America/New_York").date(): float(r["h"]) - float(r["l"]) for r in rows})
        return s.sort_index()

    # ---------- orders / positions ----------
    def place(self, account_id: int, contract_id: str, side: int, size: int, type_: int,
              limit_price: float | None = None, stop_price: float | None = None) -> int:
        body = {"accountId": account_id, "contractId": contract_id, "type": type_, "side": side, "size": size}
        if limit_price is not None:
            body["limitPrice"] = limit_price      # verified field name
        if stop_price is not None:
            body["stopPrice"] = stop_price        # verified field name
        return self._post("/api/Order/place", body)["orderId"]

    def cancel(self, account_id: int, order_id: int) -> None:
        self._post("/api/Order/cancel", {"accountId": account_id, "orderId": order_id})  # verified

    def open_orders(self, account_id: int) -> list[dict]:
        return self._post("/api/Order/searchOpen", {"accountId": account_id}).get("orders", [])  # VERIFY

    def open_positions(self, account_id: int) -> list[dict]:
        return self._post("/api/Position/searchOpen", {"accountId": account_id}).get("positions", [])  # VERIFY

    def flatten(self, account_id: int, contract_id: str) -> None:
        """Cancel all working orders and close the position. Used by the kill switch and time exits."""
        for o in self.open_orders(account_id):
            try:
                self.cancel(account_id, o["id"])
            except TopstepXError:
                pass
        self._post("/api/Position/closeContract", {"accountId": account_id, "contractId": contract_id})  # verified

    def flatten_orders(self, account_id: int, contract_id: str) -> None:
        """Cancel every working order on this contract (no position change)."""
        for o in self.open_orders(account_id):
            if str(o.get("contractId", contract_id)) == str(contract_id):
                try:
                    self.cancel(account_id, o["id"])
                except TopstepXError:
                    pass
