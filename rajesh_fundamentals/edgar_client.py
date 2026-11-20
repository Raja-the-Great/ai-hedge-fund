"""EdgarDataClient — a DataClient for ai-hedge-fund backed by SEC EDGAR.

Drop-in alternative to the shipped FDClient (Financial Datasets, paid):
same protocol (hedge_fund/data/protocol.py), no API key, no inheritance —
structural typing does the work. Only the persona/snapshot path is fully
served:

    get_financial_metrics(ticker, end_date, period="ttm", limit)
    get_company_facts(ticker)

POINT-IN-TIME CONTRACT. FDClient asks the FD API for rows with
filing_date <= end_date. EDGAR's companyfacts JSON carries a `filed`
date on every fact, but paper_agent.edgar's extractor cannot be used
for this: extract_financials() drops `filed`, and even
edgar._merged_entries() collapses each (start, end) period to its
LATEST filing — a period's original 10-Q filing vanishes, replaced by
next year's comparative restatement (verified live: Mastercard's
periods all carry filed dates ~1 year late, and its annual
NetIncomeLoss survives only as a DEF 14A entry that wins the collapse
and then fails the 10-K form filter, anchoring TTM in 2014). So this
module loads PIT entries itself — every filed variant of every period
kept, tag priority resolved per (start, end, filed) — and lets the
as-of filter + edgar's series helpers (which dedupe by period end,
latest filed first, among the entries they are given) pick exactly
what was public on the cutoff date:

- A TTM *observation* is anchored at a filing event: for each distinct
  revenue filing date F (chronological), the TTM is computed from ONLY
  the facts with filed <= F. The observation's report_period is the
  period end that TTM resolves to; each anchor end is emitted once, at
  the earliest filing that made it public.
- TTM value = sum of the last 4 consecutive quarters, else the latest
  annual value — the same rule as edgar.ttm_value(), re-implemented
  here because the stock helper cannot see filing dates.
- Balance-sheet (instant) values for an observation are taken at the
  anchor period end, again from facts filed <= F only.
- Ratios are computed here from the raw statements (FD precomputes
  them server-side; EDGAR serves facts, not ratios).

MARKET CAP / P/E. EDGAR has no prices. When the Alpaca paper keys from
paper_agent/.env are reachable, market_cap uses the point-in-time close
on or before each row's filing date (Alpaca data API, IEX feed, raw —
unadjusted — closes, matching the as-filed share counts) times shares
outstanding as filed by that date. Any price failure disables prices
for the run and leaves market_cap / P/E null (the snapshot renders
"-"); it never blocks the fundamentals path.

EDGAR SHARE-COUNT CORRUPTION — handled explicitly (all three cases were
verified live against the starter tickers, and match what the earlier
PIT-universe audit found independently):
- Some filers' dei cover counts in companyfacts are stale stubs
  (Visa: only 2009-2010 entries, 469M shares; Mastercard: same era,
  122M). A cover count whose period end is >400 days before the
  observation's period end is treated as missing, never used.
- When the cover count is missing/stale, diluted weighted-average
  shares stand in for shares outstanding (within ~1% for these names).
- When no share count exists at all (Visa has NO weighted-average or
  EPS tags in companyfacts), market_cap falls back to dei
  EntityPublicFloat as filed by the row's filing date — a lower bound
  on market cap, flagged per row in `cap_source`. Visa's
  earnings_per_share stays null: EDGAR as served carries no Visa EPS
  or share-count series to derive it from. P/E is then market_cap /
  net_income, the same ratio computed from the float bound.
- Net income merges NetIncomeLoss AND ProfitLoss: Mastercard files
  quarterly ProfitLoss but only annual NetIncomeLoss, so the single
  tag silently anchors TTM in 2014.

NOT SERVED (no EDGAR equivalent; personas never call them): news,
insider trades, earnings estimates — they return empty/None per the
protocol contract. get_prices/get_snapshot ARE served from Alpaca when
keys are present, else empty/None. screen() returns [] — the
point-in-time screener is an FD-API feature this adapter does not
replicate.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate and import the existing paper_agent EDGAR pipeline (read-only use).
# This file: <goal>/files/repo_agents/ai_hedge_fund/repo/rajesh_fundamentals/
# paper_agent: <goal>/files/paper_agent/
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_FILES_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
if _FILES_DIR not in sys.path:
    sys.path.insert(0, _FILES_DIR)

from paper_agent import edgar as _edgar  # noqa: E402
from paper_agent import universe as _universe  # noqa: E402

from hedge_fund.data.models import (  # noqa: E402
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsRecord,
    FinancialMetrics,
    InsiderTrade,
    Price,
    PriceSnapshot,
)
from hedge_fund.data.screener import (  # noqa: E402
    OrderBy,
    ScreenerFields,
    ScreenFilter,
    ScreenRow,
)

_SECTOR_LABELS = {
    "tech": "Technology",
    "healthcare": "Healthcare",
    "financials": "Financial Services",
    "consumer": "Consumer",
    "industrials": "Industrials",
    "energy": "Energy",
}

# Flow concepts whose TTM the snapshot ratios need.
_FLOWS = ["revenue", "gross_profit", "operating_income", "net_income",
          "cfo", "capex", "interest_expense"]
_INSTANTS = ["assets", "liabilities", "equity", "cash", "assets_current",
             "liabilities_current", "debt_current", "debt_noncurrent"]


# ---------------------------------------------------------------------------
# Filed-date-preserving TTM machinery (PIT entries; see module docstring)
# ---------------------------------------------------------------------------

def _asof(entries: list[dict], cutoff: str) -> list[dict]:
    """Entries publicly filed on or before *cutoff* (YYYY-MM-DD)."""
    return [e for e in entries if e["filed"] and e["filed"] <= cutoff]


def _augmented_quarters(entries: list[dict]) -> list[tuple]:
    """Quarter series with the fiscal Q4 synthesized where untagged.

    Most filers never tag a standalone Q4 flow value — the 10-K tags
    the full year and the Q3 10-Q tags the 9-month YTD — so a naive
    4-quarter walk breaks every year (verified live: AAPL has no
    tagged Q4 revenue after FY21). The standard derivation
    Q4 = annual - 9M YTD (same fiscal-year start) restores it from
    as-filed facts only. Returns [(end, value)] sorted by end.
    """
    q = dict(_edgar._quarter_series(entries))
    annual: dict[tuple, tuple[str, float]] = {}
    for e in entries:
        if e["start"] and 300 <= e["dur"] <= 380 and (
                e["form"].startswith("10-K") or e["fp"] == "FY"):
            key = (e["start"], e["end"])
            cur = annual.get(key)
            if cur is None or e["filed"] > cur[0]:
                annual[key] = (e["filed"], e["val"])
    ytd: dict[tuple, tuple[str, float]] = {}
    for e in entries:
        if e["start"] and 150 <= e["dur"] <= 290:
            key = (e["start"], e["end"])
            cur = ytd.get(key)
            if cur is None or e["filed"] >= cur[0]:
                ytd[key] = (e["filed"], e["val"])
    for (start, aend), (_, aval) in annual.items():
        if aend in q:
            continue
        for (s2, yend), (_, yval) in ytd.items():
            if s2 == start and 60 <= (aend - yend).days <= 130:
                q[aend] = aval - yval
                break
    return sorted(q.items())


def _ttm_asof(entries: list[dict], cutoff: str) -> tuple[float | None, datetime.date | None]:
    """(TTM value, anchor period end) using only facts filed <= cutoff.

    Same rule as edgar.ttm_value: sum of the last 4 consecutive quarters,
    else the latest annual value — but filing-date aware.
    """
    sub = _asof(entries, cutoff)
    if not sub:
        return None, None
    q = _augmented_quarters(sub)
    if len(q) >= 4:
        picked = [q[-1]]
        for i in range(len(q) - 2, -1, -1):
            gap = (picked[-1][0] - q[i][0]).days
            if 60 <= gap <= 130:
                picked.append(q[i])
                if len(picked) == 4:
                    break
            elif gap > 130:
                break
        if len(picked) == 4:
            return sum(v for _, v in picked), picked[0][0]
    ann = _edgar.annual_series(sub)
    if ann:
        return ann[-1][1], ann[-1][0]
    return None, None


def _shares_wavg_asof(entries: list[dict], cutoff: str) -> float | None:
    """TTM diluted weighted-average shares: mean of the 4 TTM quarters
    (a share count is an average, not a sum), else the annual figure."""
    sub = _asof(entries, cutoff)
    if not sub:
        return None
    q = _augmented_quarters(sub)
    if len(q) >= 4:
        picked = [q[-1]]
        for i in range(len(q) - 2, -1, -1):
            gap = (picked[-1][0] - q[i][0]).days
            if 60 <= gap <= 130:
                picked.append(q[i])
                if len(picked) == 4:
                    break
            elif gap > 130:
                break
        if len(picked) == 4:
            return sum(v for _, v in picked) / 4.0
    ann = _edgar.annual_series(sub)
    return ann[-1][1] if ann else None


def _instant_at(entries: list[dict], cutoff: str,
                end: datetime.date) -> float | None:
    """Instant (balance-sheet) value at period *end*, from facts filed
    <= cutoff; falls back to the latest period end on or before *end*."""
    series = _edgar.instant_series(_asof(entries, cutoff))
    val = None
    for e, v in series:
        if e <= end:
            val = v
        else:
            break
    return val


def _pit_entries(facts: dict, tag_units: list[tuple[str, str]]) -> list[dict]:
    """Every filed variant of a concept's facts, PIT-preserving.

    Unlike edgar._merged_entries (which keeps one entry per period —
    the latest filing), this keeps one entry per (start, end, filed):
    tag priority resolves only true duplicates (same period AND same
    filing reported under two tags). Callers filter by filed <= cutoff
    first; edgar's series helpers then dedupe by period end, which —
    among the surviving entries — picks the latest filing on or before
    the cutoff. Exactly the point-in-time semantics FDClient documents.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    best: dict[tuple, tuple[int, dict]] = {}
    for prio, (tag, unit) in enumerate(tag_units):
        node = gaap.get(tag)
        if not node:
            continue
        for e in node.get("units", {}).get(unit, []):
            key = (e.get("start"), e["end"], e.get("filed", ""))
            cur = best.get(key)
            if cur is None or prio < cur[0]:
                best[key] = (prio, e)
    out = []
    for _, e in best.values():
        start, end = _edgar._d(e.get("start")), _edgar._d(e["end"])
        out.append({
            "start": start, "end": end, "val": float(e["val"]),
            "form": e.get("form", ""), "fp": e.get("fp", ""),
            "filed": e.get("filed", ""),
            "dur": (end - start).days if start and end else 0,
        })
    return out


def _concept_entries(facts: dict, concept: str) -> list[dict]:
    """PIT entries for a paper_agent concept (tag fallbacks + units
    resolved exactly as edgar.CONCEPTS declares them). Net income adds
    ProfitLoss: some filers (Mastercard) report quarterly ProfitLoss
    but only annual NetIncomeLoss."""
    if concept == "net_income":
        return _pit_entries(facts, [("NetIncomeLoss", "USD"),
                                    ("ProfitLoss", "USD")])
    tags, units_pref = _edgar.CONCEPTS[concept]
    gaap = facts.get("facts", {}).get("us-gaap", {})
    tag_units = []
    for tag in tags:
        node = gaap.get(tag)
        if not node:
            continue
        unit = next((u for u in units_pref if u in node.get("units", {})),
                    None)
        if unit is not None:
            tag_units.append((tag, unit))
    return _pit_entries(facts, tag_units)


# A cover count older than this relative to the period it prices is a
# stale stub (Visa/Mastercard dei series stop in 2010), not a share count.
_STALE_SHARES_DAYS = 400


def _shares_out_asof(facts: dict, cutoff: str,
                     end: datetime.date) -> float | None:
    """Shares outstanding (dei cover count first, us-gaap fallback) with
    period end <= *end*, from facts filed <= cutoff. Stale stubs (period
    end >400 days before *end*) are rejected."""
    dei = facts.get("facts", {}).get("dei", {}).get(
        "EntityCommonStockSharesOutstanding", {})
    units = dei.get("units", {}).get("shares", []) if dei else []
    # One count per period end: the latest filing on or before the cutoff.
    by_end: dict[str, tuple[str, float]] = {}
    for e in units:
        if e.get("filed", "") <= cutoff and e.get("end"):
            cur = by_end.get(e["end"])
            if cur is None or e["filed"] > cur[0]:
                by_end[e["end"]] = (e["filed"], float(e["val"]))
    cands = [(end, val) for end, (_, val) in by_end.items()]
    if not cands:
        entries = _asof(_concept_entries(facts, "shares_outstanding_gaap"),
                        cutoff)
        cands = [(e.isoformat(), v) for e, v in _edgar.instant_series(entries)]
    ok = [c for c in cands if c[0] <= end.isoformat()]
    if not ok:
        return None
    end_iso, val = max(ok)
    if (end - datetime.date.fromisoformat(end_iso)).days > _STALE_SHARES_DAYS:
        return None
    return val


def _public_float_asof(facts: dict, cutoff: str) -> float | None:
    """Latest dei EntityPublicFloat filed on or before *cutoff*."""
    units = facts.get("facts", {}).get("dei", {}).get(
        "EntityPublicFloat", {}).get("units", {}).get("USD", [])
    ok = [e for e in units if e.get("filed", "") <= cutoff]
    if not ok:
        return None
    return float(max(ok, key=lambda e: e["filed"])["val"])


def _eps_reported_asof(facts: dict, cutoff: str,
                       end: datetime.date) -> float | None:
    """Latest annual diluted EPS (as reported) filed <= cutoff with
    period end <= *end*. Fallback when no share count exists to derive
    EPS from net income."""
    entries = _asof(_concept_entries(facts, "eps_diluted"), cutoff)
    ann = [(e, v) for e, v in _edgar.annual_series(entries) if e <= end]
    return ann[-1][1] if ann else None


# ---------------------------------------------------------------------------
# Optional point-in-time prices from Alpaca's data API (paper keys)
# ---------------------------------------------------------------------------

class _PitPrices:
    """Close on or before a date, per (ticker, date), cached.

    Uses the same paper keys and data host as paper_agent/run_dry.py.
    Raw (unadjusted) closes: they pair with as-filed historical share
    counts for market cap, and with as-filed EPS for P/E.
    """

    def __init__(self) -> None:
        self._keys = self._load_keys()
        self._cache: dict[tuple[str, str], float | None] = {}
        self.disabled = self._keys is None

    @staticmethod
    def _load_keys() -> tuple[str, str] | None:
        kid = os.environ.get("ALPACA_API_KEY_ID")
        sec = os.environ.get("ALPACA_API_SECRET_KEY")
        if kid and sec:
            return kid, sec
        env_path = os.environ.get(
            "PAPER_AGENT_ENV",
            os.path.join(_FILES_DIR, "paper_agent", ".env"))
        try:
            env = {}
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        env[k] = v
            if env.get("ALPACA_API_KEY_ID") and env.get("ALPACA_API_SECRET_KEY"):
                return env["ALPACA_API_KEY_ID"], env["ALPACA_API_SECRET_KEY"]
        except OSError:
            pass
        return None

    def close_on_or_before(self, ticker: str, date_iso: str) -> float | None:
        if self.disabled:
            return None
        key = (ticker, date_iso)
        if key in self._cache:
            return self._cache[key]
        val = None
        try:
            end = datetime.date.fromisoformat(date_iso)
            start = (end - datetime.timedelta(days=15)).isoformat()
            q = urllib.parse.urlencode({
                "timeframe": "1Day", "start": f"{start}T00:00:00Z",
                "end": f"{date_iso}T23:59:59Z", "feed": "iex",
                "sort": "desc", "limit": 1, "adjustment": "raw",
            })
            req = urllib.request.Request(
                f"https://data.alpaca.markets/v2/stocks/{ticker}/bars?{q}",
                headers={"APCA-API-KEY-ID": self._keys[0],
                         "APCA-API-SECRET-KEY": self._keys[1]})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
            bars = data.get("bars") or []
            if bars:
                val = float(bars[0]["c"])
        except Exception as exc:  # price enrichment must never kill the run
            logger.warning("Alpaca price lookup failed (%s); disabling "
                           "prices for this run", type(exc).__name__)
            self.disabled = True
            val = None
        self._cache[key] = val
        return val

    def bars(self, ticker: str, start_date: str, end_date: str) -> list[Price]:
        if self.disabled:
            return []
        try:
            q = urllib.parse.urlencode({
                "timeframe": "1Day", "start": f"{start_date}T00:00:00Z",
                "end": f"{end_date}T23:59:59Z", "feed": "iex",
                "sort": "asc", "limit": 10000, "adjustment": "raw",
            })
            req = urllib.request.Request(
                f"https://data.alpaca.markets/v2/stocks/{ticker}/bars?{q}",
                headers={"APCA-API-KEY-ID": self._keys[0],
                         "APCA-API-SECRET-KEY": self._keys[1]})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            return [Price(open=b["o"], close=b["c"], high=b["h"],
                          low=b["l"], volume=int(b["v"]), time=b["t"])
                    for b in (data.get("bars") or [])]
        except Exception as exc:
            logger.warning("Alpaca bars failed (%s); disabling prices",
                           type(exc).__name__)
            self.disabled = True
            return []


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------

class EdgarDataClient:
    """DataClient (protocol) backed by SEC EDGAR company facts."""

    def __init__(self, prices: bool = True) -> None:
        self._facts_cache: dict[str, dict] = {}
        self._metrics_cache: dict[str, list[FinancialMetrics]] = {}
        self._tickers_extra: dict[str, int] = {}
        self._prices = _PitPrices() if prices else None
        # (ticker, report_period) -> "shares" | "float" | "none": where
        # each row's market_cap came from, so callers can flag the
        # EntityPublicFloat lower bound precisely, per row.
        self.cap_source: dict[tuple[str, str], str] = {}

    # -- identity ------------------------------------------------------
    def _cik(self, ticker: str) -> int:
        ticker = ticker.upper()
        entry = _universe.BY_TICKER.get(ticker)
        if entry:
            return entry.cik
        if ticker in self._tickers_extra:
            return self._tickers_extra[ticker]
        # Fallback: SEC's official ticker file (same UA rules as edgar.py).
        req = urllib.request.Request(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": _edgar.USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            rows = json.loads(r.read())
        for row in rows.values():
            if row["ticker"].upper() == ticker:
                self._tickers_extra[ticker] = int(row["cik_str"])
                return self._tickers_extra[ticker]
        raise ValueError(f"EdgarDataClient: no CIK known for {ticker!r}")

    def _facts(self, ticker: str) -> dict:
        ticker = ticker.upper()
        if ticker not in self._facts_cache:
            self._facts_cache[ticker] = _edgar.fetch_company_facts(
                self._cik(ticker))
        return self._facts_cache[ticker]

    # -- the persona path ----------------------------------------------
    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        """TTM metric rows that were PUBLIC as of *end_date*.

        Only period="ttm" is served (the snapshot's contract); anything
        else returns []. Rows are newest first, at most *limit*.
        """
        if period != "ttm":
            return []
        ticker = ticker.upper()
        rows = self._all_metrics(ticker)
        out = [m for m in rows if m.filing_date and m.filing_date <= end_date]
        return out[:limit]

    def _all_metrics(self, ticker: str) -> list[FinancialMetrics]:
        if ticker in self._metrics_cache:
            return self._metrics_cache[ticker]
        facts = self._facts(ticker)
        flow_entries = {c: _concept_entries(facts, c)
                        for c in _FLOWS + ["shares_diluted_wavg"]}
        inst_entries = {c: _concept_entries(facts, c)
                        for c in _INSTANTS}

        # Filing events: every distinct date a revenue fact became public.
        filed_dates = sorted({e["filed"] for e in flow_entries["revenue"]
                              if e["filed"]})
        observations: list[dict] = []
        seen_anchors: set[datetime.date] = set()
        for f in filed_dates:
            rev, anchor = _ttm_asof(flow_entries["revenue"], f)
            if rev is None or anchor is None or anchor in seen_anchors:
                continue
            seen_anchors.add(anchor)
            obs: dict = {"filed": f, "end": anchor, "ttm": {}, "inst": {}}
            for c in _FLOWS:
                obs["ttm"][c], _ = _ttm_asof(flow_entries[c], f)
            obs["ttm"]["revenue"] = rev
            obs["shares_wavg"] = _shares_wavg_asof(
                flow_entries["shares_diluted_wavg"], f)
            for c in _INSTANTS:
                obs["inst"][c] = _instant_at(inst_entries[c], f, anchor)
            obs["shares_out"] = _shares_out_asof(facts, f, anchor)
            obs["public_float"] = _public_float_asof(facts, f)
            obs["eps_reported"] = _eps_reported_asof(facts, f, anchor)
            observations.append(obs)

        rows: list[FinancialMetrics] = []
        for obs in observations:
            rows.append(self._row(ticker, obs))
        # Revenue / EPS growth need the year-ago TTM row: match by the
        # anchor period end closest to (this end - 365 days).
        by_end = {r.report_period: r for r in rows}
        ttm_rev = {o["end"].isoformat(): o["ttm"]["revenue"]
                   for o in observations}
        for r in rows:
            target = (datetime.date.fromisoformat(r.report_period)
                      - datetime.timedelta(days=365))
            anchor = None
            for cand in rows:
                d = abs((datetime.date.fromisoformat(cand.report_period)
                         - target).days)
                if d <= 100 and (anchor is None or d < anchor[0]):
                    anchor = (d, cand.report_period)
            anchor = anchor[1] if anchor else None
            cur_rev = ttm_rev.get(r.report_period)
            old_rev = ttm_rev.get(anchor) if anchor else None
            if cur_rev and old_rev:
                r.revenue_growth = round(cur_rev / old_rev - 1, 4)
            old_eps = by_end[anchor].earnings_per_share if anchor else None
            if r.earnings_per_share is not None and old_eps:
                r.earnings_per_share_growth = round(
                    r.earnings_per_share / old_eps - 1, 4)

        rows.sort(key=lambda m: m.report_period, reverse=True)
        self._metrics_cache[ticker] = rows
        return rows

    def _row(self, ticker: str, obs: dict) -> FinancialMetrics:
        ttm, inst = obs["ttm"], obs["inst"]
        rev = ttm.get("revenue")
        ni = ttm.get("net_income")
        equity = inst.get("equity")
        assets = inst.get("assets")
        shares_wavg = obs.get("shares_wavg")
        shares_out = obs.get("shares_out")

        def ratio(num, den):
            if num is None or den is None or den == 0:
                return None
            return round(num / den, 4)

        # Share count for per-share / cap math: cover count, else the
        # diluted weighted average (stale stubs already rejected above).
        shares = shares_out if shares_out else shares_wavg
        eps = ratio(ni, shares_wavg)
        if eps is None:
            eps = obs.get("eps_reported")
        fcf = None
        if ttm.get("cfo") is not None and ttm.get("capex") is not None:
            fcf = ttm["cfo"] - ttm["capex"]
        debt = None
        if inst.get("debt_current") is not None or inst.get("debt_noncurrent") is not None:
            debt = (inst.get("debt_current") or 0) + (inst.get("debt_noncurrent") or 0)

        price = None
        if self._prices is not None:
            price = self._prices.close_on_or_before(ticker, obs["filed"])
        if price is not None and shares:
            market_cap = price * shares
            cap_src = "shares"
        elif obs.get("public_float"):
            # No usable share count (Visa): public float is the floor.
            market_cap = obs["public_float"]
            cap_src = "float"
        else:
            market_cap = None
            cap_src = "none"
        self.cap_source[(ticker, obs["end"].isoformat())] = cap_src
        pe = None
        if price is not None and eps:
            pe = round(price / eps, 4)
        elif market_cap is not None and ni:
            pe = round(market_cap / ni, 4)

        return FinancialMetrics(
            ticker=ticker,
            report_period=obs["end"].isoformat(),
            period="ttm",
            currency="USD",
            filing_date=obs["filed"],
            market_cap=market_cap,
            enterprise_value=(market_cap + debt - (inst.get("cash") or 0)
                              if market_cap is not None and debt is not None
                              else None),
            price_to_earnings_ratio=pe,
            price_to_book_ratio=ratio(price, ratio(equity, shares)),
            price_to_sales_ratio=ratio(market_cap, rev),
            free_cash_flow_yield=ratio(fcf, market_cap),
            gross_margin=ratio(ttm.get("gross_profit"), rev),
            operating_margin=ratio(ttm.get("operating_income"), rev),
            net_margin=ratio(ni, rev),
            return_on_equity=ratio(ni, equity),
            return_on_assets=ratio(ni, assets),
            current_ratio=ratio(inst.get("assets_current"),
                                inst.get("liabilities_current")),
            debt_to_equity=ratio(debt, equity),
            debt_to_assets=ratio(debt, assets),
            interest_coverage=ratio(ttm.get("operating_income"),
                                    ttm.get("interest_expense")),
            earnings_per_share=eps,
            book_value_per_share=ratio(equity, shares),
            free_cash_flow_per_share=ratio(fcf, shares_wavg),
        )

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        ticker = ticker.upper()
        try:
            facts = self._facts(ticker)
            cik = self._cik(ticker)
        except Exception:
            return None
        entry = _universe.BY_TICKER.get(ticker)
        sector = _SECTOR_LABELS.get(entry.sector) if entry else None
        return CompanyFacts(
            ticker=ticker,
            name=facts.get("entityName"),
            cik=str(cik),
            sector=sector,
            sec_filings_url=f"https://www.sec.gov/edgar/browse/?CIK={cik}",
        )

    # -- prices (Alpaca enrichment; empty when unavailable) ------------
    def get_prices(self, ticker: str, start_date: str, end_date: str,
                   **kwargs) -> list[Price]:
        if self._prices is None:
            return []
        return self._prices.bars(ticker.upper(), start_date, end_date)

    def get_snapshot(self, ticker: str) -> PriceSnapshot | None:
        if self._prices is None or self._prices.disabled:
            return None
        today = datetime.date.today().isoformat()
        close = self._prices.close_on_or_before(ticker.upper(), today)
        if close is None:
            return None
        return PriceSnapshot(ticker=ticker.upper(), price=close, time=today)

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        rows = self.get_financial_metrics(ticker, end_date, period="ttm",
                                          limit=1)
        return rows[0].market_cap if rows else None

    # -- no EDGAR equivalent: empty per protocol contract ---------------
    def get_news(self, ticker: str, end_date: str,
                 start_date: str | None = None,
                 limit: int = 1000) -> list[CompanyNews]:
        return []

    def get_insider_trades(self, ticker: str, end_date: str,
                           start_date: str | None = None,
                           limit: int = 1000) -> list[InsiderTrade]:
        return []

    def get_earnings(self, ticker: str) -> Earnings | None:
        return None

    def get_earnings_history(self, ticker: str,
                             limit: int = 12) -> list[EarningsRecord]:
        return []

    def screen(self, as_of: str, filters: list[ScreenFilter], *,
               order_by: OrderBy | None = None,
               limit: int = 50) -> list[ScreenRow]:
        return []

    def screener_fields(self) -> ScreenerFields:
        return ScreenerFields(numeric=frozenset(), sectors=frozenset(),
                              industries=frozenset(), operators=frozenset())
