# ai-hedge-fund — Rajesh fundamentals wrapper: build notes

Built 2026-11-20 on the fork `Raja-the-Great/ai-hedge-fund` (upstream
v2.7.2, HEAD `5c553b5`), branch `rajesh-fundamentals`. **New files only —
no upstream code was edited.** Research/decision support only: no
trades, and the repo's paper/backtest execution paths are not used.

## What was built (`rajesh_fundamentals/`)

| File | What it does |
|---|---|
| `edgar_client.py` | `EdgarDataClient` — a drop-in `DataClient` (structural Protocol in `hedge_fund/data/protocol.py`) backed by free SEC EDGAR data instead of the paid Financial Datasets API. |
| `mandate.yaml` | Fund mandate for the repo's own spec loader: one long-only `fundamentals` strategy running the **buffett + graham + munger** personas, weekly rebalance, SPY benchmark. |
| `run_fundamentals.py` | Runner with three modes: `--check` (prerequisite validation), default (EDGAR snapshots for the starter tickers), `--personas` (persona signals → normalized JSON). Flags: `--local` (Ollama), `--tickers`, `--agents`. |
| `output/` | Snapshot proof files (`{TICKER}_snapshot.json/.txt`, `snapshot_summary.json`) and the persona run output (`AAPL.json`, `summary.json`). |

Starter tickers: **AAPL, MSFT, ADBE, V, MA**.

### Why the runner drives `predict()` directly

The repo's own runners (`hedge_fund/run.py`: TUI, backtest, paper)
hard-wire `FDClient` — there is no data-client injection point. The
personas, however, take `predict(ticker, date, data_client)`, so the
runner constructs the registry personas itself and passes
`EdgarDataClient`. The mandate YAML still loads through the repo's own
`load_spec()` (verified) and documents the intended fund shape;
`mandate.yaml`'s header records this wiring fact.

## EdgarDataClient — the data layer

Backed by the existing free pipeline
(`paper_agent/edgar.py` companyfacts client + `universe.py`
ticker→CIK map, 7-day cache). Only the two persona-path methods carry
real data — `get_financial_metrics(ticker, end_date, period="ttm",
limit)` and `get_company_facts(ticker)`; news / insider trades /
estimates / screen return empty per the protocol contract (personas
never call them).

**Point-in-time contract.** The snapshot layer filters metric rows by
*filing* date, so the client is filing-date aware throughout:

- It does **not** use `edgar._merged_entries()`: that helper collapses
  each (start, end) period to its *latest* filing, which destroys PIT
  semantics (a period's original 10-Q filing is replaced by next
  year's comparative restatement). Verified live on Mastercard: every
  period carried a filed date ~1 year late, and its annual
  NetIncomeLoss survived the collapse only as a DEF 14A entry that
  then failed the 10-K form filter, anchoring net income in 2014.
  The client instead loads PIT entries — every filed variant kept,
  tag priority resolved per (start, end, filed) — and lets the as-of
  filter plus edgar's series helpers (dedupe by period end among the
  surviving entries) reconstruct exactly what was public on a date.
- **Q4 synthesis.** Most filers never tag a standalone Q4 flow value
  (the 10-K tags the full year; the Q3 10-Q tags the 9-month YTD), so
  a naive 4-quarter TTM walk breaks every year. Missing Q4s are
  derived as `annual − 9M YTD` from as-filed facts. Without this,
  AAPL's TTM anchored on annual periods only after FY21.
- **Prices** are point-in-time raw closes from the Alpaca paper data
  API (IEX feed, close on or before each row's filing date), pairing
  with as-filed share counts/EPS. If price access fails, fundamentals
  still build; market_cap / P/E stay null.

### EDGAR share-count corruption — found and handled

Same corruption family the paper-agent's frozen-universe audit hit:

- **Visa has no per-share data in companyfacts at all** (no diluted
  share count, no diluted EPS, no shares-outstanding tags), and its
  dei cover count is a stale 2010 stub (469M shares). **Mastercard's**
  dei cover count is likewise a stale 2010 stub (~123M vs ~906M real).
- Fallbacks, in order: cover count (rejected if its period end is
  >400 days before the anchor — the staleness test), else diluted
  weighted-average shares; market cap = price × shares, else the dei
  **EntityPublicFloat** (a lower bound, flagged per row in
  `cap_source` / `market_cap_source`); EPS = NI ÷ wavg shares, else
  the latest as-reported annual diluted EPS; P/E = price ÷ EPS, else
  market cap ÷ net income. Mastercard also needed net income merged
  across tags (`NetIncomeLoss` + `ProfitLoss` — it files quarterly
  ProfitLoss only).
- Consequence: **Visa's EPS stays null** — a genuine EDGAR gap, not
  papered over. Its P/E (21.3) is market-cap ÷ net income.

## EDGAR snapshot proof (no LLM key needed)

`python -m rajesh_fundamentals.run_fundamentals` — all five tickers
built 20 filed TTM periods each (full quarterly cadence):

| Ticker | Latest period | Filed | Market cap | Cap source | P/E | ROE avg | Net margin avg | D/E |
|---|---|---|---|---|---|---|---|---|
| AAPL | 2026-09-26 | 2026-11-03 | $4,840.9B | price × shares | 27.74 | 1.48 | 0.258 | 0.70 |
| MSFT | 2026-09-30 | 2026-10-28 | $3,952.1B | price × shares | 20.85 | 0.34 | 0.365 | 0.08 |
| ADBE | 2026-08-28 | 2026-09-22 | $94.7B | price × shares | 9.86 | 0.43 | 0.280 | 0.54 |
| V | 2026-09-30 | 2026-11-05 | $501.7B | EntityPublicFloat (lower bound) | 21.35 | 0.49 | 0.521 | 0.61 |
| MA | 2026-09-30 | 2026-10-29 | $421.1B | price × shares | 24.83 | 1.81 | 0.431 | 3.55 |

Spot-checks: AAPL TTM revenue at the Q1 FY26 filing = $435.6B
(143.8 + derived-Q4 102.5 + 94.0 + 95.4); MA FY25 EPS $16.52, BVPS
$8.55. Full per-period rows are in `output/{TICKER}_snapshot.json`;
the persona-facing rendered tables in `output/{TICKER}_snapshot.txt`.

## Persona layer — local Llama route (no hosted key)

**Route found and used (wrapper config only, no upstream edits):**
the framework's `make_llm()` routes by registry model id and no
registry id names a local model, so `OPENAI_API_BASE` alone cannot
reach Ollama. But `LLMAgent` accepts an injected `LLMClient`, and the
framework's own `ChatLLM` wraps any langchain chat model. The runner
therefore builds the same `ChatOpenAI + ChatLLM` pair `make_llm()`
builds, pointed at Ollama's OpenAI-compatible endpoint
(`OLLAMA_API_BASE`, default `http://localhost:11434/v1`) with model
`OLLAMA_MODEL` (default `llama3.2:3b`), and injects it via
`LLMAgent(llm=...)`. Sandbox quirk handled in the wrapper: the VM's
proxy env vars (`no_proxy` with IPv6 entries) crash this httpx build
at client construction, so both sync and async httpx clients are
passed explicitly with `trust_env=False`.

**End-to-end run (2026-11-20):**

```
python -m rajesh_fundamentals.run_fundamentals --personas --local \
    --tickers AAPL --agents buffett
→ buffett / AAPL: signal bullish, confidence 90, value +0.90,
  verdict "buy candidate", elapsed 441.5s (total run 8m46s on this
  1.5-CPU VM), output in output/AAPL.json
```

The reasoning is coherent, in-persona, and cites the snapshot's real
figures (ROE 1.48, BVPS CAGR 0.13). **Honest quality note:** this is a
3B-parameter local model — confidence 90 is overconfident, the
reasoning praises quality without seriously engaging the P/E 27.7
valuation, and no bearish counter-reasoning was produced. Treat local
persona output as a structured second opinion and a pipeline proof,
not investment-grade analysis. A hosted frontier model (below) is the
quality route; the local route is the free, private, always-available
one.

## Keys still needed

- **None** for the EDGAR data path, and **none** for the local persona
  route above.
- For hosted persona quality: exactly **one** LLM key, chosen by
  `HEDGE_FUND_LLM_MODEL` — e.g. `ANTHROPIC_API_KEY` (default model
  `claude-opus-5-5`), `OPENAI_API_KEY`, `GOOGLE_API_KEY`,
  `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `KIMI_API_KEY`, or
  `TYPESAFE_API_KEY`. `--check` names the exact missing variable for
  the configured model and exits 2.
- `FINANCIAL_DATASETS_API_KEY` is **not** needed — the EDGAR adapter
  replaces FDClient on the persona path. (The repo's own TUI/backtest/
  paper runners still require it; they are unused here.)

## Limitations

- Visa EPS is null (no per-share tags exist in its EDGAR
  companyfacts); Visa market cap is the public-float lower bound.
- Prices come from Alpaca's free IEX feed (single-venue prints, raw
  closes) — fine for ratios, not a full-market SIP record.
- EDGAR facts are as-filed companyfacts values; the PIT loader keeps
  every filed variant, but XBRL tagging errors by filers flow through
  (the stale-stub and missing-tag cases above are handled, not
  eliminated as a class).
- One persona call on this VM takes ~7 minutes (3B model, shared
  CPUs); a 5-ticker × 3-persona run is ~2 hours locally. Hosted keys
  or a bigger machine change that math entirely.
- Persona signals are LLM judgments over a fundamentals table — the
  framework's own design — with no backtest validation performed here,
  and none claimed.
