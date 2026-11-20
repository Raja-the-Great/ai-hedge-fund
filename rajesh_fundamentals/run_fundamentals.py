"""Run the fundamentals desk: EDGAR snapshots + (optionally) personas.

Modes:
  --check      Validate prerequisites and stop. Exit 2 naming the exact
               missing env var when no LLM route is available.
  (default)    Build point-in-time fundamentals snapshots for the starter
               tickers from SEC EDGAR via EdgarDataClient — NO LLM key
               needed — save them under output/, print a summary. This
               is the proof the free data path works end to end.
  --personas   Additionally call the Buffett/Graham/Munger personas on
               each ticker and write normalized JSON per ticker (the
               shared cross-framework shape). Needs an LLM route:
               a provider key (ANTHROPIC_API_KEY etc.), or --local.
  --local      Route the personas to a local Ollama server
               (http://localhost:11434/v1, OpenAI-compatible) running
               llama3.2:3b — keyless. Uses the framework's own injection
               seam (LLMAgent(llm=ChatLLM(...))), no upstream edits.

Usage (from the repo root, inside the venv):
  python -m rajesh_fundamentals.run_fundamentals --check
  python -m rajesh_fundamentals.run_fundamentals
  python -m rajesh_fundamentals.run_fundamentals --personas --local --tickers AAPL
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

OUTPUT_DIR = os.path.join(_HERE, "output")
STARTER_TICKERS = ["AAPL", "MSFT", "ADBE", "V", "MA"]
PERSONAS = ["buffett", "graham", "munger"]
OLLAMA_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

_VERDICT_BAND = 0.25  # |mean signal value| needed to leave "watch"


# ---------------------------------------------------------------------------
# LLM routes
# ---------------------------------------------------------------------------

def _ollama_status() -> tuple[bool, bool]:
    """(server up, model pulled) for the local route."""
    base = OLLAMA_BASE.rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=3) as r:
            tags = json.loads(r.read())
        names = {m.get("name") for m in tags.get("models", [])}
        return True, OLLAMA_MODEL in names
    except Exception:
        return False, False


def _provider_key_status() -> tuple[str | None, str | None]:
    """(model id, missing env var) for the hosted-provider route.

    Mirrors hedge_fund.llm.client.make_llm's resolution: model from
    HEDGE_FUND_LLM_MODEL or the default; key var from the registry.
    """
    from hedge_fund.llm.client import DEFAULT_MODEL
    from hedge_fund.llm.registry import env_var_for, provider_for

    model = os.environ.get("HEDGE_FUND_LLM_MODEL") or DEFAULT_MODEL
    provider = provider_for(model) or "Anthropic"
    env_var = env_var_for(provider)
    if env_var and os.environ.get(env_var):
        return model, None
    return model, env_var


def check(local: bool) -> int:
    """Print prerequisite status; return a process exit code."""
    print("== ai-hedge-fund fundamentals wrapper: prerequisite check ==")
    print("Data: SEC EDGAR via EdgarDataClient — no key needed. OK")
    up, pulled = _ollama_status()
    if up and pulled:
        print(f"Local LLM: Ollama up at {OLLAMA_BASE}, "
              f"model {OLLAMA_MODEL} pulled. OK")
    elif up:
        print(f"Local LLM: Ollama up at {OLLAMA_BASE}, but model "
              f"{OLLAMA_MODEL} is NOT pulled (ollama pull {OLLAMA_MODEL}).")
    else:
        print(f"Local LLM: Ollama not reachable at {OLLAMA_BASE}.")
    model, missing = _provider_key_status()
    if missing is None:
        print(f"Hosted LLM: model {model}, provider key present. OK")
    else:
        print(f"Hosted LLM: model {model} needs env var {missing} "
              f"(not set).")
    if local:
        if up and pulled:
            return 0
        print(f"\nMISSING: local route requested but unavailable — start "
              f"Ollama and pull {OLLAMA_MODEL}, or drop --local.")
        return 2
    if missing is None or (up and pulled):
        return 0
    print(f"\nMISSING: {missing} — set it to use the hosted route, or "
          f"use --local with Ollama running {OLLAMA_MODEL}.")
    return 2


def _make_local_llm():
    """An LLMClient for Ollama, via the framework's own ChatLLM wrapper.

    make_llm()'s OpenAI branch honors OPENAI_API_BASE but routes by
    registry model id, and no registry id names a local model — so the
    wrapper constructs the same ChatOpenAI+ChatLLM pair make_llm builds,
    pointed at the local endpoint with the true model name, and injects
    it through LLMAgent(llm=...). No upstream code is touched.
    """
    import httpx
    from langchain_openai import ChatOpenAI

    from hedge_fund.llm.client import ChatLLM

    # trust_env=False: this sandbox exports http_proxy/no_proxy vars
    # (with IPv6 entries this httpx build fails to parse — it crashes
    # with "Invalid port ':1]'"), and a localhost endpoint must bypass
    # proxies anyway. BOTH clients must be passed: langchain-openai
    # otherwise builds a default async client from the env at
    # construction time and crashes before any request is made.
    http_client = httpx.Client(trust_env=False, timeout=1800)
    http_async_client = httpx.AsyncClient(trust_env=False, timeout=1800)
    chat = ChatOpenAI(
        model=OLLAMA_MODEL,
        api_key="ollama",  # Ollama ignores the key; the client requires one
        base_url=OLLAMA_BASE,
        timeout=1800,  # CPU inference on a small VM is slow; be patient
        max_retries=1,
        max_tokens=1024,  # the persona contract is one JSON verdict
        http_client=http_client,
        http_async_client=http_async_client,
    )
    return ChatLLM(OLLAMA_MODEL, chat)


# ---------------------------------------------------------------------------
# Snapshot proof (no LLM)
# ---------------------------------------------------------------------------

def run_snapshots(tickers: list[str], as_of: str) -> dict:
    from hedge_fund.features.snapshot import build_snapshot

    from rajesh_fundamentals.edgar_client import EdgarDataClient

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = EdgarDataClient()
    summary = {}
    for ticker in tickers:
        snap = build_snapshot(ticker, as_of, client)
        path = os.path.join(OUTPUT_DIR, f"{ticker}_snapshot.json")
        with open(path, "w") as f:
            json.dump(snap.model_dump(), f, indent=2)
        txt = os.path.join(OUTPUT_DIR, f"{ticker}_snapshot.txt")
        with open(txt, "w") as f:
            f.write(snap.render())
        latest = snap.periods[0]
        summary[ticker] = {
            "periods": len(snap.periods),
            "latest_period": latest.report_period,
            "latest_filed": latest.filing_date,
            "market_cap": snap.market_cap_latest,
            "market_cap_source": {
                "shares": "price x shares",
                "float": "entity_public_float (lower bound — no usable "
                         "share count in EDGAR)",
                "none": "unavailable",
            }[client.cap_source.get((ticker, latest.report_period),
                                    "none")],
            "pe_latest": latest.price_to_earnings_ratio,
            "roe_avg": snap.roe_avg,
            "net_margin_avg": snap.net_margin_avg,
            "debt_to_equity_latest": snap.debt_to_equity_latest,
            "snapshot_hash": snap.content_hash,
            "snapshot_path": path,
        }
        print(f"{ticker}: {len(snap.periods)} filed TTM periods, latest "
              f"{latest.report_period} (filed {latest.filing_date}), "
              f"mktcap={_fmt(snap.market_cap_latest)} "
              f"P/E={_fmt(latest.price_to_earnings_ratio)} "
              f"ROEavg={_fmt(snap.roe_avg)}")
    with open(os.path.join(OUTPUT_DIR, "snapshot_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Persona layer (LLM required)
# ---------------------------------------------------------------------------

def _verdict(mean_value: float) -> tuple[str, str]:
    if mean_value >= _VERDICT_BAND:
        return "bullish", "buy candidate"
    if mean_value <= -_VERDICT_BAND:
        return "bearish", "avoid"
    return "neutral", "watch"


def run_personas(tickers: list[str], as_of: str, local: bool,
                 personas: list[str]) -> dict:
    from hedge_fund.signals import ALPHA_MODEL_REGISTRY

    from rajesh_fundamentals.edgar_client import EdgarDataClient

    llm = _make_local_llm() if local else None  # None -> make_llm() default
    client = EdgarDataClient()
    out = {}
    for ticker in tickers:
        agents = []
        values = []
        for name in personas:
            agent = ALPHA_MODEL_REGISTRY[name](llm=llm) if llm \
                else ALPHA_MODEL_REGISTRY[name]()
            t0 = time.time()
            sig = agent.predict(ticker, as_of, client)
            elapsed = round(time.time() - t0, 1)
            md = sig.metadata or {}
            agents.append({
                "agent": name,
                "signal": md.get("signal"),
                "confidence": md.get("confidence"),
                "value": sig.value,
                "reasoning": sig.reasoning,
                "abstained": bool(md.get("abstained")),
                "model": md.get("model"),
                "elapsed_s": elapsed,
            })
            if not md.get("abstained"):
                values.append(sig.value)
            print(f"{ticker}/{name}: value={sig.value:+.2f} "
                  f"({elapsed}s)")
        mean_value = sum(values) / len(values) if values else 0.0
        rating, verdict = _verdict(mean_value)
        record = {
            "ticker": ticker,
            "trade_date": as_of,
            "framework": "ai-hedge-fund",
            "framework_rating": rating,
            "verdict": verdict,
            "agents": agents,
            "fundamentals_evidence": {
                "snapshot_summary": _snapshot_evidence(ticker),
                "report_path": os.path.join(
                    OUTPUT_DIR, f"{ticker}_snapshot.txt"),
            },
            "risks": {
                "bearish_reasoning": [
                    a["reasoning"] for a in agents
                    if a["signal"] == "bearish" and a["reasoning"]],
                "data_gaps": _data_gaps(ticker),
            },
            "raw_report_path": os.path.join(
                OUTPUT_DIR, f"{ticker}_snapshot.txt"),
        }
        path = os.path.join(OUTPUT_DIR, f"{ticker}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        out[ticker] = record
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def _snapshot_evidence(ticker: str) -> dict:
    path = os.path.join(OUTPUT_DIR, f"{ticker}_snapshot.json")
    try:
        with open(path) as f:
            snap = json.load(f)
        return {k: snap.get(k) for k in (
            "roe_avg", "net_margin_avg", "gross_margin_trend",
            "bvps_cagr", "debt_to_equity_latest", "market_cap_latest",
            "eps_growth_yoy")}
    except OSError:
        return {}


def _data_gaps(ticker: str) -> list[str]:
    gaps = []
    path = os.path.join(OUTPUT_DIR, f"{ticker}_snapshot.json")
    try:
        with open(path) as f:
            snap = json.load(f)
        latest = snap["periods"][0]
        if latest.get("market_cap") is None:
            gaps.append("market_cap unavailable (no point-in-time price)")
        if latest.get("price_to_earnings_ratio") is None:
            gaps.append("P/E unavailable (price or EPS missing)")
    except OSError:
        gaps.append("snapshot not built")
    return gaps


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and abs(v) >= 1e9:
        return f"{v / 1e9:.1f}B"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="validate prerequisites and stop")
    ap.add_argument("--personas", action="store_true",
                    help="also run the LLM personas (needs an LLM route)")
    ap.add_argument("--local", action="store_true",
                    help="route personas to local Ollama (llama3.2:3b)")
    ap.add_argument("--tickers", default=",".join(STARTER_TICKERS),
                    help="comma-separated tickers")
    ap.add_argument("--agents", default=",".join(PERSONAS),
                    help="comma-separated persona names")
    ap.add_argument("--as-of", default=datetime.date.today().isoformat(),
                    help="as-of date YYYY-MM-DD (default: today)")
    args = ap.parse_args(argv)

    if args.check:
        return check(args.local)
    if args.personas:
        rc = check(args.local)
        if rc != 0:
            return rc
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    run_snapshots(tickers, args.as_of)
    if args.personas:
        personas = [p.strip() for p in args.agents.split(",") if p.strip()]
        run_personas(tickers, args.as_of, args.local, personas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
