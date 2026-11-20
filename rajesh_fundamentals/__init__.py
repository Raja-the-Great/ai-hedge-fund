"""Rajesh's fundamentals wrapper around the ai-hedge-fund fork.

New code only — nothing upstream is modified. Contents:
- edgar_client.EdgarDataClient: a DataClient-protocol implementation
  backed by free SEC EDGAR data (via the paper_agent pipeline) instead
  of the paid Financial Datasets API.
- run_fundamentals: snapshot proof + (key permitting) persona runner.
- mandate.yaml: the Buffett/Graham/Munger fundamentals desk, in the
  repo's own mandate format.
"""
