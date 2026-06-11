"""Fundamental / estimate / event factors.

These read the non-OHLCV panel fields the data layer supplies (availability-
stamped and forward-filled — see :mod:`quant_fund_agent.data.fundamentals`).  A
factor declares the fields it needs in ``inputs``; the capability-gating layer
makes it visible only on a provider that supplies them (FMP / AlphaVantage on an
equity universe) and gates it out elsewhere — it never crashes on a missing
column.
"""
