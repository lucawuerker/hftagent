"""Deprecated stub package.

The original ``agents/strategy`` subgraph was a scaffold of ``# TODO``
no-op nodes.  The genuine strategy-building pipeline is

    Selector -> Architect -> Statistician -> persist

and lives in the ``selector`` / ``architect`` / ``statistician`` agent
packages, wired together by :mod:`quant_fund_agent.pipeline`.  This
package is intentionally left empty.
"""
