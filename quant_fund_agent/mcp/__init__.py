"""Anthropic MCP integration for the quant fund.

Every agent's deterministic toolbox is exposed as an MCP stdio server, with a
synchronous client facade the (synchronous LangGraph) agents call.  Each facade
falls back to calling the same underlying ``*_service`` functions in-process
when MCP is disabled, so the two paths compute identical results.

Servers (``*_server``) / clients (``*_client``) / shared logic (``*_service``):

- ``modeling_server`` / ``client`` — Architect's model toolbox
  (``list_models``, ``fit_and_backtest``).
- ``catalog_server`` / ``catalog_client`` — Selector's factor catalog
  (``load_factor_catalog``).
- ``research_server`` / ``research_client`` — Factor Researcher's deterministic
  backends (paper loading, codegen materialisation, IC backtest, persistence).
- ``statistics_server`` / ``statistics_client`` — Statistician's tests
  (``list_tests``, ``run_tests``).
- ``portfolio_server`` / ``portfolio_client`` — Portfolio Manager's analytics
  (``screen_strategies``, ``construct_portfolio``, ``expected_portfolio_metrics``).

``_bridge`` is the shared transport: one persistent stdio session per server on
a single background event loop.

Toggle MCP off with the global ``QF_USE_MCP=0`` or a per-server override
(``MODELING_USE_MCP``, ``CATALOG_USE_MCP``, ``RESEARCH_USE_MCP``,
``STATISTICS_USE_MCP``, ``PORTFOLIO_USE_MCP``).

Note: this package is ``quant_fund_agent.mcp``; the third-party SDK is the
top-level ``mcp`` package.  Absolute imports keep the two distinct.
"""
