"""Generate Excalidraw diagrams for the QuantFundAgent project.

Produces, under ``diagrams/``:

  * ``00_system_architecture.excalidraw`` – the high-level structure: every
    agent, its MCP tool server ("core functions"), and the connected
    databases / data stores.
  * ``agent_factor_researcher.excalidraw`` … ``agent_portfolio_manager.excalidraw``
    – one LangGraph node-flow diagram per agent.

The files open in https://excalidraw.com, the VS Code "Excalidraw" extension,
or Obsidian.  Re-run ``./venv/bin/python build_diagrams.py`` after editing the
layout / labels here to regenerate them.

Geometry note: arrows attach to the *edge* of each box (computed from the line
joining the two box centres), so nudging a box in the script keeps its arrows
attached.  When opened and a box is dragged in Excalidraw, the binding keeps the
arrow attached too.
"""

from __future__ import annotations

import json
import os
import random

random.seed(7)

OUT_DIR = "diagrams"

# (stroke, fill) per semantic role — Excalidraw's standard palette.
PALETTE = {
    "agent":    ("#1971c2", "#a5d8ff"),   # blue
    "tool":     ("#2f9e44", "#b2f2bb"),   # green
    "db":       ("#f08c00", "#ffec99"),   # yellow
    "data":     ("#495057", "#dee2e6"),   # grey
    "startend": ("#1e1e1e", "#ffffff"),   # white
    "note":     ("#1e1e1e", "#fff3bf"),   # light amber strip
    "process":  ("#1971c2", "#a5d8ff"),   # graph node = blue
}

GREY = "#868e96"
INK = "#1e1e1e"


def _id() -> str:
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEF0123456789", k=18))


def _nonce() -> int:
    return random.randint(1, 2_000_000_000)


def _nlines(text: str) -> int:
    return text.count("\n") + 1


class Diagram:
    def __init__(self) -> None:
        self.elements: list[dict] = []
        self._by_id: dict[str, dict] = {}

    # ── primitives ────────────────────────────────────────────────────
    def _register(self, el: dict) -> dict:
        self.elements.append(el)
        self._by_id[el["id"]] = el
        return el

    def box(self, x, y, w, h, text, kind="agent", shape="rectangle", font=15):
        stroke, fill = PALETTE[kind]
        roundness = None
        if shape == "rectangle":
            roundness = {"type": 3}
        rid = _id()
        rect = {
            "id": rid, "type": shape, "x": x, "y": y, "width": w, "height": h,
            "angle": 0, "strokeColor": stroke, "backgroundColor": fill,
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
            "roundness": roundness, "seed": _nonce(), "version": 1,
            "versionNonce": _nonce(), "isDeleted": False, "boundElements": [],
            "updated": 1, "link": None, "locked": False,
        }
        self._register(rect)
        if text:
            self._bound_text(rect, text, font=font, color=stroke if kind == "startend" else INK)
        return rid

    def _bound_text(self, container, text, font=15, color=INK):
        tid = _id()
        n = _nlines(text)
        lh = 1.25
        th = n * font * lh
        widest = max(len(line) for line in text.split("\n"))
        tw = min(container["width"] - 12, widest * font * 0.58 + 8)
        tx = container["x"] + (container["width"] - tw) / 2
        ty = container["y"] + (container["height"] - th) / 2
        t = {
            "id": tid, "type": "text", "x": tx, "y": ty, "width": tw, "height": th,
            "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
            "roundness": None, "seed": _nonce(), "version": 1,
            "versionNonce": _nonce(), "isDeleted": False, "boundElements": None,
            "updated": 1, "link": None, "locked": False, "fontSize": font,
            "fontFamily": 2, "text": text, "textAlign": "center",
            "verticalAlign": "middle", "containerId": container["id"],
            "originalText": text, "autoResize": True, "lineHeight": lh,
        }
        container["boundElements"].append({"type": "text", "id": tid})
        self._register(t)
        return tid

    def text(self, x, y, text, font=14, color=INK, align="left", bold=False):
        n = _nlines(text)
        lh = 1.25
        widest = max(len(line) for line in text.split("\n"))
        tw = widest * font * 0.58 + 8
        th = n * font * lh
        t = {
            "id": _id(), "type": "text", "x": x, "y": y, "width": tw, "height": th,
            "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
            "roundness": None, "seed": _nonce(), "version": 1,
            "versionNonce": _nonce(), "isDeleted": False, "boundElements": None,
            "updated": 1, "link": None, "locked": False, "fontSize": font,
            "fontFamily": 2, "text": text, "textAlign": align,
            "verticalAlign": "top", "containerId": None,
            "originalText": text, "autoResize": True, "lineHeight": lh,
        }
        return self._register(t)["id"]

    # ── arrows ────────────────────────────────────────────────────────
    @staticmethod
    def _center(b):
        return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2

    @staticmethod
    def _edge_point(b, tx, ty):
        cx, cy = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
        dx, dy = tx - cx, ty - cy
        if dx == 0 and dy == 0:
            return cx, cy
        hw, hh = b["width"] / 2, b["height"] / 2
        sx = hw / abs(dx) if dx != 0 else float("inf")
        sy = hh / abs(dy) if dy != 0 else float("inf")
        t = min(sx, sy)
        return cx + dx * t, cy + dy * t

    def arrow(self, src_id, dst_id, label=None, dashed=False, double=False,
              waypoints=None, color=INK, label_font=12):
        src, dst = self._by_id[src_id], self._by_id[dst_id]
        scx, scy = self._center(src)
        dcx, dcy = self._center(dst)
        if waypoints:
            sx, sy = self._edge_point(src, *waypoints[0])
            ex, ey = self._edge_point(dst, *waypoints[-1])
            abs_pts = [(sx, sy)] + list(waypoints) + [(ex, ey)]
        else:
            sx, sy = self._edge_point(src, dcx, dcy)
            ex, ey = self._edge_point(dst, scx, scy)
            abs_pts = [(sx, sy), (ex, ey)]
        pts = [[round(px - sx, 2), round(py - sy, 2)] for px, py in abs_pts]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        aid = _id()
        arrow = {
            "id": aid, "type": "arrow", "x": sx, "y": sy,
            "width": max(xs) - min(xs), "height": max(ys) - min(ys),
            "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 2,
            "strokeStyle": "dashed" if dashed else "solid", "roughness": 1,
            "opacity": 100, "groupIds": [], "frameId": None,
            "roundness": {"type": 2}, "seed": _nonce(), "version": 1,
            "versionNonce": _nonce(), "isDeleted": False, "boundElements": [],
            "updated": 1, "link": None, "locked": False, "points": pts,
            "lastCommittedPoint": None,
            "startBinding": {"elementId": src_id, "focus": 0, "gap": 4},
            "endBinding": {"elementId": dst_id, "focus": 0, "gap": 4},
            "startArrowhead": "arrow" if double else None,
            "endArrowhead": "arrow",
        }
        src["boundElements"].append({"type": "arrow", "id": aid})
        dst["boundElements"].append({"type": "arrow", "id": aid})
        self._register(arrow)
        if label:
            mid = abs_pts[len(abs_pts) // 2] if len(abs_pts) > 2 else (
                (sx + ex) / 2, (sy + ey) / 2)
            tid = _id()
            n = _nlines(label)
            tw = max(len(l) for l in label.split("\n")) * label_font * 0.58 + 8
            th = n * label_font * 1.25
            t = {
                "id": tid, "type": "text", "x": mid[0] - tw / 2,
                "y": mid[1] - th / 2, "width": tw, "height": th, "angle": 0,
                "strokeColor": color, "backgroundColor": "#ffffff",
                "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
                "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
                "roundness": None, "seed": _nonce(), "version": 1,
                "versionNonce": _nonce(), "isDeleted": False,
                "boundElements": None, "updated": 1, "link": None,
                "locked": False, "fontSize": label_font, "fontFamily": 2,
                "text": label, "textAlign": "center", "verticalAlign": "middle",
                "containerId": aid, "originalText": label, "autoResize": True,
                "lineHeight": 1.25,
            }
            arrow["boundElements"].append({"type": "text", "id": tid})
            self._register(t)
        return aid

    # ── serialise ─────────────────────────────────────────────────────
    def save(self, path):
        doc = {
            "type": "excalidraw", "version": 2,
            "source": "https://github.com/QuantFundAgent",
            "elements": self.elements,
            "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
            "files": {},
        }
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=2)
        print(f"wrote {path}  ({len(self.elements)} elements)")


# ===========================================================================
# Diagram 1 — system architecture
# ===========================================================================

def build_system() -> Diagram:
    d = Diagram()
    AW, AH = 220, 84            # agent box
    SW, SH = 230, 86           # server box
    agent_cx = {"fr": 180, "sel": 470, "arc": 760, "stat": 1050, "pm": 1340}

    d.text(60, 18, "QuantFundAgent — System Architecture", font=30, bold=True)
    d.text(62, 56,
           "LangGraph multi-agent pipeline mirroring a quant fund's org chart.  "
           "Each agent calls a deterministic MCP tool server; shared state lives in the databases.",
           font=14, color=GREY)

    # orchestration strip
    orch = d.box(60, 92, 1500, 46,
                 "Orchestration  ·  quant_fund_agent/pipeline.py     "
                 "run_research_session → run_strategy_pipeline → persist_approved_strategy → run_pm_rebalance",
                 kind="note", font=14)

    agents = {
        "fr": "Factor Researcher\nreads papers · invents alpha\nfactors · IC-gates them",
        "sel": "Selector\nforms a hypothesis ·\npicks factors",
        "arc": "Architect\nfits a model · backtests ·\nrefinement loop",
        "stat": "Statistician\nOOS · deflated Sharpe ·\naccept / reject gate",
        "pm": "Portfolio Manager\nscreen · allocate ·\nmonitor / retire",
    }
    servers = {
        "fr": "research_server\npapers · codegen ·\nIC backtest · persist",
        "sel": "catalog_server\nfactor catalog",
        "arc": "modeling_server\nmodel toolbox ·\nfit + backtest",
        "stat": "statistics_server\ndeflated Sharpe · OOS test",
        "pm": "portfolio_server\nscreen · construct ·\nexpected metrics",
    }

    AY, SY = 170, 320
    aid_ = {}
    sid_ = {}
    for k, cx in agent_cx.items():
        aid_[k] = d.box(cx - AW / 2, AY, AW, AH, agents[k], kind="agent", font=13)
        sid_[k] = d.box(cx - SW / 2, SY, SW, SH, servers[k], kind="tool", font=12)
        d.arrow(aid_[k], sid_[k], double=True)   # agent <-> its MCP server

    # pipeline stage order (horizontal)
    d.arrow(aid_["fr"], aid_["sel"], label="new factors")
    d.arrow(aid_["sel"], aid_["arc"], label="hypothesis\n+ factors")
    d.arrow(aid_["arc"], aid_["stat"], label="strategy spec\n+ IS metrics")
    d.arrow(aid_["stat"], aid_["pm"], label="approved\nstrategy")

    # storage band
    STY, STH = 480, 80
    store = {}
    store["paper"] = d.box(20, STY, 190, STH, "Paper DB\n+ PDF corpus", kind="data", font=12)
    store["factor"] = d.box(285, STY, 160, STH, "Factor DB", kind="db", font=13)
    store["panel"] = d.box(560, STY, 270, STH,
                           "Intraday price panel\n10s bars · ~50 tickers\n(ticker_data/)", kind="data", font=12)
    store["artifacts"] = d.box(865, STY, 175, STH, "Model artifacts\n(.joblib)", kind="db", font=12)
    store["strategy"] = d.box(1090, STY, 175, STH, "Strategy DB", kind="db", font=13)
    store["portfolio"] = d.box(1320, STY, 175, STH, "Portfolio DB", kind="db", font=13)

    # server <-> storage (dashed = read, solid = write)
    d.arrow(sid_["fr"], store["paper"], dashed=True, color=GREY)
    d.arrow(sid_["fr"], store["panel"], dashed=True, color=GREY)
    d.arrow(sid_["fr"], store["factor"])                       # write kept factors
    d.arrow(sid_["sel"], store["factor"], dashed=True, color=GREY)
    d.arrow(sid_["arc"], store["panel"], dashed=True, color=GREY)
    d.arrow(sid_["arc"], store["artifacts"])                   # write fitted model
    d.arrow(sid_["stat"], store["panel"], dashed=True, color=GREY)
    d.arrow(sid_["stat"], store["artifacts"], dashed=True, color=GREY)   # reload model
    d.arrow(sid_["pm"], store["strategy"], double=True)        # read book / set pm_status
    d.arrow(sid_["pm"], store["portfolio"])                    # write portfolio record

    d.text(1090, 566, "Strategy records written by\npipeline.persist_approved_strategy",
           font=10, color=GREY)

    # legend
    LY = 612
    swatches = [("agent", "Agent (LangGraph)"), ("tool", "MCP tool server"),
                ("db", "Database (state)"), ("data", "Data source")]
    x = 60
    for kind, lab in swatches:
        d.box(x, LY, 22, 22, "", kind=kind)
        d.text(x + 30, LY + 3, lab, font=13)
        x += 230
    d.text(x, LY + 3, "→ solid = write / call        ⇢ dashed = read",
           font=13, color=GREY)
    return d


# ===========================================================================
# Helpers for the per-agent graph diagrams
# ===========================================================================

def _graph_header(d: Diagram, title: str, subtitle: str):
    d.text(40, 18, title, font=26, bold=True)
    d.text(42, 52, subtitle, font=14, color=GREY)


def _se(d: Diagram, cx, y, label):  # start/end pill
    return d.box(cx - 55, y, 110, 40, label, kind="startend", font=14, shape="rectangle")


# ===========================================================================
# Diagram 2 — Factor Researcher
# ===========================================================================

def build_factor_researcher() -> Diagram:
    d = Diagram()
    _graph_header(d, "Factor Researcher — LangGraph subgraph",
                  "One research session: read papers → invent factors → IC-gate the survivors.")
    cx, W, H = 320, 340, 70
    L = cx - W / 2
    start = _se(d, cx, 86, "START")
    n_lp = d.box(L, 150, W, H, "load_papers\npick N papers ≤ cutoff · extract text (MCP)", kind="process", font=13)
    n_bs = d.box(L, 260, W, H, "brainstorm\n1 LLM call → K distinct factor ideas", kind="process", font=13)
    n_gc = d.box(L, 400, W, H, "generate_code\nLLM writes a BaseFactor subclass per idea (+1 retry)", kind="process", font=12)
    n_bt = d.box(L, 510, W, H, "backtest_factors\nstandard single-factor IC backtest (MCP)", kind="process", font=13)
    n_fp = d.box(L, 620, W, H, "filter_and_persist\nkeep |IC| ≥ threshold · tag source=RESEARCHER · purge rest", kind="process", font=12)
    end = _se(d, cx, 740, "END")

    d.arrow(start, n_lp)
    d.arrow(n_lp, n_bs)
    d.arrow(n_bs, n_gc, label="ideas ≥ 1")
    d.arrow(n_gc, n_bt, label="code ok")
    d.arrow(n_bt, n_fp)
    d.arrow(n_fp, end)
    # skip routes (conditional edges), routed on the right
    d.arrow(n_bs, end, label="no ideas",
            waypoints=[(700, 295), (700, 760)], color=GREY, dashed=True)
    d.arrow(n_gc, n_fp, label="no code",
            waypoints=[(620, 435), (620, 655)], color=GREY, dashed=True)
    return d


# ===========================================================================
# Diagram 3 — Selector
# ===========================================================================

def build_selector() -> Diagram:
    d = Diagram()
    _graph_header(d, "Selector — LangGraph subgraph",
                  "Turn the factor catalog into a trading hypothesis and a chosen factor set.")
    cx, W, H = 300, 320, 70
    L = cx - W / 2
    start = _se(d, cx, 96, "START")
    a = d.box(L, 160, W, H, "load_factor_catalog\nread factor DB → compact catalog (MCP)", kind="process", font=13)
    b = d.box(L, 300, W, H, "formulate_hypothesis\nLLM proposes one grounded trading hypothesis", kind="process", font=12)
    c = d.box(L, 440, W, H, "select_factors\nLLM picks complementary factors for it", kind="process", font=13)
    end = _se(d, cx, 580, "END")
    d.arrow(start, a)
    d.arrow(a, b)
    d.arrow(b, c)
    d.arrow(c, end, label="hypothesis +\nselected_factor_ids")
    return d


# ===========================================================================
# Diagram 4 — Architect
# ===========================================================================

def build_architect() -> Diagram:
    d = Diagram()
    _graph_header(d, "Architect — LangGraph subgraph (refinement loop)",
                  "Pick a model from the toolbox, fit + backtest it, then approve or revise.")
    cx, W, H = 330, 320, 72
    L = cx - W / 2
    start = _se(d, cx, 96, "START")
    dm = d.box(L, 160, W, H, "design_model\nLLM picks model_type, params, features, sizing", kind="process", font=12)
    fb = d.box(L, 300, W, H, "fit_and_backtest\nfit on IS-train · score IS-valid (modeling MCP)", kind="process", font=12)
    ev = d.box(L, 440, W, H, "evaluate_and_decide\nLLM reviews metrics + overfit gap", kind="process", font=12)
    rv = d.box(720, 300, 300, H, "revise_model\nLLM makes a targeted change", kind="process", font=13)
    end = _se(d, cx, 590, "END")

    d.arrow(start, dm)
    d.arrow(dm, fb)
    d.arrow(fb, ev)
    d.arrow(ev, end, label="approve")
    d.arrow(ev, rv, label="revise")
    d.arrow(rv, fb, label="re-fit\n(new trial)")
    d.text(560, 600, "Every trial is logged → the Statistician uses the trial\n"
                      "count as the # tests for the Deflated Sharpe ratio.",
           font=12, color=GREY)
    return d


# ===========================================================================
# Diagram 5 — Statistician
# ===========================================================================

def build_statistician() -> Diagram:
    d = Diagram()
    _graph_header(d, "Statistician — LangGraph subgraph",
                  "Run the mandatory tests, optionally add risk-driven ones, then accept / reject.")
    cx, W, H = 330, 360, 70
    L = cx - W / 2
    start = _se(d, cx, 86, "START")
    a = d.box(L, 150, W, H, "run_mandatory_tests\ndeflated Sharpe + out-of-sample backtest (MCP)", kind="process", font=12)
    b = d.box(L, 280, W, H, "assess_risk_and_pick_optional\nLLM rates risk, selects optional tests (none yet)", kind="process", font=12)
    c = d.box(L, 410, W, H, "run_optional_tests\nexecute any picked optional tests", kind="process", font=13)
    e = d.box(L, 540, W, H, "final_decision\nLLM verdict · any FAIL → forced reject", kind="process", font=13)
    end = _se(d, cx, 670, "END")
    d.arrow(start, a)
    d.arrow(a, b)
    d.arrow(b, c)
    d.arrow(c, e)
    d.arrow(e, end, label="accept / reject")
    return d


# ===========================================================================
# Diagram 6 — Portfolio Manager
# ===========================================================================

def build_portfolio_manager() -> Diagram:
    d = Diagram()
    _graph_header(d, "Portfolio Manager — LangGraph subgraph",
                  "Same topology in every mode; the monitor/screen/construct nodes branch on PM mode.")
    cx, W, H = 340, 380, 68
    L = cx - W / 2
    start = _se(d, cx, 80, "START")
    a = d.box(L, 140, W, H, "load_universe\nnon-retired strategies · KPIs · corr / cov", kind="process", font=13)
    b = d.box(L, 256, W, H, "monitor_performance\nlive vs IS · keep / flag / retire", kind="process", font=13)
    c = d.box(L, 372, W, H, "screen_strategies\npersonality filters + corr-aware roster pick", kind="process", font=13)
    e = d.box(L, 488, W, H, "construct_portfolio\nrun chosen construction method → weights", kind="process", font=13)
    f = d.box(L, 604, W, H, "finalise\nwrite PortfolioRecord · update pm_status", kind="process", font=13)
    end = _se(d, cx, 720, "END")
    for s, t in [(start, a), (a, b), (b, c), (c, e), (e, f), (f, end)]:
        d.arrow(s, t)
    d.text(720, 268,
           "SELECTOR mode → rule-based\nthresholds from the personality.\n\n"
           "ACTIVE mode → the LLM decides\n(falls back to rules on failure).",
           font=13, color=GREY)
    return d


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    build_system().save(os.path.join(OUT_DIR, "00_system_architecture.excalidraw"))
    build_factor_researcher().save(os.path.join(OUT_DIR, "agent_factor_researcher.excalidraw"))
    build_selector().save(os.path.join(OUT_DIR, "agent_selector.excalidraw"))
    build_architect().save(os.path.join(OUT_DIR, "agent_architect.excalidraw"))
    build_statistician().save(os.path.join(OUT_DIR, "agent_statistician.excalidraw"))
    build_portfolio_manager().save(os.path.join(OUT_DIR, "agent_portfolio_manager.excalidraw"))


if __name__ == "__main__":
    main()
