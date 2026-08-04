# Factors → Strategy: der deterministische „Strategy Compiler"

*Deep-Research-Ergebnis 2026-08-03. Auftrag: Die Faktoren (und das evolutionäre
Research) bleiben das Produkt und der Ort des Research-Aufwands; der Schritt
Faktoren → Strategie soll leichtgewichtig, deterministisch und zuverlässig
sein („immer gute Strategien"), Ziel Net-Sharpe > 1. Die bisherige
Selector→Architect→Statistician-Route wird hierfür bewusst beiseitegelegt.*

---

## 1. Diagnose: Das Signal ist da — die Konstruktion vernichtet es

Interne Evidenz (L4_terra_s0 `book_backtest/` + `prequential_deployment/`,
L2WF_terra_s0 `prequential.jsonl`):

| Befund | Zahl |
|---|---|
| Ehrlicher 5-J-Walk-forward-IC des L2WF-Buchs (prequential, nie gesehene Blöcke) | **+0.020 Mittel, 6/10 Blöcke positiv** |
| L4 forward IC (2024→2026, untouched) | **+0.040** |
| L4 forward Sharpe **gross** | **0.79** |
| L4 forward Sharpe **net** | **0.47** |
| Mittlerer Tagesumsatz der Konstruktionen | **13–23 %/Tag** |
| Alle Konstruktionen im Construction-Lab, 2016–2024 | **Sharpe ≤ 0** trotz positiver Block-ICs |
| Combiner-Race: lightgbm VAL-Sharpe | **1.04** (aber Race-PBO 0.84 → Modellwahl instabil) |

Der Verlustpfad ist quantifizierbar:

1. **Kostenfraß durch Turnover.** 18,6 %/Tag × 252 × 5 bps ≈ **2,3 %/Jahr**
   Drag; bei ~7 % Vol sind das **~0,33 Sharpe** — exakt die Lücke gross→net
   (0.79→0.47). Ursache: das Rohsignal wird täglich neu gerankt, obwohl der
   Forecast-Horizont 6 Tage ist (Alpha-Decay ≪ Rebalance-Frequenz).
2. **Transfer-Koeffizient-Kollaps durch Top-10-Konzentration.** Top-N-Bücher
   werfen die Breite weg (Fundamental Law: IR ≈ TC · IC · √Breadth). Die
   `full_book`-Konstruktion hatte gleiche Forward-Sharpe bei 2/3 der Vol und
   niedrigerem Umsatz — Konzentration hat nie geholfen.
3. **Kein Risiko-Management auf Portfolioebene.** Ann-Vol schwankt frei
   (4–10 %), kein Vol-Targeting, keine Beta-/Sektor-Neutralisierung → das Buch
   trägt unbeabsichtigte Markt-/Sektor-Wetten, die die IC-Wette verwässern.
4. **Instabile Modellwahl.** Der Race-Gewinner wird auf einem wiederverwendeten
   VAL-Fenster gekürt (PBO 0.84) — ein Ensemble ist robuster als ein Pick.

Externe Bestätigung: Signal-Glättung kann Umsatz um ~80 % senken und Net-Sharpe
von negativ auf positiv drehen (Turnover-adjusted IR, Zhang/Wang/Cao 2021;
FactSet Autokorrelations-Analysen); Vol-Targeting erhöht die Sharpe von
Equity-/Long-short-Faktor-Portfolios (Moreira & Muir 2017; Man Group 2019);
Standard-Praxis ist Score → Winsorize ±3 → risikoskalierte Gewichte statt
Top-N (AQR „Building a Better Long-Short Equity Portfolio").

## 2. Design: ein deterministischer Compiler, keine Agenten-Route

**Ein** fixes, LLM-freies Rezept `Faktorbuch → Portfolio`, versioniert wie
Code. Das Research-Produkt bleibt das Buch; der Compiler ist Commodity.

```
Faktorbuch (Archiv eines Arms / kuratierter Katalog)
  │  A. SIGNAL
  │  Ensemble-Kombination (ridge + lightgbm, 50/50) statt Race-Gewinner,
  │  walk-forward refit alle 126 Bars (= WF-Blöcke; identisch zur
  │  prequential_deployment-Kadenz — Forschung und Produkt driften nicht)
  ▼
  │  B. POSITIONEN  (der eigentliche Fix)
  │  1. cross-sektionaler z-Score, Winsorize ±3
  │  2. EWMA-Glättung des Scores, Halflife = Horizont (6 Bars)
  │  3. Gewichte ∝ z / EWMA-Vol(Name)  → volle Breite, KEIN Top-N
  │  4. Beta-Neutralisierung (Rolling-Beta vs Universum); Sektor-Demean
  │     (FMP `sector` liegt im Panel)
  │  5. No-Trade-Band: handle nur |Δw| > 10 % der Zielposition
  ▼
  │  C. RISIKO
  │  Vol-Targeting auf Zielvol (EWMA realisierte Portfolio-Vol),
  │  Leverage-Cap, Drawdown-Deleveraging optional
  ▼
  │  D. PERSONA-LAYER  (reine Parametrisierung, kein neues Research)
  │  {vol_target, net_exposure (market-neutral | 30 % net-long | long-only),
  │   leverage_cap, turnover_budget, universe_filter (Liquidität)}
  ▼
Tägliche Zielgewichte + Trades
```

Warum das die Produkt-These stützt: **ein** Research-Motor (Evolution →
Faktorbuch), **viele** verkaufbare Profile — Personas unterscheiden sich nur in
Risiko-Parametern, nicht im Alpha. Jede Persona ist damit automatisch „gut",
wenn das Buch gut ist; es gibt keinen fragilen Bastel-Schritt mehr pro
Strategie.

## 3. Warum Net-Sharpe > 1 erreichbar ist (Arithmetik, keine Hoffnung)

IR ≈ TC · IC · √(unabhängige Wetten/Jahr). Mit IC ≈ 0.03–0.04 pro 6-Tage-
Horizont, ~100 Namen (effektiv ~30–40 unabhängige Querschnitts-Wetten), ~42
unabhängige Perioden/Jahr:

- Status quo: TC ≈ 0.4 (Top-10, keine Glättung) → gross ≈ 0.8, − 0.33 Kosten
  → **0.47** ✓ (reproduziert die Messung)
- Compiler: TC ≈ 0.7–0.8 (volle Breite, Risiko-Skalierung, Neutralisierung),
  Kostendrag ≤ 0.10 (Glättung + Band ≈ −70 % Umsatz), Vol-Targeting +10–15 %
  → **net ≈ 0.9–1.3**. Sharpe 1 ist das Mittelfeld dieser Spanne, kein
  Streckziel — vorausgesetzt der IC hält (was der prequential Track ehrlich
  misst).

## 4. Validierungsprotokoll (ehrlich, ohne neuen Overfit)

1. **`strategy_lab`**: kleines Grid (Glättungs-Halflife × Band ×
   Neutralisierung × Vol-Target) — aber bewertet NUR auf dem gestitchten
   prequential Walk-forward-Track (Buch@Gen g handelt Block g+1; Daten, die
   keine Selektion je sah). Bericht mit DSR und CSCV-PBO über das Grid, damit
   die Konstruktionswahl selbst nicht zum neuen Overfitting wird.
2. Eine **fixe Default-Parametrisierung** (Halflife 6, Band 10 %, beta+sektor-
   neutral, Vol 10 %) wird VOR dem Grid festgeschrieben und immer mit
   ausgewiesen (pre-registered baseline).
3. Gleiches Protokoll für jeden Ladder-Arm → die Ablation bekommt eine
   Strategie-Zeile (Net-Sharpe walk-forward) zusätzlich zur IC-Zeile.

## 5. Implementierung (klein, LLM-frei)

- `quant_fund_agent/backtesting/strategy_compiler.py` — Stufen A–D als pure
  Funktionen über dem Panel; wiederverwendet `positions.py`-Primitiven und die
  prequential-Snapshot-Logik aus `scripts/prequential_deployment.py`.
- `scripts/compile_strategy.py --prerun <arm> --persona <key>` → Gewichte,
  Net-Return-Serie, Report (+ `strategy_lab`-Modus).
- `personas.yaml` — die Persona-Parameter (ersetzt LLM-Persona-Bastelei).
- Aufwand: ~1 Tag; null LLM-Kosten; vollständig backtestbar auf den bereits
  bezahlten Runs.

Quellen: [Turnover-Adjusted Information Ratio (Zhang, Wang, Cao 2021)](https://arxiv.org/pdf/2105.10306) ·
[FactSet: Portfolio Turnover & Autocorrelation](https://insight.factset.com/navigating-portfolio-turnover-with-autocorrelation-insights) ·
[Man Group: The Impact of Volatility Targeting](https://www.man.com/insights/the-impact-of-volatility-targeting) ·
[Conditional Volatility Targeting (FAJ 2020)](https://www.tandfonline.com/doi/full/10.1080/0015198X.2020.1790853) ·
[AQR: Building a Better Long-Short Equity Portfolio](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/AQR-Building-a-Better-Long-Short-Equity-Portfolio.pdf) ·
[Fundamental Law of Active Management (CFA Institute)](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/analysis-active-portfolio-management) ·
[Information Horizon, Portfolio Turnover, and Optimal Alpha Models (Sneddon)](https://www.researchgate.net/publication/247906091_Information_Horizon_Portfolio_Turnover_and_Optimal_Alpha_Models)
