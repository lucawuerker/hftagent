# Faktor-Level-Analyse: L4_terra_s0 (GPT-5.6) vs. L2_opus5_s0 (Opus 5) vs. 101 formulaic alphas

*2026-08-04 · Skripte: `scripts/analyze_l4_factor_books.py` → CSV/JSON hier, `scripts/plot_l4_factor_books.py` → `figures/`.*

**Achtung Konfundierung:** Ein Opus-5-*L4*-Lauf existiert nicht (Opus-Ladder-Budget nach L2
erschöpft, $514/$600; L3–L7 nie gestartet). Verglichen: **L4_terra_s0** (GraphRAG, 8 Gruppen,
44 Faktoren, $86.70) vs. **L2_opus5_s0** (retrieval none, 1×4 Demes, 18 Faktoren, $452.13) —
Modell und Ladder-Stufe unterscheiden sich gleichzeitig.

## Setup
- Panel `quant.config.nasdaq100_2010_forward.yaml`: 4.165 × 232, 2010-01-04 → 2026-07-27, h=6.
- Fenster (Split-Konvention der Läufe, 60/20/20 auf dem Research-Panel bis 2024-07-27):
  **DEV** 2010-01-04→2021-08-25 (2.932 Bars, Suchfenster), **TEST** 2021-08-26→2024-07-26
  (733, nie enthüllt), **FORWARD** 2024-07-29→2026-07-27 (500, außerhalb des Research-Panels).
- IC = pooled per-underlying Pearson (Harness-Statistik); cross-sectional Daily-IC als Zweitspalte.
- 4/44 Terra-Faktoren degeneriert auf diesem Panel (kein Signal), u. a. `consensus_revision_whipsaw_correction`.

## Kombinierte Bücher (LightGBM, Fit nur auf DEV) — pooled IC
| Buch | DEV (in-sample) | TEST | FORWARD | cs-IC TEST | cs-IC FWD |
|---|---:|---:|---:|---:|---:|
| 101 alphas allein | 0.210 | 0.026 | 0.024 | −0.001 | −0.008 |
| Terra-Buch (44) | 0.213 | 0.024 | 0.033 | 0.009 | 0.002 |
| Terra + 101 (145) | 0.236 | **0.031** | **0.041** | −0.004 | −0.005 |
| Opus-Buch (18) | 0.158 | **0.045** | 0.027 | **0.023** | **0.012** |
| Opus + 101 (119) | 0.216 | 0.039 | 0.027 | 0.002 | −0.006 |
| Terra + Opus (62) | 0.221 | 0.041 | 0.039 | – | – |
| Terra + Opus + 101 (163) | 0.244 | 0.039 | 0.040 | – | – |

- **Opus**: beste OOS-Generalisierung (TEST 0.045 = 28 % Retention vs. Terra 11 %, Zoo 12 %);
  einziges Buch mit positivem cross-sectional OOS-IC.
- **Terra**: allein Zoo-Niveau, aber stark *marginal*: hebt FORWARD des Zoos 0.024 → 0.041 (+73 %) —
  konsistent mit dem LOCO-Marginalwert-Selektionsziel gegen das fixe 101er-Referenzbuch.

## Per-Faktor-ICs
| Buch | n | ø\|IC\| DEV | ø\|IC\| TEST | ø\|IC\| FWD | Sign-Retention DEV→TEST | ρ(DEV,TEST) |
|---|---:|---:|---:|---:|---:|---:|
| Terra | 40 | 0.0082 | 0.0106 | 0.0133 | 80 % | 0.55 |
| Opus | 18 | 0.0135 | 0.0176 | 0.0152 | 83 % | 0.65 |
| Zoo | 101 | 0.0143 | 0.0110 | 0.0126 | 75 % | 0.66 |

Keine mittlere Degradation DEV→TEST bei den LLM-Büchern (Selektion lief auf Marginalwert, nicht
Standalone-IC). Beste Einzelfaktoren nach |TEST-IC|: Opus `comparable_basket_repair_rate_activity`
**0.062** (DEV nur 0.024), `rice_consistency_wedge_gated_crossing` 0.041, `run_surprisal_vs_flip`
0.037; Terra `nonlinear_downside_coskew_compensation` 0.034, `peer_hawkes_contagion_underreaction`
0.029 (DEV 0.036, FWD 0.037 — stabilster Terra-Faktor); Zoo-Bester `alpha_024` 0.052.

## Korrelationen / Diversität (DEV, cs-z-Signale)
- Intern: Terra ø|ρ| 0.093 (eff. N 19.7/41), Opus 0.094 (**12.7/18** = 71 % Effizienz),
  Zoo 0.120 (25.8/101 = 26 %).
- Kreuzblöcke schwach: **Terra×Opus ø|ρ| 0.056** (max 0.62) — fast orthogonale Bücher;
  Terra×Zoo 0.076, Opus×Zoo 0.093.
- Pool-Zugewinn: eff. N 101 alphas 25.8 → +Terra 32.6, +Opus 28.7, alle 160 → 35.0.
- Neuheit: Median max|ρ| zum Zoo 0.28 (Terra) / 0.29 (Opus); zoo-ähnlichste Faktoren docken
  an `alpha_033`/`alpha_038` an (max 0.78). Terras Redundanz ist hausgemacht
  (Liquiditäts-/Downside-Cluster bis |ρ| 0.85), nicht Zoo-Nachbau.
- Kategorien: Opus 12/18 microstructure; Terra breiter (20 micro, 9 sentiment, 5 vola, 5 other, 2 mr, 3 statarb).

## Regime-Befund (Abb. 8)
Der rollierende cs-IC aller drei Bücher — auch des *nicht gefitteten* Zoos — kollabiert
gleichzeitig ab 2021: das cross-sectionale Faktor-Regime im Nasdaq-100 hat sich geändert
(Mega-Cap-Konzentration), der DEV→TEST-Kollaps ist nicht nur Overfitting. Opus hält sich OOS am
häufigsten über Null; der Zoo dreht 2025–26 klar negativ.

## Fazit
Qualität pro Faktor: **Opus** (stärker, stabiler, cs-tauglich, aber 5× teurer, nur 18 Faktoren).
Breite/Marginalwert: **Terra** (mehr Neuheit, größter Diversitätszugewinn, bester FORWARD nur in
Kombination mit dem Zoo). Ensemble beider Bücher wäre die natürliche Fortsetzung. Modell- vs.
Konfigurationseffekt bleibt ohne den nie gelaufenen L4_opus5-Arm untrennbar.

Figuren: `figures/fig1_ic_scatter.png` … `fig8_rolling_ic.png`. Daten: `per_factor_ic.csv`,
`combined_book_ic.csv`, `signal_corr_dev.csv`, `nearest_zoo_corr.csv`, `summary.json`,
`combined_cs_ic_series.csv`.
