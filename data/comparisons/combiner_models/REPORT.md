# Combiner-Modell-Studie (2026-08-05, rev. Blockmetrik)

Frage: Der IS→OOS-Kollaps des kombinierten Buch-IC — Modell-Fit oder Faktoren? Und:
Ridge/Lasso vs. LightGBM vs. LightGBM mit faktorzahl-abhängigem L2 (`reg_lambda`≈N…10N).

Browser-Report (Artifact): https://claude.ai/code/artifact/ca5040c4-c95f-45e0-bc46-ff0378fb470b
Skripte: `scripts/analyze_combiner_models.py --panel forward|wf`,
`scripts/analyze_wf_block_metric.py` (Blockmetrik), `scripts/plot_combiner_models.py`.
Daten: `combiner_{forward,wf}.jsonl`, `combined_wf_blocks.csv`,
`per_factor_{forward,wf}.csv`, `per_factor_block{s,metric}_wf.csv`, `figures/`.

## Metrik-Konvention (User-Entscheid 2026-08-05)
Die WF-OOS-Kennzahl ist der **Mittelwert der 10 per-Block-ICs** über die prequentiellen
~6-Monats-Blöcke 2021→2026 (ein IC pro Walk-forward-Step; Konvention des
Prequential-Records). In-sample dieselbe Metrik: Fit-Fenster (2010→2021-07) in 23
126-Bar-Blöcke zerlegt, ICs gemittelt. Blockweise ICs sind systematisch HÖHER als
Ganzfenster-gepoolte (lokalere Korrelation) — beide Seiten müssen gleich gemessen werden.
Forward-Panel (Terra/Opus/Zoo) bleibt Fenster-basiert (TEST/FORWARD).

## Kernergebnisse (WF-Panel, ø IC je Block: IS → WF, Hit, Retention)
| Buch | Ridge | Lasso | LightGBM | GBM λ₂=N | GBM λ₂=10N |
|---|---|---|---|---|---|
| L2WF (19) | 0.066→**0.053**, 10/10, 81 % | 0.064→**0.055**, 10/10, 86 % | 0.176→0.015, 5/10, 9 % | 0.016, 6/10 | 0.020, 6/10 |
| L4WF (57) | 0.089→**0.064**, 10/10, 72 % | 0.083→0.058, 10/10, 69 % | 0.201→0.065, 8/10, 32 % (σ 0.051 vs. Ridge 0.023) | 0.065, 8/10 | 0.064, 8/10 |
| 101 alphas | 0.106→**0.062**, 10/10 | 0.071→0.038 | 0.185→0.055, 10/10 | 0.059 | 0.048 |
| L2WF+L4WF (76) | 0.094→**0.067**, 10/10 | 0.081→0.058 | 0.231→0.045, 7/10 | 0.055 | 0.058 |

1. **Kollaps = Modell, nicht Faktoren** (auch auf dem Server): Einzelfaktoren halten
   (ø Block-|IC| L2WF 105 %, L4WF 96 %, Sign-Ret. 74–82 %); GBM-Kombi 9–32 % Retention.
2. **Ridge = bester WF-Combiner überall**: 10/10 positive Blöcke bei jedem Buch,
   ø 0.053–0.067, halbe Block-Streuung vs. GBM. Schlägt sogar den Prequential-Record der
   Läufe (L4WF 0.064 vs. 0.035; L2WF 0.053 vs. 0.020) — Caveat: der Record handelte das
   *damalige*, unfertige Archiv je Block (Teil des Vorsprungs ist Buch-Reife).
3. **λ₂≈N…10N verbessert GBM** dort, wo es schwach ist (Union 0.045→0.058), erreicht
   Ridge auf dem WF-Panel aber nicht. Forward-Panel (Fenster-Metrik, große Pools):
   GBM+λ₂=N bleibt am besten (163 F. TEST 0.045 vs. Ridge 0.020).
4. **Lasso-Caveat**: meist nur 1–8 Faktoren genutzt (Opus-Ausnahme 16/18);
   **RidgeCV wählt IMMER α=10⁴ = Rand des Katalog-Gitters** → Gitter auf 10⁵–10⁶ erweitern.
5. Forward-Panel (Terra/Opus): kleine Bücher linear (Opus FWD Ridge 0.037 vs. GBM 0.027;
   Terra Lasso FWD 0.042), 163-F.-Pool GBM+λ₂.

## Empfehlungen
- WF-/Deployment-Kontext: **Ridge als Default-Combiner** (Lasso nur mit
  Mindest-Faktorzahl-Guard); häufiges GBM-Refitten bringt keinen Mehrwert.
- Große cross-sektionale Pools: LightGBM mit reg_lambda≈N.
- In-Sample-Kombi-ICs nie als Qualitätsmaß zitieren.
- Offen: Ridge-Gitter 10⁵–10⁶; Ensemble Ridge+GBM-λ₂; num_leaves/colsample-Dämpfung.

Caveat: WF-Bücher-Faktorselektion sah Post-2021 prequentiell-dann-adaptiv → per-Faktor-OOS
leicht optimistisch; die Combiner-Fits selbst nutzen strikt Pre-2021.
