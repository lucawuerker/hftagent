"""Figure + REPORT.md for the KG nonlinear-combiner study."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/comparisons/kg_nonlinear_combiners"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

# validated categorical palette (dataviz reference, light mode)
COL = {"nn": "#2a78d6", "rf": "#eb6834", "lightgbm": "#1baf7a",
       "ridge_full": "#52514e"}
LABEL = {"nn": "TabM-NN (k=8)", "rf": "Random Forest",
         "lightgbm": "LightGBM (λ₂=5N)",
         "ridge_full": "Ridge, voller Pool 2015 F. (lagias)"}

rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text().splitlines()]
df = pd.DataFrame([r for r in rows if r["model"] in ("nn", "rf", "lightgbm")])
df = df.sort_values(["model", "gen"])

# linear reference: kg_ic_worker cum ridge, run 17 (2,015 factors, full book)
RIDGE_FULL = [0.0564753320271595, 0.1384469980036403, 0.08993501620269165,
              0.10925331279051695, 0.00927447320101307, 0.038883794705574945,
              0.032333623644965004, 0.06837273968924956, 0.06332245726803185,
              0.051085864502277645]

gens = sorted(df["gen"].unique())
starts = {r["gen"]: r["start"] for r in rows}
xlab = [f"{g-10}\n{starts[g][:7]}" for g in gens]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2),
                         gridspec_kw={"width_ratios": [1.9, 1.0]})
ax = axes[0]
ax.axhline(0, color="#d9d8d4", lw=1)
ax.plot(range(10), RIDGE_FULL, color=COL["ridge_full"], lw=1.6, ls="--",
        marker="o", ms=3.5, label=LABEL["ridge_full"])
for m in ("lightgbm", "rf", "nn"):
    ics = df[df.model == m].set_index("gen")["ic"].reindex(gens)
    ax.plot(range(10), ics.values, color=COL[m], lw=2, marker="o", ms=4,
            label=LABEL[m])
    ax.annotate(LABEL[m].split(" (")[0], (9.12, ics.values[-1]),
                color=COL[m], fontsize=8.5, va="center")
ax.annotate("Ridge (voll)", (9.12, RIDGE_FULL[-1]), color=COL["ridge_full"],
            fontsize=8.5, va="center")
ax.set_xticks(range(10), xlab, fontsize=7.5)
ax.set_xlim(-0.4, 11.2)
ax.set_xlabel("Walk-Forward-Block (126 Bars, Refit ab 2021-07-20)", fontsize=9)
ax.set_ylabel("gepoolter per-underlying IC", fontsize=9)
ax.set_title("Per-Block-IC der Kombinierer (Clean-Pool, 1.632 Faktoren)",
             fontsize=10, loc="left")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#eeede9", lw=0.7)
ax.set_axisbelow(True)

ax = axes[1]
summ = (df.groupby("model")["ic"]
        .agg(mean="mean", std="std", n="count")).loc[["nn", "rf", "lightgbm"]]
ridge_mean, ridge_std = float(np.mean(RIDGE_FULL)), float(np.std(RIDGE_FULL, ddof=1))
names = ["nn", "rf", "lightgbm", "ridge_full"]
means = list(summ["mean"]) + [ridge_mean]
ses = list(summ["std"] / np.sqrt(summ["n"])) + [ridge_std / np.sqrt(10)]
ax.bar(range(4), means, color=[COL[n] for n in names], width=0.62, zorder=3)
ax.errorbar(range(4), means, yerr=ses, fmt="none", ecolor="#0b0b0b",
            elinewidth=1.1, capsize=3, zorder=4)
for i, v in enumerate(means):
    ax.annotate(f"{v:.3f}", (i, v + ses[i] + 0.003), ha="center", fontsize=8.5,
                color="#0b0b0b")
ax.set_xticks(range(4), ["TabM-NN", "RF", "LightGBM", "Ridge\n(voll)"],
              fontsize=8.5)
ax.set_ylabel("mittlerer Block-IC ± SE", fontsize=9)
ax.set_title("WF-Statistik (Mittel der 10 Block-ICs)", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#eeede9", lw=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG / "kg_nonlinear_combiners.png", dpi=200)
fig.savefig(FIG / "kg_nonlinear_combiners.pdf")
print("figure saved")

# REPORT.md
per_block = df.pivot(index="gen", columns="model", values="ic").loc[gens]
per_block["ridge_full_2015F"] = RIDGE_FULL
tbl = per_block.round(4).to_string()
meta = {m: [r for r in rows if r["model"] == m] for m in ("nn", "rf", "lightgbm")}
sec = {m: sum(r["seconds"] for r in meta[m]) / 60 for m in meta}
report = f"""# Nichtlineare Kombinierer auf dem KG-Kampagnen-Clean-Pool

**Setup** (Design-Abnahme 2026-08-17): Clean-Pool des kumulativen KG-Buchs
(rho_med < 0.9 auf dem Dev-Fenster; 1.632 von 2.522 Faktoren, 270 nicht
berechenbar/degeneriert), Panel `nasdaq100_2010_wf` (4.165×232), Horizont 6.
Protokoll = die 10 prequentialen 126-Bar-Bloecke ab 2021-07-20; jedes Modell
wird pro Block auf allen Bars strikt davor neu gefittet (expanding) und auf
dem Block per gepooltem per-underlying IC gescort (harness._pooled_ic).
Baeume/NN fitten auf den ersten 90 % der Fit-Bars, die letzten 10 % sind
temporales Tail-Val (LightGBM: Early Stopping auf Val-l2; NN: Early Stopping
auf Val-pooled-IC). Modelle: LightGBM (lambda_l2=5N, lambda_l1=50,
feature_fraction 0.3, num_leaves 63, lr 0.03), Random Forest
(LightGBM rf-Modus: Bagging 0.632, per-Node sqrt(N)-Feature-Sampling,
300 Baeume, num_leaves 2047), TabM-artiges MLP-Ensemble (BatchEnsemble k=8,
512-128-32, Input-Dropout 0.15, Huber-Loss, AdamW, PyTorch/MPS).
Alle Modelle/z-Statistiken/Prediktionen gespeichert unter `models/`/`preds/`.

## Ergebnis (WF-Statistik = Mittel der 10 Block-ICs)

| Modell | Block-Mean | Block-Std | Hit | Fit-Zeit gesamt |
|---|---|---|---|---|
| TabM-NN (k=8) | {summ.loc['nn','mean']:.4f} | {summ.loc['nn','std']:.4f} | 10/10 | {sec['nn']:.0f} min |
| Random Forest | {summ.loc['rf','mean']:.4f} | {summ.loc['rf','std']:.4f} | 10/10 | {sec['rf']:.0f} min |
| LightGBM | {summ.loc['lightgbm','mean']:.4f} | {summ.loc['lightgbm','std']:.4f} | 10/10 | {sec['lightgbm']:.0f} min |
| Ridge (voller Pool, 2.015 F., lagias run 17) | {ridge_mean:.4f} | {ridge_std:.4f} | 10/10 | — |

## Per-Block-ICs

{tbl}

## Lesart

* **TabM-NN und RF schlagen das lineare Plateau** (~0.066): NN {summ.loc['nn','mean']:.3f}
  und RF {summ.loc['rf','mean']:.3f} vs. Ridge {ridge_mean:.3f} — beide 10/10 positiv.
  Das NN ist dabei deutlich blockstabiler (Std {summ.loc['nn','std']:.3f} vs. RF
  {summ.loc['rf','std']:.3f}); der RF-Mean haengt stark an den Ausreisser-Bloecken
  2 und 9.
* **LightGBM bleibt trotz N-skalierter Regulierung auf Ridge-Niveau**
  ({summ.loc['lightgbm','mean']:.3f}) — konsistent mit der Combiner-Studie vom
  2026-08-05 (GBM auf dem WF-Panel nie ueber Ridge).
* Vergleichshinweis: Die Ridge-Referenz laeuft auf dem VOLLEN (unbereinigten)
  kumulativen Buch der Kampagne (run 17, 2.015 F.); die Clean-Pool-Ridge
  (Block 1: 0.046 vs. voll 0.056) liegt eher etwas darunter — der
  Nichtlinearitaets-Vorsprung von NN/RF ist also nicht durch die Pool-Wahl
  geschenkt.
* Frueh-Stopp-Findung: IC-basiertes Early Stopping fuer LightGBM ist
  pathologisch (stoppt auf der verrauschten Val-IC-Kurve bei Iteration 8 mit
  einem Quasi-Null-Modell, Block-IC -0.006); Val-l2-Stopping (best_iteration
  ~800) ist die korrekte Konvention. Fuers NN (epochenweises Val-IC-Stopping)
  tritt das Problem nicht auf.

Figur: `figures/kg_nonlinear_combiners.png|pdf`. Rohdaten: `results.jsonl`,
`summary.csv`; Modelle unter `models/` (LightGBM-Booster .txt, NN state_dicts
.pt + z-Statistiken .npz je Block), Block-Prediktionen unter `preds/`.
"""
(OUT / "REPORT.md").write_text(report)
print("report written")
