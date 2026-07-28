# Deferred Alphas — Industry / Market-Cap Dependencies

These alphas required industry classification data (`IndNeutralize`) or
market capitalisation (`cap`) that was not originally wired into the data
pipeline.  As of 2026-07-27 the FMP Premium archive serves the canonical
categorical label fields `sector` / `industry` / `subindustry` (wide
object-dtype frames) and a daily `marketCap` series, so **all 101
formulaic alphas are now implemented**.

## Label fallback convention

Every `IndNeutralize` alpha resolves its grouping through
`factors/_labels.py::neutralize`: the paper's `IndClass.subindustry`
maps to `data["subindustry"]` but **falls back to `industry`, then
`sector`** when the finer label is entirely missing (logged once per
alpha); `industry` falls back to `sector`.  With no label field at all
the neutralisation step is skipped with a warning — the historical
Alpha#48/#58/#59 behaviour.  Fractional windows from the paper are
rounded to the nearest integer (house convention, cf. Alpha#58; the
paper's own stated rule is floor).

## Implemented earlier (with graceful fallback)

| Alpha | Dependency | File | Behaviour |
|-------|-----------|------|-----------|
| #48 | `data["subindustry"]` | `statistical_arbitrage/alpha_048.py` | Skips neutralisation, logs warning |
| #56 | `data["marketCap"]` (canonical) | `momentum/alpha_056.py` | Prefers `marketCap`, then legacy `cap`, else volume proxy |
| #58 | `data["sector"]` | `statistical_arbitrage/alpha_058.py` | Skips neutralisation, logs warning |
| #59 | `data["industry"]` | `statistical_arbitrage/alpha_059.py` | Skips neutralisation, logs warning |

## Implemented (2026-07-27, FMP fundamentals)

All formulas were re-verified against the source paper (Kakushadze,
"101 Formulaic Alphas", Appendix A) before implementation.  Corrections
vs the previous version of this table: the paper's outer `* -1` on
#63, #67, #69, #70, #76, #80, #82, #87, #90, #91 and #97 had been
dropped here and is restored in the implementations; #87's delta window
is 2 (paper 1.91233, this table previously said 1), #97's delta window
is 3 and its decay window 16 (previously 4 and 15), and #76's decay
windows are 12 and 17 (previously 11 and 18).

| Alpha | Neutralisation | File |
|-------|---------------|------|
| #63 | `IndNeutralize(close, industry)` | `statistical_arbitrage/alpha_063.py` |
| #67 | `IndNeutralize(vwap, sector)`, `IndNeutralize(adv20, subindustry)` | `statistical_arbitrage/alpha_067.py` |
| #69 | `IndNeutralize(vwap, industry)` | `momentum/alpha_069.py` |
| #70 | `IndNeutralize(close, industry)` | `momentum/alpha_070.py` |
| #76 | `IndNeutralize(low, sector)` | `statistical_arbitrage/alpha_076.py` |
| #79 | `IndNeutralize(close*0.607+open*0.393, sector)` | `momentum/alpha_079.py` |
| #80 | `IndNeutralize(open*0.868+high*0.132, industry)` | `momentum/alpha_080.py` |
| #82 | `IndNeutralize(volume, sector)` | `statistical_arbitrage/alpha_082.py` |
| #87 | `IndNeutralize(adv81, industry)` | `statistical_arbitrage/alpha_087.py` |
| #89 | `IndNeutralize(vwap, industry)` | `statistical_arbitrage/alpha_089.py` |
| #90 | `IndNeutralize(adv40, subindustry)` | `mean_reversion/alpha_090.py` |
| #91 | `IndNeutralize(close, industry)` | `statistical_arbitrage/alpha_091.py` |
| #93 | `IndNeutralize(vwap, industry)` | `statistical_arbitrage/alpha_093.py` |
| #97 | `IndNeutralize(low*0.721+vwap*0.279, industry)` | `statistical_arbitrage/alpha_097.py` |
| #100 | `IndNeutralize(…, subindustry)` ×3 | `microstructure/alpha_100.py` |

Tests: `tests/test_formulaic_alphas_fundamental.py`.
