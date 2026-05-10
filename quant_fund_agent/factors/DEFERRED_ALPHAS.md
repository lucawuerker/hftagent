# Deferred Alphas — Industry / Market-Cap Dependencies

These alphas require industry classification data (`IndNeutralize`) or
market capitalisation (`cap`) that is not yet wired into the data pipeline.
Come back to implement them once those fields are available.

## Already implemented (with graceful fallback)

| Alpha | Dependency | File | Fallback behaviour |
|-------|-----------|------|--------------------|
| #48 | `data["subindustry"]` | `statistical_arbitrage/alpha_048.py` | Skips neutralisation, logs warning |
| #56 | `data["cap"]` | `momentum/alpha_056.py` | Uses `volume` as proxy, logs warning |
| #58 | `data["sector"]` | `statistical_arbitrage/alpha_058.py` | Skips neutralisation, logs warning |
| #59 | `data["industry"]` | `statistical_arbitrage/alpha_059.py` | Skips neutralisation, logs warning |

## Not yet implemented

| Alpha | Dependencies | Original formula |
|-------|-------------|-----------------|
| #63 | `IndNeutralize(close, industry)` | `rank(decay_linear(delta(IndNeutralize(close, industry), 2), 8)) - rank(decay_linear(corr(vwap*0.318+open*0.682, sum(adv180,37), 14), 12))` |
| #67 | `IndNeutralize(vwap, sector)`, `IndNeutralize(adv20, subindustry)` | `rank(high - ts_min(high, 2))^rank(corr(IndNeutralize(vwap, sector), IndNeutralize(adv20, subindustry), 6))` |
| #69 | `IndNeutralize(vwap, industry)` | `rank(ts_max(delta(IndNeutralize(vwap, industry), 3), 5))^Ts_Rank(corr(close*0.49+vwap*0.51, adv20, 5), 9)` |
| #70 | `IndNeutralize(close, industry)` | `rank(delta(vwap, 1))^Ts_Rank(corr(IndNeutralize(close, industry), adv50, 18), 18)` |
| #76 | `IndNeutralize(low, sector)` | `max(rank(decay_linear(delta(vwap,1),11)), Ts_Rank(decay_linear(Ts_Rank(corr(IndNeutralize(low,sector),adv81,8),20),18),19))` |
| #79 | `IndNeutralize(close*0.607+open*0.393, sector)` | `rank(delta(IndNeutralize(close*0.607+open*0.393, sector),1)) < rank(corr(Ts_Rank(vwap,4),Ts_Rank(adv150,9),15))` |
| #80 | `IndNeutralize(open*0.868+high*0.132, industry)` | `rank(sign(delta(IndNeutralize(open*0.868+high*0.132, industry),4)))^Ts_Rank(corr(high,adv10,5),6)` |
| #82 | `IndNeutralize(volume, sector)` | `min(rank(decay_linear(delta(open,1),15)), Ts_Rank(decay_linear(corr(IndNeutralize(volume,sector), open, 17), 7), 13))` |
| #87 | `IndNeutralize(adv81, industry)` | `max(rank(decay_linear(delta(close*0.37+vwap*0.63, 1), 3)), Ts_Rank(decay_linear(abs(corr(IndNeutralize(adv81,industry), close, 13)), 5), 14))` |
| #89 | `IndNeutralize(vwap, industry)` | `Ts_Rank(decay_linear(corr(low, adv10, 7), 6), 4) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, industry), 3), 10), 15)` |
| #90 | `IndNeutralize(adv40, subindustry)` | `rank(close - ts_max(close, 5))^Ts_Rank(corr(IndNeutralize(adv40, subindustry), low, 5), 3)` |
| #91 | `IndNeutralize(close, industry)` | `Ts_Rank(decay_linear(decay_linear(corr(IndNeutralize(close,industry), volume, 10), 16), 4), 5) - rank(decay_linear(corr(vwap, adv30, 4), 3))` |
| #93 | `IndNeutralize(vwap, industry)` | `Ts_Rank(decay_linear(corr(IndNeutralize(vwap,industry), adv81, 17), 20), 8) / rank(decay_linear(delta(close*0.524+vwap*0.476, 3), 16))` |
| #97 | `IndNeutralize(low*0.721+vwap*0.279, industry)` | `rank(decay_linear(delta(IndNeutralize(low*0.721+vwap*0.279,industry), 4), 20)) - Ts_Rank(decay_linear(Ts_Rank(corr(Ts_Rank(low,8),Ts_Rank(adv60,17),5),19),15),7)` |
| #100 | `IndNeutralize(…, subindustry)` ×3 | `0 - 1.5*scale(indneutralize(indneutralize(rank(CLV*volume), subindustry), subindustry)) - scale(indneutralize(corr(close,rank(adv20),5)-rank(ts_argmin(close,30)), subindustry)) * (volume/adv20)` |
