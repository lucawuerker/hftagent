# FMP point-in-time membership build

- built: 2026-07-27
- since: 2004-01-01
- source: FMP `historical-*-constituent` (backward walk) + current constituent list

## sp500

- spells: **991** over **955** distinct tickers
- change log earliest event: **1957-03-03**
- still-active spells: 503
- left-censored names (spell start is a floor, not a fact): 289
- month-end constituent count: min 496, max 505, mean 501.9 (band (475, 515))

Audit: all invariants passed.

## nasdaq100

- spells: **345** over **295** distinct tickers
- change log earliest event: **1985-01-30**
- still-active spells: 102
- left-censored names (spell start is a floor, not a fact): 90
- month-end constituent count: min 101, max 111, mean 103.3 (band (95, 115))

Audit: all invariants passed.

## Reconciliation vs the free public reconstruction (sp500_public)

Per-year month-end Jaccard. A thin change log shows as a low early-year score; the free reconstruction remains the fallback for those years.

| year | months | mean Jaccard | min Jaccard |
|---|---|---|---|
| 2004 | 12 | 0.853 | 0.849 |
| 2005 | 12 | 0.853 | 0.849 |
| 2006 | 12 | 0.863 | 0.855 |
| 2007 | 12 | 0.862 | 0.851 |
| 2008 | 12 | 0.866 | 0.855 |
| 2009 | 12 | 0.872 | 0.865 |
| 2010 | 12 | 0.882 | 0.878 |
| 2011 | 12 | 0.880 | 0.878 |
| 2012 | 12 | 0.878 | 0.871 |
| 2013 | 12 | 0.880 | 0.874 |
| 2014 | 12 | 0.877 | 0.872 |
| 2015 | 12 | 0.884 | 0.875 |
| 2016 | 12 | 0.898 | 0.886 |
| 2017 | 12 | 0.902 | 0.900 |
| 2018 | 12 | 0.900 | 0.897 |
| 2019 | 12 | 0.912 | 0.902 |
| 2020 | 12 | 0.934 | 0.924 |
| 2021 | 12 | 0.944 | 0.942 |
| 2022 | 12 | 0.952 | 0.944 |
| 2023 | 12 | 0.963 | 0.959 |
| 2024 | 12 | 0.981 | 0.971 |
| 2025 | 12 | 0.984 | 0.982 |
| 2026 | 6 | 0.989 | 0.980 |

Overall mean Jaccard: **0.9029** over 270 month-ends.

## Vendor coverage — tickers with no usable price history

Of **1107** tickers in the downloaded universe, **168** returned no bars under any candidate symbol and **110** more were priced for less than half of their membership window.

These are a **vendor limit, not a resolution bug**: FMP carries no security for them at all (`search-symbol` and `search-name` both come back empty for Bear Stearns, AT&T Wireless, Countrywide, Cephalon, Andrew, BEA Systems). Where a modern ticker exists for a post-bankruptcy successor (`ABKFQ` → `AMBC`) it is a **different security** and must not be spliced onto the old series.

Index column: `S` = FMP S&P 500, `N` = Nasdaq-100, `P` = free reconstruction (`sp500_public`).

### Unresolved, by the era they left the index

**left 2004–2009 — 93 names**

| ticker | company | membership window | index |
|---|---|---|---|
| `ABKFQ` | — | 2004-01-01 → 2008-06-11 | P |
| `ABS` | Albertson's Inc. | 2004-01-01 → 2006-06-02 | PS |
| `ACAS` | American Capital Ltd | 2007-07-06 → 2009-03-04 | PS |
| `AMLN` | Amylin Pharmaceutical Corp | 2006-01-09 → 2008-12-22 | N |
| `ANDW` | Andrew Corp. | 2004-01-01 → 2006-10-02 | PS |
| `APCC` | American Power Conversion Corp | 2004-01-01 → 2007-02-15 | NPS |
| `ASN` | — | 2004-12-17 → 2007-10-08 | PS |
| `ATYT` | ATI Technologies Inc. | 2004-01-01 → 2006-10-24 | N |
| `AV` | Avaya Inc. | 2004-01-01 → 2007-10-26 | PS |
| `AW` | Allied Waste Industries Inc. | 2004-01-01 → 2008-12-05 | PS |
| `AWE` | AT&T Wireless | 2004-01-01 → 2004-10-27 | PS |
| `BEAS` | BEA Systems Inc | 2004-01-01 → 2008-04-30 | N |
| `BLS` | BellSouth Corp. | 2004-01-01 → 2007-01-03 | PS |
| `BMET` | Biomet Inc. | 2004-01-01 → 2007-07-12 | NPS |
| `BRL` | Barr Pharmaceuticals Inc. | 2006-02-24 → 2008-12-23 | PS |
| `BSC` | Bear Stearns Co | 2004-01-01 → 2008-06-02 | PS |
| `CBSS` | Compass Bancshares Inc | 2004-12-17 → 2007-09-07 | PS |
| `CCTYQ` | — | 2004-01-01 → 2008-03-31 | P |
| `CDWC` | CDW Corp | 2004-01-01 → 2007-10-08 | N |
| `CFC` | Countrywide Financial Corp. | 2004-01-01 → 2008-07-01 | PS |
| `CIN` | Cinergy Corp. | 2004-01-01 → 2006-04-03 | PS |
| `CITGQ` | — | 2004-10-27 → 2009-07-27 | P |
| `CKFR` | Checkfree Corp | 2005-12-19 → 2007-12-04 | N |
| `CMVT` | Comverse Technology Inc | 2004-01-01 → 2007-02-01 | NPS |
| `CMX` | Caremark Rx | 2004-03-24 → 2007-03-23 | PS |
| `CPNLQ` | — | 2004-01-01 → 2005-12-02 | P |
| `DALRQ` | — | 2004-01-01 → 2005-08-19 | P |
| `DCNAQ` | — | 2004-01-01 → 2006-03-03 | P |
| `DJ` | Dow Jones & Co. Inc. | 2004-01-01 → 2007-12-14 | PS |
| `DPH` | Delphi Corp. | 2004-01-01 → 2005-10-10 | S |
| `DPHIQ` | — | 2004-01-01 → 2005-10-11 | P |
| `EDS` | Electronic Data Systems Corp | 2004-01-01 → 2008-08-26 | PS |
| `EOP` | Equity Office Properties Trust | 2004-01-01 → 2007-02-12 | PS |
| `FBF` | FleetBoston Financial Group Inc | 2004-01-01 → 2004-04-01 | PS |
| `FDC` | First Data Corp. | 2004-01-01 → 2007-09-25 | PS |
| `FHCC` | First Health Group Corp | 2004-01-01 → 2004-12-20 | N |
| `FMCN` | Focus Media Holding | 2007-12-24 → 2009-01-20 | N |
| `FSH` | Fisher Scientific Int'l | 2004-08-02 → 2006-11-10 | PS |
| `FSL` | — | 2004-12-03 → 2006-12-04 | P |
| `FSLB` | Freescale Semiconductor Inc | 2004-12-02 → 2006-12-01 | S |
| `GDW` | Golden West Financial | 2004-01-01 → 2006-10-02 | PS |
| `GLK` | Great Lakes Chemical Corp | 2004-01-01 → 2005-07-05 | PS |
| `GTW` | Gateway Inc. | 2004-01-01 → 2006-08-01 | PS |
| `HET` | — | 2004-01-01 → 2008-01-29 | P |
| `HPC` | Hercules Inc. | 2004-01-01 → 2008-11-14 | PS |
| `ISIL` | Intersil Corp | 2004-01-01 → 2005-12-19 | N |
| `JDSU` | JDS Uniphase Corp. | 2004-01-01 → 2006-12-18 | N |
| `JHF` | John Hancock Financial Services Inc | 2004-01-01 → 2004-04-29 | PS |
| `JNY` | Jones Group | 2004-01-01 → 2009-03-04 | PS |
| `KRB` | MBNA Corp. | 2004-01-01 → 2006-01-03 | PS |
| `KRI` | — | 2004-01-01 → 2006-06-28 | PS |
| `KSE` | KeySpan Corp | 2004-01-01 → 2007-08-27 | PS |
| `LEHMQ` | Lehman Brothers Holding | 2004-01-01 → 2008-09-17 | PS |
| `LNCR` | Lincare Holdings Inc | 2004-01-01 → 2006-12-18 | N |
| `LWIN` | Leap Wireless Int'l Inc | 2007-10-08 → 2008-12-22 | N |
| `MAY` | May Dept Stores | 2004-01-01 → 2005-08-29 | PS |
| `MCIP` | MCI, Inc. | 2004-12-20 → 2006-01-09 | N |
| `MEL` | Mellon Financial Corp | 2004-01-01 → 2007-07-02 | PS |
| `MLNM` | Millennium Pharmaceuticals Inc | 2004-01-01 → 2005-12-19 | N |
| `MTLQQ` | — | 2004-01-01 → 2009-06-03 | P |
| `MYG` | Maytag Corp. | 2004-01-01 → 2006-04-03 | PS |
| `NCC` | National City Corp. | 2004-01-01 → 2009-01-02 | PS |
| `NFB` | North Fork BanCorp | 2004-01-01 → 2006-12-01 | PS |
| `NXTL` | Nextel Communications Inc | 2004-01-01 → 2005-08-15 | NPS |
| `OMX` | OfficeMax Inc | 2004-01-01 → 2008-06-23 | PS |
| `PCS` | Sprint PCS Group | 2004-01-01 → 2004-04-23 | PS |
| `PGL` | Peoples Energy Corp | 2004-01-01 → 2007-02-22 | PS |
| `PIXR` | Pixar Inc | 2004-01-01 → 2006-05-08 | N |
| `PPDI` | Pharmaceutical Product Development | 2008-12-22 → 2009-12-21 | N |
| `PVN` | Providian Corp. | 2004-01-01 → 2005-10-03 | PS |
| `RBK` | Reebok Int'l Ltd | 2004-01-01 → 2006-02-01 | PS |
| `ROH` | Rohm & Hass Co | 2004-01-01 → 2009-04-02 | PS |
| `SBL` | Symbol Technologies Inc | 2004-01-01 → 2007-01-10 | PS |
| `SEBL` | Siebel Systems Inc. | 2004-01-01 → 2006-02-01 | NPS |
| `SEPR` | Sepracor Inc | 2005-08-15 → 2007-12-24 | N |
| `SFA` | — | 2004-01-01 → 2006-02-27 | PS |
| `SLR` | Solectron Corp. | 2004-01-01 → 2007-10-02 | PS |
| `SOTR` | SouthTrust Corp | 2004-01-01 → 2004-11-01 | PS |
| `TAP-B` | Travelers Property Casualty | 2004-01-01 → 2004-04-01 | S |
| `TIN` | — | 2004-01-01 → 2007-12-31 | PS |
| `TNB` | Thomas & Betts Corp | 2004-01-01 → 2004-08-03 | PS |
| `TOY` | Toys "R" Us, Inc. | 2004-01-01 → 2005-07-22 | PS |
| `TRB` | Tribune Media | 2004-01-01 → 2007-12-21 | P |
| `TXU` | TXU Gas Co. | 2004-01-01 → 2007-10-10 | PS |
| `UVN` | Univision Comm Inc | 2004-01-01 → 2007-03-29 | PS |
| `VSTNQ` | — | 2004-01-01 → 2006-01-03 | P |
| `WAMUQ` | — | 2004-01-01 → 2008-09-30 | P |
| `WFT` | Weatherford Int'l plc | 2005-07-21 → 2009-02-26 | PS |
| `WLP` | Wellpoint Health Networks Inc | 2004-01-01 → 2004-12-01 | PS |
| `WNDXQ` | — | 2004-01-01 → 2004-12-03 | P |
| `WWY` | Wrigley Co. | 2004-01-01 → 2008-10-07 | PS |
| `WYE` | Wyeth Corp. | 2004-01-01 → 2009-10-16 | PS |
| `XMSR` | XM Satellite Radio Holdings | 2004-12-20 → 2007-12-24 | N |

**left 2010–2014 — 39 names**

| ticker | company | membership window | index |
|---|---|---|---|
| `ANR` | American Natural Resource Co | 2011-06-01 → 2012-10-01 | S |
| `ANRZQ` | — | 2011-06-02 → 2012-10-02 | P |
| `AYE` | Allegheny Energy Inc. | 2004-01-01 → 2011-02-28 | PS |
| `BDK` | Black & Decker Manufacturing | 2004-01-01 → 2010-03-15 | PS |
| `BJS` | BJ Services Co. | 2004-01-01 → 2010-04-29 | PS |
| `BMC` | BMC Software Inc | 2004-01-01 → 2013-09-11 | NPS |
| `BTUUQ` | — | 2006-11-20 → 2014-09-22 | P |
| `CBE` | Cooper Industries plc | 2004-01-01 → 2012-12-03 | PS |
| `CEPH` | Cephalon Inc. | 2008-11-14 → 2011-10-14 | NPS |
| `CVH` | Coventry Corp | 2005-08-29 → 2013-05-07 | PS |
| `DF` | Dean Foods Co | 2006-03-31 → 2013-05-24 | PS |
| `EKDKQ` | — | 2004-01-01 → 2010-12-20 | P |
| `FRX` | Forest Laboratories Inc | 2004-01-01 → 2014-07-01 | PS |
| `FWLT` | Foster Wheeler AG | 2007-07-12 → 2010-12-20 | N |
| `GR` | Goodrich Corp. | 2004-01-01 → 2012-07-27 | PS |
| `HNZ` | H. J. Heinz Company | 2004-01-01 → 2013-06-07 | PS |
| `HSH` | Hillshire Brands Co | 2004-01-01 → 2012-06-29 | PS |
| `MEE` | Massey Energy Co. | 2008-06-20 → 2011-06-02 | PS |
| `MFE` | McAfee Inc. | 2008-12-22 → 2011-03-01 | PS |
| `MHS` | Marriott Corp. | 2004-01-01 → 2012-04-02 | PS |
| `MIL` | Millipore Corp | 2004-01-01 → 2010-07-15 | PS |
| `MOLX` | Molex Inc. | 2004-01-01 → 2013-12-09 | NPS |
| `MWW` | Monster Worldwide Inc. | 2004-01-01 → 2011-12-19 | NPS |
| `NOVL` | Novell Inc | 2004-01-01 → 2011-04-28 | PS |
| `NVLS` | Novellus Systems Inc | 2004-01-01 → 2012-06-05 | NPS |
| `NYX` | NYSE Euronext Inc | 2007-10-24 → 2013-11-13 | PS |
| `PBG` | Pepsi Bottling Group Inc. | 2004-01-01 → 2010-03-01 | PS |
| `PGN` | Progress Energy Inc. | 2004-01-01 → 2012-07-02 | PS |
| `PTV` | Pactiv Corp | 2004-01-01 → 2010-11-17 | PS |
| `RDC` | Rowan Companies plc | 2004-01-01 → 2014-08-19 | PS |
| `RSH` | RadioShack Corp. | 2004-01-01 → 2011-06-30 | S |
| `RSHCQ` | — | 2004-01-01 → 2011-07-01 | P |
| `RX` | IMS Health | 2004-01-01 → 2010-02-26 | P |
| `STRZA` | Starz Series | 2013-01-15 → 2013-03-18 | N |
| `TIE` | Titanium Metal Corp | 2007-10-26 → 2012-12-24 | PS |
| `TLAB` | Tellabs Inc. | 2004-01-01 → 2011-12-21 | NPS |
| `VMED` | Virgin Media Inc | 2004-12-20 → 2013-06-05 | N |
| `WCRX` | Warner Chilcott PLC | 2008-12-22 → 2012-12-24 | N |
| `XTO` | XTO Energy Inc. | 2004-12-28 → 2010-06-28 | PS |

**left 2015–2019 — 36 names**

| ticker | company | membership window | index |
|---|---|---|---|
| `BCR` | C. R. Bard Inc | 2004-01-01 → 2017-12-29 | PS |
| `BXLT` | Baxalta | 2015-06-30 → 2016-06-03 | PS |
| `CFN` | CareFusion Corp | 2009-08-31 → 2015-03-17 | PS |
| `CMCSK` | Comcast Corp | 2015-09-18 → 2015-12-14 | NPS |
| `COV` | Covidien Plc | 2007-06-29 → 2015-01-27 | PS |
| `CTRX` | Catamaran Corp. | 2012-12-24 → 2015-07-29 | N |
| `DPS` | Dr Pepper Snapple Group Inc | 2008-10-03 → 2018-06-29 | S |
| `DWDP` | DuPont | 2017-09-01 → 2019-06-03 | S |
| `ESV` | Ensco | 2007-01-04 → 2016-03-30 | P |
| `FDO` | Family Dollar Stores Inc | 2004-01-01 → 2015-07-07 | PS |
| `GAS` | Nicor Inc. | 2004-01-01 → 2016-07-01 | PS |
| `GGP` | GGP Inc | 2007-06-29 → 2018-08-28 | PS |
| `HAR` | Harvey Aluminium Inc | 2006-01-31 → 2017-03-13 | PS |
| `HCBK` | Hudson City Bancorp Inc | 2007-02-14 → 2015-11-02 | PS |
| `HRS` | Harris Corporation | 2008-09-22 → 2019-06-01 | P |
| `HSP` | Hospira Inc. | 2004-04-30 → 2015-09-03 | PS |
| `KORS` | Michael Kors | 2013-11-13 → 2018-09-19 | P |
| `KRFT` | Kraft Foods | 2012-10-02 → 2015-07-06 | P |
| `LMCA` | Liberty Media Class A | 2012-12-24 → 2016-06-20 | N |
| `LMCK` | Liberty Media C | 2014-12-22 → 2016-06-20 | N |
| `LO` | Lorillard Inc. | 2008-06-10 → 2015-06-12 | PS |
| `LVNTA` | Liberty Ventures | 2014-11-06 → 2018-03-08 | N |
| `MWV` | — | 2004-01-01 → 2015-07-02 | P |
| `PCL` | Plum Creek Timber Company | 2004-01-01 → 2016-02-22 | PS |
| `PETM` | PetSmart Inc | 2012-10-04 → 2015-03-12 | NPS |
| `RHT` | Red Hat Inc. | 2009-07-24 → 2019-07-15 | NPS |
| `SCG` | SCANA Corp. | 2008-12-31 → 2019-01-02 | PS |
| `SIAL` | Sigma-Aldrich | 2004-01-01 → 2015-11-18 | NPS |
| `SNI` | Scripps Networks Interactive Inc | 2008-06-30 → 2018-03-07 | PS |
| `SWY` | Safeway Inc. | 2004-01-01 → 2015-01-27 | PS |
| `TEG` | Integrys Energy Group | 2007-02-21 → 2015-06-30 | PS |
| `TMK` | — | 2004-01-01 → 2019-08-08 | P |
| `TWC` | Time Warner Cable Inc. | 2009-03-27 → 2016-05-18 | PS |
| `TYC` | Tyco International | 2004-01-01 → 2016-09-06 | S |
| `WIN` | Windstream Holdings Inc | 2006-07-17 → 2015-04-07 | PS |
| `YHOO` | Yahoo! Inc | 2004-01-01 → 2017-06-19 | NS |

### Ticker reused by a different company

**95 tickers** returned bars that do not overlap the membership window at all — the symbol was recycled after the original constituent left. **The point-in-time mask already excludes every one of these bars** (verified: zero survive `membership_mask`), so the panel is unaffected; they are listed because reading the archive *without* the mask would silently splice one company's prices onto another's history.

| ticker | company | coverage | bars | vendor history | membership window |
|---|---|---|---|---|---|
| `AABA` | — | 0% | 577 | 2017-06-19 → 2019-10-02 | 2004-01-01 → 2017-06-19 |
| `PLL` | Pall Corp. | 0% | 1926 | 2018-01-02 → 2025-08-29 | 2004-01-01 → 2015-08-31 |
| `PD` | Phelps Dodge Corp | 0% | 1832 | 2019-04-11 → 2026-07-27 | 2004-01-01 → 2007-03-20 |
| `PCP` | Precision Castparts Corp | 0% | 194 | 2018-08-28 → 2019-06-07 | 2007-05-31 → 2016-02-01 |
| `ONE` | Bank One Corp. | 0% | 971 | 2018-03-28 → 2022-04-29 | 2004-01-01 → 2004-07-01 |
| `NSM` | National Semiconductor | 0% | 1609 | 2012-03-09 → 2018-07-31 | 2004-01-01 → 2011-09-26 |
| `NE` | Noble Corp plc | 0% | 1288 | 2021-06-09 → 2026-07-27 | 2004-01-01 → 2015-07-20 |
| `NBL` | Noble Energy Inc. | 0% | 7 | 2021-01-19 → 2021-01-27 | 2007-10-05 → 2020-10-12 |
| `NAV` | Navistar Int'l Corp | 0% | 3274 | 2008-06-30 → 2021-06-30 | 2004-01-01 → 2006-12-20 |
| `MON` | Monsanto Co. | 0% | 356 | 2021-03-16 → 2022-12-23 | 2004-01-01 → 2018-06-07 |
| `MNK` | Mallinckrodt plc | 0% | 223 | 2022-10-28 → 2023-09-19 | 2014-08-18 → 2017-07-26 |
| `POM` | Pepco Holdings Inc. | 0% | 200 | 2025-10-08 → 2026-07-27 | 2007-11-08 → 2016-03-24 |
| `MMI` | Motorola Mobility | 0% | 3201 | 2013-10-31 → 2026-07-27 | 2011-01-03 → 2012-05-22 |
| `MI` | Marshall & Ilsley Corp | 0% | 2680 | 2015-11-25 → 2026-07-27 | 2004-01-01 → 2011-07-06 |
| `MERQ` | Mercury Interactive Corp | 0% | 231 | 2018-08-28 → 2025-08-04 | 2004-01-01 → 2006-01-04 |
| `MER` | Merrill Lynch & Co | 0% | 231 | 2018-08-28 → 2025-08-04 | 2004-01-01 → 2009-01-02 |
| `MEDI` | MedImmune Inc. | 0% | 923 | 2022-11-17 → 2026-07-27 | 2004-01-01 → 2007-06-01 |
| `LU` | Lucent Technology | 0% | 1439 | 2020-10-30 → 2026-07-27 | 2004-01-01 → 2006-12-01 |
| `LLL` | L3 Technologies Inc | 0% | 249 | 2021-10-13 → 2022-10-07 | 2004-11-30 → 2019-07-01 |
| `LIFE` | Life Technologies Corp | 0% | 123 | 2026-01-29 → 2026-07-27 | 2008-11-21 → 2014-01-24 |
| `LB` | — | 0% | 520 | 2024-06-28 → 2026-07-27 | 2004-01-01 → 2021-08-03 |
| `KODK` | Eastman Kodak Co. | 0% | 3229 | 2013-09-23 → 2026-07-27 | 2004-01-01 → 2010-12-17 |
| `WYND` | Wyndham Destinations Inc | 0% | 682 | 2018-06-01 → 2021-02-16 | 2006-07-31 → 2018-05-31 |
| `MICC` | Millicom Int'l Cellular | 0% | 158 | 2025-12-08 → 2026-07-27 | 2006-05-08 → 2011-05-27 |
| `KATE` | Kate Spade & Co | 0% | 850 | 2014-02-26 → 2017-07-11 | 2004-01-01 → 2008-12-02 |
| `PSFT` | People Soft Inc | 0% | 3704 | 2008-04-10 → 2023-05-23 | 2004-01-01 → 2004-12-29 |
| `PX` | — | 0% | 1121 | 2021-10-21 → 2026-04-10 | 2004-01-01 → 2018-10-31 |
| `WB` | Wachovia Corp. | 0% | 3086 | 2014-04-17 → 2026-07-27 | 2004-01-01 → 2009-01-02 |
| `VSNT` | — | 0% | 153 | 2025-12-15 → 2026-07-27 | 2004-01-01 → 2025-01-08 |
| `VRTS` | Veritas Software Corp | 0% | 4417 | 2009-01-02 → 2026-07-27 | 2004-01-01 → 2005-07-05 |
| `VC` | Visteon Corp. | 0% | 3981 | 2010-09-27 → 2026-07-27 | 2004-01-01 → 2005-12-30 |
| `VAL` | Valaris plc | 0% | 1314 | 2021-05-03 → 2026-07-27 | 2007-01-03 → 2016-03-29 |
| `UST` | UST Inc. | 0% | 4152 | 2010-01-22 → 2026-07-27 | 2004-01-01 → 2009-01-06 |
| `UPC` | Union Planters Corp. | 0% | 1342 | 2021-03-23 → 2026-07-27 | 2004-01-01 → 2004-07-01 |
| `UCL` | Unocal Co | 0% | 1539 | 2020-06-10 → 2026-07-27 | 2004-01-01 → 2005-08-11 |
| `TRCO` | Tribune Media Co | 0% | 1204 | 2014-12-05 → 2019-09-18 | 2004-01-01 → 2007-12-20 |
| `TEK` | Tektronix Inc. | 0% | 440 | 2024-10-22 → 2026-07-27 | 2004-01-01 → 2007-11-16 |
| `PWER` | — | 0% | 665 | 2023-11-29 → 2026-07-27 | 2004-01-01 → 2005-03-14 |
| `TE` | TECO Energy Inc | 0% | 1642 | 2020-01-13 → 2026-07-27 | 2004-01-01 → 2016-07-01 |
| `STI` | SunTrust Banks Inc | 0% | 1062 | 2022-05-02 → 2026-07-27 | 2004-01-01 → 2019-12-09 |
| `SPOT` | Panamsat Inc | 0% | 2090 | 2018-04-03 → 2026-07-27 | 2004-01-01 → 2004-08-19 |
| `SHLD` | Sears Holdings Corp | 0% | 719 | 2023-09-13 → 2026-07-27 | 2005-03-24 → 2012-09-05 |
| `SGP` | Schering-Plough | 0% | 117 | 2026-02-06 → 2026-07-27 | 2004-01-01 → 2009-11-04 |
| `SE` | Spectra Energy Corp. | 0% | 2201 | 2017-10-20 → 2026-07-27 | 2006-12-29 → 2017-02-27 |
| `SDS` | Sungard Data Systems Inc | 0% | 5040 | 2006-07-13 → 2026-07-27 | 2004-01-01 → 2005-08-12 |
| `SAF` | Safeco Corp. | 0% | 756 | 2018-08-29 → 2021-08-30 | 2004-01-01 → 2008-09-23 |
| `S` | Sears Roebuck & Co | 0% | 1273 | 2021-06-30 → 2026-07-27 | 2004-01-01 → 2013-07-09 |
| `RMG` | Cablevision Systems Corp. | 0% | 819 | 2008-01-25 → 2020-12-29 | 2010-12-17 → 2016-06-23 |
| `RLGY` | Realogy Holdings Corp | 0% | 2430 | 2012-10-11 → 2022-06-08 | 2006-07-31 → 2007-04-09 |
| `STR` | Questar Corp. | 0% | 1996 | 2017-09-08 → 2025-08-18 | 2006-11-30 → 2010-07-01 |
| `JP` | — | 0% | 1719 | 2015-07-16 → 2022-06-24 | 2004-01-01 → 2006-04-03 |
| `KG` | King Pharmaceuticals Inc. | 0% | 303 | 2025-05-12 → 2026-07-27 | 2004-01-01 → 2010-12-20 |
| `JAVA` | Sun Microsystems | 0% | 1206 | 2021-10-05 → 2026-07-27 | 2004-01-01 → 2010-01-27 |
| `COOP` | Mr. Cooper Group Inc | 0% | 3399 | 2012-03-28 → 2025-10-02 | 2004-01-01 → 2008-09-29 |
| `CHK` | Chesapeake Energy Corp | 0% | 936 | 2021-02-10 → 2024-10-29 | 2006-03-02 → 2018-03-19 |
| `CC` | Circuit City Stores Inc. | 0% | 2791 | 2015-06-19 → 2026-07-27 | 2004-01-01 → 2008-03-28 |
| `CBH` | Commerce Bancorp (New Jersey) | 0% | 1806 | 2017-06-28 → 2024-08-30 | 2006-06-05 → 2008-03-31 |
| `CAM` | Cameron International Corp | 0% | 202 | 2025-10-06 → 2026-07-27 | 2008-01-28 → 2016-04-04 |
| `CA` | CA Inc | 0% | 611 | 2023-12-15 → 2026-05-26 | 2004-01-01 → 2018-11-06 |
| `BUD` | Anheuser Busch | 0% | 4293 | 2009-07-01 → 2026-07-27 | 2004-01-01 → 2008-11-18 |
| `BTU` | Peabody Energy Corp | 0% | 2341 | 2017-04-03 → 2026-07-27 | 2006-11-17 → 2014-09-19 |
| `BOL` | Bausch & Lomb Inc. | 0% | 3 | 2018-05-16 → 2018-05-18 | 2004-01-01 → 2007-10-29 |
| `BEAM` | Beam Suntory Inc | 0% | 1625 | 2020-02-06 → 2026-07-27 | 2004-01-01 → 2014-05-01 |
| `ASO` | AmSouth Bancorp | 0% | 1459 | 2020-10-02 → 2026-07-27 | 2004-01-01 → 2006-11-06 |
| `APC` | Alpha Portland Inds Inc | 0% | 115 | 2026-02-10 → 2026-07-27 | 2004-01-01 → 2019-08-09 |
| `AMBC` | Ambac Financial Group Inc. | 0% | 3199 | 2013-05-01 → 2026-01-16 | 2004-01-01 → 2008-06-10 |
| `AM` | Armour & Co | 0% | 2320 | 2017-05-03 → 2026-07-27 | 2004-01-01 → 2004-05-03 |
| `ALTR` | Altera Corp. | 0% | 1859 | 2017-11-01 → 2025-03-26 | 2004-01-01 → 2015-12-28 |
| `ADT` | ADT Corp. | 0% | 2140 | 2018-01-19 → 2026-07-27 | 2012-09-28 → 2016-05-02 |
| `ADCT` | ADC Telecommunications Inc | 0% | 1556 | 2020-05-15 → 2026-07-27 | 2004-01-01 → 2007-07-02 |
| `ACV` | — | 0% | 2811 | 2015-05-21 → 2026-07-27 | 2004-01-01 → 2006-11-17 |
| `ACS` | American Crystal Sugar Co | 0% | 2 | 2018-02-20 → 2018-02-23 | 2004-04-01 → 2010-02-08 |
| `ACE` | Chubb Corp. | 0% | 1 | 2020-01-24 → 2020-01-24 | 2004-01-01 → 2016-01-15 |
| `ABI` | Applied Biosystems Inc. | 0% | 272 | 2025-06-26 → 2026-07-27 | 2004-01-01 → 2008-11-24 |
| `CPN` | Calpine Corp. | 0% | 2553 | 2008-01-17 → 2018-03-08 | 2004-01-01 → 2005-12-01 |
| `DAN` | Dana Inc | 0% | 4670 | 2008-01-02 → 2026-07-27 | 2004-01-01 → 2006-03-02 |
| `CHIR` | Chiron Corp. | 0% | 1317 | 2018-12-11 → 2025-06-11 | 2004-01-01 → 2006-04-20 |
| `DNB` | Dun & Bradstreet Corp | 0% | 1294 | 2020-07-01 → 2025-08-25 | 2008-12-01 → 2017-04-05 |
| `INFO` | IHS Markit Ltd. | 0% | 448 | 2024-10-10 → 2026-07-27 | 2017-06-01 → 2022-03-02 |
| `IMS` | IMS Health Holdings Inc | 0% | 629 | 2014-04-04 → 2016-09-30 | 2004-01-01 → 2010-02-26 |
| `IHRT` | iHeartMedia Inc | 0% | 1809 | 2019-05-07 → 2026-07-27 | 2004-01-01 → 2008-07-30 |
| `HMA` | Health Management Associates Inc | 0% | 342 | 2022-03-15 → 2023-07-25 | 2004-01-01 → 2007-03-02 |
| `HCP` | — | 0% | 807 | 2021-12-09 → 2025-02-27 | 2008-03-31 → 2019-11-05 |
| `H` | — | 0% | 4204 | 2009-11-05 → 2026-07-27 | 2006-08-01 → 2007-04-10 |
| `GP` | — | 0% | 2799 | 2015-02-10 → 2026-07-27 | 2004-01-01 → 2005-12-20 |
| `GOLD` | Randgold Resources | 0% | 3109 | 2014-03-17 → 2026-07-27 | 2011-12-19 → 2013-11-18 |
| `GDT` | Guidant Corp | 0% | 128 | 2026-01-22 → 2026-07-27 | 2004-01-01 → 2006-04-24 |
| `G` | Gillette Co. | 0% | 4775 | 2007-08-02 → 2026-07-27 | 2004-01-01 → 2005-10-03 |
| `EQ` | Embarq Corporation | 0% | 1955 | 2018-10-12 → 2026-07-27 | 2006-05-17 → 2009-07-01 |
| `XL` | XL Group Ltd | 0% | 477 | 2020-12-22 → 2022-11-11 | 2004-01-01 → 2018-09-12 |
| `DNR` | Denbury Resources Inc. | 0% | 34 | 2020-08-04 → 2020-09-23 | 2009-04-01 → 2015-03-23 |
| `DO` | Diamond Offshore Drilling Inc | 0% | 609 | 2022-03-31 → 2024-09-03 | 2009-02-25 → 2016-10-03 |
| `EC` | Engelhard Corp. | 0% | 4490 | 2008-09-18 → 2026-07-27 | 2004-01-01 → 2006-06-06 |
| `EMC` | EMC Corp. | 0% | 802 | 2023-05-15 → 2026-07-27 | 2004-01-01 → 2016-09-07 |
| `DYN` | Dynegy Inc. | 0% | 1470 | 2020-09-17 → 2026-07-27 | 2004-01-01 → 2009-12-21 |

### Priced, but for under half of their membership window

The vendor's history starts after the name joined the index. These bars are real and correctly masked; the earlier part of the spell is simply absent.

| ticker | company | coverage | bars | vendor history | membership window |
|---|---|---|---|---|---|
| `ARNC` | Arconic | 0% | 852 | 2020-03-31 → 2023-08-17 | 2004-01-01 → 2020-04-06 |
| `SUN` | Sunoco Inc | 0% | 3480 | 2012-09-20 → 2026-07-27 | 2004-01-01 → 2012-10-05 |
| `SOLS` | Solstice Advanced Materials | 1% | 192 | 2025-10-20 → 2026-07-27 | 2004-01-01 → 2025-12-22 |
| `SNDK` | Sandisk Corporation | 5% | 363 | 2025-02-13 → 2026-07-27 | 2006-04-19 → 2026-07-27 |
| `Q` | Qnity Electronics, Inc. | 9% | 187 | 2025-10-27 → 2026-07-27 | 2004-01-01 → 2026-07-27 |
| `DELL` | Dell Technologies | 11% | 2498 | 2016-08-17 → 2026-07-27 | 2004-01-01 → 2026-07-27 |
| `SII` | Smith Int'l Inc | 13% | 4132 | 2010-02-22 → 2026-07-27 | 2006-09-29 → 2010-08-27 |
| `BHGE` | — | 14% | 577 | 2017-07-05 → 2019-10-17 | 2004-01-01 → 2019-10-18 |
| `NLOK` | NortonLifeLock Inc | 16% | 758 | 2019-11-05 → 2022-11-07 | 2004-01-01 → 2022-11-08 |
| `FOXA` | Fox Corporation (Class A) | 30% | 1864 | 2019-02-26 → 2026-07-27 | 2004-12-17 → 2026-07-27 |
| `DOW` | Dow Inc. | 34% | 1848 | 2019-03-20 → 2026-07-27 | 2004-01-01 → 2026-07-27 |
| `GENZ` | Genzyme Corp. | 43% | 4655 | 2008-01-24 → 2026-07-27 | 2004-01-01 → 2011-04-04 |
| `CEG` | Constellation Energy | 43% | 1133 | 2022-01-19 → 2026-07-27 | 2004-01-01 → 2026-07-27 |
| `ECHO` | EchoStar Corporation | 47% | 43 | 2026-05-26 → 2026-07-27 | 2026-03-23 → 2026-07-27 |
| `JOY` | Joy Manufacturing Co | 49% | 1341 | 2011-12-06 → 2017-04-05 | 2011-02-25 → 2015-10-08 |
